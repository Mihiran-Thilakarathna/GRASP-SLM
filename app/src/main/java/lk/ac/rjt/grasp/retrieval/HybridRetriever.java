package lk.ac.rjt.grasp.retrieval;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.scoring.ScoringModel;
import lk.ac.rjt.grasp.config.GraspProperties;
import lk.ac.rjt.grasp.config.SharedResources;
import lk.ac.rjt.grasp.model.Chunk;
import lk.ac.rjt.grasp.model.ScoredChunk;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.stereotype.Component;

/**
 * Field-specific retrieval over one paper: BM25 and/or dense search, Reciprocal Rank Fusion,
 * optional cross-encoder re-ranking, then a round-robin pick across fields so that every field
 * gets evidence before any field gets a second chunk.
 */
@Component
public class HybridRetriever {

    private static final Logger log = LoggerFactory.getLogger(HybridRetriever.class);

    private final EmbeddingModel embedder;
    private final ObjectProvider<ScoringModel> reranker;
    private final List<FieldQuery> fieldQueries;
    private final GraspProperties.Retrieval cfg;

    public HybridRetriever(EmbeddingModel embedder, ObjectProvider<ScoringModel> reranker,
                           SharedResources shared, GraspProperties props) {
        this.embedder = embedder;
        this.reranker = reranker;
        this.fieldQueries = shared.fieldQueries();
        this.cfg = props.getRetrieval();
    }

    public RetrievalOptions defaults() {
        return new RetrievalOptions(cfg.getMode(), cfg.isRerank(), cfg.getMaxContextChunks());
    }

    public RetrievalResult retrieve(List<Chunk> chunks, RetrievalOptions options) {
        ScoringModel scorer = options.rerank() ? reranker.getIfAvailable() : null;
        if (options.rerank() && scorer == null) {
            log.warn("Re-ranking requested but no re-ranker model is configured; continuing without it");
        }
        Map<String, List<ScoredChunk>> rankings = new LinkedHashMap<>();
        try (PaperIndex index = PaperIndex.build(chunks, embedder)) {
            for (FieldQuery q : fieldQueries) {
                List<ScoredChunk> ranking = rank(index, q, options.mode());
                if (scorer != null) {
                    ranking = rerank(scorer, q, ranking);
                }
                rankings.put(q.field(), ranking);
            }
        }
        return new RetrievalResult(options, scorer != null, rankings, select(rankings, options.maxContextChunks()));
    }

    private List<ScoredChunk> rank(PaperIndex index, FieldQuery q, RetrievalMode mode) {
        int n = cfg.getCandidatesPerRetriever();
        List<ScoredChunk> ranking = switch (mode) {
            case BM25 -> index.bm25(q.bm25(), n);
            case DENSE -> index.dense(queryVector(q), n);
            case HYBRID -> ReciprocalRankFusion.fuse(cfg.getRrfK(),
                    List.of(index.bm25(q.bm25(), n), index.dense(queryVector(q), n)));
        };
        return ranking.subList(0, Math.min(ranking.size(), cfg.getRerankCandidates()));
    }

    private float[] queryVector(FieldQuery q) {
        return embedder.embed(cfg.getQueryInstruction() + q.dense()).content().vector();
    }

    private static List<ScoredChunk> rerank(ScoringModel scorer, FieldQuery q, List<ScoredChunk> candidates) {
        if (candidates.isEmpty()) {
            return candidates;
        }
        List<TextSegment> segments = candidates.stream().map(s -> TextSegment.from(s.chunk().text())).toList();
        List<Double> scores = scorer.scoreAll(segments, q.dense()).content();
        List<ScoredChunk> out = new ArrayList<>();
        for (int i = 0; i < candidates.size(); i++) {
            out.add(new ScoredChunk(candidates.get(i).chunk(), scores.get(i)));
        }
        out.sort(Comparator.comparingDouble(ScoredChunk::score).reversed());
        return out;
    }

    static List<Chunk> select(Map<String, List<ScoredChunk>> rankings, int max) {
        Set<Chunk> picked = new LinkedHashSet<>();
        boolean progress = true;
        while (picked.size() < max && progress) {
            progress = false;
            for (List<ScoredChunk> ranking : rankings.values()) {
                if (picked.size() >= max) {
                    break;
                }
                // Each pass, take this field's best chunk that has not been picked yet.
                for (int i = 0; i < ranking.size(); i++) {
                    if (picked.add(ranking.get(i).chunk())) {
                        progress = true;
                        break;
                    }
                }
            }
        }
        return picked.stream().sorted(Comparator.comparingInt(Chunk::index)).toList();
    }
}
