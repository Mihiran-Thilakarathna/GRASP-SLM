package lk.ac.rjt.grasp.retrieval;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.model.scoring.ScoringModel;
import lk.ac.rjt.grasp.config.GraspProperties;
import lk.ac.rjt.grasp.config.SharedResources;
import lk.ac.rjt.grasp.model.Chunk;
import lk.ac.rjt.grasp.model.ScoredChunk;
import lk.ac.rjt.grasp.testsupport.Fakes;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.support.StaticListableBeanFactory;

class HybridRetrieverTest {

    private static final List<Chunk> CHUNKS = List.of(
            chunk(0, "Abstract", "Fine-tuning large models is costly. This problem limits question answering for small labs."),
            chunk(1, "2 Method", "We propose sparse adapters, a novel method that inserts small matrices into attention layers."),
            chunk(2, "3 Results", "The main results: our model achieves an F1 score of 92.7 on the SQuAD dataset and outperforms the baseline by 1.3 points."),
            chunk(3, "4 Limitations", "A limitation is that we only evaluate English data; future work will study other languages."),
            chunk(4, "Appendix", "Hyper-parameters are listed for completeness with batch size 32 and six epochs."));

    private static Chunk chunk(int i, String section, String text) {
        return new Chunk(Chunk.idFor(i), i, section, i + 1, i + 1, text, 30);
    }

    private HybridRetriever retriever(ScoringModel scorer) {
        StaticListableBeanFactory beans = new StaticListableBeanFactory();
        if (scorer != null) {
            beans.addBean("scorer", scorer);
        }
        return new HybridRetriever(new Fakes.BagOfWordsEmbedding(), beans.getBeanProvider(ScoringModel.class),
                new SharedResources(new ObjectMapper()), new GraspProperties());
    }

    @Test
    void bm25FindsTheChunkWithResults() {
        try (PaperIndex index = PaperIndex.build(CHUNKS, new Fakes.BagOfWordsEmbedding())) {
            List<ScoredChunk> hits = index.bm25("F1 score results outperforms baseline", 3);
            assertThat(hits.getFirst().chunk().id()).isEqualTo("c2");
        }
    }

    @Test
    void denseSearchReturnsNearestChunk() {
        try (PaperIndex index = PaperIndex.build(CHUNKS, new Fakes.BagOfWordsEmbedding())) {
            float[] q = new Fakes.BagOfWordsEmbedding().embed("limitation English future work languages").content().vector();
            assertThat(index.dense(q, 2).getFirst().chunk().id()).isEqualTo("c3");
        }
    }

    @Test
    void everyFieldGetsEvidenceAndContextIsInDocumentOrder() {
        RetrievalResult r = retriever(new Fakes.OverlapScorer())
                .retrieve(CHUNKS, new RetrievalOptions(RetrievalMode.HYBRID, true, 4));

        assertThat(r.reranked()).isTrue();
        assertThat(r.rankings()).containsOnlyKeys("core_problem", "proposed_method", "key_metrics", "limitations");
        assertThat(r.context()).hasSize(4);
        assertThat(r.context()).extracting(Chunk::index).isSorted();
        assertThat(r.context()).extracting(Chunk::id).contains("c2", "c3");
    }

    @Test
    void rerankIsSkippedWhenNoModelIsAvailable() {
        RetrievalResult r = retriever(null).retrieve(CHUNKS, new RetrievalOptions(RetrievalMode.DENSE, true, 6));
        assertThat(r.reranked()).isFalse();
        assertThat(r.context()).hasSizeLessThanOrEqualTo(5);
    }

    @Test
    void roundRobinSelectionTakesEachFieldsBestFirst() {
        Map<String, List<ScoredChunk>> rankings = new LinkedHashMap<>();
        rankings.put("a", List.of(new ScoredChunk(CHUNKS.get(4), 1), new ScoredChunk(CHUNKS.get(0), 0.5)));
        rankings.put("b", List.of(new ScoredChunk(CHUNKS.get(4), 1), new ScoredChunk(CHUNKS.get(1), 0.5)));
        rankings.put("c", List.of(new ScoredChunk(CHUNKS.get(2), 1)));

        List<Chunk> picked = HybridRetriever.select(rankings, 3);

        assertThat(picked).extracting(Chunk::id).containsExactly("c1", "c2", "c4");
    }
}
