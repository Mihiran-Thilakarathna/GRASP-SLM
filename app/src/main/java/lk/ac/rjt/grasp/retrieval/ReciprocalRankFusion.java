package lk.ac.rjt.grasp.retrieval;

import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import lk.ac.rjt.grasp.model.Chunk;
import lk.ac.rjt.grasp.model.ScoredChunk;

/** Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum over rankings of 1 / (k + rank(d)). */
public final class ReciprocalRankFusion {

    private ReciprocalRankFusion() {
    }

    public static List<ScoredChunk> fuse(int k, List<List<ScoredChunk>> rankings) {
        Map<Chunk, Double> scores = new LinkedHashMap<>();
        for (List<ScoredChunk> ranking : rankings) {
            for (int rank = 0; rank < ranking.size(); rank++) {
                scores.merge(ranking.get(rank).chunk(), 1.0 / (k + rank + 1), Double::sum);
            }
        }
        return scores.entrySet().stream()
                .map(e -> new ScoredChunk(e.getKey(), e.getValue()))
                .sorted(Comparator.comparingDouble(ScoredChunk::score).reversed()
                        .thenComparingInt(s -> s.chunk().index()))
                .toList();
    }
}
