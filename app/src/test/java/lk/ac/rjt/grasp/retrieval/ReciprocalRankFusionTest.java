package lk.ac.rjt.grasp.retrieval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.List;

import lk.ac.rjt.grasp.model.Chunk;
import lk.ac.rjt.grasp.model.ScoredChunk;
import org.junit.jupiter.api.Test;

class ReciprocalRankFusionTest {

    private static Chunk chunk(int i) {
        return new Chunk(Chunk.idFor(i), i, "S", 1, 1, "text " + i, 10);
    }

    @Test
    void documentRankedHighByBothListsWins() {
        Chunk a = chunk(0), b = chunk(1), c = chunk(2);
        List<ScoredChunk> bm25 = List.of(new ScoredChunk(b, 9), new ScoredChunk(a, 5), new ScoredChunk(c, 1));
        List<ScoredChunk> dense = List.of(new ScoredChunk(a, 0.9), new ScoredChunk(b, 0.8));

        List<ScoredChunk> fused = ReciprocalRankFusion.fuse(60, List.of(bm25, dense));

        assertThat(fused).extracting(s -> s.chunk().id()).containsExactly("c0", "c1", "c2");
        assertThat(fused.getFirst().score()).isCloseTo(1.0 / 62 + 1.0 / 61, within(1e-12));
        assertThat(fused.getLast().score()).isCloseTo(1.0 / 63, within(1e-12));
    }

    @Test
    void emptyInputGivesEmptyRanking() {
        assertThat(ReciprocalRankFusion.fuse(60, List.of(List.of(), List.of()))).isEmpty();
    }
}
