package lk.ac.rjt.grasp.ingest;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import lk.ac.rjt.grasp.config.GraspProperties;
import lk.ac.rjt.grasp.model.Chunk;
import lk.ac.rjt.grasp.model.ParsedDocument.Paragraph;
import org.junit.jupiter.api.Test;

class ChunkerTest {

    private final GraspProperties props = new GraspProperties();
    private final Chunker chunker = new Chunker(props);

    private static String words(int n, String word) {
        return String.join(" ", Collections.nCopies(n, word));
    }

    @Test
    void chunksStayWithinTokenBudget() {
        List<Paragraph> paragraphs = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            paragraphs.add(new Paragraph("2 Method", 2 + i / 10, words(60, "token") + "."));
        }
        List<Chunk> chunks = chunker.chunk(paragraphs);

        assertThat(chunks).hasSizeGreaterThan(2);
        for (Chunk c : chunks) {
            assertThat(c.approxTokens()).isLessThanOrEqualTo(props.getChunking().getMaxTokens() + props.getChunking().getMinTokens() / 2);
        }
        assertThat(chunks.subList(0, chunks.size() - 1)).allMatch(c -> c.approxTokens() >= props.getChunking().getMinTokens());
    }

    @Test
    void idsAreSequentialAndPagesTracked() {
        List<Paragraph> paragraphs = List.of(
                new Paragraph("1 Introduction", 1, words(250, "intro") + "."),
                new Paragraph("1 Introduction", 2, words(250, "intro") + "."),
                new Paragraph("2 Method", 3, words(350, "method") + "."));
        List<Chunk> chunks = chunker.chunk(paragraphs);

        for (int i = 0; i < chunks.size(); i++) {
            assertThat(chunks.get(i).id()).isEqualTo("c" + i);
            assertThat(chunks.get(i).index()).isEqualTo(i);
        }
        assertThat(chunks.getFirst().section()).isEqualTo("1 Introduction");
        assertThat(chunks.getFirst().pageStart()).isEqualTo(1);
        assertThat(chunks.getLast().section()).isEqualTo("2 Method");
        assertThat(chunks.getLast().pageEnd()).isEqualTo(3);
    }

    @Test
    void splitsOversizedParagraphAtSentences() {
        String sentence = words(40, "result") + ". ";
        String huge = sentence.repeat(30);
        List<Chunk> chunks = chunker.chunk(List.of(new Paragraph("3 Results", 5, huge)));
        assertThat(chunks).hasSizeGreaterThan(1);
        assertThat(chunks).allMatch(c -> c.approxTokens() <= props.getChunking().getMaxTokens() + props.getChunking().getMinTokens() / 2);
    }

    @Test
    void shortSectionsAreMergedRatherThanLeftTiny() {
        List<Paragraph> paragraphs = List.of(
                new Paragraph("Abstract", 1, words(120, "abstract") + "."),
                new Paragraph("1 Introduction", 1, words(120, "intro") + "."));
        List<Chunk> chunks = chunker.chunk(paragraphs);
        assertThat(chunks).hasSize(1);
        assertThat(chunks.getFirst().section()).isEqualTo("Abstract");
    }
}
