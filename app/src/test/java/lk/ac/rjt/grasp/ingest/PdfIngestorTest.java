package lk.ac.rjt.grasp.ingest;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import lk.ac.rjt.grasp.model.ParsedDocument;
import lk.ac.rjt.grasp.model.ParsedDocument.Paragraph;
import lk.ac.rjt.grasp.testsupport.TestPdfs;
import org.junit.jupiter.api.Test;

class PdfIngestorTest {

    private final PdfIngestor ingestor = new PdfIngestor();

    @Test
    void extractsTitleSectionsAndDropsReferences() throws Exception {
        ParsedDocument doc = ingestor.parse(TestPdfs.samplePaper());

        assertThat(doc.title()).isEqualTo(TestPdfs.TITLE);
        assertThat(doc.numPages()).isGreaterThanOrEqualTo(1);
        List<String> sections = doc.paragraphs().stream().map(Paragraph::section).distinct().toList();
        assertThat(sections).containsSubsequence("Abstract", "1 Introduction", "2 Method", "3 Results", "4 Limitations");
        assertThat(sections).doesNotContain("References");
        assertThat(doc.paragraphs()).noneMatch(p -> p.text().contains("Something about adapters"));
    }

    @Test
    void keepsNumbersExactly() throws Exception {
        ParsedDocument doc = ingestor.parse(TestPdfs.samplePaper());
        String results = String.join(" ", doc.paragraphs().stream().filter(p -> p.section().equals("3 Results"))
                .map(Paragraph::text).toList());
        assertThat(results).contains("87.4").contains("92.7").contains("SQuAD v1.1");
    }

    @Test
    void repairsLineBreakHyphenation() {
        StringBuilder sb = new StringBuilder("an exam-");
        PdfIngestor.append(sb, "ple of text");
        assertThat(sb.toString()).isEqualTo("an example of text");

        StringBuilder keep = new StringBuilder("state-of-the-");
        PdfIngestor.append(keep, "Art");
        assertThat(keep.toString()).isEqualTo("state-of-the- Art");
    }

    private static PdfIngestor.Line line(String text, int page, float size, boolean bold, boolean italic) {
        return new PdfIngestor.Line(text, page, size, bold, italic, true);
    }

    @Test
    void detectsHeadings() {
        float body = 10f;
        assertThat(PdfIngestor.isHeading(line("Introduction", 1, 10f, false, false), body)).isTrue();
        assertThat(PdfIngestor.isHeading(line("3.2 Training Details", 4, 11f, true, false), body)).isTrue();
        assertThat(PdfIngestor.isHeading(line("3 We train the model for ten epochs.", 4, 10f, false, false), body)).isFalse();
        assertThat(PdfIngestor.isHeading(line("This is an ordinary sentence in the body text of the paper.", 2, 10f, false, false), body)).isFalse();
    }

    @Test
    void detectsIeeeStyleHeadings() {
        float body = 9f;
        // Small-caps section heading, smaller than body text and not bold.
        assertThat(PdfIngestor.isHeading(line("II. SYSTEM MODEL", 2, 7.7f, false, false), body)).isTrue();
        // Italic lettered subsection heading at body size.
        assertThat(PdfIngestor.isHeading(line("B. Our Approach and Contribution", 2, 9f, false, true), body)).isTrue();
        assertThat(PdfIngestor.isHeading(line("B. Our Approach and Contribution", 2, 9f, false, false), body)).isFalse();
    }

    @Test
    void dropsTextDrawnInsideFigures() {
        float body = 9f;
        assertThat(PdfIngestor.isFigureText(line("Silence", 2, 18f, false, false), body)).isTrue();
        assertThat(PdfIngestor.isFigureText(line("ai", 3, 9f, false, false), body)).isTrue();
        assertThat(PdfIngestor.isFigureText(line("t1 t2 t3", 3, 5f, false, false), body)).isTrue();
        assertThat(PdfIngestor.isFigureText(line("Probabilistic Accumulate-then-Transmit", 1, 23f, false, false), body)).isFalse();
        assertThat(PdfIngestor.isFigureText(line("the transmitter sends its information", 3, 9f, false, false), body)).isFalse();
    }

    @Test
    void cleanRemovesUnmappedSymbolGlyphs() {
        assertThat(PdfIngestor.clean(" x  Abstract�In")).isEqualTo("x Abstract In");
        assertThat(PdfIngestor.clean("ﬁne-tuning")).isEqualTo("fine-tuning");
    }
}
