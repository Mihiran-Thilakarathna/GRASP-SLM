package lk.ac.rjt.grasp.testsupport;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.apache.pdfbox.pdmodel.font.Standard14Fonts;

/** Builds a small, realistic research-paper PDF for tests. */
public final class TestPdfs {

    public static final String TITLE = "Sparse Adapters for Efficient Question Answering";

    private TestPdfs() {
    }

    public static byte[] samplePaper() throws IOException {
        List<Block> blocks = new ArrayList<>();
        blocks.add(new Block(TITLE, 18, true));
        blocks.add(new Block("Abstract", 12, true));
        blocks.add(body(repeat("Large language models are expensive to adapt and existing fine-tuning methods "
                + "update every parameter, which makes question answering systems costly to build.", 3)));
        blocks.add(new Block("1 Introduction", 12, true));
        blocks.add(body(repeat("The core problem we address is the high memory cost of adapting pretrained models "
                + "to new question answering domains with limited hardware.", 6)));
        blocks.add(new Block("2 Method", 12, true));
        blocks.add(body(repeat("We propose sparse adapters, a method that inserts small trainable matrices into "
                + "each attention layer while the backbone stays frozen.", 8)));
        blocks.add(new Block("3 Results", 12, true));
        blocks.add(body("Our model achieves an exact match of 87.4 on SQuAD v1.1 and an F1 score of 92.7 on "
                + "the same dataset, outperforming full fine-tuning by 1.3 points. "
                + repeat("Training uses a single GPU for six hours and the results table reports the mean of three runs.", 5)));
        blocks.add(new Block("4 Limitations", 12, true));
        blocks.add(body(repeat("A limitation of this work is that we only evaluate English datasets and the "
                + "method has not been tested on generation tasks.", 4)));
        blocks.add(new Block("References", 12, true));
        blocks.add(body("[1] A. Author. Something about adapters. In Proceedings of 2019, pages 1-10. "
                + "[2] B. Author. Another paper on SQuAD 2018."));
        return render(blocks);
    }

    private static Block body(String text) {
        return new Block(text, 10, false);
    }

    private static String repeat(String sentence, int n) {
        return String.join(" ", java.util.Collections.nCopies(n, sentence));
    }

    private record Block(String text, float size, boolean bold) {
    }

    private static byte[] render(List<Block> blocks) throws IOException {
        PDType1Font regular = new PDType1Font(Standard14Fonts.FontName.HELVETICA);
        PDType1Font bold = new PDType1Font(Standard14Fonts.FontName.HELVETICA_BOLD);
        float margin = 60;
        try (PDDocument doc = new PDDocument(); ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            PDPage page = new PDPage(PDRectangle.A4);
            doc.addPage(page);
            PDPageContentStream cs = new PDPageContentStream(doc, page);
            float y = page.getMediaBox().getHeight() - margin;
            for (Block b : blocks) {
                PDType1Font font = b.bold() ? bold : regular;
                float leading = b.size() * 1.4f;
                for (String line : wrap(b.text(), b.bold() ? 60 : 95)) {
                    if (y < margin) {
                        cs.close();
                        page = new PDPage(PDRectangle.A4);
                        doc.addPage(page);
                        cs = new PDPageContentStream(doc, page);
                        y = page.getMediaBox().getHeight() - margin;
                    }
                    cs.beginText();
                    cs.setFont(font, b.size());
                    cs.newLineAtOffset(margin, y);
                    cs.showText(line);
                    cs.endText();
                    y -= leading;
                }
                y -= leading;
            }
            cs.close();
            doc.save(out);
            return out.toByteArray();
        }
    }

    private static List<String> wrap(String text, int width) {
        List<String> lines = new ArrayList<>();
        StringBuilder line = new StringBuilder();
        for (String word : text.split(" ")) {
            if (line.length() + word.length() + 1 > width && !line.isEmpty()) {
                lines.add(line.toString());
                line.setLength(0);
            }
            if (!line.isEmpty()) {
                line.append(' ');
            }
            line.append(word);
        }
        if (!line.isEmpty()) {
            lines.add(line.toString());
        }
        return lines;
    }
}
