package lk.ac.rjt.grasp.ingest;

import java.io.IOException;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;

import lk.ac.rjt.grasp.model.ParsedDocument;
import lk.ac.rjt.grasp.model.ParsedDocument.Paragraph;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.pdfbox.text.TextPosition;
import org.springframework.stereotype.Component;

/**
 * Turns a research-paper PDF into section-tagged paragraphs using Apache PDFBox.
 * Headings are detected from line font size, boldness and common section names; the
 * reference list is dropped because citation numbers and years pollute keyword search.
 */
@Component
public class PdfIngestor {

    static final String FRONT_MATTER = "Title and abstract";

    private static final Pattern KNOWN_SECTION = Pattern.compile(
            "(?i)(abstract|introduction|background|related work[s]?|preliminar(y|ies)|problem (statement|formulation|definition)"
                    + "|method[s]?|methodology|approach|proposed (method|approach|model)|model|framework|experiment[s]?"
                    + "|experimental (setup|settings|results|evaluation)|evaluation|results( and discussion)?|discussion"
                    + "|analysis|ablation (study|studies)|conclusion[s]?( and future work)?|limitation[s]?"
                    + "|limitations and future work|future work|broader impact[s]?|ethics statement|acknowledg(e)?ment[s]?"
                    + "|references|bibliography|appendix|appendices|supplementary material)\\.?");
    private static final Pattern REFERENCES = Pattern.compile("(?i)(references|bibliography|works cited)");
    private static final Pattern NUMBERING = Pattern.compile("^(\\d+(\\.\\d+){0,3}\\.?|[IVX]+\\.|[A-H](\\.\\d+){0,2}\\.?)\\s+");
    private static final Pattern NUMBERED_HEADING = Pattern.compile(
            "^(\\d+(\\.\\d+){0,3}\\.?|[IVX]+\\.|[A-H](\\.\\d+){0,2}\\.?)\\s+\\p{Lu}.*");
    private static final Pattern NOISE = Pattern.compile(
            "^(\\d{1,4}|arXiv:\\S+.*\\d{4}|preprint.*|under review.*|page \\d+( of \\d+)?)$", Pattern.CASE_INSENSITIVE);
    /** IEEE style: "II. SYSTEM MODEL", set in small capitals that are smaller than body text. */
    private static final Pattern ROMAN_CAPS_HEADING = Pattern.compile("^[IVX]+\\.\\s+[A-Z][A-Z0-9 \\-:,&/]{2,}$");
    private static final Pattern BOLD_FONT = Pattern.compile(
            "(?i).*(bold|black|heavy|semibold|demi|medi|cmbx|\\.b$|-b$).*");
    private static final Pattern ITALIC_FONT = Pattern.compile("(?i).*(ital|oblique|cmti|-it$|\\.i$).*");

    public ParsedDocument parse(byte[] pdf) throws IOException {
        try (PDDocument doc = Loader.loadPDF(pdf)) {
            LineCollector collector = new LineCollector();
            collector.setSortByPosition(false);
            collector.getText(doc);
            List<Line> lines = collector.lines;
            float body = bodyFontSize(lines);
            float titleSize = titleFontSize(lines);
            String title = title(doc, lines, titleSize);
            return new ParsedDocument(title, doc.getNumberOfPages(), paragraphs(lines, body, titleSize));
        }
    }

    private List<Paragraph> paragraphs(List<Line> lines, float body, float titleSize) {
        List<Paragraph> out = new ArrayList<>();
        String section = FRONT_MATTER;
        boolean skipping = false;
        StringBuilder text = new StringBuilder();
        int page = 1;

        for (Line line : lines) {
            String t = line.text();
            if (NOISE.matcher(t).matches() || isFigureText(line, body)) {
                continue;
            }
            boolean onTitle = line.page() == 1 && line.fontSize() >= titleSize - 0.25f;
            if (!onTitle && isHeading(line, body)) {
                flush(out, section, page, text, skipping);
                String name = NUMBERING.matcher(t).replaceFirst("").replaceAll("\\.$", "").strip();
                skipping = REFERENCES.matcher(name).matches();
                section = t.replaceAll("\\.$", "");
                page = line.page();
                continue;
            }
            if (line.paragraphStart() || line.page() != page) {
                flush(out, section, page, text, skipping);
                page = line.page();
            }
            append(text, t);
        }
        flush(out, section, page, text, skipping);
        return out;
    }

    private static void flush(List<Paragraph> out, String section, int page, StringBuilder text, boolean skipping) {
        String t = text.toString().strip();
        if (!skipping && !t.isEmpty()) {
            out.add(new Paragraph(section, page, t));
        }
        text.setLength(0);
    }

    /** Joins a line onto the paragraph, repairing words hyphenated across line breaks. */
    static void append(StringBuilder text, String line) {
        if (text.isEmpty()) {
            text.append(line);
            return;
        }
        int last = text.length() - 1;
        boolean hyphenated = text.charAt(last) == '-' && last > 0 && Character.isLetter(text.charAt(last - 1))
                && !line.isEmpty() && Character.isLowerCase(line.charAt(0));
        if (hyphenated) {
            text.setLength(last);
            text.append(line);
        } else {
            text.append(' ').append(line);
        }
    }

    static boolean isHeading(Line line, float body) {
        String t = line.text();
        if (t.length() < 3 || t.length() > 90) {
            return false;
        }
        int words = t.split("\\s+").length;
        if (words > 12 || !Character.isLetterOrDigit(t.charAt(0))) {
            return false;
        }
        String name = NUMBERING.matcher(t).replaceFirst("").strip();
        if (words <= 6 && KNOWN_SECTION.matcher(name).matches()) {
            return true;
        }
        if (ROMAN_CAPS_HEADING.matcher(t).matches()) {
            return true;
        }
        boolean larger = line.fontSize() >= body + 0.9f;
        boolean emphasised = larger || line.bold() || line.italic();
        if (t.endsWith(".") || t.endsWith(",") || t.endsWith(":")) {
            return false;
        }
        if (emphasised && NUMBERED_HEADING.matcher(t).matches() && letterRatio(name) > 0.6) {
            return true;
        }
        return larger && words <= 8 && Character.isUpperCase(t.charAt(0)) && letterRatio(t) > 0.7;
    }

    /**
     * Text drawn inside figures and diagrams: much larger than body text (outside the title page),
     * tiny fragments such as axis labels split into letters, or very small short labels.
     */
    static boolean isFigureText(Line line, float body) {
        String t = line.text();
        if (t.length() <= 3) {
            return true;
        }
        if (line.page() > 1 && line.fontSize() >= body * 1.75f) {
            return true;
        }
        return line.fontSize() < body * 0.75f && t.split("\\s+").length <= 4;
    }

    private static double letterRatio(String s) {
        long letters = s.chars().filter(c -> Character.isLetter(c) || c == ' ').count();
        return s.isEmpty() ? 0 : (double) letters / s.length();
    }

    /** Most common font size, weighted by characters: the body text size. */
    static float bodyFontSize(List<Line> lines) {
        Map<Float, Integer> counts = new HashMap<>();
        for (Line l : lines) {
            counts.merge(Math.round(l.fontSize() * 2) / 2f, l.text().length(), Integer::sum);
        }
        return counts.entrySet().stream().max(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse(10f);
    }

    private static float titleFontSize(List<Line> lines) {
        return (float) lines.stream().filter(l -> l.page() == 1).limit(30)
                .mapToDouble(Line::fontSize).max().orElse(Float.MAX_VALUE);
    }

    private static String title(PDDocument doc, List<Line> lines, float titleSize) {
        String meta = doc.getDocumentInformation() == null ? null : doc.getDocumentInformation().getTitle();
        if (meta != null) {
            String m = clean(meta);
            String lower = m.toLowerCase(Locale.ROOT);
            if (m.length() > 8 && !lower.contains("untitled") && !lower.endsWith(".dvi") && !lower.endsWith(".pdf")
                    && !lower.startsWith("microsoft word")) {
                return m;
            }
        }
        StringBuilder sb = new StringBuilder();
        for (Line l : lines.stream().filter(l -> l.page() == 1).limit(30).toList()) {
            if (l.fontSize() >= titleSize - 0.25f) {
                append(sb, l.text());
            } else if (!sb.isEmpty()) {
                break;
            }
        }
        return sb.isEmpty() ? "Untitled paper" : sb.toString();
    }

    static String clean(String s) {
        String n = Normalizer.normalize(s, Normalizer.Form.NFKC);
        // Private-use glyphs and U+FFFD come from symbol fonts without a Unicode mapping.
        return n.replaceAll("[\\p{Cntrl}&&[^\\n\\t]]|[\\uE000-\\uF8FF]", "").replace('�', ' ')
                .replaceAll("\\s+", " ").strip();
    }

    record Line(String text, int page, float fontSize, boolean bold, boolean italic, boolean paragraphStart) {
    }

    /** Collects text line by line together with its average font size and boldness. */
    private static final class LineCollector extends PDFTextStripper {
        final List<Line> lines = new ArrayList<>();
        private final StringBuilder current = new StringBuilder();
        private double sizeSum;
        private int glyphs;
        private int boldGlyphs;
        private int italicGlyphs;
        private boolean paragraphPending = true;
        private int page;

        LineCollector() throws IOException {
            super();
        }

        @Override
        protected void startPage(PDPage pdPage) throws IOException {
            super.startPage(pdPage);
            page = getCurrentPageNo();
            paragraphPending = true;
        }

        @Override
        protected void endPage(PDPage pdPage) throws IOException {
            flushLine();
            super.endPage(pdPage);
        }

        @Override
        protected void writeString(String text, List<TextPosition> positions) {
            current.append(text);
            for (TextPosition tp : positions) {
                sizeSum += tp.getFontSizeInPt();
                glyphs++;
                String font = tp.getFont() == null ? null : tp.getFont().getName();
                if (font != null && BOLD_FONT.matcher(font).matches()) {
                    boldGlyphs++;
                }
                if (font != null && ITALIC_FONT.matcher(font).matches()) {
                    italicGlyphs++;
                }
            }
        }

        @Override
        protected void writeWordSeparator() {
            current.append(' ');
        }

        @Override
        protected void writeLineSeparator() {
            flushLine();
        }

        @Override
        protected void writeParagraphEnd() {
            flushLine();
            paragraphPending = true;
        }

        private void flushLine() {
            String t = clean(current.toString());
            if (!t.isEmpty() && glyphs > 0) {
                lines.add(new Line(t, page, (float) (sizeSum / glyphs), boldGlyphs * 2 > glyphs,
                        italicGlyphs * 2 > glyphs, paragraphPending));
                paragraphPending = false;
            }
            current.setLength(0);
            sizeSum = 0;
            glyphs = 0;
            boldGlyphs = 0;
            italicGlyphs = 0;
        }
    }
}
