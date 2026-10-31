package lk.ac.rjt.grasp.model;

import java.util.List;

/** Output of PDF ingestion: section-tagged paragraphs in reading order. */
public record ParsedDocument(String title, int numPages, List<Paragraph> paragraphs) {

    public record Paragraph(String section, int page, String text) {
    }
}
