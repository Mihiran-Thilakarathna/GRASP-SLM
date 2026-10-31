package lk.ac.rjt.grasp.service;

import java.util.List;
import java.util.Map;

import lk.ac.rjt.grasp.model.Chunk;
import lk.ac.rjt.grasp.model.ParsedDocument;
import lk.ac.rjt.grasp.retrieval.RetrievalResult;

/** Everything up to (but not including) generation: shared by the web app and the exporter. */
public record PreparedPaper(ParsedDocument document, List<Chunk> chunks, RetrievalResult retrieval,
                            String systemPrompt, String userPrompt, Map<String, Long> timingsMs) {
}
