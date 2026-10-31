package lk.ac.rjt.grasp.service;

import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.JsonNode;
import lk.ac.rjt.grasp.model.Chunk;
import lk.ac.rjt.grasp.model.ScoredChunk;
import lk.ac.rjt.grasp.retrieval.RetrievalOptions;

/**
 * Result of {@code POST /api/analyze}. It carries the retrieved chunks and timings as well as the
 * analysis so that the Python evaluation can compute Ragas, numeric-accuracy and latency metrics.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record AnalysisResponse(
        String title,
        int numPages,
        int numChunks,
        RetrievalOptions retrieval,
        boolean reranked,
        List<ChunkView> context,
        Map<String, List<Ranked>> rankings,
        String model,
        String rawOutput,
        JsonNode analysis,
        String markdown,
        boolean parsed,
        boolean schemaValid,
        List<String> errors,
        List<String> unknownCitations,
        Map<String, Long> timingsMs,
        Map<String, Integer> tokens,
        Long peakRssBytes,
        Prompt prompt) {

    public record ChunkView(String id, String section, int pageStart, int pageEnd, String text) {
        public static ChunkView of(Chunk c) {
            return new ChunkView(c.id(), c.section(), c.pageStart(), c.pageEnd(), c.text());
        }
    }

    public record Ranked(String id, double score) {
        public static Ranked of(ScoredChunk s) {
            return new Ranked(s.chunk().id(), s.score());
        }
    }

    public record Prompt(String system, String user) {
    }
}
