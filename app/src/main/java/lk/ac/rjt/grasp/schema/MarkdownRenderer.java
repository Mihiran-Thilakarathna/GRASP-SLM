package lk.ac.rjt.grasp.schema;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import com.fasterxml.jackson.databind.JsonNode;
import lk.ac.rjt.grasp.model.Chunk;

/** Renders a validated analysis as Markdown, turning chunk citations into page references. */
public final class MarkdownRenderer {

    private MarkdownRenderer() {
    }

    public static String render(String title, JsonNode analysis, List<Chunk> context) {
        Map<String, Chunk> byId = context.stream().collect(Collectors.toMap(Chunk::id, Function.identity()));
        StringBuilder md = new StringBuilder("# ").append(title).append("\n\n");

        md.append("## Core problem\n\n").append(claim(analysis.path("core_problem"), byId)).append("\n\n");
        md.append("## Proposed method\n\n").append(claim(analysis.path("proposed_method"), byId)).append("\n\n");

        md.append("## Key metrics\n\n");
        JsonNode metrics = analysis.path("key_metrics");
        if (metrics.isEmpty()) {
            md.append("_No metrics reported in the retrieved context._\n\n");
        } else {
            md.append("| Metric | Value | Dataset | Source |\n|---|---|---|---|\n");
            for (JsonNode m : metrics) {
                md.append("| ").append(cell(m.path("metric").asText()))
                        .append(" | ").append(cell(m.path("value").asText()))
                        .append(" | ").append(cell(m.path("dataset").asText()))
                        .append(" | ").append(cite(m.path("citations"), byId)).append(" |\n");
            }
            md.append('\n');
        }

        md.append("## Limitations\n\n");
        JsonNode limitations = analysis.path("limitations");
        if (limitations.isEmpty()) {
            md.append("_No limitations stated in the retrieved context._\n");
        } else {
            for (JsonNode l : limitations) {
                md.append("- ").append(claim(l, byId)).append('\n');
            }
        }
        return md.toString();
    }

    private static String claim(JsonNode claim, Map<String, Chunk> byId) {
        return claim.path("text").asText() + " " + cite(claim.path("citations"), byId);
    }

    private static String cite(JsonNode citations, Map<String, Chunk> byId) {
        StringBuilder sb = new StringBuilder();
        for (JsonNode c : citations) {
            Chunk chunk = byId.get(c.asText());
            if (!sb.isEmpty()) {
                sb.append(", ");
            }
            sb.append(c.asText()).append(chunk == null ? " (unknown)" : ", " + chunk.pageLabel());
        }
        return sb.isEmpty() ? "" : "[" + sb + "]";
    }

    private static String cell(String s) {
        return s.replace("|", "\\|").replace("\n", " ");
    }
}
