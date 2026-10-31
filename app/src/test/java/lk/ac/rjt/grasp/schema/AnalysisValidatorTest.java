package lk.ac.rjt.grasp.schema;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import com.fasterxml.jackson.databind.ObjectMapper;
import lk.ac.rjt.grasp.config.SharedResources;
import lk.ac.rjt.grasp.model.Chunk;
import org.junit.jupiter.api.Test;

class AnalysisValidatorTest {

    private final ObjectMapper mapper = new ObjectMapper();
    private final AnalysisValidator validator = new AnalysisValidator(mapper, new SharedResources(mapper));

    private static final String VALID = """
            {"core_problem":{"text":"P","citations":["c1"]},
             "proposed_method":{"text":"M","citations":["c2"]},
             "key_metrics":[{"metric":"F1","value":"92.7","dataset":"SQuAD","citations":["c3"]}],
             "limitations":[]}
            """;

    @Test
    void acceptsValidAnalysis() {
        var report = validator.validate(VALID, List.of("c1", "c2", "c3"));
        assertThat(report.parsed()).isTrue();
        assertThat(report.schemaValid()).isTrue();
        assertThat(report.errors()).isEmpty();
        assertThat(report.unknownCitations()).isEmpty();
    }

    @Test
    void flagsCitationsOutsideTheContext() {
        var report = validator.validate(VALID, List.of("c1", "c2"));
        assertThat(report.schemaValid()).isTrue();
        assertThat(report.unknownCitations()).containsExactly("c3");
    }

    @Test
    void rejectsMissingFieldsAndBadCitationIds() {
        String bad = """
                {"core_problem":{"text":"P","citations":[]},
                 "proposed_method":{"text":"M","citations":["chunk-2"]},
                 "key_metrics":[]}
                """;
        var report = validator.validate(bad, List.of("c1"));
        assertThat(report.parsed()).isTrue();
        assertThat(report.schemaValid()).isFalse();
        assertThat(report.errors()).anyMatch(e -> e.contains("limitations"));
    }

    @Test
    void reportsMalformedJson() {
        var report = validator.validate("{\"core_problem\": {", List.of());
        assertThat(report.parsed()).isFalse();
        assertThat(report.schemaValid()).isFalse();
    }

    @Test
    void acceptsCodeFencedJson() {
        var report = validator.validate("```json\n" + VALID + "\n```", List.of("c1", "c2", "c3"));
        assertThat(report.schemaValid()).isTrue();
    }

    @Test
    void markdownShowsPagesForCitations() throws Exception {
        var chunks = List.of(new Chunk("c1", 1, "Intro", 2, 2, "t", 1), new Chunk("c2", 2, "Method", 3, 4, "t", 1),
                new Chunk("c3", 3, "Results", 5, 5, "t", 1));
        String md = MarkdownRenderer.render("Paper", mapper.readTree(VALID), chunks);
        assertThat(md).contains("## Core problem").contains("[c1, p. 2]").contains("[c2, pp. 3-4]")
                .contains("| F1 | 92.7 | SQuAD | [c3, p. 5] |").contains("No limitations stated");
    }
}
