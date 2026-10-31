package lk.ac.rjt.grasp.schema;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.networknt.schema.JsonSchema;
import com.networknt.schema.JsonSchemaFactory;
import com.networknt.schema.SpecVersion;
import com.networknt.schema.ValidationMessage;
import lk.ac.rjt.grasp.config.SharedResources;
import org.springframework.stereotype.Component;

/**
 * Checks model output against the JSON Schema and checks that every cited chunk id was actually
 * part of the context the model was given.
 */
@Component
public class AnalysisValidator {

    private final ObjectMapper mapper;
    private final JsonSchema schema;

    public AnalysisValidator(ObjectMapper mapper, SharedResources shared) {
        this.mapper = mapper;
        this.schema = JsonSchemaFactory.getInstance(SpecVersion.VersionFlag.V202012).getSchema(shared.schema());
    }

    public ValidationReport validate(String rawOutput, Collection<String> contextIds) {
        JsonNode json;
        try {
            json = mapper.readTree(stripCodeFence(rawOutput));
        } catch (JsonProcessingException e) {
            return new ValidationReport(false, false, null, List.of("Output is not valid JSON: " + e.getOriginalMessage()), List.of());
        }
        if (json == null || !json.isObject()) {
            return new ValidationReport(false, false, null, List.of("Output is not a JSON object"), List.of());
        }
        List<String> errors = new ArrayList<>();
        for (ValidationMessage m : schema.validate(json)) {
            errors.add(m.getMessage());
        }
        Set<String> unknown = new LinkedHashSet<>();
        for (JsonNode citations : json.findValues("citations")) {
            citations.forEach(c -> {
                if (!contextIds.contains(c.asText())) {
                    unknown.add(c.asText());
                }
            });
        }
        return new ValidationReport(true, errors.isEmpty(), json, errors, List.copyOf(unknown));
    }

    /** Unconstrained backends sometimes wrap JSON in a Markdown code fence. */
    static String stripCodeFence(String s) {
        String t = s == null ? "" : s.strip();
        if (t.startsWith("```")) {
            t = t.replaceFirst("^```[a-zA-Z]*\\s*", "").replaceFirst("\\s*```$", "");
        }
        return t;
    }

    /**
     * @param unknownCitations cited chunk ids that were not in the model's context (fabricated citations)
     */
    public record ValidationReport(boolean parsed, boolean schemaValid, JsonNode analysis,
                                   List<String> errors, List<String> unknownCitations) {
    }
}
