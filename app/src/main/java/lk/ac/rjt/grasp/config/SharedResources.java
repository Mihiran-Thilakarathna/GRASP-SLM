package lk.ac.rjt.grasp.config;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lk.ac.rjt.grasp.retrieval.FieldQuery;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

/**
 * Files under {@code shared/} in the repository: the output JSON Schema, the prompt templates and
 * the field queries. The Python training code reads the same files, so a change here changes both.
 */
@Component
public class SharedResources {

    private final String schema;
    private final String systemPrompt;
    private final String userTemplate;
    private final List<FieldQuery> fieldQueries;

    public SharedResources(ObjectMapper mapper) {
        this.schema = read("shared/analysis.schema.json");
        this.systemPrompt = read("shared/prompt/system.txt").strip();
        this.userTemplate = read("shared/prompt/user.txt").strip();
        try {
            this.fieldQueries = mapper.readValue(read("shared/field_queries.json"), new TypeReference<>() {
            });
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static String read(String path) {
        try (InputStream in = new ClassPathResource(path).getInputStream()) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("Missing shared resource " + path, e);
        }
    }

    public String schema() { return schema; }
    public String systemPrompt() { return systemPrompt; }
    public String userTemplate() { return userTemplate; }
    public List<FieldQuery> fieldQueries() { return fieldQueries; }
}
