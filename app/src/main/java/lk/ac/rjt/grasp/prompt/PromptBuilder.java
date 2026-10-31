package lk.ac.rjt.grasp.prompt;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import dev.langchain4j.model.input.PromptTemplate;
import lk.ac.rjt.grasp.config.SharedResources;
import lk.ac.rjt.grasp.model.Chunk;
import org.springframework.stereotype.Component;

/** Fills the shared prompt templates with the paper title and the retrieved chunks. */
@Component
public class PromptBuilder {

    private final SharedResources shared;
    private final PromptTemplate userTemplate;

    public PromptBuilder(SharedResources shared) {
        this.shared = shared;
        this.userTemplate = PromptTemplate.from(shared.userTemplate());
    }

    public String systemPrompt() {
        return shared.systemPrompt();
    }

    public String userPrompt(String title, List<Chunk> context) {
        return userTemplate.apply(Map.of("title", title, "context", renderContext(context))).text();
    }

    public static String renderContext(List<Chunk> context) {
        return context.stream()
                .map(c -> "[" + c.id() + "] Section: " + c.section() + " | " + c.pageLabel() + "\n" + c.text())
                .collect(Collectors.joining("\n\n"));
    }
}
