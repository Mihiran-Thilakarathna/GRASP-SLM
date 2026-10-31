package lk.ac.rjt.grasp.llm;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lk.ac.rjt.grasp.config.GraspProperties;

/**
 * Calls any OpenAI-compatible chat endpoint (llama.cpp server, llama-cpp-python, vLLM). This is
 * the fallback from the risk table, and it is how the GPU evaluation serves the unquantised
 * models for ablation configurations A-D.
 */
public class OpenAiCompatibleLlmClient implements LlmClient {

    private final GraspProperties.Llm cfg;
    private final ObjectMapper mapper;
    private final HttpClient http;

    public OpenAiCompatibleLlmClient(GraspProperties.Llm cfg, ObjectMapper mapper) {
        this.cfg = cfg;
        this.mapper = mapper;
        this.http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(20)).build();
    }

    @Override
    public LlmResponse generate(String systemPrompt, String userPrompt, String jsonSchema) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", cfg.getModelName());
        body.put("messages", List.of(
                Map.of("role", "system", "content", systemPrompt),
                Map.of("role", "user", "content", userPrompt)));
        body.put("temperature", cfg.getTemperature());
        body.put("max_tokens", cfg.getMaxTokens());
        body.put("seed", cfg.getSeed());
        Object format = responseFormat(jsonSchema);
        if (format != null) {
            body.put("response_format", format);
        }
        try {
            HttpRequest request = HttpRequest.newBuilder(URI.create(cfg.getBaseUrl().replaceAll("/$", "") + "/chat/completions"))
                    .timeout(Duration.ofSeconds(cfg.getTimeoutSeconds()))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + cfg.getApiKey())
                    .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(body)))
                    .build();
            HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() / 100 != 2) {
                throw new LlmUnavailableException("LLM server returned " + response.statusCode() + ": " + response.body());
            }
            JsonNode json = mapper.readTree(response.body());
            String text = json.path("choices").path(0).path("message").path("content").asText("");
            JsonNode usage = json.path("usage");
            return new LlmResponse(text, usage.path("prompt_tokens").asInt(-1), usage.path("completion_tokens").asInt(-1));
        } catch (IOException e) {
            throw new LlmUnavailableException("Cannot reach LLM server at " + cfg.getBaseUrl(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new LlmUnavailableException("Interrupted while waiting for the LLM server", e);
        }
    }

    private Object responseFormat(String jsonSchema) throws LlmUnavailableException {
        if (!cfg.isGrammar() || jsonSchema == null) {
            return null;
        }
        try {
            JsonNode schema = mapper.readTree(jsonSchema);
            return switch (cfg.getResponseFormat()) {
                case "json_schema" -> Map.of("type", "json_schema",
                        "json_schema", Map.of("name", "analysis", "strict", true, "schema", schema));
                case "json_object" -> Map.of("type", "json_object", "schema", schema);
                default -> null;
            };
        } catch (IOException e) {
            throw new IllegalArgumentException("Invalid JSON schema", e);
        }
    }

    @Override
    public String describe() {
        return "OpenAI-compatible " + cfg.getModelName() + " @ " + cfg.getBaseUrl();
    }
}
