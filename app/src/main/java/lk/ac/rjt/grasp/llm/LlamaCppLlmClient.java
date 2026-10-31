package lk.ac.rjt.grasp.llm;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.ModelParameters;
import de.kherud.llama.args.LogFormat;
import jakarta.annotation.PreDestroy;
import lk.ac.rjt.grasp.config.GraspProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Runs the 4-bit GGUF model inside the JVM through java-llama.cpp. Decoding is constrained by a
 * GBNF grammar generated from the JSON Schema, so the output always parses.
 */
public class LlamaCppLlmClient implements LlmClient {

    private static final Logger log = LoggerFactory.getLogger(LlamaCppLlmClient.class);

    private final GraspProperties.Llm cfg;
    private final Map<String, String> grammars = new ConcurrentHashMap<>();
    private LlamaModel model;

    public LlamaCppLlmClient(GraspProperties.Llm cfg) {
        this.cfg = cfg;
    }

    private synchronized LlamaModel model() {
        if (model == null) {
            Path path = Path.of(cfg.getModelPath());
            if (!Files.isRegularFile(path)) {
                throw new LlmUnavailableException("GGUF model not found at " + path.toAbsolutePath()
                        + " (set GRASP_LLM_MODEL_PATH)");
            }
            LlamaModel.setLogger(LogFormat.TEXT, (level, msg) -> log.debug("llama.cpp: {}", msg.strip()));
            ModelParameters params = new ModelParameters()
                    .setModel(path.toString())
                    .setCtxSize(cfg.getContextSize())
                    .setGpuLayers(0);
            if (cfg.getThreads() > 0) {
                params.setThreads(cfg.getThreads());
            }
            log.info("Loading GGUF model {}", path);
            model = new LlamaModel(params);
        }
        return model;
    }

    @Override
    public synchronized LlmResponse generate(String systemPrompt, String userPrompt, String jsonSchema) {
        LlamaModel m = model();
        String prompt = ChatMl.prompt(systemPrompt, userPrompt);
        InferenceParameters params = new InferenceParameters(prompt)
                .setTemperature(cfg.getTemperature())
                .setNPredict(cfg.getMaxTokens())
                .setSeed(cfg.getSeed())
                .setCachePrompt(false)
                .setStopStrings(ChatMl.END);
        if (cfg.isGrammar() && jsonSchema != null) {
            params.setGrammar(grammars.computeIfAbsent(jsonSchema, LlamaModel::jsonSchemaToGrammar));
        }
        String text = m.complete(params);
        return new LlmResponse(text, m.encode(prompt).length, m.encode(text).length);
    }

    @Override
    public String describe() {
        return "llama.cpp (in-process) " + Path.of(cfg.getModelPath()).getFileName();
    }

    @PreDestroy
    public synchronized void close() {
        if (model != null) {
            model.close();
            model = null;
        }
    }
}
