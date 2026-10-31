package lk.ac.rjt.grasp.config;

import java.nio.file.Files;
import java.nio.file.Path;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.OnnxEmbeddingModel;
import dev.langchain4j.model.embedding.onnx.PoolingMode;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.scoring.ScoringModel;
import dev.langchain4j.model.scoring.onnx.OnnxScoringModel;
import lk.ac.rjt.grasp.llm.LlamaCppLlmClient;
import lk.ac.rjt.grasp.llm.LlmClient;
import lk.ac.rjt.grasp.llm.OpenAiCompatibleLlmClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Condition;
import org.springframework.context.annotation.ConditionContext;
import org.springframework.context.annotation.Conditional;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Lazy;
import org.springframework.core.type.AnnotatedTypeMetadata;

import com.fasterxml.jackson.databind.ObjectMapper;

/** Creates the ONNX embedding model, the ONNX cross-encoder and the language-model client. */
@Configuration
public class ModelConfig {

    private static final Logger log = LoggerFactory.getLogger(ModelConfig.class);

    @Bean
    @Lazy
    @ConditionalOnMissingBean
    public EmbeddingModel embeddingModel(GraspProperties props) {
        GraspProperties.Models m = props.getModels();
        if (Files.isRegularFile(Path.of(m.getEmbeddingModel())) && Files.isRegularFile(Path.of(m.getEmbeddingTokenizer()))) {
            log.info("Embedding model: {}", m.getEmbeddingModel());
            return new OnnxEmbeddingModel(m.getEmbeddingModel(), m.getEmbeddingTokenizer(), PoolingMode.CLS);
        }
        log.info("Embedding model: bundled quantised bge-small-en-v1.5 ({} not found)", m.getEmbeddingModel());
        return new BgeSmallEnV15QuantizedEmbeddingModel();
    }

    @Bean
    @Lazy
    @ConditionalOnMissingBean
    @Conditional(RerankerFilesPresent.class)
    public ScoringModel rerankerModel(GraspProperties props) {
        GraspProperties.Models m = props.getModels();
        log.info("Re-ranker model: {}", m.getRerankerModel());
        return new OnnxScoringModel(m.getRerankerModel(), m.getRerankerTokenizer());
    }

    @Bean
    @Lazy
    @ConditionalOnMissingBean
    public LlmClient llmClient(GraspProperties props, ObjectMapper mapper) {
        GraspProperties.Llm llm = props.getLlm();
        return switch (llm.getBackend().toLowerCase()) {
            case "llamacpp" -> new LlamaCppLlmClient(llm);
            case "openai" -> new OpenAiCompatibleLlmClient(llm, mapper);
            case "none" -> LlmClient.unavailable("grasp.llm.backend is 'none'");
            default -> throw new IllegalStateException("Unknown grasp.llm.backend: " + llm.getBackend());
        };
    }

    static final class RerankerFilesPresent implements Condition {
        @Override
        public boolean matches(ConditionContext context, AnnotatedTypeMetadata metadata) {
            String model = context.getEnvironment().getProperty("grasp.models.reranker-model",
                    "models/bge-reranker-base/model.onnx");
            String tokenizer = context.getEnvironment().getProperty("grasp.models.reranker-tokenizer",
                    "models/bge-reranker-base/tokenizer.json");
            boolean present = Files.isRegularFile(Path.of(model)) && Files.isRegularFile(Path.of(tokenizer));
            if (!present) {
                log.warn("Re-ranker files not found ({}); hybrid retrieval will run without re-ranking", model);
            }
            return present;
        }
    }
}
