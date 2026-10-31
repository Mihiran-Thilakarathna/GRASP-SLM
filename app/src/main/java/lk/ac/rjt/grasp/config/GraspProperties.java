package lk.ac.rjt.grasp.config;

import lk.ac.rjt.grasp.retrieval.RetrievalMode;
import org.springframework.boot.context.properties.ConfigurationProperties;

/** All tunable settings, bound from {@code grasp.*} in application.yml or environment variables. */
@ConfigurationProperties("grasp")
public class GraspProperties {

    private final Models models = new Models();
    private final Chunking chunking = new Chunking();
    private final Retrieval retrieval = new Retrieval();
    private final Llm llm = new Llm();

    public Models getModels() { return models; }
    public Chunking getChunking() { return chunking; }
    public Retrieval getRetrieval() { return retrieval; }
    public Llm getLlm() { return llm; }

    public static class Models {
        /** ONNX export of bge-small-en-v1.5. When missing, the bundled quantised copy is used. */
        private String embeddingModel = "models/bge-small-en-v1.5/model.onnx";
        private String embeddingTokenizer = "models/bge-small-en-v1.5/tokenizer.json";
        /** ONNX export of bge-reranker-base. When missing, re-ranking is disabled. */
        private String rerankerModel = "models/bge-reranker-base/model.onnx";
        private String rerankerTokenizer = "models/bge-reranker-base/tokenizer.json";

        public String getEmbeddingModel() { return embeddingModel; }
        public void setEmbeddingModel(String v) { embeddingModel = v; }
        public String getEmbeddingTokenizer() { return embeddingTokenizer; }
        public void setEmbeddingTokenizer(String v) { embeddingTokenizer = v; }
        public String getRerankerModel() { return rerankerModel; }
        public void setRerankerModel(String v) { rerankerModel = v; }
        public String getRerankerTokenizer() { return rerankerTokenizer; }
        public void setRerankerTokenizer(String v) { rerankerTokenizer = v; }
    }

    public static class Chunking {
        private int minTokens = 400;
        private int targetTokens = 500;
        private int maxTokens = 600;
        /** Approximate sub-word tokens per whitespace word for English scientific text. */
        private double tokensPerWord = 1.3;

        public int getMinTokens() { return minTokens; }
        public void setMinTokens(int v) { minTokens = v; }
        public int getTargetTokens() { return targetTokens; }
        public void setTargetTokens(int v) { targetTokens = v; }
        public int getMaxTokens() { return maxTokens; }
        public void setMaxTokens(int v) { maxTokens = v; }
        public double getTokensPerWord() { return tokensPerWord; }
        public void setTokensPerWord(double v) { tokensPerWord = v; }
    }

    public static class Retrieval {
        private RetrievalMode mode = RetrievalMode.HYBRID;
        /** Candidates taken from each of BM25 and dense search before fusion. */
        private int candidatesPerRetriever = 20;
        private int rrfK = 60;
        private boolean rerank = true;
        /** Fused candidates per field passed to the cross-encoder. */
        private int rerankCandidates = 10;
        /** Final number of chunks given to the model (proposal: four to six). */
        private int maxContextChunks = 6;
        /** bge-small-en-v1.5 query instruction for retrieval. */
        private String queryInstruction = "Represent this sentence for searching relevant passages: ";

        public RetrievalMode getMode() { return mode; }
        public void setMode(RetrievalMode v) { mode = v; }
        public int getCandidatesPerRetriever() { return candidatesPerRetriever; }
        public void setCandidatesPerRetriever(int v) { candidatesPerRetriever = v; }
        public int getRrfK() { return rrfK; }
        public void setRrfK(int v) { rrfK = v; }
        public boolean isRerank() { return rerank; }
        public void setRerank(boolean v) { rerank = v; }
        public int getRerankCandidates() { return rerankCandidates; }
        public void setRerankCandidates(int v) { rerankCandidates = v; }
        public int getMaxContextChunks() { return maxContextChunks; }
        public void setMaxContextChunks(int v) { maxContextChunks = v; }
        public String getQueryInstruction() { return queryInstruction; }
        public void setQueryInstruction(String v) { queryInstruction = v; }
    }

    public static class Llm {
        /** {@code llamacpp} (in-process GGUF), {@code openai} (any OpenAI-compatible server) or {@code none}. */
        private String backend = "llamacpp";
        private String modelPath = "models/llm/model.gguf";
        private int contextSize = 8192;
        /** 0 lets llama.cpp choose. */
        private int threads = 0;
        private int maxTokens = 1024;
        private float temperature = 0.0f;
        private int seed = 42;
        /** Constrain decoding with a grammar generated from the JSON Schema. */
        private boolean grammar = true;
        private String baseUrl = "http://localhost:8081/v1";
        private String apiKey = "none";
        private String modelName = "grasp-slm";
        /** {@code json_schema} (OpenAI style), {@code json_object} (llama-cpp-python style) or {@code none}. */
        private String responseFormat = "json_schema";
        private int timeoutSeconds = 900;

        public String getBackend() { return backend; }
        public void setBackend(String v) { backend = v; }
        public String getModelPath() { return modelPath; }
        public void setModelPath(String v) { modelPath = v; }
        public int getContextSize() { return contextSize; }
        public void setContextSize(int v) { contextSize = v; }
        public int getThreads() { return threads; }
        public void setThreads(int v) { threads = v; }
        public int getMaxTokens() { return maxTokens; }
        public void setMaxTokens(int v) { maxTokens = v; }
        public float getTemperature() { return temperature; }
        public void setTemperature(float v) { temperature = v; }
        public int getSeed() { return seed; }
        public void setSeed(int v) { seed = v; }
        public boolean isGrammar() { return grammar; }
        public void setGrammar(boolean v) { grammar = v; }
        public String getBaseUrl() { return baseUrl; }
        public void setBaseUrl(String v) { baseUrl = v; }
        public String getApiKey() { return apiKey; }
        public void setApiKey(String v) { apiKey = v; }
        public String getModelName() { return modelName; }
        public void setModelName(String v) { modelName = v; }
        public String getResponseFormat() { return responseFormat; }
        public void setResponseFormat(String v) { responseFormat = v; }
        public int getTimeoutSeconds() { return timeoutSeconds; }
        public void setTimeoutSeconds(int v) { timeoutSeconds = v; }
    }
}
