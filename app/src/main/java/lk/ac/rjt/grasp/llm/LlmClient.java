package lk.ac.rjt.grasp.llm;

/** A chat model that answers with JSON constrained to a schema. */
public interface LlmClient {

    LlmResponse generate(String systemPrompt, String userPrompt, String jsonSchema);

    /** Human-readable description of the model and backend, reported with every analysis. */
    String describe();

    record LlmResponse(String text, int promptTokens, int completionTokens) {
    }

    static LlmClient unavailable(String reason) {
        return new LlmClient() {
            @Override
            public LlmResponse generate(String systemPrompt, String userPrompt, String jsonSchema) {
                throw new LlmUnavailableException(reason);
            }

            @Override
            public String describe() {
                return "unavailable (" + reason + ")";
            }
        };
    }

    class LlmUnavailableException extends RuntimeException {
        public LlmUnavailableException(String message) {
            super(message);
        }

        public LlmUnavailableException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
