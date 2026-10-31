package lk.ac.rjt.grasp.llm;

/**
 * The Qwen2.5 chat format. It is identical to what {@code tokenizer.apply_chat_template} produces
 * in the Python training code for a system + user turn with a generation prompt.
 */
public final class ChatMl {

    public static final String END = "<|im_end|>";

    private ChatMl() {
    }

    public static String prompt(String system, String user) {
        return "<|im_start|>system\n" + system + END + "\n"
                + "<|im_start|>user\n" + user + END + "\n"
                + "<|im_start|>assistant\n";
    }
}
