package lk.ac.rjt.grasp.retrieval;

/** Per-request retrieval settings; the evaluation notebook changes these to run the ablation. */
public record RetrievalOptions(RetrievalMode mode, boolean rerank, int maxContextChunks) {
}
