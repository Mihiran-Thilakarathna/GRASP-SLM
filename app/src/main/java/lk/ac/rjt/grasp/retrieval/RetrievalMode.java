package lk.ac.rjt.grasp.retrieval;

import java.util.Locale;

public enum RetrievalMode {
    /** Dense vector search only (ablation configuration A). */
    DENSE,
    /** BM25 keyword search only (extra ablation). */
    BM25,
    /** BM25 + dense fused with Reciprocal Rank Fusion (configurations B-E). */
    HYBRID;

    public static RetrievalMode parse(String value) {
        return valueOf(value.strip().toUpperCase(Locale.ROOT));
    }
}
