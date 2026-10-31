package lk.ac.rjt.grasp.retrieval;

/**
 * Retrieval query for one output field. {@code dense} is a natural-language question used for
 * embedding search and re-ranking; {@code keywords} is added to it for BM25.
 */
public record FieldQuery(String field, String dense, String keywords) {

    public String bm25() {
        return dense + " " + keywords;
    }
}
