package lk.ac.rjt.grasp.model;

/**
 * A retrievable piece of a paper. {@code id} ("c0", "c1", ...) is what the model cites, and it
 * maps back to the section and pages it came from.
 */
public record Chunk(String id, int index, String section, int pageStart, int pageEnd, String text, int approxTokens) {

    public static String idFor(int index) {
        return "c" + index;
    }

    /** Text indexed for search: the section title helps both BM25 and dense matching. */
    public String searchText() {
        return section + "\n" + text;
    }

    public String pageLabel() {
        return pageStart == pageEnd ? "p. " + pageStart : "pp. " + pageStart + "-" + pageEnd;
    }
}
