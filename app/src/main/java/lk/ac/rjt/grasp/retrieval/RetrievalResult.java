package lk.ac.rjt.grasp.retrieval;

import java.util.List;
import java.util.Map;

import lk.ac.rjt.grasp.model.Chunk;
import lk.ac.rjt.grasp.model.ScoredChunk;

/**
 * @param rankings final ranking for each output field
 * @param context  chunks given to the model, in document order
 */
public record RetrievalResult(RetrievalOptions options, boolean reranked,
                              Map<String, List<ScoredChunk>> rankings, List<Chunk> context) {
}
