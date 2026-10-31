package lk.ac.rjt.grasp.testsupport;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.scoring.ScoringModel;
import lk.ac.rjt.grasp.llm.LlmClient;

/** Deterministic stand-ins for the ONNX models and the LLM. */
public final class Fakes {

    private Fakes() {
    }

    /** Hashed bag-of-words embedding: texts sharing words get similar vectors. */
    public static final class BagOfWordsEmbedding implements EmbeddingModel {
        private static final int DIM = 256;

        @Override
        public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
            return Response.from(segments.stream().map(s -> Embedding.from(vector(s.text()))).toList());
        }

        static float[] vector(String text) {
            float[] v = new float[DIM];
            for (String w : words(text)) {
                v[Math.floorMod(w.hashCode(), DIM)] += 1f;
            }
            double norm = Math.sqrt(sumSquares(v));
            if (norm == 0) {
                v[0] = 1f;
                return v;
            }
            for (int i = 0; i < DIM; i++) {
                v[i] /= (float) norm;
            }
            return v;
        }

        private static double sumSquares(float[] v) {
            double s = 0;
            for (float x : v) {
                s += x * x;
            }
            return s;
        }
    }

    /** Scores a passage by how many distinct query words it contains. */
    public static final class OverlapScorer implements ScoringModel {
        @Override
        public Response<List<Double>> scoreAll(List<TextSegment> segments, String query) {
            Set<String> q = Set.copyOf(words(query));
            return Response.from(segments.stream()
                    .map(s -> (double) words(s.text()).stream().distinct().filter(q::contains).count())
                    .toList());
        }
    }

    /** Returns a schema-valid analysis citing the first chunk id that appears in the prompt. */
    public static final class CannedLlm implements LlmClient {
        private static final Pattern FIRST_ID = Pattern.compile("\\[(c\\d+)]");

        @Override
        public LlmResponse generate(String systemPrompt, String userPrompt, String jsonSchema) {
            Matcher m = FIRST_ID.matcher(userPrompt);
            String id = m.find() ? m.group(1) : "c0";
            String json = """
                    {"core_problem":{"text":"High memory cost of adaptation.","citations":["%1$s"]},
                     "proposed_method":{"text":"Sparse adapters.","citations":["%1$s"]},
                     "key_metrics":[{"metric":"F1","value":"92.7","dataset":"SQuAD v1.1","citations":["%1$s"]}],
                     "limitations":[{"text":"Only English datasets.","citations":["%1$s"]}]}
                    """.formatted(id);
            return new LlmResponse(json, 100, 50);
        }

        @Override
        public String describe() {
            return "canned test model";
        }
    }

    static List<String> words(String text) {
        return Arrays.stream(text.toLowerCase(Locale.ROOT).split("[^a-z0-9.]+"))
                .map(w -> w.replaceAll("^\\.+|\\.+$", ""))
                .filter(w -> w.length() > 2)
                .collect(Collectors.toList());
    }
}
