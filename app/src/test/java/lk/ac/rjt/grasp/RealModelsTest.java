package lk.ac.rjt.grasp;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;

import lk.ac.rjt.grasp.retrieval.RetrievalMode;
import lk.ac.rjt.grasp.retrieval.RetrievalOptions;
import lk.ac.rjt.grasp.service.AnalysisResponse;
import lk.ac.rjt.grasp.service.AnalysisService;
import lk.ac.rjt.grasp.testsupport.TestPdfs;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

/**
 * End-to-end run with the real ONNX models and GGUF model. Run from app/ with
 * {@code ./mvnw test -Pmodels} after {@code python scripts/download_models.py}.
 */
@Tag("models")
@SpringBootTest(properties = {
        "grasp.models.embedding-model=../models/bge-small-en-v1.5/model.onnx",
        "grasp.models.embedding-tokenizer=../models/bge-small-en-v1.5/tokenizer.json",
        "grasp.models.reranker-model=../models/bge-reranker-base/model.onnx",
        "grasp.models.reranker-tokenizer=../models/bge-reranker-base/tokenizer.json",
        "grasp.llm.backend=llamacpp",
        "grasp.llm.model-path=../models/llm/model.gguf"})
class RealModelsTest {

    @Autowired
    private AnalysisService service;

    @BeforeAll
    static void modelsPresent() {
        assumeTrue(Files.isRegularFile(Path.of("../models/llm/model.gguf")), "run scripts/download_models.py first");
        assumeTrue(Files.isRegularFile(Path.of("../models/bge-reranker-base/model.onnx")), "re-ranker missing");
    }

    @Test
    void analysesAPaperWithSchemaValidGroundedOutput() throws Exception {
        AnalysisResponse r = service.analyze(TestPdfs.samplePaper(),
                new RetrievalOptions(RetrievalMode.HYBRID, true, 6), false);

        System.out.println(r.markdown());
        System.out.println("timings ms: " + r.timingsMs() + ", tokens: " + r.tokens());
        assertThat(r.reranked()).isTrue();
        assertThat(r.parsed()).isTrue();
        assertThat(r.schemaValid()).as("errors: %s", r.errors()).isTrue();
        assertThat(r.unknownCitations()).isEmpty();
    }
}
