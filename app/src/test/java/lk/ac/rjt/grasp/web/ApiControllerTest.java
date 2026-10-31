package lk.ac.rjt.grasp.web;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.scoring.ScoringModel;
import lk.ac.rjt.grasp.llm.LlmClient;
import lk.ac.rjt.grasp.testsupport.Fakes;
import lk.ac.rjt.grasp.testsupport.TestPdfs;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest(properties = "grasp.llm.backend=none")
@AutoConfigureMockMvc
class ApiControllerTest {

    @TestConfiguration
    static class FakeModels {
        @Bean
        @Primary
        EmbeddingModel fakeEmbedding() {
            return new Fakes.BagOfWordsEmbedding();
        }

        @Bean
        @Primary
        ScoringModel fakeScorer() {
            return new Fakes.OverlapScorer();
        }

        @Bean
        @Primary
        LlmClient fakeLlm() {
            return new Fakes.CannedLlm();
        }
    }

    @Autowired
    private MockMvc mvc;

    private static MockMultipartFile pdf() throws Exception {
        return new MockMultipartFile("file", "paper.pdf", "application/pdf", TestPdfs.samplePaper());
    }

    @Test
    void analyzeReturnsValidatedAnalysisWithEvidence() throws Exception {
        mvc.perform(multipart("/api/analyze").file(pdf()).param("includePrompt", "true"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.title").value(TestPdfs.TITLE))
                .andExpect(jsonPath("$.schemaValid").value(true))
                .andExpect(jsonPath("$.unknownCitations").isEmpty())
                .andExpect(jsonPath("$.analysis.key_metrics[0].value").value("92.7"))
                .andExpect(jsonPath("$.context").isNotEmpty())
                .andExpect(jsonPath("$.retrieval.mode").value("HYBRID"))
                .andExpect(jsonPath("$.reranked").value(true))
                .andExpect(jsonPath("$.prompt.user").value(org.hamcrest.Matchers.containsString("Context chunks:")))
                .andExpect(jsonPath("$.timingsMs.total").isNumber());
    }

    @Test
    void retrievalModeCanBeOverriddenPerRequest() throws Exception {
        mvc.perform(multipart("/api/retrieve").file(pdf()).param("retrieval", "dense").param("rerank", "false"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.retrieval.mode").value("DENSE"))
                .andExpect(jsonPath("$.reranked").value(false))
                .andExpect(jsonPath("$.analysis").doesNotExist());
    }

    @Test
    void rejectsNonPdf() throws Exception {
        var notPdf = new MockMultipartFile("file", "x.pdf", "application/pdf", "hello".getBytes());
        mvc.perform(multipart("/api/analyze").file(notPdf))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").exists());
    }

    @Test
    void schemaAndHealthEndpoints() throws Exception {
        mvc.perform(get("/api/schema")).andExpect(status().isOk())
                .andExpect(jsonPath("$.required[0]").value("core_problem"));
        mvc.perform(get("/api/health")).andExpect(status().isOk())
                .andExpect(jsonPath("$.model").value("canned test model"));
    }

    @Test
    void webInterfaceRendersResult() throws Exception {
        mvc.perform(get("/")).andExpect(status().isOk())
                .andExpect(content().string(org.hamcrest.Matchers.containsString("Analyse paper")));
        mvc.perform(multipart("/analyze").file(pdf()).param("retrieval", "HYBRID").param("rerank", "true"))
                .andExpect(status().isOk())
                .andExpect(content().string(org.hamcrest.Matchers.containsString("Sparse adapters.")))
                .andExpect(content().string(org.hamcrest.Matchers.containsString("SQuAD v1.1")));
    }
}
