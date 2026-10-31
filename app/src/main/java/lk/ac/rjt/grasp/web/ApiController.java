package lk.ac.rjt.grasp.web;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lk.ac.rjt.grasp.config.SharedResources;
import lk.ac.rjt.grasp.llm.LlmClient;
import lk.ac.rjt.grasp.retrieval.RetrievalMode;
import lk.ac.rjt.grasp.retrieval.RetrievalOptions;
import lk.ac.rjt.grasp.service.AnalysisResponse;
import lk.ac.rjt.grasp.service.AnalysisService;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api")
public class ApiController {

    private final AnalysisService service;
    private final ObjectProvider<LlmClient> llm;
    private final SharedResources shared;
    private final ObjectMapper mapper;

    public ApiController(AnalysisService service, ObjectProvider<LlmClient> llm, SharedResources shared, ObjectMapper mapper) {
        this.service = service;
        this.llm = llm;
        this.shared = shared;
        this.mapper = mapper;
    }

    /** Full analysis of one PDF. Optional parameters override the configured retrieval settings. */
    @PostMapping(path = "/analyze", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public AnalysisResponse analyze(@RequestParam("file") MultipartFile file,
                                    @RequestParam(required = false) String retrieval,
                                    @RequestParam(required = false) Boolean rerank,
                                    @RequestParam(required = false) Integer maxContextChunks,
                                    @RequestParam(defaultValue = "false") boolean includePrompt) throws IOException {
        return service.analyze(file.getBytes(), options(retrieval, rerank, maxContextChunks), includePrompt);
    }

    /** Retrieval only, without generation: used to evaluate retrieval quality on its own. */
    @PostMapping(path = "/retrieve", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public AnalysisResponse retrieve(@RequestParam("file") MultipartFile file,
                                     @RequestParam(required = false) String retrieval,
                                     @RequestParam(required = false) Boolean rerank,
                                     @RequestParam(required = false) Integer maxContextChunks,
                                     @RequestParam(defaultValue = "false") boolean includePrompt) throws IOException {
        return service.retrieveOnly(file.getBytes(), options(retrieval, rerank, maxContextChunks), includePrompt);
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("status", "ok");
        out.put("model", llm.getObject().describe());
        out.put("retrievalDefaults", service.defaultOptions());
        return out;
    }

    @GetMapping(path = "/schema", produces = MediaType.APPLICATION_JSON_VALUE)
    public JsonNode schema() throws IOException {
        return mapper.readTree(shared.schema());
    }

    RetrievalOptions options(String retrieval, Boolean rerank, Integer maxContextChunks) {
        RetrievalOptions d = service.defaultOptions();
        RetrievalMode mode = retrieval == null || retrieval.isBlank() ? d.mode() : RetrievalMode.parse(retrieval);
        int max = maxContextChunks == null ? d.maxContextChunks() : maxContextChunks;
        if (max < 1 || max > 12) {
            throw new IllegalArgumentException("maxContextChunks must be between 1 and 12");
        }
        return new RetrievalOptions(mode, rerank == null ? d.rerank() : rerank, max);
    }
}
