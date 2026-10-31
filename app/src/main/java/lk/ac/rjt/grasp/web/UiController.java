package lk.ac.rjt.grasp.web;

import java.io.IOException;

import lk.ac.rjt.grasp.llm.LlmClient.LlmUnavailableException;
import lk.ac.rjt.grasp.retrieval.RetrievalMode;
import lk.ac.rjt.grasp.retrieval.RetrievalOptions;
import lk.ac.rjt.grasp.service.AnalysisResponse;
import lk.ac.rjt.grasp.service.AnalysisService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

/** Thymeleaf web interface: upload a PDF, see the structured analysis with page citations. */
@Controller
public class UiController {

    private final AnalysisService service;

    public UiController(AnalysisService service) {
        this.service = service;
    }

    @GetMapping("/")
    public String index(Model model) {
        model.addAttribute("defaults", service.defaultOptions());
        model.addAttribute("modes", RetrievalMode.values());
        return "index";
    }

    @PostMapping("/analyze")
    public String analyze(@RequestParam("file") MultipartFile file,
                          @RequestParam(defaultValue = "HYBRID") String retrieval,
                          @RequestParam(defaultValue = "false") boolean rerank,
                          Model model) {
        if (file.isEmpty()) {
            return error(model, "Please choose a PDF file.");
        }
        try {
            RetrievalOptions options = new RetrievalOptions(RetrievalMode.parse(retrieval), rerank,
                    service.defaultOptions().maxContextChunks());
            AnalysisResponse result = service.analyze(file.getBytes(), options, false);
            model.addAttribute("result", result);
            model.addAttribute("fileName", file.getOriginalFilename());
            return "result";
        } catch (IOException e) {
            return error(model, "Could not read the PDF: " + e.getMessage());
        } catch (IllegalArgumentException | LlmUnavailableException e) {
            return error(model, e.getMessage());
        }
    }

    private String error(Model model, String message) {
        model.addAttribute("error", message);
        return index(model);
    }
}
