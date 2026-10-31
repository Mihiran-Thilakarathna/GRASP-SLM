package lk.ac.rjt.grasp.web;

import java.io.IOException;
import java.util.Map;

import lk.ac.rjt.grasp.llm.LlmClient.LlmUnavailableException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice(assignableTypes = ApiController.class)
public class ApiExceptionHandler {

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Map<String, String>> badRequest(IllegalArgumentException e) {
        return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
    }

    @ExceptionHandler(IOException.class)
    public ResponseEntity<Map<String, String>> unreadable(IOException e) {
        return ResponseEntity.badRequest().body(Map.of("error", "Could not read the PDF: " + e.getMessage()));
    }

    @ExceptionHandler(LlmUnavailableException.class)
    public ResponseEntity<Map<String, String>> llmUnavailable(LlmUnavailableException e) {
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(Map.of("error", e.getMessage()));
    }
}
