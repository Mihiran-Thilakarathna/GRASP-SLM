package lk.ac.rjt.grasp.cli;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

import com.fasterxml.jackson.databind.ObjectMapper;
import lk.ac.rjt.grasp.model.Chunk;
import lk.ac.rjt.grasp.retrieval.RetrievalMode;
import lk.ac.rjt.grasp.retrieval.RetrievalOptions;
import lk.ac.rjt.grasp.service.AnalysisService;
import lk.ac.rjt.grasp.service.PreparedPaper;
import org.springframework.stereotype.Component;

/**
 * {@code java -jar grasp-slm.jar export --input papers/ --output data/export.jsonl}
 * <p>
 * Parses, chunks and retrieves every PDF with the same code the web app uses, and writes one JSON
 * line per paper with all chunks, the selected context and the exact chat messages. The Python
 * side drafts labels and builds the SFT/DPO datasets from this file.
 */
@Component
public class ExportCommand {

    public static final String NAME = "export";

    private final AnalysisService service;
    private final ObjectMapper mapper;

    public ExportCommand(AnalysisService service, ObjectMapper mapper) {
        this.service = service;
        this.mapper = mapper;
    }

    public int run(String[] args) {
        Map<String, String> opts = parse(args);
        if (!opts.containsKey("input") || !opts.containsKey("output")) {
            System.err.println("Usage: export --input <pdf-or-dir> --output <file.jsonl> "
                    + "[--retrieval hybrid|dense|bm25] [--rerank true|false] [--max-context-chunks 6]");
            return 2;
        }
        RetrievalOptions d = service.defaultOptions();
        RetrievalOptions options = new RetrievalOptions(
                opts.containsKey("retrieval") ? RetrievalMode.parse(opts.get("retrieval")) : d.mode(),
                opts.containsKey("rerank") ? Boolean.parseBoolean(opts.get("rerank")) : d.rerank(),
                opts.containsKey("max-context-chunks") ? Integer.parseInt(opts.get("max-context-chunks")) : d.maxContextChunks());

        Path input = Path.of(opts.get("input"));
        Path output = Path.of(opts.get("output"));
        try {
            List<Path> pdfs = listPdfs(input);
            Set<String> done = alreadyExported(output);
            if (output.toAbsolutePath().getParent() != null) {
                Files.createDirectories(output.toAbsolutePath().getParent());
            }
            int ok = 0;
            int failed = 0;
            try (BufferedWriter out = Files.newBufferedWriter(output, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
                for (Path pdf : pdfs) {
                    String id = paperId(pdf);
                    if (done.contains(id)) {
                        continue;
                    }
                    try {
                        PreparedPaper p = service.prepare(Files.readAllBytes(pdf), options);
                        out.write(mapper.writeValueAsString(record(id, pdf, p)));
                        out.newLine();
                        out.flush();
                        ok++;
                        System.out.printf("[%d/%d] %s: %d chunks%n", ok + failed + done.size(), pdfs.size(), id, p.chunks().size());
                    } catch (Exception e) {
                        failed++;
                        System.err.printf("FAILED %s: %s%n", id, e.getMessage());
                    }
                }
            }
            System.out.printf("Exported %d papers (%d skipped as already exported, %d failed) to %s%n",
                    ok, done.size(), failed, output);
            return failed > 0 && ok == 0 ? 1 : 0;
        } catch (IOException e) {
            System.err.println("Export failed: " + e.getMessage());
            return 1;
        }
    }

    private Map<String, Object> record(String id, Path pdf, PreparedPaper p) {
        Map<String, Object> r = new LinkedHashMap<>();
        r.put("paper_id", id);
        r.put("source_file", pdf.getFileName().toString());
        r.put("title", p.document().title());
        r.put("num_pages", p.document().numPages());
        r.put("retrieval", Map.of(
                "mode", p.retrieval().options().mode().name().toLowerCase(),
                "rerank_requested", p.retrieval().options().rerank(),
                "reranked", p.retrieval().reranked(),
                "max_context_chunks", p.retrieval().options().maxContextChunks()));
        r.put("chunks", p.chunks().stream().map(ExportCommand::chunk).toList());
        r.put("context_chunk_ids", p.retrieval().context().stream().map(Chunk::id).toList());
        Map<String, List<String>> rankings = new LinkedHashMap<>();
        p.retrieval().rankings().forEach((f, list) -> rankings.put(f, list.stream().map(s -> s.chunk().id()).toList()));
        r.put("rankings", rankings);
        r.put("messages", List.of(
                Map.of("role", "system", "content", p.systemPrompt()),
                Map.of("role", "user", "content", p.userPrompt())));
        return r;
    }

    private static Map<String, Object> chunk(Chunk c) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("id", c.id());
        m.put("section", c.section());
        m.put("page_start", c.pageStart());
        m.put("page_end", c.pageEnd());
        m.put("approx_tokens", c.approxTokens());
        m.put("text", c.text());
        return m;
    }

    private static List<Path> listPdfs(Path input) throws IOException {
        if (Files.isRegularFile(input)) {
            return List.of(input);
        }
        try (Stream<Path> s = Files.list(input)) {
            return s.filter(p -> p.getFileName().toString().toLowerCase().endsWith(".pdf")).sorted().toList();
        }
    }

    private Set<String> alreadyExported(Path output) throws IOException {
        Set<String> ids = new HashSet<>();
        if (Files.isRegularFile(output)) {
            for (String line : Files.readAllLines(output, StandardCharsets.UTF_8)) {
                if (!line.isBlank()) {
                    ids.add(mapper.readTree(line).path("paper_id").asText());
                }
            }
        }
        return ids;
    }

    static String paperId(Path pdf) {
        String name = pdf.getFileName().toString();
        return name.substring(0, name.length() - ".pdf".length());
    }

    static Map<String, String> parse(String[] args) {
        Map<String, String> out = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            if (args[i].startsWith("--") && i + 1 < args.length && !args[i + 1].startsWith("--")) {
                out.put(args[i].substring(2), args[++i]);
            }
        }
        return out;
    }
}
