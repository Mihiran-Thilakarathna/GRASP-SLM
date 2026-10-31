package lk.ac.rjt.grasp.service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import lk.ac.rjt.grasp.config.SharedResources;
import lk.ac.rjt.grasp.ingest.Chunker;
import lk.ac.rjt.grasp.ingest.PdfIngestor;
import lk.ac.rjt.grasp.llm.LlmClient;
import lk.ac.rjt.grasp.model.Chunk;
import lk.ac.rjt.grasp.model.ParsedDocument;
import lk.ac.rjt.grasp.prompt.PromptBuilder;
import lk.ac.rjt.grasp.retrieval.HybridRetriever;
import lk.ac.rjt.grasp.retrieval.RetrievalOptions;
import lk.ac.rjt.grasp.retrieval.RetrievalResult;
import lk.ac.rjt.grasp.schema.AnalysisValidator;
import lk.ac.rjt.grasp.schema.MarkdownRenderer;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.stereotype.Service;

/** The online pipeline: parse, chunk, retrieve, prompt, generate, validate, render. */
@Service
public class AnalysisService {

    private final PdfIngestor ingestor;
    private final Chunker chunker;
    private final HybridRetriever retriever;
    private final PromptBuilder prompts;
    private final ObjectProvider<LlmClient> llm;
    private final AnalysisValidator validator;
    private final SharedResources shared;

    public AnalysisService(PdfIngestor ingestor, Chunker chunker, HybridRetriever retriever, PromptBuilder prompts,
                           ObjectProvider<LlmClient> llm, AnalysisValidator validator, SharedResources shared) {
        this.ingestor = ingestor;
        this.chunker = chunker;
        this.retriever = retriever;
        this.prompts = prompts;
        this.llm = llm;
        this.validator = validator;
        this.shared = shared;
    }

    public RetrievalOptions defaultOptions() {
        return retriever.defaults();
    }

    public PreparedPaper prepare(byte[] pdf, RetrievalOptions options) throws IOException {
        Map<String, Long> timings = new LinkedHashMap<>();
        long t0 = System.nanoTime();
        ParsedDocument doc = ingestor.parse(pdf);
        long t1 = System.nanoTime();
        List<Chunk> chunks = chunker.chunk(doc.paragraphs());
        if (chunks.isEmpty()) {
            throw new IllegalArgumentException("No text could be extracted from the PDF (is it a scanned image?)");
        }
        long t2 = System.nanoTime();
        RetrievalResult retrieval = retriever.retrieve(chunks, options);
        long t3 = System.nanoTime();
        timings.put("parse", ms(t0, t1));
        timings.put("chunk", ms(t1, t2));
        timings.put("retrieve", ms(t2, t3));
        return new PreparedPaper(doc, chunks, retrieval, prompts.systemPrompt(),
                prompts.userPrompt(doc.title(), retrieval.context()), timings);
    }

    public AnalysisResponse analyze(byte[] pdf, RetrievalOptions options, boolean includePrompt) throws IOException {
        long start = System.nanoTime();
        PreparedPaper p = prepare(pdf, options);
        LlmClient client = llm.getObject();

        long g0 = System.nanoTime();
        LlmClient.LlmResponse out = client.generate(p.systemPrompt(), p.userPrompt(), shared.schema());
        long g1 = System.nanoTime();

        List<String> ids = p.retrieval().context().stream().map(Chunk::id).toList();
        AnalysisValidator.ValidationReport report = validator.validate(out.text(), ids);
        String markdown = report.parsed()
                ? MarkdownRenderer.render(p.document().title(), report.analysis(), p.retrieval().context())
                : null;

        Map<String, Long> timings = new LinkedHashMap<>(p.timingsMs());
        timings.put("generate", ms(g0, g1));
        timings.put("total", ms(start, System.nanoTime()));
        return toResponse(p, client.describe(), out, report, markdown, timings, includePrompt);
    }

    public AnalysisResponse retrieveOnly(byte[] pdf, RetrievalOptions options, boolean includePrompt) throws IOException {
        PreparedPaper p = prepare(pdf, options);
        return toResponse(p, null, null, null, null, p.timingsMs(), includePrompt);
    }

    private static AnalysisResponse toResponse(PreparedPaper p, String model, LlmClient.LlmResponse out,
                                               AnalysisValidator.ValidationReport report, String markdown,
                                               Map<String, Long> timings, boolean includePrompt) {
        Map<String, List<AnalysisResponse.Ranked>> rankings = new LinkedHashMap<>();
        p.retrieval().rankings().forEach((field, ranking) ->
                rankings.put(field, ranking.stream().map(AnalysisResponse.Ranked::of).toList()));
        Map<String, Integer> tokens = out == null ? null
                : Map.of("prompt", out.promptTokens(), "completion", out.completionTokens());
        return new AnalysisResponse(
                p.document().title(),
                p.document().numPages(),
                p.chunks().size(),
                p.retrieval().options(),
                p.retrieval().reranked(),
                p.retrieval().context().stream().map(AnalysisResponse.ChunkView::of).toList(),
                rankings,
                model,
                out == null ? null : out.text(),
                report == null ? null : report.analysis(),
                markdown,
                report != null && report.parsed(),
                report != null && report.schemaValid(),
                report == null ? null : report.errors(),
                report == null ? null : report.unknownCitations(),
                timings,
                tokens,
                peakRssBytes(),
                includePrompt ? new AnalysisResponse.Prompt(p.systemPrompt(), p.userPrompt()) : null);
    }

    private static long ms(long fromNanos, long toNanos) {
        return (toNanos - fromNanos) / 1_000_000;
    }

    /** Peak resident memory of the whole process (JVM + native llama.cpp), Linux only. */
    static Long peakRssBytes() {
        Path status = Path.of("/proc/self/status");
        if (!Files.isReadable(status)) {
            return null;
        }
        try {
            for (String line : Files.readAllLines(status)) {
                if (line.startsWith("VmHWM:")) {
                    return Long.parseLong(line.replaceAll("\\D+", "")) * 1024;
                }
            }
        } catch (IOException | NumberFormatException ignored) {
            // memory reporting is best effort
        }
        return null;
    }
}
