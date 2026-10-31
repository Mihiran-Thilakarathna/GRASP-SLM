package lk.ac.rjt.grasp.ingest;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

import lk.ac.rjt.grasp.config.GraspProperties;
import lk.ac.rjt.grasp.model.Chunk;
import lk.ac.rjt.grasp.model.ParsedDocument.Paragraph;
import org.springframework.stereotype.Component;

/**
 * Packs paragraphs into chunks of roughly 400-600 tokens. Chunks prefer to end at section
 * boundaries, and oversized paragraphs are split at sentence boundaries. Token counts are
 * approximated from word counts so that chunking needs no tokenizer and is fully reproducible.
 */
@Component
public class Chunker {

    private static final Pattern SENTENCE_END = Pattern.compile("(?<=[.!?])\\s+(?=[\\p{Lu}(\\[])");

    private final GraspProperties.Chunking cfg;

    public Chunker(GraspProperties props) {
        this.cfg = props.getChunking();
    }

    public List<Chunk> chunk(List<Paragraph> paragraphs) {
        List<Chunk> chunks = new ArrayList<>();
        Builder current = null;
        for (Paragraph p : paragraphs) {
            for (String piece : pieces(p.text())) {
                int tokens = approxTokens(piece);
                if (current != null) {
                    boolean sectionChanged = !current.section.equals(p.section());
                    boolean reachedTarget = current.tokens + tokens > cfg.getTargetTokens();
                    boolean overflow = current.tokens + tokens > cfg.getMaxTokens();
                    if (overflow || (current.tokens >= cfg.getMinTokens() && (reachedTarget || sectionChanged))) {
                        chunks.add(current.build(chunks.size()));
                        current = null;
                    }
                }
                if (current == null) {
                    current = new Builder(p.section(), p.page());
                }
                current.add(piece, tokens, p.page());
            }
        }
        if (current != null) {
            Chunk last = current.build(chunks.size());
            if (!chunks.isEmpty() && last.approxTokens() < cfg.getMinTokens() / 2
                    && chunks.getLast().approxTokens() + last.approxTokens() <= cfg.getMaxTokens() + cfg.getMinTokens() / 2) {
                Chunk prev = chunks.removeLast();
                chunks.add(new Chunk(prev.id(), prev.index(), prev.section(), prev.pageStart(), last.pageEnd(),
                        prev.text() + "\n\n" + last.text(), prev.approxTokens() + last.approxTokens()));
            } else {
                chunks.add(last);
            }
        }
        return chunks;
    }

    /** Splits a paragraph that is too long into sentence groups, and over-long sentences into word runs. */
    private List<String> pieces(String paragraph) {
        if (approxTokens(paragraph) <= cfg.getMaxTokens()) {
            return List.of(paragraph);
        }
        List<String> out = new ArrayList<>();
        StringBuilder group = new StringBuilder();
        for (String sentence : SENTENCE_END.split(paragraph)) {
            for (String part : splitWords(sentence)) {
                if (!group.isEmpty() && approxTokens(group + " " + part) > cfg.getTargetTokens()) {
                    out.add(group.toString());
                    group.setLength(0);
                }
                if (!group.isEmpty()) {
                    group.append(' ');
                }
                group.append(part);
            }
        }
        if (!group.isEmpty()) {
            out.add(group.toString());
        }
        return out;
    }

    private List<String> splitWords(String sentence) {
        if (approxTokens(sentence) <= cfg.getTargetTokens()) {
            return List.of(sentence);
        }
        String[] words = sentence.split("\\s+");
        int perPiece = Math.max(1, (int) (cfg.getTargetTokens() / cfg.getTokensPerWord()));
        List<String> out = new ArrayList<>();
        for (int i = 0; i < words.length; i += perPiece) {
            out.add(String.join(" ", java.util.Arrays.copyOfRange(words, i, Math.min(words.length, i + perPiece))));
        }
        return out;
    }

    public int approxTokens(String text) {
        String t = text.strip();
        if (t.isEmpty()) {
            return 0;
        }
        return (int) Math.ceil(t.split("\\s+").length * cfg.getTokensPerWord());
    }

    private static final class Builder {
        final String section;
        final int pageStart;
        int pageEnd;
        int tokens;
        final StringBuilder text = new StringBuilder();

        Builder(String section, int page) {
            this.section = section;
            this.pageStart = page;
            this.pageEnd = page;
        }

        void add(String piece, int pieceTokens, int page) {
            if (!text.isEmpty()) {
                text.append("\n\n");
            }
            text.append(piece);
            tokens += pieceTokens;
            pageEnd = Math.max(pageEnd, page);
        }

        Chunk build(int index) {
            return new Chunk(Chunk.idFor(index), index, section, pageStart, pageEnd, text.toString(), tokens);
        }
    }
}
