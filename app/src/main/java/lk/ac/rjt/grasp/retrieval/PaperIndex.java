package lk.ac.rjt.grasp.retrieval;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import lk.ac.rjt.grasp.model.Chunk;
import lk.ac.rjt.grasp.model.ScoredChunk;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.ByteBuffersDirectory;

/**
 * In-memory Lucene index over the chunks of one paper. Each chunk is a single document with a
 * BM25-scored text field and an HNSW-indexed embedding field.
 */
public final class PaperIndex implements AutoCloseable {

    static final String TEXT = "text";
    static final String VECTOR = "vector";
    static final String INDEX = "index";

    private final List<Chunk> chunks;
    private final Analyzer analyzer = new EnglishAnalyzer();
    private final ByteBuffersDirectory directory = new ByteBuffersDirectory();
    private final DirectoryReader reader;
    private final IndexSearcher searcher;

    private PaperIndex(List<Chunk> chunks, List<Embedding> embeddings) throws IOException {
        this.chunks = chunks;
        try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig(analyzer))) {
            for (int i = 0; i < chunks.size(); i++) {
                Document doc = new Document();
                doc.add(new StoredField(INDEX, i));
                doc.add(new TextField(TEXT, chunks.get(i).searchText(), Field.Store.NO));
                doc.add(new KnnFloatVectorField(VECTOR, embeddings.get(i).vector(), VectorSimilarityFunction.DOT_PRODUCT));
                writer.addDocument(doc);
            }
        }
        this.reader = DirectoryReader.open(directory);
        this.searcher = new IndexSearcher(reader);
    }

    public static PaperIndex build(List<Chunk> chunks, EmbeddingModel embedder) {
        List<TextSegment> segments = chunks.stream().map(c -> TextSegment.from(c.searchText())).toList();
        List<Embedding> embeddings = segments.isEmpty() ? List.of() : embedder.embedAll(segments).content();
        try {
            return new PaperIndex(chunks, embeddings);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public List<ScoredChunk> bm25(String query, int k) {
        try {
            Query q = new QueryParser(TEXT, analyzer).parse(QueryParser.escape(query));
            return search(q, k);
        } catch (ParseException e) {
            throw new IllegalArgumentException("Bad BM25 query: " + query, e);
        }
    }

    public List<ScoredChunk> dense(float[] queryVector, int k) {
        return search(new KnnFloatVectorQuery(VECTOR, queryVector, Math.max(1, k)), k);
    }

    private List<ScoredChunk> search(Query query, int k) {
        if (chunks.isEmpty()) {
            return List.of();
        }
        try {
            TopDocs top = searcher.search(query, Math.max(1, k));
            List<ScoredChunk> out = new ArrayList<>();
            for (ScoreDoc sd : top.scoreDocs) {
                int idx = searcher.storedFields().document(sd.doc).getField(INDEX).numericValue().intValue();
                out.add(new ScoredChunk(chunks.get(idx), sd.score));
            }
            return out;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public void close() {
        try {
            reader.close();
            directory.close();
            analyzer.close();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
