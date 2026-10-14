package embeddings;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import embeddings.util.Plotter;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.jspecify.annotations.NonNull;

import java.io.IOException;
import java.util.List;

/**
 * Java port of visualize_embeddings_ollama_matplot.py: embed texts, reduce with PCA, plot in 2D.
 */
public class VisualizeEmbeddingsOllama {

    private static final List<String> TEXT_SAMPLE = List.of(
            "Java ist eine Programmiersprache",
            "Python ist auch eine Programmiersprache",
            "Esperanto ist eine internationale Plansprache",
            "Balu der Bär",
            "Shir Khan der Tiger");


    static void main() throws IOException {
        EmbeddingModel model = OllamaEmbeddingModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName("nomic-embed-text")
                .build();

        List<Embedding> embeddings = model.embedAll(
                TEXT_SAMPLE.stream().map(TextSegment::from).toList()).content();
        System.out.printf("%d embeddings created.%n", embeddings.size());

        double[][] data = extractDataFromEmbeddings(embeddings);
        double[][] visDims = pca(data, 2);
        Plotter.visualize2D(TEXT_SAMPLE, visDims);
    }

    private static double @NonNull [][] extractDataFromEmbeddings(List<Embedding> embeddings) {
        double[][] data = new double[embeddings.size()][];
        for (int i = 0; i < data.length; i++) {
            float[] v = embeddings.get(i).vector();
            data[i] = new double[v.length];
            for (int j = 0; j < v.length; j++) data[i][j] = v[j];
        }
        return data;
    }

    /**
     * PCA via SVD of the mean-centered data; returns the projection onto the first k components.
     */
    private static double[][] pca(double[][] data, int k) {
        int n = data.length, d = data[0].length;
        double[] mean = new double[d];
        for (double[] row : data) for (int j = 0; j < d; j++) mean[j] += row[j] / n;
        double[][] centered = new double[n][d];
        for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) centered[i][j] = data[i][j] - mean[j];

        SingularValueDecomposition svd = new SingularValueDecomposition(new Array2DRowRealMatrix(centered, false));
        double[] sv = svd.getSingularValues();
        double total = 0;
        for (double s : sv) total += s * s;
        for (int c = 0; c < k; c++) {
            System.out.printf("PC%d explained variance: %.1f%%%n", c + 1, 100 * sv[c] * sv[c] / total);
        }
        // scores = U * S for the first k components
        RealMatrix u = svd.getU();
        double[][] out = new double[n][k];
        for (int i = 0; i < n; i++) for (int c = 0; c < k; c++) out[i][c] = u.getEntry(i, c) * sv[c];
        return out;
    }
}
