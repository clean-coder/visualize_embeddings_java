package embeddings.util;

import dev.langchain4j.data.embedding.Embedding;
import org.jspecify.annotations.NonNull;

import java.util.List;

public class EmbeddingsUtil {

    public static double @NonNull [][] extractDataFromEmbeddings(List<Embedding> embeddings) {
        double[][] data = new double[embeddings.size()][];
        for (int i = 0; i < data.length; i++) {
            float[] v = embeddings.get(i).vector();
            data[i] = new double[v.length];
            for (int j = 0; j < v.length; j++) data[i][j] = v[j];
        }
        return data;
    }
}
