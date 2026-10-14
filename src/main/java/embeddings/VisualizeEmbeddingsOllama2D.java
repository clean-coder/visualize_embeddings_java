package embeddings;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import embeddings.util.Scatter2D;

import java.io.IOException;
import java.util.List;

import static embeddings.util.EmbeddingsUtil.extractDataFromEmbeddings;
import static embeddings.util.ProjectionUtil.pca;

/**
 * Java port of visualize_embeddings_ollama_matplot.py: embed texts, reduce with PCA, plot in 2D.
 */
public class VisualizeEmbeddingsOllama2D {

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
        printEmbeddingsInfos(visDims);

        new Scatter2D().paint(TEXT_SAMPLE, visDims);
    }

    private static void printEmbeddingsInfos(double[][] visDims) {
        for (int i = 0; i < visDims.length; i++) {
            System.out.printf("%-48s x=%8.4f y=%8.4f%n",
                    TEXT_SAMPLE.get(i), visDims[i][0], visDims[i][1]);
        }
    }
}
