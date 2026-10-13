package embeddings;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;

import java.util.List;

public class CreateEmbeddingsOllama {

    public static void main(String[] args) {
        EmbeddingModel localEmbeddingModel = OllamaEmbeddingModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName("nomic-embed-text")
                .build();

        List<TextSegment> textSample = List.of(
                TextSegment.from("hallo und willkommen"),
                TextSegment.from("dies ist ein weiterer Beispielsatz"));
        List<Embedding> embeddings = localEmbeddingModel.embedAll(textSample).content();

        System.out.printf("%d embeddings created.%n", embeddings.size());
        System.out.printf("first embedding len: %d%n", embeddings.get(0).dimension());
        System.out.printf("second embedding len: %d%n", embeddings.get(1).dimension());
        System.out.println("first embedding: " + embeddings.get(0).toString());
        System.out.println("second embedding: " + embeddings.get(1).toString());
    }
}
