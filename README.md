# Visualize Embeddings (Java)

Create text embeddings with a local [Ollama](https://ollama.com) model, reduce them to 2 or 3 dimensions with PCA, and plot them. This project is a Java port of the Python project [clean-coder/visualize_embeddings](https://github.com/clean-coder/visualize_embeddings), specifically its examples `visualize_embeddings_ollama_matplot.py` and `visualize_embeddings_ollama_plotly_3d.py`.

## How it works

1. Embed a list of sample sentences with `nomic-embed-text` through [LangChain4j](https://docs.langchain4j.dev/)'s Ollama integration.
2. Project the high-dimensional vectors onto the first 2 or 3 principal components. PCA is computed by SVD of the mean-centered data with Apache Commons Math. The explained variance of each component is printed.
3. Plot the result:
   - **2D**: JFreeChart scatter plot, saved to `embeddings_pca_2d.png` and shown in a window.
   - **3D**: custom Swing scatter plot, saved to `embeddings_pca_3d.png` and shown in a window. Drag to rotate, use the wheel to zoom, double-click to reset the view.

## Requirements

- JDK 25 (the code uses instance `main` methods)
- Maven
- Ollama running on `http://localhost:11434` with the model pulled:

  ```sh
  ollama pull nomic-embed-text
  ```

## Build

```sh
mvn clean package
```

## Run

The classes have no `public static void main`, so run them with the `java` launcher from the compiled classes (the classpath must include the dependencies):

```sh
mvn compile dependency:build-classpath -Dmdep.outputFile=cp.txt
java -cp "target/classes:$(cat cp.txt)" embeddings.VisualizeEmbeddingsOllama2D
java -cp "target/classes:$(cat cp.txt)" embeddings.VisualizeEmbeddingsOllama3D
java -cp "target/classes:$(cat cp.txt)" embeddings.CreateEmbeddingsOllama
```

Or run the main classes directly from your IDE.

| Class | Purpose |
|---|---|
| `CreateEmbeddingsOllama` | Creates embeddings for two sentences and prints their dimension and values |
| `VisualizeEmbeddingsOllama2D` | Embeds the samples, PCA to 2D, scatter plot |
| `VisualizeEmbeddingsOllama3D` | Embeds the samples, PCA to 3D, interactive scatter plot |

## Project structure

```
src/main/java/embeddings/
├── CreateEmbeddingsOllama.java
├── VisualizeEmbeddingsOllama2D.java
├── VisualizeEmbeddingsOllama3D.java
└── util/
    ├── EmbeddingsUtil.java   # List<Embedding> -> double[][]
    ├── ProjectionUtil.java   # PCA via SVD
    ├── Scatter2D.java        # JFreeChart 2D plot
    └── Scatter3D.java        # Swing 3D plot
```

## Customizing

Change the sample texts (`TEXT_SAMPLE`), the Ollama URL, or the model name directly in the `VisualizeEmbeddingsOllama2D` / `VisualizeEmbeddingsOllama3D` classes.

## Dependencies

- `langchain4j-ollama` 1.12.2
- `jfreechart` 1.5.5
- `commons-math3` 3.6.1
- `slf4j-nop` 2.0.20
