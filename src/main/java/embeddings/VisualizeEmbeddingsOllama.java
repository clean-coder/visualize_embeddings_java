package embeddings;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.annotations.XYTextAnnotation;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.ui.TextAnchor;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.io.File;
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

    // fixed color for each point, as in the Python version
    private static final List<Color> COLORS = List.of(
            Color.RED, Color.BLUE, new Color(230, 180, 0), Color.GRAY, new Color(128, 0, 128));

    public static void main(String[] args) throws IOException {
        EmbeddingModel model = OllamaEmbeddingModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName("nomic-embed-text")
                .build();

        List<Embedding> embeddings = model.embedAll(
                TEXT_SAMPLE.stream().map(TextSegment::from).toList()).content();
        System.out.printf("%d embeddings created.%n", embeddings.size());

        double[][] data = new double[embeddings.size()][];
        for (int i = 0; i < data.length; i++) {
            float[] v = embeddings.get(i).vector();
            data[i] = new double[v.length];
            for (int j = 0; j < v.length; j++) data[i][j] = v[j];
        }

        double[][] visDims = pca(data, 2);

        // one series per text so each point gets its own fixed color
        XYSeriesCollection dataset = new XYSeriesCollection();
        for (int i = 0; i < visDims.length; i++) {
            System.out.printf("%-48s x=%8.4f y=%8.4f%n", TEXT_SAMPLE.get(i), visDims[i][0], visDims[i][1]);
            XYSeries series = new XYSeries(TEXT_SAMPLE.get(i));
            series.add(visDims[i][0], visDims[i][1]);
            dataset.addSeries(series);
        }

        JFreeChart chart = ChartFactory.createScatterPlot(
                "Embeddings Visualization with PCA (2D)", "PC1", "PC2",
                dataset, PlotOrientation.VERTICAL, true, true, false);
        chart.getLegend().setPosition(org.jfree.chart.ui.RectangleEdge.BOTTOM);
        chart.getLegend().setItemFont(new Font(Font.SANS_SERIF, Font.PLAIN, 12));
        XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);

        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer(false, true);
        for (int i = 0; i < visDims.length; i++) {
            renderer.setSeriesShape(i, new Ellipse2D.Double(-9, -9, 18, 18));
            renderer.setSeriesPaint(i, COLORS.get(i));
        }
        plot.setRenderer(renderer);

        // point labels: above each point, except odd indices go below so close points don't collide
        Font labelFont = new Font(Font.SANS_SERIF, Font.PLAIN, 12);
        for (int i = 0; i < visDims.length; i++) {
            boolean below = i % 2 == 1;
            double offset = below ? -0.03 : 0.03;
            XYTextAnnotation label = new XYTextAnnotation(TEXT_SAMPLE.get(i), visDims[i][0], visDims[i][1] + offset);
            label.setFont(labelFont);
            label.setPaint(Color.DARK_GRAY);
            label.setTextAnchor(below ? TextAnchor.TOP_CENTER : TextAnchor.BOTTOM_CENTER);
            plot.addAnnotation(label);
        }
        // leave room around the points so the labels stay inside the plot area
        NumberAxis xAxis = (NumberAxis) plot.getDomainAxis();
        NumberAxis yAxis = (NumberAxis) plot.getRangeAxis();
        xAxis.setAutoRangeIncludesZero(false);
        yAxis.setAutoRangeIncludesZero(false);
        xAxis.setLowerMargin(0.25);
        xAxis.setUpperMargin(0.25);
        yAxis.setUpperMargin(0.15);

        ChartUtils.saveChartAsPNG(new File("embeddings_pca_2d.png"), chart, 1000, 800);
        System.out.println("saved embeddings_pca_2d.png");
        if (!java.awt.GraphicsEnvironment.isHeadless()) {
            SwingUtilities.invokeLater(() -> {
                JFrame frame = new JFrame("Embeddings PCA");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.setContentPane(new ChartPanel(chart));
                frame.setSize(1000, 800);
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
            });
        }
    }

    /**
     * PCA via SVD of the mean-centered data; returns the projection onto the first k components.
     */
    static double[][] pca(double[][] data, int k) {
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
