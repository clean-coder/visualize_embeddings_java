package embeddings.util;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.annotations.XYTextAnnotation;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.ui.RectangleEdge;
import org.jfree.chart.ui.TextAnchor;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.io.File;
import java.io.IOException;
import java.util.List;

import static java.awt.Color.*;

public class Plotter {

    private static final List<Color> COLORS = List.of(
            RED, BLUE, PINK, BLACK, YELLOW, ORANGE, CYAN, GRAY);

    public static void visualize2D(List<String> textSamples, double[][] visDims) throws IOException {

        // one series per text so each point gets its own fixed color
        XYSeriesCollection dataset = new XYSeriesCollection();
        for (int i = 0; i < visDims.length; i++) {
            System.out.printf("%-48s x=%8.4f y=%8.4f%n", textSamples.get(i), visDims[i][0], visDims[i][1]);
            XYSeries series = new XYSeries(textSamples.get(i));
            series.add(visDims[i][0], visDims[i][1]);
            dataset.addSeries(series);
        }

        JFreeChart chart = ChartFactory.createScatterPlot(
                "Embeddings Visualization with PCA (2D)", "PC1", "PC2",
                dataset, PlotOrientation.VERTICAL, true, true, false);

        chart.getLegend().setPosition(RectangleEdge.BOTTOM);
        chart.getLegend().setItemFont(new Font(Font.SANS_SERIF, Font.PLAIN, 12));
        XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(WHITE);
        plot.setDomainGridlinePaint(LIGHT_GRAY);
        plot.setRangeGridlinePaint(LIGHT_GRAY);

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
            XYTextAnnotation label = new XYTextAnnotation(textSamples.get(i), visDims[i][0], visDims[i][1] + offset);
            label.setFont(labelFont);
            label.setPaint(DARK_GRAY);
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

        showPlot(chart);
    }

    private static void showPlot(JFreeChart chart) {
        if (!GraphicsEnvironment.isHeadless()) {
            SwingUtilities.invokeLater(() -> {
                JFrame frame = new JFrame("Embeddings 2D Visualization");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.setContentPane(new ChartPanel(chart));
                frame.setSize(1000, 800);
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
            });
        }
    }
}