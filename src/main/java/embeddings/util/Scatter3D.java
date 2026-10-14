package embeddings.util;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static java.awt.Color.*;
import static java.awt.Color.BLACK;
import static java.awt.Color.CYAN;
import static java.awt.Color.GRAY;
import static java.awt.Color.ORANGE;
import static java.awt.Color.YELLOW;

public class Scatter3D {

    private static final List<Color> COLORS = List.of(
            RED, BLUE, PINK, BLACK, YELLOW, ORANGE, CYAN, GRAY);

    /**
     * Interactive 3D scatter: drag = rotate, wheel = zoom, double-click = reset view.
     */
    class Scatter3DPanel extends JPanel {
        private static final double DEFAULT_YAW = Math.toRadians(35), DEFAULT_PITCH = Math.toRadians(20);

        private final double[][] pts;      // normalized to [-1, 1] per axis
        private final List<String> labels;
        private final List<Color> colors;
        private double yaw = DEFAULT_YAW, pitch = DEFAULT_PITCH, zoom = 1.0;
        private int lastX, lastY;

        Scatter3DPanel(double[][] raw, List<String> labels, List<Color> colors) {
            this.labels = labels;
            this.colors = colors;
            this.pts = new double[raw.length][3];
            for (int c = 0; c < 3; c++) {
                double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
                for (double[] r : raw) {
                    min = Math.min(min, r[c]);
                    max = Math.max(max, r[c]);
                }
                double mid = (min + max) / 2, half = Math.max((max - min) / 2, 1e-9);
                for (int i = 0; i < raw.length; i++) pts[i][c] = (raw[i][c] - mid) / half;
            }
            setBackground(Color.WHITE);
            MouseAdapter mouse = new MouseAdapter() {
                @Override
                public void mousePressed(MouseEvent e) {
                    lastX = e.getX();
                    lastY = e.getY();
                }

                @Override
                public void mouseDragged(MouseEvent e) {
                    yaw += (e.getX() - lastX) * 0.01;
                    pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, pitch + (e.getY() - lastY) * 0.01));
                    lastX = e.getX();
                    lastY = e.getY();
                    repaint();
                }

                @Override
                public void mouseWheelMoved(java.awt.event.MouseWheelEvent e) {
                    zoom = Math.max(0.3, Math.min(4, zoom * Math.pow(1.1, -e.getWheelRotation())));
                    repaint();
                }

                @Override
                public void mouseClicked(MouseEvent e) {
                    if (e.getClickCount() == 2) {
                        yaw = DEFAULT_YAW;
                        pitch = DEFAULT_PITCH;
                        zoom = 1;
                        repaint();
                    }
                }
            };
            addMouseListener(mouse);
            addMouseMotionListener(mouse);
            addMouseWheelListener(mouse);
        }

        /**
         * Rotates a point and returns {screenX, screenY, depth} with a mild perspective.
         */
        private double[] project(double x, double y, double z, double scale, double cx, double cy) {
            double x1 = x * Math.cos(yaw) + z * Math.sin(yaw);
            double z1 = -x * Math.sin(yaw) + z * Math.cos(yaw);
            double y2 = y * Math.cos(pitch) - z1 * Math.sin(pitch);
            double z2 = y * Math.sin(pitch) + z1 * Math.cos(pitch);
            double persp = 1 / (1 - z2 * 0.15);
            return new double[]{cx + x1 * scale * persp, cy - y2 * scale * persp, z2};
        }

        @Override
        protected void paintComponent(Graphics graphics) {
            super.paintComponent(graphics);
            Graphics2D g = (Graphics2D) graphics;
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            int w = getWidth(), h = getHeight();
            double cx = w / 2.0, cy = h / 2.0 + 10, scale = Math.min(w, h) * 0.24 * zoom;

            g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 16));
            String title = "3D PCA of Embeddings | Java, Python, Esperanto, Balu, Shir Khan";
            g.setColor(Color.DARK_GRAY);
            g.drawString(title, (w - g.getFontMetrics().stringWidth(title)) / 2, 28);

            // bounding box edges
            g.setColor(new Color(200, 200, 200));
            g.setStroke(new BasicStroke(1f));
            for (int a = 0; a < 8; a++) {
                for (int bit = 0; bit < 3; bit++) {
                    if ((a & (1 << bit)) != 0) continue;
                    int b = a | (1 << bit);
                    double[] p = project(sign(a, 0), sign(a, 1), sign(a, 2), scale, cx, cy);
                    double[] q = project(sign(b, 0), sign(b, 1), sign(b, 2), scale, cx, cy);
                    g.drawLine((int) p[0], (int) p[1], (int) q[0], (int) q[1]);
                }
            }

            // axes through the origin with titles
            String[] names = {"PCA Component 1", "PCA Component 2", "PCA Component 3"};
            g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 12));
            for (int c = 0; c < 3; c++) {
                double[] from = new double[3], to = new double[3];
                from[c] = -1;
                to[c] = 1;
                double[] p = project(from[0], from[1], from[2], scale, cx, cy);
                double[] q = project(to[0], to[1], to[2], scale, cx, cy);
                g.setColor(Color.GRAY);
                g.setStroke(new BasicStroke(1.5f));
                g.drawLine((int) p[0], (int) p[1], (int) q[0], (int) q[1]);
                double[] end = project(c == 0 ? 1.12 : 0, c == 1 ? 1.12 : 0, c == 2 ? 1.12 : 0, scale, cx, cy);
                g.drawString(names[c], (int) end[0], (int) end[1]);
            }

            // points, far to near (painter's algorithm)
            double[][] proj = new double[pts.length][];
            for (int i = 0; i < pts.length; i++) proj[i] = project(pts[i][0], pts[i][1], pts[i][2], scale, cx, cy);
            Integer[] order = new Integer[pts.length];
            Arrays.setAll(order, i -> i);
            Arrays.sort(order, Comparator.comparingDouble(i -> proj[i][2]));

            g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 13));
            FontMetrics fm = g.getFontMetrics();
            for (int i : order) {
                double depthScale = 1 + proj[i][2] * 0.15; // nearer points look larger
                int r = (int) Math.round(9 * depthScale);
                int px = (int) proj[i][0], py = (int) proj[i][1];
                g.setColor(colors.get(i % colors.size()));
                g.fillOval(px - r, py - r, 2 * r, 2 * r);
                g.setColor(Color.DARK_GRAY);
                int ly = i % 2 == 1 ? py + r + fm.getAscent() + 2 : py - r - 5; // odd labels below, so close points don't collide
                g.drawString(labels.get(i), px - fm.stringWidth(labels.get(i)) / 2, ly);
            }
        }

        private static double sign(int corner, int bit) {
            return (corner >> bit & 1) == 0 ? -1 : 1;
        }
    }

    public void scatter(List<String> textSamples, double[][] visDims) throws IOException {
        Scatter3DPanel panel = new Scatter3DPanel(visDims, textSamples, COLORS);
        panel.setPreferredSize(new Dimension(1000, 800));

        BufferedImage img = new BufferedImage(1000, 800, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        panel.setSize(1000, 800);
        panel.paint(g);
        g.dispose();
        ImageIO.write(img, "png", new File("embeddings_pca_3d.png"));
        System.out.println("saved embeddings_pca_3d.png");

        if (!java.awt.GraphicsEnvironment.isHeadless()) {
            SwingUtilities.invokeLater(() -> {
                JFrame frame = new JFrame("Embeddings PCA (3D) - drag to rotate, wheel to zoom");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.setContentPane(panel);
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
            });
        }

    }
}
