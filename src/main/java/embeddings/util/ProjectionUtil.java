package embeddings.util;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

public class ProjectionUtil {

    public static double[][] pca(double[][] data, int k) {
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
