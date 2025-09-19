package cnn;
public class Utils {
    // public static double matrixDotProduct(double[][] A, double[][] B) {
    //     int rows = A.length;
    //     int cols = A[0].length;
    //     double sum = 0.0;

    //     for (int i = 0; i < rows; i++) {
    //         for (int j = 0; j < cols; j++) {
    //             sum += A[i][j] * B[i][j];
    //         }
    //     }

    //     return sum;
    // }

    // Rework to input any D Tensor
    public static double matrixElementProduct(double[][] A, double[][] B) {
        int rows = A.length;
        int cols = A[0].length;
        double sum = 0.0;

        for (int h = 0; h < rows; h++) {
            for (int w = 0; w < cols; w++) {
                sum += A[h][w] * B[h][w];
            }
        }

        return sum;
    }

    // return a zero matrix of input shape
    public static double[][] zeroMatrix(int rows, int cols) {
        double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = 0.0;
            }
        }
        return matrix;
    }

    // Rework to input any D Tensor
    public static double[][]getSubMatrix(double[][] M, int[] shape, int[] padding) {
        // Check input shapes
        if (M == null || shape == null || padding == null) {
            throw new IllegalArgumentException("Input arrays must not be null");
        }
        if (shape.length != 2 || padding.length != 2) {
            throw new IllegalArgumentException("Shape and padding must be 2D arrays");
        }
        if (padding[0] < 0 || padding[1] < 0 || padding[0] + shape[0] > M.length || padding[1] + shape[1] > M[0].length) {
            throw new IllegalArgumentException("Invalid shape or padding");
        }

        double[][] sub_M = new double[shape[0]][shape[1]];

        for (int h = 0; h < shape[0]; h++) {
            for (int w = 1; w < shape[1]; w++) {
                sub_M[h][w] = M[h + padding[0]][w + padding[1]];
            }
        }

        return sub_M;
    }

    public static void displayFeatureMaps(double[][][] featureMaps) {
        System.out.println("*****************");
        System.out.println(featureMaps.length + " feature maps:");
        System.out.println("*****************");

        for (double[][] featureMap : featureMaps) {
            for (double[] row : featureMap) {
                for (double value : row) {
                    System.out.print(String.format("%.3f", value) + " ");
                }
                System.out.println();
            }
            System.out.println("*****************\n");
        }
    }
}
