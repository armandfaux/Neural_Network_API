package cnn;
public class FlattenLayer extends Layer {
    private int c_in;
    private int h_in;
    private int w_in;

    // Converts 3D vector into 1D array
    public double[] flatten_3d(double[][][] data) {
        int shape[] = new int[data.length * data[0].length * data[0][0].length];
        double[] flat_vector = new double[shape[0] * shape[1] * shape[2]];

        for (int z = 0; z < shape[0]; z++) {
            for (int y = 0; y < shape[1]; y++) {
                for (int x = 0; x < shape[2]; x++) {
                    flat_vector[z * shape[0] + y * shape[1] + x * shape[2]] = data[z][y][x];
                }
            }
        }

        return flat_vector;
    }

    public double[][][] forward(double[][][] input) {
        this.c_in = input.length;
        this.h_in = input[0].length;
        this.w_in = input[0][0].length;

        double[][][] output = new double[1][1][c_in * h_in * w_in];

        for (int z = 0; z < c_in; z++) {
            for (int y = 0; y < h_in; y++) {
                for (int x = 0; x < w_in; x++) {
                    int index = z * (h_in * w_in) + y * w_in + x;
                    output[0][0][index] = input[z][y][x];
                }
            }
        }

        return output;
    }

    public double[][][] backward(double[][][] gradient, double learningRate) {
        int c_in = this.c_in;
        int h_in = this.h_in;
        int w_in = this.w_in;

        double[][][] new_delta = new double[c_in][h_in][w_in];
        int delta_idx = 0;

        for (int c = 0; c < c_in; c++) {
            for (int h = 0; h < h_in; h++) {
                for (int w = 0; w < w_in; w++) {
                    new_delta[c][h][w] = gradient[0][0][delta_idx++];
                }
            }
        }

        return new_delta;
    }
}
