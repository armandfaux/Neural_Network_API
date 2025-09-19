package cnn;
import java.util.Random;

class ConvLayer extends Layer {
    public int kernelNum;
    private int kernelChannels;
    private int kernelHeight;
    private int kernelWidth;

    public double[][][][] kernels; // [k][channel][h][w]
    public double[] biases; // tmp public access for debugging

    private int output_height;
    private int output_width;

    private int stride;
    private int padding;

    // Cache for backpropagation
    public double[][][] input_tensor;

    public ConvLayer(int kernelNum, int channels, int kernelHeight, int kernelWidth) {
        this.type = Type.CONV;

        this.kernelNum = kernelNum;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        this.stride = 1;
        this.padding = 0;

        this.output_height = 0;
        this.output_width = 0;

        this.input_tensor = new double[0][0][0];

        // Each kernel (filter) is represented by a matrix of weights
        this.kernels = new double[kernelNum][channels][kernelHeight][kernelWidth];
        this.biases = new double[kernelNum];

        init();
    }

    private void init() {
        Random rand = new Random();
        for (int k = 0; k < this.kernelNum; k++) {
            this.biases[k] = 0; // Initialize biases

            for (int c = 0; c < this.kernelChannels; c++) {
                for (int y = 0; y < this.kernelHeight; y++) {
                    for (int x = 0; x < this.kernelWidth; x++) {
                        this.kernels[k][c][y][x] = rand.nextGaussian() * 0.01;
                    }
                }
            }
        }
    }

    public double[][][] forward(double[][][] input) {
        int c_in = this.kernelChannels;

        if (input.length != this.kernelChannels) {
            System.out.println("[WARNING] input channels do not match kernel channels");
            c_in = Math.min(this.kernelChannels, this.input_tensor.length);
        }

        // print input
        // for (double[][] featureMap : input) {
        //     for (double[] row : featureMap) {
        //         for (double value : row) {
        //             System.out.print(value + " ");
        //         }
        //         System.out.println();
        //     }
        //     System.out.println();
        // }

        this.input_tensor = input;
        this.output_height = (input[0].length - this.kernelHeight + 2 * this.padding) / this.stride + 1;
        this.output_width = (input[0][1].length - this.kernelWidth + 2 * this.padding) / this.stride + 1;

        int h_out = this.output_height;
        int w_out = this.output_width;

        if (Config.verbose()) {
            System.out.println("[Conv Layer] initiating forward pass");
            System.out.println("outputHeight = " + h_out);
            System.out.println("outputWidth = " + w_out);
        }

        if (h_out < 1 || w_out < 1 || this.kernelNum < 1) {
            throw new IllegalArgumentException("Invalid output dimensions");
        }

        double[][][] output = new double[this.kernelNum][h_out][w_out];

        // For each kernel
        for (int k = 0; k < this.kernelNum; k++) {

            // Scan the input
            for (int channel = 0; channel < c_in; channel++) {
                for (int outputY = 0; outputY < h_out; outputY++) {
                    for (int outputX = 0; outputX < w_out; outputX++) {
                        double sum = 0.0;

                        // Compute product of kernel and input region
                        for (int ky = 0; ky < this.kernelHeight; ky++) {
                            for (int kx = 0; kx < this.kernelWidth; kx++) {
                                int inputY = outputY * this.stride + ky - this.padding;
                                int inputX = outputX * this.stride + kx - this.padding;
                                sum += input[channel][inputY][inputX] * this.kernels[k][channel][ky][kx];
                            }
                        }

                        output[k][outputY][outputX] = Activation.relu(sum + this.biases[k]);
                    }
                }
            }
        }

        // PRINT FEATURE MAPS
        if (Config.verbose()) {
            Utils.displayFeatureMaps(output);
        }

        return output;
    }

    // To be implemented
    public double[][][] backward(double[][][] delta_O, double learningRate) {
        // input tensor shape (chanels, height width)
        int c_in = this.input_tensor.length;
        int h_in = this.input_tensor[0].length;
        int w_in = this.input_tensor[0][0].length;

        // output tensor shape (chanels, height width)
        int c_out = this.kernelNum;
        int h_out = this.output_height;
        int w_out = this.output_width;

        // In case of mismatch between input channels and kernel channels
        if (c_in != this.kernelChannels) {
            System.out.println("[WARNING] input channels do not match kernel channels");
        }

        System.out.println("Shapes :");
        System.out.println("c_in =" + c_in);
        System.out.println("h_in =" + h_in);
        System.out.println("w_in =" + w_in);
        System.out.println("c_out =" + c_out);
        System.out.println("h_out =" + h_out);
        System.out.println("w_out =" + w_out);

        System.out.println("\n************");

        // System.out.println("\nDelta_O:");
        // Utils.displayFeatureMaps(delta_O);
        // System.out.println("\ninput tensor:");
        // Utils.displayFeatureMaps(this.input_tensor);

        // delta shapes (N = batch size) :
        // delta_I  (N)[c_in][h_in][w_in]
        // delta_B  [c_out]
        // delta_F  [c_out][c_in][k_h][k_w]
        // delta_O  (N)[c_out][h_out][w_out]
        double[] delta_B = new double[c_out];
        double[][][] delta_I = new double[c_in][h_in][w_in];
        double[][][][] delta_F =  new double[c_out][c_in][kernelHeight][kernelWidth]; // also called delta K in papers

        // Apply derivative on delta_O, to obtain pre-activation gradient (delta Z)

        System.out.println("Delta_O before ReLU derivative:");
        Utils.displayFeatureMaps(delta_O);

        for (int c = 0; c < c_out; c++) {
            for (int h = 0; h < h_out; h++) {
                for (int w = 0; w < w_out; w++) {
                    delta_O[c][h][w] = Activation.reluDerivative(delta_O[c][h][w]);
                }
            }
        }

        // Compute Delta I
        // For each input channel
        for (int c = 0; c < c_in; c++) {
            delta_I[c] = Utils.zeroMatrix(h_in, w_in);

            // For each filter
            for (int k = 0; k < c_out; k++) {

            // For every element of delta_O[k]
                for (int h = 0; h < h_out; h++) {
                    for (int w = 0; w < w_out; w++) {
                        
                        // Convolution of delta_O and rotated filter to compute delta_I
                        for (int k_h = 0; k_h < this.kernelHeight; k_h++) {
                            for (int k_w = 0; k_w < this.kernelWidth; k_w++) {
                                // 180° rotation of filter is like browsing values from the end: [h][w] -> [h][0] -> [0][w] -> [0][0]
                                delta_I[c][h + k_h][w + k_w] +=
                                delta_O[k][h][w] * this.kernels[k][c][this.kernelHeight - k_h - 1][this.kernelWidth - k_w - 1];
                            }
                        }
                    }
                }
            }
        }

        // Compute delta_B and delta_F (for each filter)
        for (int k = 0; k < c_out; k++) {
            // Compute bias gradient (sum of elements in dO[k])
            delta_B[k] = 0;
            for (int h = 0; h < h_out; h++) {
                for (int w = 0; w < w_out; w++) {
                    delta_B[k] += delta_O[k][h][w]; // adding the values of kernel's output delta (kernel, height, width)
                }
            }

            // Compute delta_F for each input chanel
            for (int c = 0; c < c_in; c++) {

                // Y padding applied to input tensor
                for (int y = 0; y < this.kernelHeight; y++) {

                    // X padding applied to input tensor
                    for (int x = 0; x < this.kernelWidth; x++) {
                        
                        double delta_F_sum = 0;
                        // Compute local gradient
                        for (int h = 0; h < h_out; h++) {
                            for (int w = 0; w < w_out; w++) {
                                int in_h = h * this.stride + y - padding;
                                int in_w = w * this.stride + x - padding;

                                delta_F_sum += this.input_tensor[c][in_h][in_w] * delta_O[k][h][w];
                            }
                        }
                        delta_F[k][c][y][x] = delta_F_sum;
                    }
                }
            }

            Utils.displayFeatureMaps(delta_F[k]);
        }

        // OPTIMISER STEP : TO BE SEPARATED FROM BACKWARD LATER
        for (int k = 0; k < this.kernelNum; k++) {
            biases[k] -= delta_B[k] * learningRate;

            for (int c = 0; c < this.kernelChannels; c++) {
                for (int y = 0; y < this.kernelHeight; y++) {
                    for (int x = 0; x < this.kernelWidth; x++) {
                        this.kernels[k][c][y][x] -= delta_F[k][c][y][x] * learningRate;
                    }
                }
            }
        }

        return delta_I;
    }

    public void displayKernels() {
        System.out.println("[Conv Layer] Kernels:");
        for (int k = 0; k < this.kernelNum; k++) {
            System.out.println("Kernel " + k + ":");
            for (int y = 0; y < this.kernelHeight; y++) {
                for (int x = 0; x < this.kernelWidth; x++) {
                    System.out.print(String.format("%.3f", this.kernels[k][y][x]) + " ");
                }
                System.out.println();
            }
            System.out.println("Bias: " + String.format("%.3f", this.biases[k]));
            System.out.println();
        }
    }

    // Setters and getters for layer properties
    public void setStride(int stride) {
        this.stride = stride;
    }

    public void setPadding(int padding) {
        this.padding = padding;
    }

    public int getStride() {
        return stride;
    }

    public int getPadding() {
        return padding;
    }
}