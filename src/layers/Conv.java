package layers;

import data.Tensor;
import tools.Activation;
import tools.Config;
import tools.Utils;

// Weights initialization : HE
// Biases initialization :  zero
// Activation :             ReLU

public class Conv extends LayerTensor {
    public int kernelNum;
    private int kernelChannels;
    private int kernelHeight;
    private int kernelWidth;

    private Tensor kernels; // [k][channel][h][w]
    private Tensor biases;

    private int output_height;
    private int output_width;

    private int stride;
    private int padding;

    // Cache for backpropagation
    public Tensor input_tensor;

    public Conv(int kernelNum, int channels, int kernelHeight, int kernelWidth) {
        this.type = Type.CONV;

        this.kernelNum = kernelNum;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        this.stride = 1;
        this.padding = 0;

        this.output_height = 0;
        this.output_width = 0;

        this.input_tensor = new Tensor(new int[]{0, 0, 0});

        // Each kernel (filter) is represented by a matrix of weights
        this.kernels = new Tensor(new int[]{
            kernelNum, // Number of kernels
            channels, // Input channels (c_in)
            kernelHeight,
            kernelWidth
        });

        this.biases = new Tensor(new int[]{kernelNum});

        init();
    }

    private void init() {
        this.kernels.init_he(this.kernelChannels * this.kernelHeight * this.kernelWidth);
        this.biases.init_zero();
    }

    public Tensor forward(Tensor input) {
        int c_in = this.kernelChannels;

        // if (input.length != this.kernelChannels) {
        //     System.out.println("[WARNING] input channels do not match kernel channels");
        //     c_in = Math.min(this.kernelChannels, this.input_tensor.length);
        // }

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
        this.output_height = (input.shape()[1] - this.kernelHeight + 2 * this.padding) / this.stride + 1;
        this.output_width = (input.shape()[2] - this.kernelWidth + 2 * this.padding) / this.stride + 1;

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

        Tensor output = new Tensor(new int[]{this.kernelNum, h_out, w_out});

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
                                sum += input.get(channel, inputY, inputX) * this.kernels.get(k, channel, ky, kx);
                            }
                        }

                        output.set(Activation.relu(sum + this.biases.get(k)),k, outputY, outputX);
                    }
                }
            }
        }

        // PRINT FEATURE MAPS
        if (Config.verbose()) {
            output.display();
        }

        return output;
    }

    // To be implemented
    public Tensor backward(Tensor delta_O, double learningRate) {
        // input tensor shape (chanels, height width)
        int c_in = this.input_tensor.shape()[0];
        int h_in = this.input_tensor.shape()[1];
        int w_in = this.input_tensor.shape()[2];

        // output tensor shape (chanels, height width)
        int c_out = this.kernelNum;
        int h_out = this.output_height;
        int w_out = this.output_width;

        // In case of mismatch between input channels and kernel channels
        // if (c_in != this.kernelChannels) {
        //     System.out.println("[WARNING] input channels do not match kernel channels");
        // }

        // delta shapes (N = batch size) :
        // delta_I  (N)[c_in][h_in][w_in]
        // delta_B  [c_out]
        // delta_F  [c_out][c_in][k_h][k_w]
        // delta_O  (N)[c_out][h_out][w_out]
        Tensor delta_I = new Tensor(new int[]{c_in, h_in, w_in});
        double[] delta_B = new double[c_out];
        double[][][][] delta_F =  new double[c_out][c_in][kernelHeight][kernelWidth]; // also called delta K in papers

        // Apply derivative on delta_O, to obtain pre-activation gradient (delta Z)
        delta_O.map(Activation::derivativeReLU);
        delta_I.init_zero();

        // Compute Delta I
        // For each input channel
        for (int c = 0; c < c_in; c++) {

            // For each filter
            for (int k = 0; k < c_out; k++) {

            // For every element of delta_O[k]
                for (int h = 0; h < h_out; h++) {
                    for (int w = 0; w < w_out; w++) {
                        
                        // Convolution of delta_O and rotated filter to compute delta_I
                        for (int k_h = 0; k_h < this.kernelHeight; k_h++) {
                            for (int k_w = 0; k_w < this.kernelWidth; k_w++) {
                                // 180° rotation of filter is like browsing values from the end: [h][w] -> [h][0] -> [0][w] -> [0][0]
                                delta_I.set(delta_O.get(k, h, w) * this.kernels.get(k, c, this.kernelHeight - k_h - 1, this.kernelWidth - k_w - 1),
                                    c, h + k_h, w + k_w
                                );
                                
                                // OLD
                                // delta_I[c][h + k_h][w + k_w] +=
                                // delta_O.get(k, h, w) * this.kernels.get(k, c, this.kernelHeight - k_h - 1, this.kernelWidth - k_w - 1);
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
                    delta_B[k] += delta_O.get(k, h, w); // adding the values of kernel's output delta (kernel, height, width)
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

                                delta_F_sum += this.input_tensor.get(c, in_h, in_w) * delta_O.get(k, h, w);
                            }
                        }
                        delta_F[k][c][y][x] = delta_F_sum;
                    }
                }
            }
        }

        // OPTIMISER STEP : TO BE SEPARATED FROM BACKWARD LATER
        for (int k = 0; k < this.kernelNum; k++) {
            // Substract gradient * learning rate
            this.biases.inc(-delta_B[k] * learningRate, k);

            for (int c = 0; c < this.kernelChannels; c++) {
                for (int y = 0; y < this.kernelHeight; y++) {
                    for (int x = 0; x < this.kernelWidth; x++) {
                        // Update kernels with gradient descent
                        this.kernels.inc(-delta_F[k][c][y][x] * learningRate, k, c, y, x);
                        // OLD
                        // this.kernels[k][c][y][x] -= delta_F[k][c][y][x] * learningRate;
                    }
                }
            }
        }

        return delta_I;
    }

    public void displayKernels() {
        System.out.println("[Conv Layer] Kernels:");

        this.kernels.display();

        // LEGACY
        // for (int k = 0; k < this.kernelNum; k++) {
        //     System.out.println("Kernel " + k + ":");
        //     for (int y = 0; y < this.kernelHeight; y++) {
        //         for (int x = 0; x < this.kernelWidth; x++) {
        //             System.out.print(String.format("%.3f", this.kernels[k][y][x]) + " ");
        //         }
        //         System.out.println();
        //     }
        //     System.out.println("Bias: " + String.format("%.3f", this.biases[k]));
        //     System.out.println();
        // }
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