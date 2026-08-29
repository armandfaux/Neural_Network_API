package layers;

import data.Tensor;
import tools.Activation;
import tools.Config;

// Weights initialization : HE
// Biases initialization :  zero
// Activation :             ReLU

public class Conv extends LayerTensor {
    public int kernelCount;
    private int channelCount;
    private int kernelHeight;
    private int kernelWidth;

    private Tensor kernels; // [k][channel][h][w]
    private Tensor biases;

    private int stride;
    
    // Unique padding applied in all 4 directions
    private int padding;

    // Cache for backpropagation
    public Tensor cachedInput;
    public Tensor cachedOutput;

    public Conv(int kernelCount, int channelCount, int kernelHeight, int kernelWidth) {
        validateConstructorArguments(kernelCount, channelCount, kernelHeight, kernelWidth);

        this.type = Type.CONV;

        this.kernelCount = kernelCount;
        this.channelCount = channelCount;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;

        this.stride = 1;
        this.padding = 0;

        this.cachedInput = null;
        this.cachedOutput = null;

        // Each kernel (filter) is represented by a matrix of weights
        this.kernels = new Tensor(new int[]{
            kernelCount, // Number of kernels
            channelCount, // Input channels (c_in)
            kernelHeight,
            kernelWidth
        });

        this.biases = new Tensor(new int[]{kernelCount});

        init();
    }

    private void validateConstructorArguments(int kernelCount, int channelCount, int kernelHeight, int kernelWidth) {
        if (kernelCount <= 0) {
            throw new IllegalArgumentException("Conv.Conv() - kernelCount must be positive");
        }

        if (channelCount <= 0) {
            throw new IllegalArgumentException("Conv.Conv() - channelCount must be positive");
        }

        if (kernelHeight <= 0) {
            throw new IllegalArgumentException("Conv.Conv() - kernelHeight must be positive");
        }

        if (kernelWidth <= 0) {
            throw new IllegalArgumentException("Conv.Conv() - kernelWidth must be positive");
        }
    }

    private void init() {
        this.kernels.init_he(this.channelCount * this.kernelHeight * this.kernelWidth);
        this.biases.init_zero();
    }

    public Tensor forward(Tensor input) {
        validateForwardInput(input);

        this.cachedInput = input;

        int h_out = (input.shape()[1] - this.kernelHeight + 2 * this.padding) / this.stride + 1;
        int w_out = (input.shape()[2] - this.kernelWidth + 2 * this.padding) / this.stride + 1;

        if (Config.verbose()) {
            System.out.println("[Conv Layer] initiating forward pass");
            System.out.println("outputHeight = " + h_out);
            System.out.println("outputWidth = " + w_out);
        }

        Tensor output = new Tensor(new int[]{this.kernelCount, h_out, w_out});

        // For each kernel
        for (int k = 0; k < this.kernelCount; k++) {

            // Scan the input
            for (int outputY = 0; outputY < h_out; outputY++) {
                for (int outputX = 0; outputX < w_out; outputX++) {
                    double sum = 0.0;
                    for (int c = 0; c < this.channelCount; c++) {

                        // Compute product of kernel and input region
                        for (int ky = 0; ky < this.kernelHeight; ky++) {
                            for (int kx = 0; kx < this.kernelWidth; kx++) {
                                int inputY = outputY * this.stride + ky - this.padding;
                                int inputX = outputX * this.stride + kx - this.padding;

                                // Ignore padding zeros
                                if (inputY >= 0 && inputY < input.size(1) && inputX >= 0 && inputX < input.size(2)) {
                                    sum += input.get(c, inputY, inputX) * this.kernels.get(k, c, ky, kx);
                                }
                            }
                        }
                    }
                    output.set(Activation.relu(sum + this.biases.get(k)), k, outputY, outputX);
                }
            }
        }

        // PRINT FEATURE MAPS
        if (Config.verbose()) {
            output.display();
        }

        this.cachedOutput = output;

        return output;
    }

    // To be implemented
    public Tensor backward(Tensor delta_O, double learningRate) {
        // input tensor shape (chanels, height width)
        int c_in = this.cachedInput.shape()[0];
        int h_in = this.cachedInput.shape()[1];
        int w_in = this.cachedInput.shape()[2];

        // output tensor shape (chanels, height width)
        int c_out = this.kernelCount;
        int h_out = this.cachedOutput.shape()[1];
        int w_out = this.cachedOutput.shape()[2];

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
        for (int k = 0; k < c_out; k++) {
            for (int h = 0; h < h_out; h++) {
                for (int w = 0; w < w_out; w++) {
                    double z = Activation.derivativeReLU(this.cachedOutput.get(k, h, w)) * delta_O.get(k, h, w);
                    delta_O.set(z, k, h, w);
                }
            }
        }

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
                                delta_I.inc(delta_O.get(k, h, w) * this.kernels.get(k, c, this.kernelHeight - k_h - 1, this.kernelWidth - k_w - 1),
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

                                delta_F_sum += this.cachedInput.get(c, in_h, in_w) * delta_O.get(k, h, w);
                            }
                        }
                        delta_F[k][c][y][x] = delta_F_sum;
                    }
                }
            }
        }

        // OPTIMISER STEP : TO BE SEPARATED FROM BACKWARD LATER
        for (int k = 0; k < this.kernelCount; k++) {
            // Substract gradient * learning rate
            this.biases.inc(-delta_B[k] * learningRate, k);

            for (int c = 0; c < this.channelCount; c++) {
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

    private void validateForwardInput(Tensor input) {
        if (input == null) {
            throw new IllegalArgumentException("Conv.validateForwardInput() - input cannot be null");
        }

        if (input.shape().length != 3) {
            throw new IllegalArgumentException("Conv.validateForwardInput() - input shape must be 3D");
        }

        if (input.size(0) != this.channelCount) {
            throw new IllegalArgumentException("Conv.validateForwardInput() - input channels do not match kernel channels");
        }

        if (input.size(1) + 2 * padding < this.kernelHeight) {
            throw new IllegalArgumentException("Conv.validateForwardInput() - input shape height is too small to perform a convolution");
        }

        if (input.size(2) + 2 * padding < this.kernelWidth) {
            throw new IllegalArgumentException("Conv.validateForwardInput() - input shape width is too small to perform a convolution");
        }

        // Strict convolution
        if ((input.size(1) - this.kernelHeight + 2 * this.padding) % this.stride != 0) {
            throw new IllegalArgumentException("Conv.validateForwardInput() - stride does not evenly tile input height");
        }
        
        if ((input.size(2) - this.kernelWidth + 2 * this.padding) % this.stride != 0) {
            throw new IllegalArgumentException("Conv.validateForwardInput() - stride does not evenly tile input width");
        }
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

    public void setKernels(Tensor kernels) {
        validateKernels(kernels);
        this.kernels = kernels;
    }

    public Tensor getKernels() {
        return kernels;
    }

    public void setBiases(Tensor biases) {
        validateBiases(biases);
        this.biases = biases;
    }

    private void validateKernels(Tensor kernels) {
        if (kernels == null) {
            throw new IllegalArgumentException("Conv.setKernels() - kernels cannot be null");
        }

        int[] shape = kernels.shape();

        if (shape.length != 4) {
            throw new IllegalArgumentException("Conv.setKernels() - kernels must have shape [kernelCount, channelCount, kernelHeight, kernelWidth]");
        }

        if (
            shape[0] != this.kernelCount ||
            shape[1] != this.channelCount ||
            shape[2] != this.kernelHeight ||
            shape[3] != this.kernelWidth
        ) {
            throw new IllegalArgumentException("Conv.setKernels() - kernels shape does not match layer configuration");
        }
    }

    private void validateBiases(Tensor biases) {
        if (biases == null) {
            throw new IllegalArgumentException("Conv.setBiases() - biases cannot be null");
        }

        int[] shape = biases.shape();

        if (shape.length != 1 || shape[0] != this.kernelCount) {
            throw new IllegalArgumentException("Conv.setBiases() - biases must have shape [kernelCount]");
        }
    }

    public Tensor getBiases() {
        return biases;
    }

    public void setStride(int stride) {
        if (stride < 1) {
            throw new IllegalArgumentException("Conv.setStride() - stride must be positive");
        }

        this.stride = stride;
    }

    public void setPadding(int padding) {
        if (padding < 0) {
            throw new IllegalArgumentException("Conv.setPadding() - padding cannot be negative");
        }

        this.padding = padding;
    }

    public int getStride() {
        return stride;
    }

    public int getPadding() {
        return padding;
    }
}
