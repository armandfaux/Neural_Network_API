package cnn;
public class PoolLayer extends Layer {
    private int poolHeight;
    private int poolWidth;

    private int stride;
    // pooling type ? Max or average
    // pooling "mode" -> valid (only pool complete windows)
    // private int padding;

    public PoolLayer(int poolHeight, int poolWidth) {
        this.type = Type.POOLING;
        this.poolHeight = poolHeight;
        this.poolWidth = poolWidth;
        this.stride = 2;
    }

    public double[][][] forward(double[][][] input) {
        System.out.println("[POOL LAYER] initiating forward pass");
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;

        // Calculate output dimensions (downsampling)
        int outputHeight = (inputHeight - this.poolHeight) / this.stride + 1;
        int outputWidth = (inputWidth - poolWidth) / this.stride + 1;

        System.out.println("outputHeight = " + outputHeight);
        System.out.println("outputWidth = " + outputWidth);

        double[][][] output = new double[input.length][outputHeight][outputWidth];

        // For each feature map
        for (double[][] featureMap : input) {

            // For each pooling window
            for (int outY = 0; outY < outputHeight; outY += this.stride) {
                for (int outX = 0; outX < outputWidth; outX += this.stride) {
                    double max = Double.NEGATIVE_INFINITY;

                    // Find max value in the pooling window (MaxPooling)
                    for (int windowY = 0; windowY < this.poolHeight; windowY++) {
                        for (int windowX = 0; windowX < this.poolWidth; windowX++) {
                            int inY = outY + windowY;
                            int inX = outX + windowX;

                            if (inY < inputHeight && inX < inputWidth) {
                                max = Math.max(max, featureMap[inY][inX]);
                            }
                        }
                    }

                    output[0][outY][outX] = max;
                }
            }
        }

        // DISPLAY OUTPUT
        System.out.println("*****************");
        System.out.println("dimensionality reduction");
        System.out.println("*****************");
        for (double[][] featureMap : output) {
            for (double[] row : featureMap) {
                for (double value : row) {
                    System.out.print(value + " ");
                }
                System.out.println();
            }
        }

        return output;
    }

    public void setStride(int stride) {
        this.stride = stride;
    }

    public void setPoolHeight(int poolHeight) {
        this.poolHeight = poolHeight;
    }

    public void setPoolWidth(int poolWidth) {
        this.poolWidth = poolWidth;
    }

    public int getStride() {
        return this.stride;
    }

    public int getPoolHeight() {
        return this.poolHeight;
    }

    public int getPoolWidth() {
        return this.poolWidth;
    }
}