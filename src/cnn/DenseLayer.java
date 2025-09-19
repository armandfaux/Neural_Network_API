package cnn;
public class DenseLayer extends Layer {
    private int size;
    public int previousLayerSize;

    private double[] biases;
    private double[][] weights;

    private double[][][] lastOutput;
    private double[][][] lastInput;

    public DenseLayer(int size, int previousLayerSize) {
        this.size = size;
        init(previousLayerSize);
    }

    public void init(int previousLayerSize) {
        this.previousLayerSize = previousLayerSize;

        this.biases = new double[size];
        this.weights = new double[size][previousLayerSize];

        this.lastOutput = new double[1][1][size];
        this.lastInput = new double[1][1][previousLayerSize];

        this.type = Type.DENSE;

        // Set biases to random values from 0 to 1
        for (int i = 0; i < this.size; i++) {
            biases[i] = Math.random();
        }

        // Set all weights to random values from 0 to 1
        for (int i = 0; i < this.size; i++) {
            for (int j = 0; j < this.previousLayerSize; j++) {
                // this.weights[i][j] = Math.random();
                this.weights[i][j] = (Math.random() - 0.5) * 2;
            }
        }
    }

    public double[][][] forward(double[][][] input) {
        System.out.println("[Dense Layer] Initiating forward pass");
        // display();

        if (input[0][0].length != this.previousLayerSize) {
            throw new IllegalArgumentException("Input size does not match the previous layer size.");
        }

        // (can also use the existing member lastOutput instead)
        double[][][] output = new double[1][1][this.size];

        for (int neuron = 0; neuron < this.weights.length; neuron++) {
            double sum_weighted_input = 0;

            // Sum of every input * corresponding weight
            for (int k = 0; k < this.previousLayerSize; k++) {
                sum_weighted_input += input[0][0][k] * this.weights[neuron][k];
            }

            // Add bias and activation function
            output[0][0][neuron] = Activation.sigmoid(sum_weighted_input + this.biases[neuron]);
        }

        this.lastInput = input;
        this.lastOutput = output;

        if (Config.verbose()) {
            System.out.println("[Dense Layer] Output:");
            for (double value : output[0][0]) {
                System.out.printf("%.6f ", value);
            }
            System.out.println("\n====================");
        }

        return output;
    }

    public double[][][] backward(double[][][] delta, double learningRate) {
        System.out.println("[Dense Layer] delta");
        Utils.displayFeatureMaps(delta);

        double[][][] newDelta = new double[1][1][this.previousLayerSize];

        // For each neuron in this layer
        for (int neuron = 0; neuron < this.size; neuron++) {
            // Compute delta (error)
            double derivative = Activation.derivativeSigmoid(this.lastOutput[0][0][neuron]);
            double delta_i = delta[0][0][neuron] * derivative;

            for (int i = 0; i < this.previousLayerSize; i++) {
                // Update weight using gradient descent
                double gradient = delta_i * this.lastInput[0][0][i];
                weights[neuron][i] -= learningRate * gradient;

                // Accumulate delta to propagate to previous layer
                newDelta[0][0][i] += delta_i * weights[neuron][i];
            }

            // Update bias
            biases[neuron] -= learningRate * delta_i;
        }

        System.out.println("[Dense Layer] new delta");
        Utils.displayFeatureMaps(newDelta);

        return newDelta;
    }

    public double[][][] getLastOutput() {
        return lastOutput;
    }

    public void display() {
        System.out.println("=== Layer Details ===");
    
        System.out.printf("%-10s %-15s %s\n", "Neuron", "Bias", "Weights");
        System.out.println("------------------------------------------------------");
    
        for (int i = 0; i < biases.length; i++) {
            // Print neuron index and bias
            System.out.printf("%-10d %-15.6f ", i, biases[i]);
    
            // Print all weights for this neuron
            for (int j = 0; j < weights[i].length; j++) {
                System.out.printf("%.6f ", weights[i][j]);
            }
            System.out.println();
        }

        System.out.println("======================================================\n");
    }
}