package layers;

import data.Tensor;
import tools.Activation;
import tools.Config;

public class DenseTensor extends LayerTensor {
    private int size;
    public int previousLayerSize;

    private Tensor biases; // 1D
    private Tensor weights; // 2D

    private Tensor lastOutput; // 3D -> 1D
    private Tensor lastInput; // 3D -> 1D

    public DenseTensor(int size, int previousLayerSize) {
        this.size = size;
        init(previousLayerSize);
    }

    public void init(int previousLayerSize) {
        this.previousLayerSize = previousLayerSize;

        this.biases = new Tensor(
            new int[]{size}
        );

        this.weights = new Tensor(
            new int[]{size, previousLayerSize}
        );

        this.lastOutput = new Tensor(
            new int[]{size}
        );

        this.lastInput = new Tensor(
            new int[]{previousLayerSize}
        );

        biases.randomise();
        weights.randomise();
    }

    public Tensor forward(Tensor input) {
        // if (input[0][0].length != this.previousLayerSize) {
        //     System.out.println("[WARNING] Adjusting layer size to " + input[0][0].length);
        //     init(input[0][0].length);
        // }

        // (can also use the existing member lastOutput instead)
        Tensor output = new Tensor(
            new int[]{this.size}
        );

        for (int neuron = 0; neuron < this.weights.size(0); neuron++) {
            double sum_weighted_input = 0;

            // Sum of every input * corresponding weight
            for (int k = 0; k < this.previousLayerSize; k++) {
                sum_weighted_input += input.get(k) * this.weights.get(neuron, k);
            }

            // Add bias and activation function
            double post_activation = Activation.sigmoid(sum_weighted_input + this.biases.get(neuron));
            output.set(post_activation, neuron);
        }

        this.lastInput = input;
        this.lastOutput = output;

        if (Config.verbose()) {
            System.out.println("[Dense Layer] Output:");
            for (double value : output.raw_data()) {
                System.out.printf("%.6f ", value);
            }
            System.out.println("\n====================");
        }

        return output;
    }

    public Tensor backward(Tensor delta_O, double learningRate) {
        Tensor newDelta = new Tensor(new int[]{this.previousLayerSize});

        // For each neuron in this layer
        for (int neuron = 0; neuron < this.size; neuron++) {
            // Compute delta (error)
            double derivative = Activation.derivativeSigmoid(this.lastOutput.get(neuron));
            double delta_I = delta_O.get(neuron) * derivative;

            for (int i = 0; i < this.previousLayerSize; i++) {
                // Accumulate delta to propagate to previous layer
                newDelta.inc(delta_I * weights.get(neuron, i), i);
                
                // Update weight using gradient descent
                double gradient = delta_I * this.lastInput.get(i);
                
                // weights[neuron][i] -= learningRate * gradient; // OLD IMPLEMENTATION
                weights.inc(learningRate * gradient * -1, neuron, i);
            }

            // Update bias
            biases.inc(learningRate * delta_I * -1, neuron);
        }

        return newDelta;
    }

    public Tensor getLastOutput() {
        return lastOutput;
    }

    public void display() {
        System.out.println("=== Layer Details ===");
    
        System.out.printf("%-10s %-15s %s\n", "Neuron", "Bias", "Weights");
        System.out.println("------------------------------------------------------");
    
        // LEGACY
        // for (int i = 0; i < biases.length; i++) {
        //     // Print neuron index and bias
        //     System.out.printf("%-10d %-15.6f ", i, biases[i]);
    
        //     // Print all weights for this neuron
        //     for (int j = 0; j < weights[i].length; j++) {
        //         System.out.printf("%.6f ", weights[i][j]);
        //     }
        //     System.out.println();
        // }

        System.out.println("======================================================\n");
    }

    public Tensor getBiases() {
        return biases;
    }

    public void setBiases(Tensor biases) {
        this.biases = biases;
    }

    public Tensor getWeights() {
        return weights;
    }

    public void setWeights(Tensor weights) {
        this.weights = weights;
    }
}