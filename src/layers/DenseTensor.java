package layers;

import data.Tensor;
import tools.Activation;
import tools.Config;

public class DenseTensor extends LayerTensor {
    private int size;
    private int input_size; // Necessary to initialize weights

    private Tensor biases; // 1D
    private Tensor weights; // 2D

    private Tensor last_output; // 1D
    private Tensor last_input; // 1D

    private java.util.function.Function<Double, Double> activation_function;
    private java.util.function.Function<Double, Double> activation_derivative;

    public DenseTensor(int size, int input_size) {
        this.size = size;
        init(input_size);

        this.type = Type.DENSE;
        this.activation_function = Activation::relu;
        this.activation_derivative = Activation::derivativeReLU;
    }

    public void init(int input_size) {
        this.input_size = input_size;

        this.biases = new Tensor(
            new int[]{size}
        );
        this.biases.randomise();   

        this.weights = new Tensor(
            new int[]{size, input_size}
        );
        this.weights.randomise();

        this.last_output = new Tensor(
            new int[]{size}
        );

        this.last_input = new Tensor(
            new int[]{input_size}
        );

        biases.randomise();
        weights.randomise();
    }

    public Tensor forward(Tensor input) {
        Tensor output = new Tensor(
            new int[]{this.size}
        );

        for (int neuron = 0; neuron < this.weights.size(0); neuron++) {
            double sum_weighted_input = 0;

            // Sum of every input * corresponding weight
            for (int k = 0; k < this.input_size; k++) {
                sum_weighted_input += input.get(k) * this.weights.get(neuron, k);
            }

            // Add bias and activation function
            double post_activation = this.activation_function.apply(sum_weighted_input + this.biases.get(neuron));
            output.set(post_activation, neuron);
        }

        this.last_input = input;
        this.last_output = output;

        if (Config.verbose()) {
            System.out.println("[Dense Layer] Output:");
            for (double value : output.raw_data()) {
                System.out.printf("%.6f ", value);
            }
            System.out.println("\n====================");
        }

        return output;
    }

    public Tensor backward(Tensor delta_O, double learning_rate) {
        Tensor new_delta = new Tensor(new int[]{this.input_size});

        // For each neuron in this layer
        for (int neuron = 0; neuron < this.size; neuron++) {
            // Compute delta (error)
            // delta_I = delta_O * derivative(last_output)
            double derivative = this.activation_derivative.apply(this.last_output.get(neuron));
            double delta_I = delta_O.get(neuron) * derivative;

            for (int i = 0; i < this.input_size; i++) {
                // Accumulate delta to propagate to previous layer
                new_delta.inc(delta_I * weights.get(neuron, i), i);
                
                // Update weight using gradient descent
                double gradient = delta_I * this.last_input.get(i);
                weights.inc(learning_rate * gradient * -1, neuron, i);
            }

            // Update bias
            biases.inc(learning_rate * delta_I * -1, neuron);
        }

        return new_delta;
    }

    public Tensor getLastOutput() {
        return last_output;
    }

    public java.util.function.Function<Double, Double> getActivationFunction() {
        return activation_function;
    }

    public void setActivationFunction(java.util.function.Function<Double, Double> activation_function) {
        this.activation_function = activation_function;
    }

    public java.util.function.Function<Double, Double> getActivationDerivative() {
        return activation_derivative;
    }

    public void setActivationDerivative(java.util.function.Function<Double, Double> activation_derivative) {
        this.activation_derivative = activation_derivative;
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