package core;

import data.Tensor;
import layers.Conv;
import layers.Dense;
import layers.DenseTensor;
import layers.Flatten;
import tools.Activation;
import tools.Config;
import tools.Utils;

public class Main {
    public static void main(String[] args) {
        NN network = new NN();

        Tensor[] inputs = {
            new Tensor(new int[]{2}, new double[]{0.0, 0.0}),
            new Tensor(new int[]{2}, new double[]{0.0, 1.0}),
            new Tensor(new int[]{2}, new double[]{1.0, 0.0}),
            new Tensor(new int[]{2}, new double[]{1.0, 1.0}),
        };

        Tensor[] expectedOutputs = {
            new Tensor(new int[]{1}, new double[]{0.0}),
            new Tensor(new int[]{1}, new double[]{1.0}),
            new Tensor(new int[]{1}, new double[]{1.0}),
            new Tensor(new int[]{1}, new double[]{0.0}),
        };

        DenseTensor denseLayer1 = new DenseTensor(2, 2);
        DenseTensor denseLayer2 = new DenseTensor(4, 2);
        DenseTensor denseLayer4 = new DenseTensor(1, 4);

        // todo pair activation function with derivative
        denseLayer4.setActivationFunction(Activation::sigmoid);
        denseLayer4.setActivationDerivative(Activation::derivativeSigmoid);

        network.addLayer(denseLayer1);
        network.addLayer(denseLayer2);
        network.addLayer(denseLayer4);

        for (int epoch = 0; epoch < 100000; epoch++) {
            Tensor output = network.forward(inputs[epoch % inputs.length]);
            Tensor lossGradient = output.subtract(expectedOutputs[epoch % inputs.length]);
            network.backward(lossGradient);
        }

        System.out.println("Final outputs after training:");
        Tensor finalOutput = network.forward(inputs[0]);
        finalOutput.display();

        System.out.println("Final outputs after training:");
        finalOutput = network.forward(inputs[1]);
        finalOutput.display();

        System.out.println("Final outputs after training:");
        finalOutput = network.forward(inputs[2]);
        finalOutput.display();

        System.out.println("Final outputs after training:");
        finalOutput = network.forward(inputs[3]);
        finalOutput.display();
    }
}
