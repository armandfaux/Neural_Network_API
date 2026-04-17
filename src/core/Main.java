package core;

import data.Tensor;
import layers.DenseTensor;
import tools.Activation;

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

        DenseTensor denseLayer1 = new DenseTensor(4, 2);
        DenseTensor denseLayer2 = new DenseTensor(2, 4);
        DenseTensor denseLayer4 = new DenseTensor(1, 2);

        // todo pair activation function with derivative
        denseLayer4.setActivationFunction(Activation::relu);
        denseLayer4.setActivationDerivative(Activation::derivativeReLU);

        network.addLayer(denseLayer1);
        network.addLayer(denseLayer2);
        network.addLayer(denseLayer4);

        // for (int epoch = 0; epoch < 10; epoch++) {
        //     System.out.println(epoch % inputs.length);
        //     inputs[epoch % inputs.length].display();
        //     Tensor output = network.forward(inputs[epoch % inputs.length]);

        //     Tensor lossGradient = output.subtract(expectedOutputs[epoch % inputs.length]);
        //     // System.out.println("OUTPUT");
        //     // output.display();
        //     // System.out.println("LOSS GRADIENT");
        //     // lossGradient.display();
        //     network.backward(lossGradient);
        // }

        System.out.println("Final outputs after training:");
        Tensor finalOutput = network.forward(inputs[0]);
        finalOutput.display();

        Tensor output2 = network.forward(inputs[1]);
        output2.display();

        Tensor output3 = network.forward(inputs[2]);
        output3.display();

        Tensor output4= network.forward(inputs[3]);
        output4.display();

    }
}
