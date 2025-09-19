package cnn;
public class Main {
    public static void main(String[] args) {
        // Example usage
        ConvLayer convLayer = new ConvLayer(1, 1, 2, 2);
        // Backward pass to be implemented
        // PoolLayer poolLayer = new PoolLayer(2, 2);
        FlattenLayer flattenLayer = new FlattenLayer();
        DenseLayer denseLayer = new DenseLayer(10, 2);
        Config.setVerbose(true);

        double[][][] input = new double[1][3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                input[0][i][j] = i + j;
            }
        }

        // Create a CNN and add layers
        CNN network = new CNN();
        network.addLayer(convLayer);
        // network.addLayer(poolLayer);
        network.addLayer(flattenLayer);
        network.addLayer(denseLayer);

        // Forward pass
        double[][][] cnnOutput = network.forward(input);
        network.backward(cnnOutput);
    }
}
