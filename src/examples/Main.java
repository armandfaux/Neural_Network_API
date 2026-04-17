// package examples;

// import core.NN;
// import layers.Conv;
// import layers.Dense;
// import layers.Flatten;
// import tools.Config;
// import tools.Utils;

// public class Main {
//     public static void main(String[] args) {
//         // Example usage
//         Conv convLayer = new Conv(1, 1, 2, 2);
//         // Backward pass to be implemented
//         // PoolLayer poolLayer = new PoolLayer(2, 2);
//         Flatten flattenLayer = new Flatten();
//         Dense denseLayer = new Dense(10, 2);
//         Config.setVerbose(false);

//         double[][][] input = new double[1][3][3];
//         for (int i = 0; i < 3; i++) {
//             for (int j = 0; j < 3; j++) {
//                 input[0][i][j] = i + j;
//             }
//         }

//         // Create a CNN and add layers
//         NN network = new NN();
//         network.addLayer(convLayer);
//         // network.addLayer(poolLayer);
//         network.addLayer(flattenLayer);
//         network.addLayer(denseLayer);

//         // Forward pass

//         double[][][] expectedOutput = {{{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}}};
//         double[][][] delta = new double[1][1][10];

//         double[][][] cnnOutput = network.forward(input);

//         for (int e = 0; e < 1000; e++) {
//             cnnOutput = network.forward(input);
//             for (int i = 0; i < 10; i++) {
//                 delta[0][0][i] = cnnOutput[0][0][i] - expectedOutput[0][0][i];
//             }
//             network.backward(delta);
//         }

//         Utils.displayFeatureMaps(cnnOutput);
//     }
// }
