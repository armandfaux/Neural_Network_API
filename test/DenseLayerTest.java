// package test;
// import static org.junit.jupiter.api.Assertions.*;

// import javax.swing.Action;

// import org.junit.jupiter.api.Test;

// import layers.Dense;
// import tools.Activation;

// public class DenseLayerTest {

//     // ===== Helpers =====
//     // Build 1×1×D input from flat vector
//     private static double[][][] asInput(double[] v) {
//         double[][][] x = new double[1][1][v.length];
//         System.arraycopy(v, 0, x[0][0], 0, v.length);
//         return x;
//     }

//     // Mean squared error: 0.5 * sum (a - t)^2
//     private static double mse(double[] a, double[] t) {
//         double s = 0.0;
//         for (int i = 0; i < a.length; i++) {
//             double d = a[i] - t[i];
//             s += 0.5 * d * d;
//         }
//         return s;
//     }

//     // Upstream delta for MSE with identity loss derivative wrt activation: dL/dA = (A - T)
//     private static double[][][] deltaFromMSE(double[][][] activations, double[] target) {
//         double[][][] delta = new double[1][1][activations[0][0].length];
//         for (int i = 0; i < activations[0][0].length; i++) {
//             delta[0][0][i] = activations[0][0][i] - target[i];
//         }
//         return delta;
//     }

//     // ===== Tests =====

//     @Test
//     public void forward_producesExpectedValuesWithFixedParams() {
//         // Dense: Din=3 -> Dout=2
//         Dense layer = new Dense(2, 3);

//         // Fix weights and biases deterministically
//         double[][] W = {
//             { 0.10, -0.20, 0.30 },   // neuron 0
//             { -0.40, 0.50, -0.60 }   // neuron 1
//         };
//         double[] b = { 0.05, -0.10 };

//         layer.setWeights(W);
//         layer.setBiases(b);

//         // Input x = [1.0, 2.0, -1.0]
//         double[] x = { 1.0, 2.0, -1.0 };
//         double[][][] out = layer.forward(asInput(x));

//         // Expected pre-activations z = x·W^T + b
//         double z0 = 1.0*0.10 + 2.0*(-0.20) + (-1.0)*0.30 + 0.05;  // = 0.10 -0.40 -0.30 + 0.05 = -0.55
//         double z1 = 1.0*(-0.40) + 2.0*(0.50) + (-1.0)*(-0.60) - 0.10; // = -0.40 +1.00 +0.60 -0.10 = 1.10

//         double a0 = Activation.sigmoid(z0);
//         double a1 = Activation.sigmoid(z1);

//         assertEquals(2, out[0][0].length);
//         assertEquals(a0, out[0][0][0], 1e-12);
//         assertEquals(a1, out[0][0][1], 1e-12);
//     }

//     @Test
//     public void weightGradient_matchesNumericalFiniteDifference() {
//         // Small dense: Din=3 -> Dout=2
//         Dense layer = new Dense(2, 3);

//         // Fix params for determinism
//         double[][] W = {
//             { 0.10, -0.20, 0.30 },
//             { -0.40, 0.50, -0.60 }
//         };
//         double[] b = { 0.05, -0.10 };
//         layer.setWeights(W);
//         layer.setBiases(b);

//         double[] x = { 0.2, -0.3, 0.4 };
//         double[] t = { 0.7, 0.1 };   // target for MSE
//         double lr = 1e-3;            // small learning rate so updates ≈ gradient * lr

//         // Forward
//         double[][][] out = layer.forward(asInput(x));

//         // Analytical gradient implied by weight update:
//         // Call backward once; then grad_ij = -(W_new_ij - W_old_ij)/lr
//         double[][][] delta = deltaFromMSE(out, t);
//         double[][] W_before = new double[2][3];
//         for (int i=0;i<2;i++) System.arraycopy(W[i], 0, W_before[i], 0, 3);
//         layer.backward(delta, lr);

//         double[][] W_after = layer.getWeights();
//         double[][] gradAnalytic = new double[2][3];
//         for (int i=0;i<2;i++) {
//             for (int j=0;j<3;j++) {
//                 gradAnalytic[i][j] = -(W_after[i][j] - W_before[i][j]) / lr;
//             }
//         }

//         // Numerical gradient by finite difference on a few entries
//         double eps = 1e-6;
//         int[][] picks = { {0,0}, {0,2}, {1,1} };
//         for (int[] idx : picks) {
//             int i = idx[0], j = idx[1];

//             // Save original
//             double w0 = layer.getWeights()[i][j];

//             // f(w + eps)
//             layer.getWeights()[i][j] = w0 + eps;
//             double Lpos = mse(layer.forward(asInput(x))[0][0], t);

//             // f(w - eps)
//             layer.getWeights()[i][j] = w0 - eps;
//             double Lneg = mse(layer.forward(asInput(x))[0][0], t);

//             // restore
//             layer.getWeights()[i][j] = w0;

//             double gradNum = (Lpos - Lneg) / (2*eps);
//             assertEquals(gradNum, gradAnalytic[i][j], 1e-4, "weight grad mismatch at ("+i+","+j+")");
//         }
//     }

//     @Test
//     public void biasGradient_matchesNumericalFiniteDifference() {
//         Dense layer = new Dense(2, 3);

//         double[][] W = {
//             { 0.10, -0.20, 0.30 },
//             { -0.40, 0.50, -0.60 }
//         };
//         double[] b = { 0.05, -0.10 };
//         layer.setWeights(W);
//         layer.setBiases(b);

//         double[] x = { 0.2, -0.3, 0.4 };
//         double[] t = { 0.7, 0.1 };
//         double lr = 1e-3;

//         double[][][] out = layer.forward(asInput(x));
//         double[][][] delta = deltaFromMSE(out, t);

//         // Backward once to infer grad from bias update
//         double[] b_before = b.clone();
//         layer.backward(delta, lr);
//         double[] b_after = layer.getBiases();

//         double[] gradAnalytic = new double[b.length];
//         for (int i=0;i<b.length;i++) {
//             gradAnalytic[i] = -(b_after[i] - b_before[i]) / lr;
//         }

//         // Numerical grads
//         double eps = 1e-6;
//         for (int i=0;i<b.length;i++) {
//             double b0 = layer.getBiases()[i];

//             // f(b + eps)
//             layer.getBiases()[i] = b0 + eps;
//             double Lpos = mse(layer.forward(asInput(x))[0][0], t);

//             // f(b - eps)
//             layer.getBiases()[i] = b0 - eps;
//             double Lneg = mse(layer.forward(asInput(x))[0][0], t);

//             // restore
//             layer.getBiases()[i] = b0;

//             double gradNum = (Lpos - Lneg) / (2*eps);
//             assertEquals(gradNum, gradAnalytic[i], 1e-4, "bias grad mismatch at index "+i);
//         }
//     }

//     @Test
//     public void backward_usesOriginalWeightsForNewDelta() {
//         Dense layer = new Dense(2, 3);

//         double[][] W = {
//             { 0.1, -0.2, 0.3 },
//             { -0.4, 0.5, -0.6 }
//         };
//         double[] b = { 0.01, -0.02 };
//         layer.setWeights(W);
//         layer.setBiases(b);

//         double[] x = { 0.5, -0.1, 0.2 };
//         double[][][] out = layer.forward(asInput(x));

//         // Build an arbitrary upstream delta (e.g., from MSE)
//         double[] t = { 0.6, 0.4 };
//         double[][][] delta = deltaFromMSE(out, t);

//         // Compute what newDelta SHOULD be using the ORIGINAL weights:
//         // For Dense with sigmoid activation inside the layer:
//         // delta_i = delta[n]*sigmoid'(lastOutput[n])
//         // newDelta = W^T * delta_i   (should use W BEFORE any update)
//         double[] delta_i = new double[2];
//         for (int n = 0; n < 2; n++) {
//             double a = out[0][0][n];
//             double dphi = a * (1.0 - a);  // derivativeSigmoid on activation
//             delta_i[n] = delta[0][0][n] * dphi;
//         }
//         double[] expected = new double[3];
//         for (int i = 0; i < 3; i++) {
//             expected[i] = W[0][i] * delta_i[0] + W[1][i] * delta_i[1];
//         }

//         double lr = 1e-3;
//         double[][][] newDelta = layer.backward(delta, lr); // this uses UPDATED W in your code

//         // Compare to expected using ORIGINAL W -> this will fail until you fix backward
//         for (int i = 0; i < 3; i++) {
//             assertEquals(expected[i], newDelta[0][0][i], 1e-8,
//                 "newDelta must be computed with original weights (pre-update)");
//         }
//     }

//     @Test
//     public void training_reduces_loss() {
//         Dense layer = new Dense(2, 3);

//         // Deterministic params
//         double[][] W = {
//             { 0.10, -0.20, 0.30 },
//             { -0.40, 0.50, -0.60 }
//         };
//         double[] b = { 0.05, -0.10 };
//         layer.setWeights(W);
//         layer.setBiases(b);

//         double[] x = { 0.2, -0.3, 0.4 };
//         double[] t = { 0.9, 0.2 };

//         // One small SGD step should reduce MSE
//         double[][][] out1 = layer.forward(asInput(x));
//         double L1 = mse(out1[0][0], t);

//         double[][][] delta = deltaFromMSE(out1, t);
//         layer.backward(delta, 1e-2); // small lr

//         double[][][] out2 = layer.forward(asInput(x));
//         double L2 = mse(out2[0][0], t);

//         assertTrue(L2 <= L1, "Loss should not increase after a small SGD step");
//     }
// }
