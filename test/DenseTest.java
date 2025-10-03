package test;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import data.Tensor;
import layers.DenseTensor;
import tools.Activation;

/**
 * Unit tests for DenseTensor forward/backward passes, parameter updates,
 * activation swapping, and state tracking.
 *
 * Assumptions about Tensor API based on provided class:
 *  - new Tensor(new int[]{...}) constructs a zero-initialized tensor
 *  - get(int... idx), set(double v, int... idx), inc(double dv, int... idx)
 *  - size(int dim) -> length of given dimension
 *  - raw_data() returns the flat storage for logging (not used in assertions)
 */
public class DenseTest {

    // ---------- Helpers ----------

    private Tensor tensor1D(double[] vals) {
        Tensor t = new Tensor(new int[]{vals.length});
        for (int i = 0; i < vals.length; i++) t.set(vals[i], i);
        return t;
    }

    private Tensor tensor2D(double[][] vals) {
        int r = vals.length, c = vals[0].length;
        Tensor t = new Tensor(new int[]{r, c});
        for (int i = 0; i < r; i++) {
            assertEquals(c, vals[i].length, "All rows must have the same length");
            for (int j = 0; j < c; j++) t.set(vals[i][j], i, j);
        }
        return t;
    }

    private void assertTensor1DClose(Tensor actual, double[] expected, double tol, String msg) {
        assertEquals(expected.length, actual.size(0), msg + " (shape mismatch)");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual.get(i), tol, msg + " (idx " + i + ")");
        }
    }

    private void assertTensor2DClose(Tensor actual, double[][] expected, double tol, String msg) {
        assertEquals(expected.length, actual.size(0), msg + " (rows mismatch)");
        assertEquals(expected[0].length, actual.size(1), msg + " (cols mismatch)");
        for (int i = 0; i < expected.length; i++) {
            for (int j = 0; j < expected[0].length; j++) {
                assertEquals(expected[i][j], actual.get(i, j), tol,
                    msg + " (idx " + i + "," + j + ")");
            }
        }
    }

    // ---------- Tests ----------

    @Test
    @DisplayName("Forward (identity activation): y = W x + b (deterministic)")
    public void testForwardIdentity() {
        int size = 2, input = 3;
        DenseTensor layer = new DenseTensor(size, input);

        // Override activation with identity for easy arithmetic checks
        layer.setActivationFunction(x -> x);
        layer.setActivationDerivative(y -> 1.0);

        // Set deterministic parameters
        double[][] W = {
            { 1.0,  2.0,  3.0},   // neuron 0
            {-1.0,  0.5,  2.0}    // neuron 1
        };
        double[] b = { 0.1, -0.2 };

        layer.setWeights(tensor2D(W));
        layer.setBiases(tensor1D(b));

        // Input
        double[] x = { 0.5, -1.0, 2.0 }; // column vector conceptually
        Tensor in = tensor1D(x);

        // Forward
        Tensor out = layer.forward(in);

        // Expected y = W x + b
        // neuron0: 1*0.5 + 2*(-1) + 3*2 + 0.1 = 0.5 - 2 + 6 + 0.1 = 4.6
        // neuron1: -1*0.5 + 0.5*(-1) + 2*2 - 0.2 = -0.5 -0.5 + 4 - 0.2 = 2.8
        double[] expected = { 4.6, 2.8 };
        assertTensor1DClose(out, expected, 1e-9, "Forward output mismatch (identity)");
    }

    @Test
    @DisplayName("Forward (ReLU): nonlinearity applied element-wise")
    public void testForwardReLU() {
        int size = 3, input = 2;
        DenseTensor layer = new DenseTensor(size, input);

        // Use built-in ReLU (default) explicitly
        layer.setActivationFunction(Activation::relu);
        layer.setActivationDerivative(Activation::derivativeReLU);

        // Parameters chosen to produce both negative and positive pre-activations
        double[][] W = {
            {  2.0,  1.0},  // -> positive
            { -3.0, -2.0},  // -> negative
            {  0.5, -1.5}   // -> depends on bias
        };
        double[] b = { -1.0, 0.5, 0.25 };

        layer.setWeights(tensor2D(W));
        layer.setBiases(tensor1D(b));

        // Input
        double[] x = { 0.5, 1.0 };
        // Pre-acts:
        // n0: 2*0.5 + 1*1.0 - 1.0 = 1.0 + 1.0 - 1.0 = 1.0 => ReLU=1.0
        // n1: -3*0.5 + -2*1.0 + 0.5 = -1.5 - 2 + 0.5 = -3.0 => ReLU=0.0
        // n2: 0.5*0.5 + -1.5*1.0 + 0.25 = 0.25 - 1.5 + 0.25 = -1.0 => ReLU=0.0
        double[] expected = { 1.0, 0.0, 0.0 };

        Tensor out = layer.forward(tensor1D(x));
        assertTensor1DClose(out, expected, 1e-9, "Forward output mismatch (ReLU)");
    }

    @Test
    @DisplayName("Backward (identity): returns W^T * delta and updates W,b with GD")
    public void testBackwardIdentityUpdates() {
        int size = 2, input = 3;
        DenseTensor layer = new DenseTensor(size, input);

        // Identity activation -> derivative = 1
        layer.setActivationFunction(x -> x);
        layer.setActivationDerivative(y -> 1.0);

        // Parameters
        double[][] W = {
            { 1.0,  2.0,  3.0},
            {-1.0,  0.5,  2.0}
        };
        double[] b = { 0.1, -0.2 };
        layer.setWeights(tensor2D(W));
        layer.setBiases(tensor1D(b));

        // Input and forward (to set last_input/last_output)
        double[] x = { 0.5, -1.0, 2.0 };
        Tensor in = tensor1D(x);
        Tensor out = layer.forward(in); // not used directly, but sets state

        // Upstream delta_O (same shape as out)
        double[] deltaO = { 0.2, -0.4 };
        Tensor deltaTensor = tensor1D(deltaO);

        double lr = 0.1;

        // Expected:
        // delta_I = delta_O * d(act) = delta_O
        // new_delta = W^T * delta_I
        // W_update = -lr * (delta_I outer last_input)
        // b_update = -lr * delta_I
        // new_delta:
        // [ i=0 ] = 1.0*0.2 + (-1.0)*(-0.4) = 0.2 + 0.4 = 0.6
        // [ i=1 ] = 2.0*0.2 +  0.5*(-0.4)   = 0.4 - 0.2 = 0.2
        // [ i=2 ] = 3.0*0.2 +  2.0*(-0.4)   = 0.6 - 0.8 = -0.2
        double[] expectedNewDelta = { 0.6, 0.2, -0.2 };

        Tensor newDelta = layer.backward(deltaTensor, lr);
        assertTensor1DClose(newDelta, expectedNewDelta, 1e-9, "new_delta mismatch");

        // Check parameter updates
        // For neuron 0: delta_I0 = 0.2
        // W0j' = W0j - lr * (0.2 * xj)
        //  j0: 1.0 - 0.1*(0.2*0.5)= 1.0 - 0.01 = 0.99
        //  j1: 2.0 - 0.1*(0.2*-1.0)= 2.0 + 0.02 = 2.02
        //  j2: 3.0 - 0.1*(0.2*2.0)= 3.0 - 0.04 = 2.96
        // b0' = 0.1 - 0.1*0.2 = 0.08
        // For neuron 1: delta_I1 = -0.4
        // W1j' = W1j - 0.1*(-0.4 * xj)
        //  j0: -1.0 - 0.1*(-0.4*0.5)= -1.0 + 0.02 = -0.98
        //  j1:  0.5 - 0.1*(-0.4*-1.0)= 0.5 - 0.04 = 0.46
        //  j2:  2.0 - 0.1*(-0.4*2.0)= 2.0 + 0.08 = 2.08
        // b1' = -0.2 - 0.1*(-0.4) = -0.16
        double[][] expectedW = {
            {0.99, 2.02, 2.96},
            {-0.98, 0.46, 2.08}
        };
        double[] expectedB = {0.08, -0.16};

        assertTensor2DClose(layer.getWeights(), expectedW, 1e-9, "Weights update mismatch");
        assertTensor1DClose(layer.getBiases(), expectedB, 1e-9, "Biases update mismatch");
    }

    @Test
    @DisplayName("last_output is stored after forward")
    public void testLastOutputStored() {
        DenseTensor layer = new DenseTensor(2, 2);

        // Identity to keep it simple
        layer.setActivationFunction(x -> x);
        layer.setActivationDerivative(y -> 1.0);

        // Set trivial params to make output = input
        layer.setWeights(tensor2D(new double[][]{
            {1.0, 0.0},
            {0.0, 1.0}
        }));
        layer.setBiases(tensor1D(new double[]{0.0, 0.0}));

        double[] x = {3.14, -2.5};
        Tensor out = layer.forward(tensor1D(x));

        assertTensor1DClose(out, x, 1e-12, "Forward result mismatch");
        assertTensor1DClose(layer.getLastOutput(), x, 1e-12, "last_output not stored correctly");
    }

    @Test
    @DisplayName("Custom activation set via setters is used in forward/backward")
    public void testCustomActivationIsUsed() {
        DenseTensor layer = new DenseTensor(1, 1);

        // Custom activation: f(x) = 2x + 1, f'(y)=2 (note: derivative takes post-activation arg per your code)
        layer.setActivationFunction(z -> 2*z + 1);
        layer.setActivationDerivative(y -> 2.0);

        // Set weights= [3], bias= [0]
        layer.setWeights(tensor2D(new double[][]{{3.0}}));
        layer.setBiases(tensor1D(new double[]{0.0}));

        // Forward on x=[4] -> pre = 3*4 + 0 = 12; out = 2*12 + 1 = 25
        Tensor out = layer.forward(tensor1D(new double[]{4.0}));
        assertTensor1DClose(out, new double[]{25.0}, 1e-12, "Custom activation forward mismatch");

        // Backward with delta_O=[10], lr=0.1
        // derivative = 2.0, so delta_I = 10 * 2 = 20
        // new_delta (to previous) = W^T * delta_I = 3 * 20 = 60
        // weight update: w' = 3 - 0.1*(20*4) = 3 - 8 = -5
        // bias update:  b' = 0 - 0.1*(20) = -2
        Tensor newDelta = layer.backward(tensor1D(new double[]{10.0}), 0.1);
        assertTensor1DClose(newDelta, new double[]{60.0}, 1e-12, "Custom activation new_delta mismatch");
        assertTensor2DClose(layer.getWeights(), new double[][]{{-5.0}}, 1e-12, "Custom activation weight update mismatch");
        assertTensor1DClose(layer.getBiases(), new double[]{-2.0}, 1e-12, "Custom activation bias update mismatch");
    }

    @Test
    @DisplayName("Shapes: output length equals layer size; backward delta length equals input size")
    public void testShapes() {
        DenseTensor layer = new DenseTensor(4, 3);
        layer.setActivationFunction(x -> x);
        layer.setActivationDerivative(y -> 1.0);

        Tensor out = layer.forward(tensor1D(new double[]{1, 2, 3}));
        assertEquals(4, out.size(0), "Forward output length should equal layer size");

        Tensor newDelta = layer.backward(tensor1D(new double[]{1, 1, 1, 1}), 0.01);
        assertEquals(3, newDelta.size(0), "Backward delta length should equal input size");
    }
}
