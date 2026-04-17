package test;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import data.Tensor;
import layers.Conv;

/**
 * Unit tests for layers.Conv (forward/backward).
 *
 * Assumptions from provided code and messages:
 *  - Tensor supports new Tensor(int[]{...}) and set/get/inc, shape(), size(dim).
 *  - Conv now correctly:
 *      * stores kernelChannels,
 *      * accumulates over channels in forward,
 *      * accumulates ΔI with inc(...),
 *      * applies ReLU derivative by gating δO with last output (or preact),
 *      * exposes getKernels()/setKernels(), getBiases()/setBiases(), and public lastOutput.
 *  - Stride/padding supported via setters; tests use padding=0.
 *  - Activations are ReLU; tests keep all pre-activations positive to avoid zero gradients.
 */
public class ConvTest {

    // ---------- Helpers ----------

    private Tensor tensor1D(double[] vals) {
        Tensor t = new Tensor(new int[]{vals.length});
        for (int i = 0; i < vals.length; i++) t.set(vals[i], i);
        return t;
    }

    // shape [C][H][W]
    private Tensor tensor3D(double[][][] vals) {
        int C = vals.length, H = vals[0].length, W = vals[0][0].length;
        Tensor t = new Tensor(new int[]{C, H, W});
        for (int c = 0; c < C; c++) {
            assertEquals(H, vals[c].length);
            for (int h = 0; h < H; h++) {
                assertEquals(W, vals[c][h].length);
                for (int w = 0; w < W; w++) {
                    t.set(vals[c][h][w], c, h, w);
                }
            }
        }
        return t;
    }

    // shape [K][C][KH][KW]
    private Tensor tensor4D(double[][][][] vals) {
        int K = vals.length, C = vals[0].length, KH = vals[0][0].length, KW = vals[0][0][0].length;
        Tensor t = new Tensor(new int[]{K, C, KH, KW});
        for (int k = 0; k < K; k++) {
            assertEquals(C, vals[k].length);
            for (int c = 0; c < C; c++) {
                assertEquals(KH, vals[k][c].length);
                for (int h = 0; h < KH; h++) {
                    assertEquals(KW, vals[k][c][h].length);
                    for (int w = 0; w < KW; w++) {
                        t.set(vals[k][c][h][w], k, c, h, w);
                    }
                }
            }
        }
        return t;
    }

    private void assertTensor3DClose(Tensor t, double[][][] exp, double tol, String msg) {
        assertEquals(exp.length, t.size(0), msg + " (C mismatch)");
        assertEquals(exp[0].length, t.size(1), msg + " (H mismatch)");
        assertEquals(exp[0][0].length, t.size(2), msg + " (W mismatch)");
        for (int c = 0; c < exp.length; c++) {
            for (int h = 0; h < exp[0].length; h++) {
                for (int w = 0; w < exp[0][0].length; w++) {
                    assertEquals(exp[c][h][w], t.get(c, h, w), tol,
                            msg + String.format(" at (%d,%d,%d)", c, h, w));
                }
            }
        }
    }

    // Sum all elements in a [K][H][W] output (useful for finite-difference)
    private double sumAll(Tensor out) {
        int K = out.size(0), H = out.size(1), W = out.size(2);
        double s = 0;
        for (int k = 0; k < K; k++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    s += out.get(k, h, w);
        return s;
    }

    // ---------- Forward tests ----------

    @Test
    @DisplayName("Forward: single-channel, 2x2 kernel, stride=1, padding=0")
    public void testForwardSingleChannel() {
        Conv conv = new Conv(/*kernels*/1, /*channels*/1, /*kH*/2, /*kW*/2);
        conv.setStride(1);
        conv.setPadding(0);

        // Input: 1x3x3
        double[][][] x = new double[][][]{
            {
                {1, 2, 3},
                {0, 1, 2},
                {2, 1, 0}
            }
        };
        Tensor input = tensor3D(x);

        // Kernel: 1 filter, 1 channel, 2x2
        double[][][][] K = new double[][][][]{
            { // k=0
                { // c=0
                    {1, 0},
                    {0, 1}
                }
            }
        };
        Tensor kernels = tensor4D(K);

        // Biases: zero
        Tensor biases = tensor1D(new double[]{0.0});

        conv.setKernels(kernels);
        conv.setBiases(biases);

        Tensor out = conv.forward(input);

        // Expected valid conv, then ReLU (all positive anyway):
        // Positions (h,w): sum of x[h:h+2,w:w+2] * kernel
        double[][][] expected = new double[][][]{
            {
                {1*1 + 1*1, 2*1 + 2*1},    // top row windows: [[1 2],[0 1]] and [[2 3],[1 2]]
                {0*1 + 2*1, 1*1 + 0*1}     // [[0 1],[2 1]] and [[1 2],[1 0]]
            }
        };
        assertTensor3DClose(out, expected, 1e-9, "Forward output mismatch");
    }

    @Test
    @DisplayName("Forward: multi-channel accumulation")
    public void testForwardMultiChannelAccumulation() {
        Conv conv = new Conv(1, 2, 2, 2);
        conv.setStride(1);
        conv.setPadding(0);

        // Input: 2 channels, 2x2 -> only one output position
        double[][][] x = new double[][][]{
            {
                {1, 2},
                {3, 4}
            },
            {
                {5, 6},
                {7, 8}
            }
        };
        Tensor input = tensor3D(x);

        // Kernels: one filter over 2 channels, each 2x2
        // Sum over channel 0 with weights all 1, and channel 1 with weights all 0.5
        double[][][][] K = new double[][][][]{
            {
                {
                    {1, 1},
                    {1, 1}
                },
                {
                    {0.5, 0.5},
                    {0.5, 0.5}
                }
            }
        };
        Tensor kernels = tensor4D(K);
        Tensor biases = tensor1D(new double[]{0.0});
        conv.setKernels(kernels);
        conv.setBiases(biases);

        Tensor out = conv.forward(input);
        // Expect: sum(C0) + 0.5*sum(C1) = (1+2+3+4) + 0.5*(5+6+7+8) = 10 + 0.5*26 = 23
        double[][][] expected = new double[][][]{ { {23.0} } };
        assertTensor3DClose(out, expected, 1e-9, "Multi-channel accumulation mismatch");
    }

    @Test
    @DisplayName("Forward: stride=2 downsampling")
    public void testForwardStride2() {
        Conv conv = new Conv(1, 1, 2, 2);
        conv.setStride(2);
        conv.setPadding(0);

        double[][][] x = new double[][][]{
            {
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9,10,11,12},
                {13,14,15,16}
            }
        };
        Tensor input = tensor3D(x);

        // Kernel that sums its 2x2 window
        double[][][][] K = new double[][][][]{
            { { {1,1},{1,1} } }
        };
        Tensor kernels = tensor4D(K);
        Tensor biases = tensor1D(new double[]{0.0});
        conv.setKernels(kernels);
        conv.setBiases(biases);

        Tensor out = conv.forward(input);
        // stride=2 windows:
        // top-left: 1+2+5+6=14; top-right: 3+4+7+8=22
        // bot-left: 9+10+13+14=46; bot-right: 11+12+15+16=54
        double[][][] expected = new double[][][]{
            { {14,22},{46,54} }
        };
        assertTensor3DClose(out, expected, 1e-9, "Stride=2 forward mismatch");
    }

    // ---------- Backward / gradients ----------

    @Test
    @DisplayName("Backward: bias gradient equals sum of δ over spatial positions")
    public void testBackwardBiasGradient() {
        Conv conv = new Conv(2, 1, 2, 2);
        conv.setStride(1);
        conv.setPadding(0);

        // Input 1x3x3, all positive so ReLU gate = 1
        double[][][] x = new double[][][]{ { {1,2,3},{4,5,6},{7,8,9} } };
        Tensor input = tensor3D(x);

        // Kernels produce positive outputs: use all-ones kernels
        double[][][][] K = new double[][][][]{
            { { {1,1},{1,1} } },
            { { {1,1},{1,1} } }
        };
        Tensor kernels = tensor4D(K);
        Tensor biases  = tensor1D(new double[]{0.0, 0.0});
        conv.setKernels(kernels);
        conv.setBiases(biases);

        // Forward to fill caches (lastOutput used for ReLU')
        Tensor out = conv.forward(input);
        // Output shape for 3x3 with 2x2 kernel, stride=1: 2x2 per filter
        assertEquals(2, out.size(1));
        assertEquals(2, out.size(2));

        // Upstream gradient δO: ones (dL/dA) so dL/dB = sum(δZ) = sum(δO) because ReLU gate=1
        Tensor deltaO = tensor3D(new double[][][]{
            { {1,1},{1,1} },
            { {1,1},{1,1} }
        });

        // Run backward with small lr; we'll compute expected bias updates
        double lr = 0.1;
        conv.backward(deltaO, lr);

        // Bias update rule in your code: b -= lr * dB
        // dB for each filter is sum of δO over its 2x2 map = 4
        double[] expectedBias = new double[]{ -lr * 4.0, -lr * 4.0 };
        assertEquals(expectedBias[0], conv.getBiases().get(0), 1e-12, "Bias 0 update mismatch");
        assertEquals(expectedBias[1], conv.getBiases().get(1), 1e-12, "Bias 1 update mismatch");
    }

    @Test
    @DisplayName("Backward: finite-difference check for kernel gradients (single filter)")
    public void testBackwardFiniteDifferenceKernels() {
        // One filter, one channel, 2x2 kernel, stride=1
        Conv conv = new Conv(1, 1, 2, 2);
        conv.setStride(1);
        conv.setPadding(0);

        // Positive input to keep ReLU active
        double[][][] x = new double[][][]{ { {1,2,1},{0,1,2},{2,1,0} } };
        Tensor input = tensor3D(x);

        // Start with a simple kernel, bias=0
        double[][][][] K = new double[][][][]{ { { {0.3, 0.2},{0.1, 0.4} } } };
        Tensor kernels = tensor4D(K);
        Tensor biases  = tensor1D(new double[]{0.0});
        conv.setKernels(kernels);
        conv.setBiases(biases);

        // Define loss L = sum(out); then dL/dA = 1 at every output location.
        // We'll compute numeric grad dL/dw via central differences and compare to analytic grad from backward update.

        // First, compute numeric gradients
        double eps = 1e-6;
        double[][] numeric = new double[2][2];

        // Capture original kernel to restore between perturbations
        double[][] wOrig = new double[][]{
            { kernels.get(0,0,0,0), kernels.get(0,0,0,1) },
            { kernels.get(0,0,1,0), kernels.get(0,0,1,1) }
        };

        // Utility to compute L = sum(out)
        java.util.function.Supplier<Double> lossFn = () -> {
            Tensor o = conv.forward(input);
            return sumAll(o);
        };

        for (int ky = 0; ky < 2; ky++) {
            for (int kx = 0; kx < 2; kx++) {
                // w+ = w + eps
                kernels.set(wOrig[ky][kx] + eps, 0, 0, ky, kx);
                conv.setKernels(kernels);
                double Lplus = lossFn.get();

                // w- = w - eps
                kernels.set(wOrig[ky][kx] - eps, 0, 0, ky, kx);
                conv.setKernels(kernels);
                double Lminus = lossFn.get();

                // restore original weight
                kernels.set(wOrig[ky][kx], 0, 0, ky, kx);
                conv.setKernels(kernels);

                numeric[ky][kx] = (Lplus - Lminus) / (2 * eps);
            }
        }

        // Now get analytic gradient magnitude by observing the optimizer step:
        // backward with δO = 1 everywhere, learningRate = alpha
        Tensor deltaO = tensor3D(new double[][][]{ { {1,1},{1,1} } });

        // Take a snapshot of kernels before update
        double[][] wBefore = new double[][]{
            { conv.getKernels().get(0,0,0,0), conv.getKernels().get(0,0,0,1) },
            { conv.getKernels().get(0,0,1,0), conv.getKernels().get(0,0,1,1) }
        };

        double alpha = 0.05;
        conv.forward(input); // ensure caches are from same weights
        conv.backward(deltaO, alpha);

        double[][] wAfter = new double[][]{
            { conv.getKernels().get(0,0,0,0), conv.getKernels().get(0,0,0,1) },
            { conv.getKernels().get(0,0,1,0), conv.getKernels().get(0,0,1,1) }
        };

        // In your optimizer: w := w - alpha * grad  ⇒ grad = -(w_after - w_before) / alpha
        double[][] analytic = new double[2][2];
        for (int ky = 0; ky < 2; ky++) {
            for (int kx = 0; kx < 2; kx++) {
                analytic[ky][kx] = -(wAfter[ky][kx] - wBefore[ky][kx]) / alpha;
            }
        }

        // Compare numeric vs analytic
        double tol = 1e-5; // with small eps, this should be tight
        assertEquals(numeric[0][0], analytic[0][0], tol, "Grad(0,0) mismatch");
        assertEquals(numeric[0][1], analytic[0][1], tol, "Grad(0,1) mismatch");
        assertEquals(numeric[1][0], analytic[1][0], tol, "Grad(1,0) mismatch");
        assertEquals(numeric[1][1], analytic[1][1], tol, "Grad(1,1) mismatch");

        // Bias finite-difference (optional quick sanity): dL/db = number of outputs (since δ=1)
        // For 3x3 input, 2x2 kernel, stride=1 → output is 2x2 = 4 positions
        // (We already tested bias update separately, so we skip an extra FD here.)
    }

    @Test
    @DisplayName("Backward: returns ΔI with correct shape and non-zero where expected")
    public void testBackwardDeltaIShapeAndSignal() {
        Conv conv = new Conv(1, 1, 2, 2);
        conv.setStride(1);
        conv.setPadding(0);

        // Simple input 1x3x3
        double[][][] x = new double[][][]{ { {1,0,0},{0,1,0},{0,0,1} } };
        Tensor input = tensor3D(x);

        // Kernel ones, bias zero -> positive outputs
        double[][][][] K = new double[][][][]{ { { {1,1},{1,1} } } };
        conv.setKernels(tensor4D(K));
        conv.setBiases(tensor1D(new double[]{0.0}));

        // Forward
        conv.forward(input);

        // δO = ones (shape 1x2x2)
        Tensor deltaO = tensor3D(new double[][][]{ { {1,1},{1,1} } });

        // Backward with zero LR so weights aren't changed (we only care about ΔI)
        Tensor deltaI = conv.backward(deltaO, 0.0);

        assertEquals(1, deltaI.size(0), "ΔI channel mismatch");
        assertEquals(3, deltaI.size(1), "ΔI height mismatch");
        assertEquals(3, deltaI.size(2), "ΔI width mismatch");

        // Because kernel is ones and δO is ones with stride=1, ΔI should be a 2D correlation of a 2x2 ones filter
        // over a 2x2 map of ones -> corners get 1, edges 2, center 4.
        double[][][] expected = new double[][][]{ {
            {1, 2, 1},
            {2, 4, 2},
            {1, 2, 1}
        } };
        assertTensor3DClose(deltaI, expected, 1e-9, "ΔI values mismatch");
    }
}
