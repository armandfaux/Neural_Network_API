package test;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

import tools.Activation;

public class ActivationTest {
    @Test
    public void testSigmoid() {
        double result = Activation.sigmoid(0);
        assertEquals(0.5, result, 1e-9, "Sigmoid at 0 should be 0.5");

        result = Activation.sigmoid(1000);
        assertEquals(1.0, result, 1e-9, "Sigmoid at large positive values should approach 1");

        result = Activation.sigmoid(-1000);
        assertEquals(0.0, result, 1e-9, "Sigmoid at large negative values should approach 0");
    }

    @Test
    public void testDerivativeSigmoid() {
        double sigmoid = Activation.sigmoid(0);
        double derivative = Activation.derivativeSigmoid(sigmoid);
        assertEquals(0.25, derivative, 1e-9, "Derivative of Sigmoid at 0 should be 0.25");

        sigmoid = Activation.sigmoid(1000);
        derivative = Activation.derivativeSigmoid(sigmoid);
        assertEquals(0.0, derivative, 1e-9, "Derivative of Sigmoid at extreme positive values should approach 0");

        sigmoid = Activation.sigmoid(-1000);
        derivative = Activation.derivativeSigmoid(sigmoid);
        assertEquals(0.0, derivative, 1e-9, "Derivative of Sigmoid at extreme negative values should approach 0");
    }

    @Test
    public void testReLU() {
        double result = Activation.relu(0);
        assertEquals(0.0, result, 1e-9, "ReLU at 0 should be 0");

        result = Activation.relu(5);
        assertEquals(5.0, result, 1e-9, "ReLU at positive values should be the value itself");

        result = Activation.relu(-5);
        assertEquals(0.0, result, 1e-9, "ReLU at negative values should be 0");
    }

    @Test
    public void testReLUDerivative() {
        double result = Activation.derivativeReLU(0);
        assertEquals(0.0, result, 1e-9, "ReLU derivative at 0 should be 0");

        result = Activation.derivativeReLU(5);
        assertEquals(1.0, result, 1e-9, "ReLU derivative at positive values should be 1");

        result = Activation.derivativeReLU(-5);
        assertEquals(0.0, result, 1e-9, "ReLU derivative at negative values should be 0");
    }
}
