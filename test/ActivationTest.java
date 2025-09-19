package test;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import cnn.*;

public class ActivationTest {
    @Test
    public void testSigmoidAtZero() {
        double result = Activation.sigmoid(0);
        assertEquals(0.5, result, 1e-9, "Sigmoid at 0 should be 0.5");
    }

    @Test
    public void testSigmoidAtPositive() {
        double result = Activation.sigmoid(2);
        assertTrue(result > 0.5 && result < 1.0, "Sigmoid at positive values should be between 0.5 and 1");
    }

    @Test
    public void testSigmoidAtNegative() {
        double result = Activation.sigmoid(-2);
        assertTrue(result > 0.0 && result < 0.5, "Sigmoid at negative values should be between 0 and 0.5");
    }

    @Test
    public void testSigmoidAtLargePositive() {
        double result = Activation.sigmoid(1000);
        assertEquals(1.0, result, 1e-9, "Sigmoid at large positive values should approach 1");
    }

    @Test
    public void testSigmoidAtLargeNegative() {
        double result = Activation.sigmoid(-1000);
        assertEquals(0.0, result, 1e-9, "Sigmoid at large negative values should approach 0");
    }

    @Test
    public void testDerivativeSigmoidAtZero() {
        double sigmoidAtZero = Activation.sigmoid(0);
        double derivative = Activation.derivativeSigmoid(sigmoidAtZero);
        assertEquals(0.25, derivative, 1e-9, "Derivative of Sigmoid at 0 should be 0.25");
    }

    @Test
    public void testDerivativeSigmoidAtPositive() {
        double sigmoidAtPositive = Activation.sigmoid(2);
        double derivative = Activation.derivativeSigmoid(sigmoidAtPositive);
        assertTrue(derivative > 0.0 && derivative < 0.25, "Derivative of Sigmoid at positive values should be between 0 and 0.25");
    }

    @Test
    public void testDerivativeSigmoidAtNegative() {
        double sigmoidAtNegative = Activation.sigmoid(-2);
        double derivative = Activation.derivativeSigmoid(sigmoidAtNegative);
        assertTrue(derivative > 0.0 && derivative < 0.25, "Derivative of Sigmoid at negative values should be between 0 and 0.25");
    }

    @Test
    public void testDerivativeSigmoidAtExtreme() {
        double sigmoidAtExtreme = Activation.sigmoid(1000);
        double derivative = Activation.derivativeSigmoid(sigmoidAtExtreme);
        assertEquals(0.0, derivative, 1e-9, "Derivative of Sigmoid at extreme values should approach 0");
    }

    @Test
    public void testDerivativeSigmoidAtExtremeNegative() {
        double sigmoidAtExtremeNegative = Activation.sigmoid(-1000);
        double derivative = Activation.derivativeSigmoid(sigmoidAtExtremeNegative);
        assertEquals(0.0, derivative, 1e-9, "Derivative of Sigmoid at extreme negative values should approach 0");
    }

    @Test
    public void testReLUAtZero() {
        double result = Activation.relu(0);
        assertEquals(0.0, result, 1e-9, "ReLU at 0 should be 0");
    }

    @Test
    public void testReLUAtPositive() {
        double result = Activation.relu(5);
        assertEquals(5.0, result, 1e-9, "ReLU at positive values should be the value itself");
    }

    @Test
    public void testReLUAtNegative() {
        double result = Activation.relu(-5);
        assertEquals(0.0, result, 1e-9, "ReLU at negative values should be 0");
    }

    @Test
    public void testReLUDerivativeAtZero() {
        double result = Activation.reluDerivative(0);
        assertEquals(0.0, result, 1e-9, "ReLU derivative at 0 should be 0");
    }

    @Test
    public void testReLUDerivativeAtPositive() {
        double result = Activation.reluDerivative(5);
        assertEquals(1.0, result, 1e-9, "ReLU derivative at positive values should be 1");
    }

    @Test
    public void testReLUDerivativeAtNegative() {
        double result = Activation.reluDerivative(-5);
        assertEquals(0.0, result, 1e-9, "ReLU derivative at negative values should be 0");
    }
}
