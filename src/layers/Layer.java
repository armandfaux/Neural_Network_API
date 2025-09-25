package layers;

public class Layer {
    public enum Type {
        DENSE,
        CONV,
        POOLING,
    }

    public Type type;

    public double[][][] forward(double[][][] input) {
        // This method should be overridden in subclasses
        throw new UnsupportedOperationException("Forward method not implemented in Layer class.");
    }

    public double[][][] backward(double[][][] gradient, double learningRate) {
        // This method should be overridden in subclasses
        throw new UnsupportedOperationException("Backward method not implemented in Layer class.");
    }
}
