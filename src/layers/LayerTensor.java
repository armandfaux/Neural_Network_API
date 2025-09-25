package layers;

import data.Tensor;

public class LayerTensor {
    enum Type {
        CONV,
        DENSE,
        FLATTEN,
        POOLING,
    }

    protected Type type;

    public Tensor forward(Tensor input) {
        // This method should be overridden in subclasses
        throw new UnsupportedOperationException("Forward method not implemented in Layer class.");
    }

    public Tensor backward(Tensor delta, double learning_rate) {
        // This method should be overridden in subclasses
        throw new UnsupportedOperationException("Backward method not implemented in Layer class.");
    }
}
