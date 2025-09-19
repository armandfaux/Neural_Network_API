package cnn;
import java.util.ArrayList;
import java.util.List;

public class CNN {
    private List<Layer> layers;
    private boolean verbose;

    // TODO implement "predict" and "train" methods
    // Current implementation is made for testing the different layers and structure of the network
    public CNN() {
        this.verbose = false;
        this.layers = new ArrayList<>();
    }

    public void addLayer(Layer layer) {
        this.layers.add(layer);
    }

    public double[][][] forward(double[][][] input) {
        if (this.verbose) {
            System.out.println("[NETWORK] Initiating forward pass through " + this.layers.size() + " layers");
        }

        double[][][] output = input;
        for (Layer layer : layers) {
            if (layer.type == Layer.Type.DENSE && output[0][0].length != ((DenseLayer) layer).previousLayerSize) {
                if (Config.verbose()) {
                    System.out.println("[WARNING] Adjusting Dense layer input size from " + ((DenseLayer) layer).previousLayerSize + " to " + output[0][0].length);
                }
                ((DenseLayer) layer).init(output[0][0].length);
            }

            output = layer.forward(output);
        }
        return output;
    }

    public double[][][] backward(double[][][] gradient) {
        double[][][] output = gradient;

        for (int i = layers.size() - 1; i >= 0; i--) {
            output = layers.get(i).backward(output, 0.1);
        }
        return output;
    }

    public void listLayers() {
        for (Layer l : this.layers) {
            switch (l.type) {
                case Layer.Type.DENSE:
                    System.out.println("Dense layer");
                    break;

                case Layer.Type.CONV:
                    System.out.println("Conv layer");
                    break;

                case Layer.Type.POOLING:
                    System.out.println("Pooling layer");
                    break;
            
                default:
                    System.out.println("[ERROR] Unrecognised layer");
                    break;
            }
        }
    }

    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }
}
