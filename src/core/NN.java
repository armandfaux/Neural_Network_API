package core;

import java.util.ArrayList;
import java.util.List;

import data.Tensor;
import junit.framework.Test;
import layers.LayerTensor;
import tools.Config;

public class NN {
    private List<LayerTensor> layers;

    // TODO implement "predict" and "train" methods
    // Current implementation is made for testing the different layers and structure of the network
    public NN() {
        this.layers = new ArrayList<>();
    }

    public void addLayer(LayerTensor layer) {
        this.layers.add(layer);
    }

    public Tensor forward(Tensor input) {
        if (Config.verbose()) {
            System.out.println("[NETWORK] Initiating forward pass through " + this.layers.size() + " layers");
        }

        Tensor output = input;
        for (LayerTensor layer : layers) {
            // if (layer.type == Layer.Type.DENSE && output[0][0].length != ((DenseLayer) layer).previousLayerSize) {
            //     if (Config.verbose()) {
            //         System.out.println("[WARNING] Adjusting Dense layer input size from " + ((DenseLayer) layer).previousLayerSize + " to " + output[0][0].length);
            //     }
            //     ((DenseLayer) layer).init(output[0][0].length);
            // }

            output = layer.forward(output);
        }
        return output;
    }

    public Tensor backward(Tensor gradient) {
        Tensor output = gradient;

        for (int i = layers.size() - 1; i >= 0; i--) {
            output = layers.get(i).backward(output, 0.1);
        }
        return output;
    }

    public void listLayers() {
        for (LayerTensor l : this.layers) {
            switch (l.getType()) {
                case LayerTensor.Type.DENSE:
                    System.out.println("Dense layer");
                    break;

                case LayerTensor.Type.CONV:
                    System.out.println("Conv layer");
                    break;

                case LayerTensor.Type.POOLING:
                    System.out.println("Pooling layer");
                    break;
            
                default:
                    System.out.println("[ERROR] Unrecognised layer");
                    break;
            }
        }
    }
}
