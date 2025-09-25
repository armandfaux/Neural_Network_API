package data;

import java.util.function.Function;

// Defines a multi-dimensional array
public class Tensor {
    private double[] data;
    private int[] shape;
    private int dim;

    public Tensor(int[] shape) {
        this.data = new double[0];
        this.shape = shape;
        this.dim = shape.length;
    }

    public Tensor(double[] data, int[] shape) {
        this.data = data;
        this.shape = shape;
        this.dim = shape.length;
    }

    // Return target element at data[index[0]][index[1]] etc...
    public double get(int... index) {
        if (index.length != this.shape.length) {
            throw new IllegalArgumentException("Tensor.get() - index doesn't match the shape");
        }

        // Compute actual index in 1D array
        int real_i = 0;
        int stride = 1;
        for (int d = this.dim - 1; d >= 0; d--) {
            real_i += index[d] * stride;
            stride *= this.shape[d];
        }

        return this.data[real_i];
    }

    // Set value of target element at data[index[0]][index[1]] etc...
    public void set(double value, int... index) {
        if (index.length != this.shape.length) {
            throw new IllegalArgumentException("Tensor.get() - index doesn't match the shape");
        }

        // Compute actual index in 1D array
        int real_i = 0;
        int stride = 1;
        for (int d = this.dim - 1; d >= 0; d--) {
            real_i += index[d] * stride;
            stride *= this.shape[d];
        }

        this.data[real_i] = value;
    }

    // Increment value to target element at data[index[0]][index[1]] etc...
    public void inc(double value, int... index) {
        if (index.length != this.shape.length) {
            throw new IllegalArgumentException("Tensor.get() - index doesn't match the shape");
        }

        // Compute actual index in 1D array
        int real_i = 0;
        int stride = 1;
        for (int d = this.dim - 1; d >= 0; d--) {
            real_i += index[d] * stride;
            stride *= this.shape[d];
        }

        System.out.println("Incrementing value " + this.data[real_i] + " by value: " + value);
        this.data[real_i] += value;
    }

    public void set_data(double[] data) {
        this.data = data;
    }

    public void reshape(int[] shape) {
        this.shape = shape;
    }

    public void randomise() {
        for (int i = 0; i < data.length; i++) {
            data[i] = Math.random();
        }
    }

    public void map(Function<Double, Double> f) {
        for (int i = 0; i < data.length; i++) {
            data[i] = f.apply(data[i]);
        }
    }

    public double[] raw_data() {
        return this.data.clone();
    }

    public int size(int dim) {
        return this.shape[dim];
    }

    public int[] shape() {
        return this.shape.clone();
    }
}
