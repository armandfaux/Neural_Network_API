package data;

import java.util.Random;
import java.util.function.Function;

// Defines a multi-dimensional array
public class Tensor {
    private double[] data;
    private int[] shape;
    private int dim;

    public Tensor(int[] shape) {

        // TODO CHECK NEGATIVE SHAPE !!!!
        int size = 1;
        for (int s : shape) {
            size *= s;
        }

        this.data = new double[size];
        this.shape = shape;
        this.dim = shape.length;
    }

    public Tensor(int[] shape, double[] data) {
        this.data = data;
        this.shape = shape;
        this.dim = shape.length;
        this.init_zero();
    }

    public void display() {
        System.out.print("Tensor shape: [");
        for (int i = 0; i < shape.length; i++) {
            System.out.print(shape[i]);
            if (i < shape.length - 1) {
                System.out.print(", ");
            }
        }
        System.out.println("]");
        System.out.print("Data: [");
        for (int i = 0; i < data.length; i++) {
            System.out.print(data[i]);
            if (i < data.length - 1) {
                System.out.print(", ");
            }
        }
        System.out.println("]");
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

        this.data[real_i] += value;
    }

    public void set_data(double[] data) {
        this.data = data;
    }

    public void reshape(int[] shape) {
        this.shape = shape;
    }

    // Set all weights to zero
    public void init_zero() {
        for (int i = 0; i < this.data.length; i++) {
            this.data[i] = 0.0;
        }
    }

    // Set all weights to given value
    public void init_constant(double value) {
        for (int i = 0; i < this.data.length; i++) {
            this.data[i] = value;
        }
    }

    // Uniform distribution between -0.5 and 0.5
    public void init_random() {
        for (int i = 0; i < this.data.length; i++) {
            this.data[i] = (Math.random() - 0.5);
        }
    }

    // Values drawn from normal distribution
    public void init_normal() {
        Random rand = new Random();
        for (int i = 0; i < this.data.length; i++) {
            this.data[i] = rand.nextGaussian() * 0.1;
        }
    }

    // Xavier initialization, uniform distribution based on number of inputs and outputs
    public void init_xavier(int in, int out) {
        double x = Math.sqrt(6.0 / (in + out));
        for (int i = 0; i < this.data.length; i++) {
            this.data[i] = (Math.random() * 2 - 1) * x;
        }
    }

    // He initialization, normal distribution based on number of inputs
    public void init_he(int in) {
        double std = Math.sqrt(2.0 / in);
        Random rand = new Random();
        for (int i = 0; i < this.data.length; i++) {
            this.data[i] = rand.nextGaussian() * std;
        }
    }

    public void map(Function<Double, Double> f) {
        for (int i = 0; i < data.length; i++) {
            data[i] = f.apply(data[i]);
        }
    }

    public Tensor subtract(Tensor other) {
        if (other.data.length != this.data.length) {
            throw new IllegalArgumentException("Tensor.subtract() - other tensor has different size");
        }

        Tensor result = new Tensor(this.shape, new double[this.data.length]);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = this.data[i] - other.data[i];
        }
        return result;
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
