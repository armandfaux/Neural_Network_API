package data;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

// Defines a multi-dimensional array
public class Tensor {
    private double[] data;
    private int[] shape;
    private int dim;

    public Tensor(int[] shape) {
        if (shape.length == 0) {
            throw new IllegalArgumentException("Tensor.Tensor() - shape cannot be empty");
        }

        for (int s : shape) {
            if (s <= 0) {
                throw new IllegalArgumentException("Tensor.Tensor() - non positive shape");
            }
        }

        // Plus tard, size sera directement dans la classe Shape
        int size = 1;
        for (int s : shape) size *= s;

        this.data = new double[size];
        this.shape = shape.clone();
        this.dim = shape.length;
    }

    public Tensor(int[] shape, double[] data) {
        this.data = data.clone();
        this.shape = shape.clone();
        this.dim = shape.length;
    }

    public void display() {
        System.out.print("-- Tensor shape: [");
        for (int i = 0; i < shape.length; i++) {
            System.out.print(shape[i]);
            if (i < shape.length - 1) {
                System.out.print(", ");
            }
        }
        System.out.println("]");
        System.out.print("[");
        for (int i = 0; i < data.length; i++) {
            System.out.print(data[i]);
            if (i < data.length - 1) {
                System.out.print(", ");
            }
        }
        System.out.println("]\n---");
    }

    // Return target element at data[index[0]][index[1]] etc...
    public double get(int... index) {
        return this.data[flatIndex(index)];
    }

    // Set value of target element at data[index[0]][index[1]] etc...
    public void set(double value, int... index) {
        this.data[flatIndex(index)] = value;
    }

    // Increment value to target element at data[index[0]][index[1]] etc...
    public void inc(double value, int... index) {
        this.data[flatIndex(index)] += value;
    }

    public void setData(double[] newData) {
        if (newData.length != this.data.length) {
            throw new IllegalArgumentException("Tensor.set_data() - data must have the same number of elements as the current shape");
        }

        this.data = newData.clone();
    }

    public void reshape(int[] newShape) {
        if (newShape.length == 0) {
            throw new IllegalArgumentException("Tensor.Tensor() - shape cannot be empty");
        }

        for (int s : newShape) {
            if (s <= 0) {
                throw new IllegalArgumentException("Tensor.Tensor() - non positive shape");
            }
        }

        int currentSize = 1;
        for (int s : shape) currentSize *= s;

        int newSize = 1;
        for (int s : newShape) newSize *= s;

        if (currentSize != newSize) {
            throw new IllegalArgumentException("Tensor.reshape() - new shape must have the same number of elements as the current shape");
        }

        this.dim = newShape.length;
        this.shape = newShape.clone();
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
    public void init_xavier(int fanIn, int fanOut) {
        double x = Math.sqrt(6.0 / (fanIn + fanOut));
        for (int i = 0; i < this.data.length; i++) {
            this.data[i] = (Math.random() * 2 - 1) * x;
        }
    }

    // He initialization, normal distribution based on number of inputs
    public void init_he(int fanIn) {
        double std = Math.sqrt(2.0 / fanIn);
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
        if (!Arrays.equals(this.shape, other.shape)) {
            throw new IllegalArgumentException("Tensor.subtract() - other tensor has different shape");
        }

        Tensor result = new Tensor(this.shape, new double[this.data.length]);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = this.data[i] - other.data[i];
        }
        return result;
    }

    public double[] raw() {
        return this.data.clone();
    }

    public int size() {
        return this.data.length;
    }

    public int size(int axis) {
        if (axis >= this.shape.length) {
            throw new IllegalArgumentException("Tensor.size() - dimension out of bounds");
        }

        if (axis < 0) {
            throw new IllegalArgumentException("Tensor.size() - negative dimension");
        }

        return this.shape[axis];
    }

    public int[] shape() {
        return this.shape.clone();
    }

    private int flatIndex(int... index) {
        if (index.length != this.shape.length) {
            throw new IllegalArgumentException("Tensor - index doesn't match the shape");
        }

        int real_i = 0;
        int stride = 1;
        for (int d = this.dim - 1; d >= 0; d--) {
            if (index[d] < 0 || index[d] >= this.shape[d]) {
                throw new IllegalArgumentException("Tensor - index out of bounds");
            }

            real_i += index[d] * stride;
            stride *= this.shape[d];
        }

        return real_i;
    }
}
