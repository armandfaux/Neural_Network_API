package test;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

public class TensorTest {
    private double[] raw_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    private double[] short_data = {1.0, 2.0, 3.0, 4.0};

    @Test
    public void tensor_init() {
        // Shape only constructor
        data.Tensor tensor = new data.Tensor(new int[]{3, 2, 2});
        tensor.set_data(raw_data);
        assertArrayEquals(raw_data, tensor.raw_data(), "Tensor data should be set properly");

        // Data + shape constructor
        tensor = new data.Tensor(new int[]{2, 3, 2}, raw_data);
        assertArrayEquals(raw_data, tensor.raw_data(), "Tensor data should be set properly");
    }

    @Test
    public void shapes_and_access() {
        data.Tensor tensor = new data.Tensor(new int[]{3, 2, 2});
        tensor.set_data(raw_data);
        
        // Checking initial data
        assertArrayEquals(new int[]{3, 2, 2}, tensor.shape(), "Tensor shape should be [3, 2, 2]");
        assertEquals(1.0, tensor.get(0, 0, 0), "Element at (0,0,0) should be 1.0");
        assertEquals(6.0, tensor.get(1, 0, 1), "Element at (1,0,1) should be 6.0");

        // Same dimension reshaping
        tensor.reshape(new int[]{2, 3, 2});
        assertArrayEquals(new int[]{2, 3, 2}, tensor.shape(), "Tensor shape should be [2, 3, 2] after reshaping");
        assertEquals(8.0, tensor.get(1, 0, 1), "Element at (1,0,1) should be 8.0 after reshaping");

        // Change vector space dimension
        // tensor.reshape(new int[]{2, 6});
        // assertArrayEquals(new int[]{2, 6}, tensor.shape(), "Tensor shape should be [2, 6] after reshaping");
        // assertEquals(11.0, tensor.get(1, 4), "Element at (1, 4) should be 11.0 after reshaping");

        // // Single dimension
        // tensor.reshape(new int[]{12});
        // assertArrayEquals(new int[]{12}, tensor.shape(), "Tensor shape should be [12] after reshaping");
        // assertEquals(3.0, tensor.get(2), "Element at (2) should be 3.0 after reshaping");

        // // Weird dimension
        // tensor.reshape(new int[]{12, 1, 1, 1, 1, 1, 1});
        // assertArrayEquals(new int[]{12, 1, 1, 1, 1, 1, 1}, tensor.shape(), "Tensor shape should be [12, 1, 1, 1, 1, 1, 1] after reshaping");
        // assertEquals(10.0, tensor.get(9, 1, 1, 1, 1, 1, 1), "Correct behaviour in weird dimension");
    }

    @Test
    public void invalid_shapes() {
        data.Tensor tensor = new data.Tensor(new int[]{3, 2, 2});
        tensor.set_data(raw_data);

        // Incompatible shape
        assertThrows(IllegalArgumentException.class, () -> {
            tensor.reshape(new int[]{5, 5});
        }, "Reshaping to incompatible shape should throw IllegalArgumentException");

        // Zero shape
        assertThrows(IllegalArgumentException.class, () -> {
            tensor.reshape(new int[]{0});
        }, "Reshaping to zero shape should throw IllegalArgumentException");

        // Zero dimension
        assertThrows(IllegalArgumentException.class, () -> {
            tensor.reshape(new int[]{2, 6, 0});
        }, "Reshaping to zero dimension should throw IllegalArgumentException");

        // Higher number of indices
        tensor.reshape(new int[]{2, 6});
        assertThrows(IllegalArgumentException.class, () -> {
            tensor.get(0, 0, 0, 0);
        }, "Accessing with higher number of indices should throw IllegalArgumentException");

        // Lower number of indices
        assertThrows(IllegalArgumentException.class, () -> {
            tensor.get(1);
        }, "Accessing with lower number of indices should throw IllegalArgumentException");

        // Out of bounds
        assertThrows(IllegalArgumentException.class, () -> {
            tensor.get(2, 0);
        }, "Accessing with out of bounds indices should throw IllegalArgumentException (hard test)");

        assertThrows(IllegalArgumentException.class, () -> {
            tensor.get(0, 6);
        }, "Accessing with out of bounds indices should throw IllegalArgumentException (soft test)");
    }

    @Test
    public void increment_and_set() {
        data.Tensor tensor = new data.Tensor(new int[]{3, 2, 2});
        tensor.set_data(raw_data);

        // Increment positive value
        tensor.inc(2, 1, 0, 1);
        assertEquals(8.0, tensor.get(1, 0, 1), "Element at (1,0,1) +2 should be 8.0");

        // Increment negative value
        tensor.inc(-3, 2, 0, 1);
        assertEquals(7.0, tensor.get(2, 0, 1), "Element at (2,0,1) -3 should be 7.0");

        // Set zero value
        tensor.set(0, 0, 0, 1);
        assertEquals(0.0, tensor.get(0, 0, 1), "Element at (0,0,1) should be set to 0.0");

        // Set negative value
        tensor.set(-4, 0, 1, 1);
        assertEquals(-4.0, tensor.get(0, 1, 1), "Element at (0,1,1) should be set to -4.0");
    }

    @Test
    public void invalid_increment_and_set() {
        data.Tensor tensor = new data.Tensor(new int[]{3, 2, 2});
        tensor.set_data(raw_data);

        // Increment lower number of indices
        assertThrows(IllegalArgumentException.class, () -> {
            tensor.inc(2, 0, 1);
        }, "Increment with lower number of indices should throw IllegalArgumentException");

        // Increment higher number of indices
        assertThrows(IllegalArgumentException.class, () -> {
            tensor.inc(2, 0, 1, 1, 1);
        }, "Increment with higher number of indices should throw IllegalArgumentException");

        // Increment out of bounds
        assertThrows(IllegalArgumentException.class, () -> {
            tensor.inc(2, 0, 1, 3);
        }, "Increment with out of bounds indices should throw IllegalArgumentException");

        // Set lower number of indices
        assertThrows(IllegalArgumentException.class, () -> {
            tensor.set(2, 0, 1);
        }, "Set with lower number of indices should throw IllegalArgumentException");

        // Set higher number of indices
        assertThrows(IllegalArgumentException.class, () -> {
            tensor.set(2, 0, 1, 1, 1);
        }, "Set with higher number of indices should throw IllegalArgumentException");

        // Set out of bounds
        assertThrows(IllegalArgumentException.class, () -> {
            tensor.set(2, 0, 1, 3);
        }, "Set with out of bounds indices should throw IllegalArgumentException");

    }

    @Test
    public void mapping() {
        // map using square function
        java.util.function.Function<Double, Double> squareFunction = (Double x) -> x * x;
        data.Tensor tensor = new data.Tensor(new int[]{4});
        tensor.set_data(short_data);

        tensor.map(squareFunction);

        assertEquals(1.0, tensor.get(0), "Element at (0) should be 1.0 after mapping");
        assertEquals(4.0, tensor.get(1), "Element at (1) should be 4.0 after mapping");
        assertEquals(9.0, tensor.get(2), "Element at (2) should be 9.0 after mapping");
        assertEquals(16.0, tensor.get(3), "Element at (3) should be 16.0 after mapping");

        // map using zero function
        java.util.function.Function<Double, Double> zeroFunction = (Double _) -> 0.0;
        tensor.map(zeroFunction);
        for (int i = 0; i < 4; i++) {
            assertEquals(0.0, tensor.get(i), "Element at (" + i + ") should be 0.0 after mapping to zero");
        }
    }

    // @Test
    // public void invalid_mapping() {
    // }
}
