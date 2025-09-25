package test;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

import layers.Flatten;

public class TensorTest {
    @Test
    public void tensor_init() {
        double[][][] data = {
            {
                {1.0, 2.0},
                {3.0, 4.0}
            },
            {
                {5.0, 6.0},
                {7.0, 8.0}
            },
            {
                {9.0, 10.0},
                {11.0, 12.0}
            }
        };
        double[] raw_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
        data.Tensor tensor = new data.Tensor(new int[]{3, 2, 2});
        tensor.set_data(raw_data);

        assertArrayEquals(raw_data, tensor.raw_data(), "Tensor data should be set properly");
    }

    @Test
    public void shapes_and_access() {
        double[][][] data = {
            {
                {1.0, 2.0},
                {3.0, 4.0}
            },
            {
                {5.0, 6.0},
                {7.0, 8.0}
            },
            {
                {9.0, 10.0},
                {11.0, 12.0}
            }
        };
        double[] raw_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
        data.Tensor tensor = new data.Tensor(new int[]{3, 2, 2});
        tensor.set_data(raw_data);
        
        assertArrayEquals(new int[]{3, 2, 2}, tensor.shape(), "Tensor shape should be [3, 2, 2]");
        assertEquals(1.0, tensor.get(0, 0, 0), "Element at (0,0,0) should be 1.0");
        assertEquals(6.0, tensor.get(1, 0, 1), "Element at (1,0,1) should be 6.0");
        
        tensor.reshape(new int[]{2, 3, 2});

        assertArrayEquals(new int[]{2, 3, 2}, tensor.shape(), "Tensor shape should be [2, 3, 2] after reshape");
        assertEquals(8.0, tensor.get(1, 0, 1), "Element at (1,0,1) should be 8.0 after reshaping");
    }
}
