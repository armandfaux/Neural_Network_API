package data;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class CsvReader {
    public static Dataset readCSV(String filename) {
        ArrayList<double[][][]> inputsList = new ArrayList<>();
        ArrayList<double[]> outputsList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line = br.readLine(); // skip header
            if (line == null) {
                throw new IOException("Empty file");
            }

            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                int label = Integer.parseInt(parts[0]);

                // Prepare input as (1,28,28)
                double[][][] input = new double[1][28][28];
                for (int i = 1; i < parts.length; i++) {
                    int pixel = Integer.parseInt(parts[i]);
                    double normalized = pixel / 255.0;

                    int idx = i - 1;           // skip label
                    int row = idx / 28;
                    int col = idx % 28;
                    input[0][row][col] = normalized;
                }

                // One-hot encode label (10 classes)
                double[] output = new double[10];
                output[label] = 1.0;

                inputsList.add(input);
                outputsList.add(output);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Convert ArrayList → arrays
        int N = inputsList.size();
        double[][][][] inputs = new double[N][1][28][28];
        for (int n = 0; n < N; n++) {
            inputs[n] = inputsList.get(n);
        }
        double[][] outputs = outputsList.toArray(new double[0][]);

        return new Dataset(inputs, outputs);
    }
}
