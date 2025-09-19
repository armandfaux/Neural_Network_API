package cnn;
public final class Config {
    private static boolean verbose;

    private Config () {
        verbose = false;
    }

    static public boolean verbose() {
        return verbose;
    }

    static public void setVerbose(boolean new_value) {
        verbose = new_value;
    }
}
