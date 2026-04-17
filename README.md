# Neural Network API

**Implemented entirely from scratch, including matrix operations, gradient computation, and backpropagation algorithms.**

A Java-based API to build and experiment with neural networks from scratch.

This project provides a fully custom implementation of neural network architectures such as MLP (Multi-Layer Perceptron) and CNN (Convolutional Neural Networks), with future support for RNN (Recurrent Neural Networks).

This project is designed for educational purposes, focusing on understanding the internal mechanics of deep learning rather than performance.

---

## Key Features

- Neural networks implemented from scratch
- Custom matrix operations (no external math libraries)
- Supports:
  - MLP (fully connected networks)
  - CNN (convolutional networks)
  - (Planned) RNN (recurrent networks)
- Manual implementation of:
  - Forward propagation
  - Backpropagation
  - Gradient computation
- API layer to interact with models
- Designed for experimentation and learning

---

## Project Philosophy

Unlike production-grade frameworks such as TensorFlow or PyTorch, this project intentionally avoids external dependencies.

The goal is to:

- Understand how neural networks work at a low level
- Implement the mathematical foundations manually
- Gain intuition on:
  - Gradient descent
  - Backpropagation
  - Convolution operations

---

## Architecture Overview

The project is structured around three main layers:

### 1. Neural Network Core
- Layers (Dense, Convolutional, etc.)
- Activation functions
- Loss functions
- Training logic

### 2. Math Engine
- flexible Tensor class
- Matrix operations (multiplication, transpose, etc.)
- Gradient calculations
- Numerical computations

### 3. API Layer
- Endpoints to:
  - Train models
  - Run predictions
  - Configure hyper-parameters

---

## Example Usage

A working example is available here:

```
src/sample/Main.java
```

## Roadmap

- Add RNN support (Recurrent Neural Networks)
- Improve performance (optimizations)
- Add model persistence (save/load)
- Visualization tools

---

## Contributing

Contributions are welcome.

Feel free to:
- Open issues
- Suggest improvements
- Submit pull requests

---

## Author

Armand Faux
