# Handwritten Digit Recognition Using Dense Neural Network using NumPy

## Model architecture

The model is a custom multi-layer perceptron implemented from scratch. The notebook defines a `DNN` class with the following architecture:

- Input layer: 784 neurons
- Hidden layer 1: 128 neurons
- Hidden layer 2: 64 neurons
- Output layer: 10 neurons

Activation functions:

- Hidden layers: sigmoid
- Output layer: softmax

Loss behavior:

- Output logits are compared against one-hot-like target vectors
- Backpropagation computes gradients of the loss with respect to the network weights
- Weights are updated iteratively using gradient descent
