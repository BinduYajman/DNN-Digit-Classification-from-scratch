# Handwritten Digit Recognition Using a Custom Neural Network

This project implements a handwritten digit classifier using a fully connected neural network trained from scratch with NumPy. The model is trained on the famous MNSIT Digit Recognizer dataset, where each sample is a flattened 784-pixel grayscale image (28 x 28), and the first value in each row is the corresponding digit label.

The implementation is intentionally manual: it performs forward propagation, computes gradients with backpropagation, updates weights using gradient descent.

The goal is to classify digits from 0 to 9 using pixel intensities as input features. Each training example consists of:

- 784 input features (one for each pixel)
- a single integer label from 0 to 9

The network learns a mapping from raw pixel values to one of ten output classes.

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
