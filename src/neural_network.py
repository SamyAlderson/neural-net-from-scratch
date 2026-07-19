# src/neural_network.py

import numpy as np
from activation_functions import sigmoid, relu, softmax
from optimizers import SGD, Adam
from utils import to_numpy

class NeuralNetwork:
    """
    A simple implementation of a neural network from scratch.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu'):
        """
        Initialize the neural network.

        :param input_dim: Dimension of the input layer.
        :param hidden_dim: Dimension of the hidden layer.
        :param output_dim: Dimension of the output layer.
        :param activation: Activation function for the hidden layer.
        """
        self.weights1 = np.random.rand(input_dim, hidden_dim)
        self.weights2 = np.random.rand(hidden_dim, output_dim)
        self.bias1 = np.zeros((1, hidden_dim))
        self.bias2 = np.zeros((1, output_dim))
        self.activation = activation

    def forward_pass(self, inputs):
        """
        Perform a forward pass through the network.

        :param inputs: Input to the network.
        :return: Output of the network.
        """
        # this was tricky
        inputs = to_numpy(inputs)

        # First layer
        hidden_layer = np.dot(inputs, self.weights1) + self.bias1
        if self.activation == 'sigmoid':
            hidden_layer = sigmoid(hidden_layer)
        elif self.activation == 'relu':
            hidden_layer = relu(hidden_layer)
        else:
            raise ValueError("Unsupported activation function")

        # Second layer
        output_layer = np.dot(hidden_layer, self.weights2) + self.bias2
        if self.activation == 'softmax':
            output_layer = softmax(output_layer)
        else:
            raise ValueError("Unsupported activation function")

        return output_layer

    def backward_pass(self, inputs, targets, output):
        """
        Perform a backward pass through the network.

        :param inputs: Input to the network.
        :param targets: Targets for the network.
        :param output: Output of the network.
        :return: Gradients of the weights and biases.
        """
        # this is a lot of work
        d_output = output - targets
        d_weights2 = np.dot(output.T, d_output)
        d_bias2 = np.sum(d_output, axis=0, keepdims=True)

        # Backpropagate through the hidden layer
        d_hidden = np.dot(d_output, self.weights2.T)
        if self.activation == 'sigmoid':
            d_hidden = d_hidden * sigmoid(np.dot(inputs, self.weights1) + self.bias1) * (1 - sigmoid(np.dot(inputs, self.weights1) + self.bias1))
        elif self.activation == 'relu':
            d_hidden = d_hidden * (relu(np.dot(inputs, self.weights1) + self.bias1) > 0)
        else:
            raise ValueError("Unsupported activation function")

        d_weights1 = np.dot(inputs.T, d_hidden)
        d_bias1 = np.sum(d_hidden, axis=0, keepdims=True)

        return d_weights1, d_bias1, d_weights2, d_bias2

    def train(self, inputs, targets, optimizer, learning_rate, num_epochs):
        """
        Train the network.

        :param inputs: Input to the network.
        :param targets: Targets for the network.
        :param optimizer: Optimizer to use.
        :param learning_rate: Learning rate for the optimizer.
        :param num_epochs: Number of epochs to train for.
        """
        for epoch in range(num_epochs):
            output = self.forward_pass(inputs)
            d_weights1, d_bias1, d_weights2, d_bias2 = self.backward_pass(inputs, targets, output)
            optimizer.update(self.weights1, self.bias1, d_weights1, d_bias1)
            optimizer.update(self.weights2, self.bias2, d_weights2, d_bias2)

            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}, Loss: {np.mean((output - targets) ** 2)}")

    def predict(self, inputs):
        """
        Make predictions on the network.

        :param inputs: Input to the network.
        :return: Output of the network.
        """
        return self.forward_pass(inputs)