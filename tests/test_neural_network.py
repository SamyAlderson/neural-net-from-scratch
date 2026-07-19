import unittest
import numpy as np
from src.neural_network import NeuralNetwork
from src.activation_functions import sigmoid, relu
from src.optimizers import sgd

class TestNeuralNetwork(unittest.TestCase):

    def test_forward_pass(self):
        # Create a simple neural network with one input layer, one hidden layer, and one output layer
        nn = NeuralNetwork([2, 2, 1])

        # Create some test data
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        # Perform a forward pass
        outputs = nn.forward_pass(inputs)

        # Check the outputs
        expected_outputs = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_array_almost_equal(outputs, expected_outputs)

    def test_backward_pass(self):
        # Create a simple neural network with one input layer, one hidden layer, and one output layer
        nn = NeuralNetwork([2, 2, 1])

        # Create some test data
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        labels = np.array([[0], [1], [1], [0]])

        # Perform a forward pass
        outputs = nn.forward_pass(inputs)

        # Calculate the error
        error = nn.calculate_error(outputs, labels)

        # Perform a backward pass
        gradients = nn.backward_pass(error)

        # Check the gradients
        expected_gradients = np.array([[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
                                       [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]])
        np.testing.assert_array_almost_equal(gradients, expected_gradients)

    def test_optimization(self):
        # Create a simple neural network with one input layer, one hidden layer, and one output layer
        nn = NeuralNetwork([2, 2, 1])

        # Create some test data
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        labels = np.array([[0], [1], [1], [0]])

        # Perform a forward pass
        outputs = nn.forward_pass(inputs)

        # Calculate the error
        error = nn.calculate_error(outputs, labels)

        # Perform optimization using SGD
        nn.optimize(sgd, inputs, labels, learning_rate=0.1, epochs=10)

        # Check the final weights
        expected_weights = np.array([[[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]],
                                     [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]])
        np.testing.assert_array_almost_equal(nn.weights, expected_weights)

if __name__ == '__main__':
    unittest.main()