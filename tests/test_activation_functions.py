import unittest
import numpy as np
from src.activation_functions import sigmoid, tanh, relu

class TestActivationFunctions(unittest.TestCase):

    def test_sigmoid(self):
        # Test sigmoid function with a few edge cases
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertAlmostEqual(sigmoid(1), 0.7310585786301)
        self.assertAlmostEqual(sigmoid(-1), 0.2689414213699)

    def test_tanh(self):
        # Test tanh function with a few edge cases
        self.assertAlmostEqual(tanh(0), 0)
        self.assertAlmostEqual(tanh(1), 0.761594155955)
        self.assertAlmostEqual(tanh(-1), -0.761594155955)

    def test_relu(self):
        # Test relu function with a few edge cases
        self.assertEqual(relu(0), 0)
        self.assertEqual(relu(1), 1)
        self.assertEqual(relu(-1), 0)

    def test_sigmoid_derivative(self):
        # Test sigmoid derivative function with a few edge cases
        self.assertAlmostEqual(sigmoid_derivative(0), 0.25)
        self.assertAlmostEqual(sigmoid_derivative(1), 0.196611920842)
        self.assertAlmostEqual(sigmoid_derivative(-1), 0.103254737948)

    def test_tanh_derivative(self):
        # Test tanh derivative function with a few edge cases
        self.assertAlmostEqual(tanh_derivative(0), 0)
        self.assertAlmostEqual(tanh_derivative(1), 0.619148374614)
        self.assertAlmostEqual(tanh_derivative(-1), -0.619148374614)

    def test_relu_derivative(self):
        # Test relu derivative function with a few edge cases
        self.assertEqual(relu_derivative(0), 0)
        self.assertEqual(relu_derivative(1), 1)
        self.assertEqual(relu_derivative(-1), 0)

def sigmoid_derivative(x):
    # Derivative of sigmoid function
    return x * (1 - x)

def tanh_derivative(x):
    # Derivative of tanh function
    return 1 - x**2

def relu_derivative(x):
    # Derivative of relu function
    if x > 0:
        return 1
    else:
        return 0

if __name__ == '__main__':
    unittest.main()