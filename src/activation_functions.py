# src/activation_functions.py

import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid of x.

    The sigmoid function maps any real-valued number to a value between 0 and 1.
    It's commonly used as the activation function in neural networks.

    :param x: Input array
    :return: Sigmoid of x
    """
    # Using np.exp and np.clip to avoid overflow
    return np.clip(1 / (1 + np.exp(-x)), 1e-8, 1 - 1e-8)


def relu(x: np.ndarray) -> np.ndarray:
    """
    Compute the rectified linear unit of x.

    ReLU maps all negative values to 0 and all non-negative values to themselves.

    :param x: Input array
    :return: ReLU of x
    """
    # Simple and efficient implementation
    return x * (x > 0)


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Compute the hyperbolic tangent of x.

    The tanh function maps any real-valued number to a value between -1 and 1.
    It's similar to the sigmoid function but maps to a different range.

    :param x: Input array
    :return: Tanh of x
    """
    # Using np.exp and np.clip to avoid overflow
    return 2 * np.clip(np.exp(x) - np.exp(-x), -1 + 1e-8, 1 - 1e-8) / np.clip(np.exp(x) + np.exp(-x), 2 * 1e-8, 2 - 2 * 1e-8)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute the softmax of x.

    The softmax function maps any real-valued array to a probability distribution.

    :param x: Input array
    :param axis: Axis to compute softmax along (default: -1)
    :return: Softmax of x
    """
    # Using np.exp to compute the exponentials
    # Using np.sum to compute the normalization factor
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)