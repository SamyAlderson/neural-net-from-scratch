import numpy as np
from typing import Callable, Tuple

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid of x.
    """
    return 1 / (1 + np.exp(-x))  # classic implementation

def derivative_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the sigmoid function.
    """
    return sigmoid(x) * (1 - sigmoid(x))  # not proud of this but it works

def relu(x: np.ndarray) -> np.ndarray:
    """
    Compute the ReLU of x.
    """
    return np.maximum(x, 0)  # simple and straightforward

def derivative_relu(x: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the ReLU function.
    """
    return np.where(x > 0, 1, 0)  # this was tricky

def l1_loss(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Compute the L1 loss between y and y_hat.
    """
    return np.mean(np.abs(y - y_hat))  # simple mean absolute error

def l2_loss(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Compute the L2 loss between y and y_hat.
    """
    return np.mean((y - y_hat) ** 2)  # simple mean squared error

def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, num_iterations: int) -> np.ndarray:
    """
    Perform gradient descent on the given problem.

    Args:
    X: design matrix
    y: target vector
    theta: initial guess for parameters
    alpha: learning rate
    num_iterations: number of iterations

    Returns:
    the optimized theta
    """
    m = len(y)
    for i in range(num_iterations):
        # compute the hypothesis
        h = np.dot(X, theta)
        
        # compute the cost
        cost = np.mean((h - y) ** 2)
        
        # compute the gradient
        gradient = (1 / m) * np.dot(X.T, h - y)
        
        # update theta
        theta -= alpha * gradient
    
    return theta