# src/optimizers.py

import numpy as np

def stochastic_gradient_descent(weights, learning_rate, gradients):
    """
    Updates the weights using stochastic gradient descent.

    Args:
        weights (numpy.ndarray): The current weights.
        learning_rate (float): The learning rate for the update.
        gradients (numpy.ndarray): The gradients computed during the backward pass.

    Returns:
        numpy.ndarray: The updated weights.
    """
    return weights - learning_rate * gradients

def momentum_optimizer(weights, learning_rate, gradients, momentum):
    """
    Updates the weights using the momentum optimization algorithm.

    Args:
        weights (numpy.ndarray): The current weights.
        learning_rate (float): The learning rate for the update.
        gradients (numpy.ndarray): The gradients computed during the backward pass.
        momentum (float): The momentum value for the update.

    Returns:
        numpy.ndarray: The updated weights.
    """
    velocity = momentum * velocity + learning_rate * gradients
    return weights - velocity

def rmsprop_optimizer(weights, learning_rate, gradients, cache, decay_rate, epsilon):
    """
    Updates the weights using the RMSProp optimization algorithm.

    Args:
        weights (numpy.ndarray): The current weights.
        learning_rate (float): The learning rate for the update.
        gradients (numpy.ndarray): The gradients computed during the backward pass.
        cache (numpy.ndarray): The RMSProp cache.
        decay_rate (float): The decay rate for the cache.
        epsilon (float): A small value to prevent division by zero.

    Returns:
        numpy.ndarray: The updated weights.
    """
    cache = decay_rate * cache + (1 - decay_rate) * gradients ** 2
    return weights - learning_rate * gradients / (np.sqrt(cache) + epsilon)

def adam_optimizer(weights, learning_rate, gradients, cache_m, cache_v, beta1, beta2, epsilon):
    """
    Updates the weights using the Adam optimization algorithm.

    Args:
        weights (numpy.ndarray): The current weights.
        learning_rate (float): The learning rate for the update.
        gradients (numpy.ndarray): The gradients computed during the backward pass.
        cache_m (numpy.ndarray): The Adam cache for the first moment.
        cache_v (numpy.ndarray): The Adam cache for the second moment.
        beta1 (float): The beta1 value for the Adam algorithm.
        beta2 (float): The beta2 value for the Adam algorithm.
        epsilon (float): A small value to prevent division by zero.

    Returns:
        numpy.ndarray: The updated weights.
    """
    cache_m = beta1 * cache_m + (1 - beta1) * gradients
    cache_v = beta2 * cache_v + (1 - beta2) * gradients ** 2
    return weights - learning_rate * cache_m / (np.sqrt(cache_v) + epsilon)