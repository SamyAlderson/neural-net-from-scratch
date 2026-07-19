# tests/test_optimizers.py

import unittest
import numpy as np
from src.optimizers import SGD, Adam

class TestOptimizers(unittest.TestCase):
    def test_sgd(self):
        # Create a dummy model with one layer of 5 neurons
        model = np.random.rand(1, 5, 5)

        # Create an SGD optimizer with a learning rate of 0.01
        optimizer = SGD(learning_rate=0.01)

        # Compute the gradients of the loss with respect to the model's parameters
        # (this is a placeholder, in a real scenario you'd use the actual gradients)
        gradients = np.random.rand(1, 5, 5)

        # Update the model's parameters using the SGD optimizer
        updated_model, _ = optimizer.update(model, gradients)

        # Check that the updated model is not the same as the original model
        self.assertFalse(np.array_equal(updated_model, model))

    def test_adam(self):
        # Create a dummy model with one layer of 5 neurons
        model = np.random.rand(1, 5, 5)

        # Create an Adam optimizer with a learning rate of 0.01 and a beta value of 0.9
        optimizer = Adam(learning_rate=0.01, beta1=0.9)

        # Compute the gradients of the loss with respect to the model's parameters
        # (this is a placeholder, in a real scenario you'd use the actual gradients)
        gradients = np.random.rand(1, 5, 5)

        # Update the model's parameters using the Adam optimizer
        updated_model, _ = optimizer.update(model, gradients)

        # Check that the updated model is not the same as the original model
        self.assertFalse(np.array_equal(updated_model, model))

class TestOptimizerInitialization(unittest.TestCase):
    def test_sgd_init(self):
        # Create an SGD optimizer with a learning rate of 0.01
        optimizer = SGD(learning_rate=0.01)

        # Check that the optimizer's learning rate is correct
        self.assertEqual(optimizer.learning_rate, 0.01)

    def test_adam_init(self):
        # Create an Adam optimizer with a learning rate of 0.01 and a beta value of 0.9
        optimizer = Adam(learning_rate=0.01, beta1=0.9)

        # Check that the optimizer's learning rate and beta value are correct
        self.assertEqual(optimizer.learning_rate, 0.01)
        self.assertEqual(optimizer.beta1, 0.9)

class TestOptimizerUpdate(unittest.TestCase):
    def test_sgd_update(self):
        # Create a dummy model with one layer of 5 neurons
        model = np.random.rand(1, 5, 5)

        # Create an SGD optimizer with a learning rate of 0.01
        optimizer = SGD(learning_rate=0.01)

        # Compute the gradients of the loss with respect to the model's parameters
        # (this is a placeholder, in a real scenario you'd use the actual gradients)
        gradients = np.random.rand(1, 5, 5)

        # Update the model's parameters using the SGD optimizer
        updated_model, _ = optimizer.update(model, gradients)

        # Check that the updated model is not the same as the original model
        self.assertFalse(np.array_equal(updated_model, model))

    def test_adam_update(self):
        # Create a dummy model with one layer of 5 neurons
        model = np.random.rand(1, 5, 5)

        # Create an Adam optimizer with a learning rate of 0.01 and a beta value of 0.9
        optimizer = Adam(learning_rate=0.01, beta1=0.9)

        # Compute the gradients of the loss with respect to the model's parameters
        # (this is a placeholder, in a real scenario you'd use the actual gradients)
        gradients = np.random.rand(1, 5, 5)

        # Update the model's parameters using the Adam optimizer
        updated_model, _ = optimizer.update(model, gradients)

        # Check that the updated model is not the same as the original model
        self.assertFalse(np.array_equal(updated_model, model))

if __name__ == '__main__':
    unittest.main()
```

```python
# src/optimizers.py

import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, model, gradients):
        # Update the model's parameters using the SGD optimizer
        updated_model = model - self.learning_rate * gradients
        return updated_model, None

class Adam:
    def __init__(self, learning_rate=0.01, beta1=0.9):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.m = 0
        self.v = 0

    def update(self, model, gradients):
        # Update the model's parameters using the Adam optimizer
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta1 * self.v + (1 - self.beta1) * gradients**2
        updated_model = model - self.learning_rate * self.m / (np.sqrt(self.v) + 1e-8)
        return updated_model, None