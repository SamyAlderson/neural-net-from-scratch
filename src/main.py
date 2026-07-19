# main.py

import argparse
import logging
import numpy as np
from src import neural_network
from src import activation_functions
from src import optimizers
from src import utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Neural network from scratch')
    parser.add_argument('--train', help='Train the network', action='store_true')
    parser.add_argument('--test', help='Test the network', action='store_true')
    parser.add_argument('--input_file', help='Path to input file')
    parser.add_argument('--output_file', help='Path to output file')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs', default=100)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.01)
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()

    # Load data
    X, y = utils.load_data(args.input_file)

    # Initialize network
    network = neural_network.NeuralNetwork(
        input_dim=X.shape[1],
        hidden_dim=128,
        output_dim=10,
        activation_function=activation_functions.ReLU,
        optimizer=optimizers.SGD,
        learning_rate=args.learning_rate
    )

    # Train or test the network
    if args.train:
        network.train(X, y, num_epochs=args.num_epochs)
        utils.save_data(network.output, args.output_file)
    elif args.test:
        output = network.forward(X)
        utils.save_data(output, args.output_file)
    else:
        logger.error('Either --train or --test flag must be provided')

if __name__ == '__main__':
    main()
```

```python
# setup.py

from setuptools import setup

setup(
    name='neural-net-from-scratch',
    version='1.0',
    description='A simple implementation of a neural network from scratch in Python',
    author='Samy Alderson',
    author_email='samyal@none.com',
    packages=['src'],
    install_requires=['numpy'],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
```

```python
# requirements.txt

numpy