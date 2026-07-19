# neural-net-from-scratch
A minimal implementation of a neural network from scratch in Python

## What it does

This project implements a basic feedforward neural network with forward and backward passes, as well as gradient descent optimization. It's a simple example to get you started with neural networks from scratch.

## Installation and Usage

```bash
pip install numpy
git clone https://your-repo-url.com/neural-net-from-scratch.git
cd neural-net-from-scratch
python main.py
```

## Running Tests

```bash
pytest tests/
```

## Project Structure

- `main.py`: Entry point for the project
- `nn.py`: Neural network implementation
- `layers.py`: Layer implementations for the neural network
- `activation.py`: Activation function implementations
- `loss.py`: Loss function implementations
- `optimizers.py`: Optimizer implementations
- `tests/`: Test suite for the project

## License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.