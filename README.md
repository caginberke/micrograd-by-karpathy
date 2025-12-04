# Scalar-Valued Autograd Engine & Neural Network

This repository contains a lightweight implementation of an **Autograd Engine** and a **Neural Network** library built from scratch in Python. It is inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

The core of this project is to understand the mathematical foundations of Deep Learning, specifically **Backpropagation** and the construction of **Computational Graphs**.

## 🚀 Features

* **`Value` Class:** A wrapper around scalar values that tracks operations and gradients.
* **Automatic Differentiation:** Implements `backward()` passes using the Chain Rule.
* **Graph Operations:** Supports basic arithmetic (`+`, `-`, `*`, `/`, `**`) and activation functions (`tanh`, `exp`).
* **Topological Sort:** Uses DFS to build the dependency graph for correct gradient propagation.
* **Neural Network API:** Includes `Neuron`, `Layer`, and `MLP` (Multi-Layer Perceptron) classes similar to PyTorch's design.

## 📂 File Structure

* `micrograd_scratch.ipynb`: The main Jupyter Notebook containing the source code, training loops, and visualizations.
* `microgradnotes(turkish).pdf`: My personal study notes

## 💻 Usage Example

Here is how the computational graph is built and backpropagation is performed:

```python
from micrograd import Value

# 1. Build a simple graph
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'a*b'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'

# 2. Run Backpropagation
L.backward()

print(f"Gradient of a: {a.grad}")
print(f"Gradient of b: {b.grad}")
