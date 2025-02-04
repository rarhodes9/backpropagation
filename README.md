# Circle Classification

A simple demonstration of binary classification on a circle vs. not-circle dataset, using both a manual NumPy approach and a PyTorch approach.

## Table of Contents
1. [Introduction](#introduction)  
2. [File 1: circle_classification.py](#file-1-circle_classificationpy)  
3. [File 2: circle_classification_pytorch.py](#file-2-circle_classification_pytorchpy)  
4. [Why PyTorch Is More Efficient](#why-pytorch-is-more-efficient-and-scalable)  
5. [Conclusion](#conclusion)

## Introduction
This repository contains two files demonstrating how to perform binary classification on a simple “circle vs. not-circle” dataset:
- **File 1 (circle_classification.py)** implements a neural network with backpropagation from scratch using NumPy.
- **File 2 (circle_classification_pytorch.py)** implements the same problem using PyTorch.

Both approaches achieve similar classification outcomes but differ significantly in how they handle the training process, efficiency, and scalability.

## File 1: circle_classification.py
**Key Points**:
- **Manual Backpropagation**:  
  - Forward pass explicitly calculates `Z1`, `A1`, `Z2`, and `A2`.
  - Backward pass manually computes gradients (`dZ2`, `dW2`, `db2`, etc.).
- **Binary Cross-Entropy (BCE) Loss**:  
  - Implemented manually via the BCE formula.
- **Training Loop**:
  - Forward pass, loss computation, backward pass, parameter update.
- **Decision Boundary Visualization**:
  - Mesh grid to visualize the boundary between inside and outside the circle.
- **Pros**:
  - Full transparency, educational value.
- **Cons**:
  - Limited efficiency, no GPU acceleration.
  - Harder to scale or extend (new layers, optimizers, etc.).

## File 2: circle_classification_pytorch.py
**Key Points**:
- **PyTorch NN Module**:  
  - Layers defined using `nn.Linear`.
  - Xavier initialization with `nn.init.xavier_uniform_`.
- **Automatic Differentiation**:
  - PyTorch autograd computes all gradients automatically.
- **BCELoss**:
  - Built-in `nn.BCELoss` for binary classification tasks.
- **Training Loop**:
  - Standard PyTorch pattern: forward pass, compute loss, `loss.backward()`, and `optimizer.step()`.
- **Decision Boundary Visualization**:
  - Similar approach using a mesh grid.
- **Pros**:
  - Efficient and scalable with GPU support.
  - Large ecosystem and community support.
  - Less error-prone (no manual gradient calculations).
- **Cons**:
  - Less transparency into the underlying math.

## Why PyTorch Is More Efficient and Scalable
- **Built-in Autograd**:  
  PyTorch dynamically tracks operations to compute gradients efficiently.
- **Optimized Operations**:  
  Matrix multiplications and other numerical routines are highly optimized in C++ and can leverage GPU acceleration.
- **Modular Ecosystem**:
  Layers, optimizers, and scheduling are easily interchangeable.
- **Community & Flexibility**:
  Library of pre-trained models and examples.


Both scripts solve the same classification task, but PyTorch generally offers better scalability and a more user-friendly framework for real-world applications.
