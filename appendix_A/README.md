# Appendix A — Introduction to PyTorch

Appendix A introduced the PyTorch foundations needed for the rest of the book.

## Main artifacts

- [`notebooks/appA_training_loop_anatomy.ipynb`](notebooks/appA_training_loop_anatomy.ipynb)  
Contains my primary notes, selected code snippets, and observations.
- [`scripts/common.py`](scripts/common.py)  
Shared model, dataset, and evaluation utilities.
- [`scripts/01_train_cpu.py`](scripts/01_train_cpu.py)  
Contains the standard CPU training loop.
- [`scripts/02_train_single_gpu.py`](scripts/02_train_single_gpu.py)  
Contains the single-GPU adaptation, intended for Google Colab.
- [`scripts/03_train_ddp.py`](scripts/03_train_ddp.py)  
Contains the multi-GPU DDP implementation, intended for a two-GPU RunPod environment.

## Learning sequence

1. Standard CPU training
2. Single-GPU training
3. Multi-GPU Distributed Data Parallel training

## Concepts retained

- Forward pass, loss calculation, backpropagation, and optimizer updates
- Training mode versus evaluation mode
- CPU and device-aware training
- One-process-per-GPU distributed training
- Gradient synchronization with DDP

## Status

- CPU implementation: executed locally
- Single-GPU implementation: prepared; Colab execution done
- DDP implementation: prepared; two-GPU RunPod execution pending

---
---

## APPENDIX A SUMMARY

- PyTorch is an open source library with three core components: a tensor library, automatic differentiation functions, and deep learning utilities.
- PyTorch’s tensor library is similar to array libraries like NumPy.
- In the context of PyTorch, tensors are array-like data structures representing scalars, vectors, matrices, and higher-dimensional arrays.
- PyTorch tensors can be executed on the CPU, but one major advantage of PyTorch’s tensor format is its GPU support to accelerate computations.
The automatic differentiation (autograd) capabilities in PyTorch allow us to conveniently train neural networks using backpropagation without manually deriving - gradients.
- The deep learning utilities in PyTorch provide building blocks for creating custom deep neural networks.
- PyTorch includes Dataset and DataLoader classes to set up efficient data-loading pipelines.
- It’s easiest to train models on a CPU or single GPU.
- Using DistributedDataParallel is the simplest way in PyTorch to accelerate the training if multiple GPUs are available.