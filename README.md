# TinyTorch

**TinyTorch** is a lightweight deep learning training framework implemented from scratch in C++. The project's goal is to closely mimic the API design of PyTorch, serving as an educational tool for developers to understand the core mechanics of a deep learning framework, including automatic differentiation, modular network structures, loss functions, and optimizers.

This project provides a complete yet simple implementation that allows developers to directly see the inner workings of each component, making it an ideal resource for learning.

For more details, please refer to my blog post: [Write a nn training framework from scratch](https://robot9.me/write-nn-framework-from-scratch-tinytorch/)

[![CMake Linux](https://github.com/keith2018/TinyTorch/actions/workflows/cmake_linux.yml/badge.svg)](https://github.com/keith2018/TinyTorch/actions/workflows/cmake_linux.yml)
[![CMake MacOS](https://github.com/keith2018/TinyTorch/actions/workflows/cmake_macos.yml/badge.svg)](https://github.com/keith2018/TinyTorch/actions/workflows/cmake_macos.yml)
[![CMake Windows](https://github.com/keith2018/TinyTorch/actions/workflows/cmake_windows.yml/badge.svg)](https://github.com/keith2018/TinyTorch/actions/workflows/cmake_windows.yml)

## Key Features

* **PyTorch-Style API**: Adopts similar class and function naming conventions as PyTorch, such as `Tensor`, `Functions`, `nn.Module`, and `Optimizer`, to make it intuitive for users familiar with the original framework.
* **Pure C++ Implementation**: The entire framework is written in C++ with no dependency on external deep learning libraries, making it perfect for understanding the fundamentals.
* **CPU and CUDA Support**: The framework is built to run on both **CPU** and **CUDA** enabled GPUs, allowing for flexible development and experimentation.

## Implemented Operators and Components

### Activation Functions
* `relu`, `gelu`, `silu`
* `softmax`, `logSoftmax`

### Mathematical Operations
* `add`, `sub`, `mul`, `div`, `matmul`
* `sin`, `cos`, `sqrt`, `pow`
* `maximum`, `minimum`

### Comparison and Logical Operations
* `lt`, `le`, `gt`, `ge`, `eq`, `ne`
* `logicNot`, `logicAnd`, `logicOr`

### Statistical and Reduction Operations
* `min`, `argmin`, `max`, `argmax`
* `sum`, `mean`, `var`

### Tensor Shape and Indexing Operations
* `reshape`, `view`, `permute`, `transpose`
* `flatten`, `unflatten`, `squeeze`, `unsqueeze`
* `split`, `concat`, `stack`, `hstack`, `vstack`, `narrow`
* `topk`, `sort`, `cumsum`
* `gather`, `scatter`

### Neural Network Layers and Loss Functions
* `linear`
* `dropout`
* `maxPool2d`
* `conv2d`
* `embedding`
* `layerNorm`
* `rmsNorm`
* `sdpAttention`
* `mseLoss`
* `nllLoss`

### Optimizers
* `SGD`, `Adagrad`, `RMSprop`, `AdaDelta`, `Adam`, `AdamW`

### Other
* `Dataset`, `DataLoader`, `data.Transform`

## Automatic differentiation

TinyTorch's automatic differentiation (AD) is implemented by building a computation graph. Each operation on a `Tensor` is represented by a `Function` object, which is responsible for both the forward and backward passes. The `Function` nodes are connected via a `nextFunctions` field, creating the dependency graph. During the `backward()` call, the framework traverses this graph in reverse order, computing and propagating gradients using the chain rule.

<img src=doc/AD.png width="400">

## Getting Started

### Prerequisites
* CMake
* C++17 or a more recent compiler
* CUDA Toolkit 11.0+ (optional)

### Build
```bash
mkdir build
cmake -B ./build -DCMAKE_BUILD_TYPE=Release
cmake --build ./build --config Release
```

### Run `MNIST` Demo
```bash
cd demo/bin
./TinyTorch_demo
```

### Run Tests
```bash
cd build
ctest
```

## License
This code is licensed under the MIT License (see [LICENSE](LICENSE)).
