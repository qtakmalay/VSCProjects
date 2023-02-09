# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 09.08.2022

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

################################################################################

In this file, we will learn how to create computational graphs, speed up our
Python code, utilize the GPU and get ready for the basics of machine learning
code with the PyTorch module.
"""

################################################################################
# General information on GPU programming
################################################################################

# There are 2 major standards for communicating with GPUs:
#   > OpenGL: Supported by most GPUs (AMD and NVIDIA).
#   > CUDA: Optimized for scientific calculations and neural networks,
#     restricted to NVIDIA GPUs.
# Both interfaces are relatively hardware dependent, especially newer NVIDIA
# GPUs benefit from the latest CUDA versions. However, not every NVIDIA GPU
# supports every CUDA version.

# There are 2 major factors that can slow down your GPU calculations:
#   > Bottleneck data transfer: If your GPU has a small memory or the transfer
#     bandwidth is slow, loading your data from CPU RAM to GPU memory will
#     become a major issue.
#   > Computation speed: The actual speed of your GPU computations might not be
#     fast enough, or the computations you issued might not be parallelized
#     optimally.
# Rule of thumb: GPU utilization should be as high as possible, GPU memory and
# bandwidth utilization should be as low as possible (you can check this in the
# NVIDIA server settings on Linux or some aftermarket tools like MSI Afterburner
# on Windows).


################################################################################
# Computational graphs
################################################################################

# Using CUDA or OpenGL directly is possible but usually tedious. However, there
# are modules that allow for easier GPU utilization and optimization of your
# code. These modules typically require you to write abstract code that will get
# translated to a computational graph, which is optimized for one or multiple
# GPU(s) and/or CPU(s). Since we often need to calculate gradients in machine
# learning, some modules use the computational graph to automatically compute
# the gradients. Commonly used modules that also provide automatic gradient
# computation are:
#   > Theano (www.deeplearning.net/software/theano/): Creates a static graph;
#     optimization for CPU(s) or a single GPU; predecessor of
#     TensorFlow/PyTorch; ceased development
#   > TensorFlow1 (www.tensorflow.org): Creates a rather static computational
#     graph; very popular in production (pushed by Google/Deepmind);
#     optimization for CPU(s), GPU(s), TPU(s); very similar to Theano; provides
#     Tensorboard (visualization tools in web browser); not very Python-like
#     code
#   > TensorFlow2 (www.tensorflow.org): Creates a more dynamic computational
#     graph; very popular in production (pushed by Google/Deepmind);
#     optimization for CPU(s), GPU(s), TPU(s); more similar to PyTorch; provides
#     Tensorboard (visualization tools in web browser); more Python-like code;
#     partly uses Keras as interface
#   > PyTorch (www.pytorch.org): Creates a more dynamic computational graph;
#     very popular in research (pushed by Facebook); optimization for CPU(s),
#     GPU(s)(, TPU(s)); good for development and research; more Python-like code
# All of these modules mainly deal with arrays and integrate nicely with
# NumPy. There also exist modules that are wrappers for, e.g., neural network
# design, such as Keras https://keras.io/.


################################################################################
# NVIDIA drivers, CUDA, cuDNN
################################################################################

# If you want to utilize your GPU for high-performance computation, you will
# have to install appropriate drivers. Having the latest driver versions can
# give you speed-ups. For NVIDIA GPUs, there are 3 components that you have to
# consider in the setup: the GPU driver, CUDA, cuDNN (make sure that whatever
# module you want to use, it can actually use the CUDA version). Important note
# on PyTorch: PyTorch already comes with prebuilt binaries for CUDA and cuDNN
# (if you install it the normal way), so there is no need to manually install
# CUDA and cuDNN. Useful links for manual installation:
# GPU driver: https://www.nvidia.com/Download/index.aspx
# CUDA: https://developer.nvidia.com/cuda-downloads
# cuDNN: https://developer.nvidia.com/cudnn


################################################################################
# PyTorch
################################################################################

# Install instructions: https://pytorch.org/get-started/locally/
# This section will give a very short introduction to PyTorch. For more detailed
# introductions, please refer to the lecture in the next semester (Programming
# in Python II) or https://pytorch.org/tutorials/.

# Import NumPy since we will be using arrays later.
import numpy as np
# Import the PyTorch module, which is called "torch".
import torch
# We will need this for time measurements later on (CPU vs. GPU)
import time

#
# Tensors and computational graphs
#

# We will start with a simple example for a computational graph. Let's assume we
# want to build a computational graph for the formula "c=a*b", where "a" and "b"
# are inputs and "c" should be our result.

# In Python, we could write such a formula as:
a = 5.0
b = 4.0
c = a * b
print(f"Python c: {c}")
# ... which will give us 3 variables that point to float values. Variable "c"
# was created from variables "a" and "b". However, "c" only points to the result
# of the computation. If we had information about how "c" was computed (in our
# case that it is the result of a multiplication of "a" and "b"), we could apply
# optimization methods, automatic differentiation (=compute gradients
# automatically), and other magic. PyTorch (and others) store this information
# in a "computational graph". In case of PyTorch, this computational graph is
# built on-the-fly and quite pythonic, as we will see in a few lines. This is in
# contrast to Theano/Tensorflow 1, where one first creates the computational
# graph (symbolically without values) and then runs the graph while supplying
# input values.

# In PyTorch, we could write our formula like this: Point "a" and "b" to a
# PyTorch tensor (=node in graph) and keep track of gradients. A PyTorch tensor
# can be seen as a multidimensional data array (similar to NumPy arrays) but
# with additional gradient information to be used in the computational graph.
# Here, we create scalars, i.e., zero-dimensional tensors, but arbitrary
# dimensional tensors can be created.
a = torch.tensor(5.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)
c = a * b  # Point "c" to multiplication of "a" and "b" (=node in graph).
# We now have defined 3 nodes in our graph (also called tensors). In PyTorch, we
# can simply evaluate it by accessing the variable:
print(f"PyTorch c: {c}")  # Prints the tensor.

# Since "c" is pointing to a PyTorch tensor, we have access to the benefits of
# the computational graph. Furthermore, the computation of "c" is optimized (by
# default for CPU). Some examples on what we can do with the tensor:

# Access the value as Python object (if it is a scalar tensor):
print(f"c.item(): {c.item()}")

# Get the computational graph for gradient computation:
print(f"c.grad_fn: {c.grad_fn}")

# Compute gradients of "c" w.r.t. its input nodes:
c.backward()
# The gradients that were computed are now accumulated in the nodes:
print(f"a.grad: {a.grad}")  # This is the derivative of "c" w.r.t. "a".
# ("c=a*b" ... derivative of this "c" w.r.t. "a" is "1*b", which has value 4.0)
print(f"b.grad: {b.grad}")  # This is the derivative of "c" w.r.t. "b".
# Important: The gradients are accumulated in the nodes. If you want to reset
# them, you have to call
a.grad.zero_()
# to reset it. This comes in handy in ML applications (resetting is easier too,
# as we will see later).

# Let's take this a step further and make our graph more complex:
a = torch.tensor(5.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)
c = a * b
c.retain_grad()  # This will keep the computed gradient for "c".
d = a * c
# Now, our graph is a little longer, but we can still get the computational
# graph for the gradient computation the same way:
print(f"d.grad_fn: {d.grad_fn}")
d.backward()  # Computes derivative of c*a = a*b*a.
print(f"a.grad: {a.grad}")
print(f"b.grad: {b.grad}")
print(f"c.grad: {c.grad}")

# We can remove/detach a node from the graph using "detach" to show the
# interaction between PyTorch and NumPy arrays further below.
a = torch.tensor(5.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)
c = a * b
c = c.detach()  # This will detach "c" from the graph (=its "history").
d = a * c
print(f"d.grad_fn: {d.grad_fn}")
d.backward()  # Computes derivative of c*a = 20*a.
print(f"a.grad: {a.grad}")
print(f"b.grad: {b.grad}")
print(f"c.grad: {c.grad}")

# We can convert tensors that have no gradient information to NumPy arrays:
# (since "c" is a scalar tensor, the return value will be a NumPy scalar)
print(f"c.detach().numpy(): {c.detach().numpy()}")

# If you want a code block where gradients are not stored, you can use:
with torch.no_grad():
    # No gradients are computed/stored in this code block.
    e = a * b
    print(e)

#
# Optimizing parameters
#

# We can easily create parameters that we want to optimize/train in PyTorch.
a = torch.tensor(5.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)
# But now, we create a trainable parameter "w" from tensor "a":
w = torch.nn.Parameter(a, requires_grad=True)
# We compute some output tensor "output" given "w" and "b":
output = w * b

# ... but we would like "output" to be close to the value "target":
target = torch.tensor(10.0)

# We can use gradient descent to change our trainable parameter "w" such that
# "output" (in ML, this would be the prediction) is closer to "target":
optimizer = torch.optim.SGD([w], lr=0.01)  # SGD = stochastic gradient descent
for update in range(25):
    w_before = w.item()
    output = w * b
    loss = (output - target) ** 2  # MSE (mean squared error) loss
    loss.backward()  # Calculate gradients
    optimizer.step()  # Do one SGD step (changes the value of "w")
    optimizer.zero_grad()  # Reset gradients (or they would be accumulated)
    print(f"update:{update:3d}: w={w_before:9.6f} --> loss={loss.item():11.6f} --> update to w={w.item():9.6f}")

#
# PyTorch and NumPy arrays
#

# PyTorch and NumPy work together nicely. Tensors in PyTorch can be arrays and,
# for a large part, used the same way as NumPy arrays (indexing, computations,
# arithmetic functions, etc.; see https://pytorch.org/docs/stable/index.html).
a = torch.arange(5 * 4).reshape(5, 4)
print(a)
print(f"a.shape: {a.shape}")
print(f"a.dtype: {a.dtype}")
print(f"a.sum(): {a.sum()}")

# We can also create tensors from NumPy arrays:
a = torch.from_numpy(np.arange(5 * 4).reshape(5, 4))
print(a)

#
# Utilizing CPU or GPU
#

# This section only works if you have a GPU and installed PyTorch with CUDA.
# PyTorch uses the CPU as default device. To perform computations on a different
# device, you can either create tensors on this device or "send" tensors from
# one device to another device (this requires a copy, since the CPU and GPU each
# have their own memory). Values for devices: "cpu" for CPU and "cuda:x" for GPU
# with ID "x" (or simply "cuda", which uses the current GPU device).
a = torch.tensor(5.0, requires_grad=True, device="cpu")  # Create on CPU ...
a = a.to(device="cuda:0")  # ... and send to GPU0
b = torch.tensor(4.0, requires_grad=True, device="cuda:0")  # Create on GPU0
c = a * b  # This is computed on GPU0 since the nodes are on GPU0
print(f"GPU c: {c}")  # Prints the tensor, which is on GPU0

# Since "c" is on the GPU, CPU operations will not work, e.g., "numpy":
# print(c.detach().numpy())  # Not possible for GPU tensors

# However, we can copy "c" to the CPU easily using "cpu":
print(f"Numpy c: {c.detach().cpu().numpy()}")


# We can compare the speed of matrix computations on CPU vs. GPU:
def compute(device, size=1000, dtype=torch.float32):
    x = torch.arange(size * size, dtype=dtype, device=device).reshape((size, size))
    y = torch.arange(size * size, dtype=dtype, device=device).reshape((size, size))
    z = torch.ones_like(x)
    num_comp = size + size % 2  # Ensure an even number of computations (-> leads to z = 1).
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_comp):
            z = x * y / z.mean()
        z = z.mean()
        end_time = time.time()
    print(f"device: {device}")
    print(f"result: {z}")
    print(f"  time: {end_time - start_time:.2f} sec")


# Note that the result might be slightly different on different devices, which
# is due to how floating point numbers and arithmetic are handled. See the
# following links for more details:
# https://discuss.pytorch.org/t/why-different-results-when-multiplying-in-cpu-than-in-gpu/1356
# https://discuss.pytorch.org/t/significance-of-the-difference-between-cpu-and-gpu-results/14739
# https://docs.nvidia.com/cuda/floating-point/index.html
compute("cpu")
compute("cuda:0")

#
# PyTorch for ML
#

# PyTorch offers many functions for ML and especially neural networks. This
# includes:
#   > Tools for creating networks (torch.nn),
#   > Handling data (torch.utils.data),
#   > Optimizing parameters (torch.optim), etc.
# We will learn more about this next semester.
