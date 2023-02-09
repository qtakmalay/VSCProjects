# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 05.08.2022

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

################################################################################

In this file, we will learn how to perform fast (vector/matrix) calculations
with NumPy (using the "numpy" module).
"""

################################################################################
# NumPy - Fast numerical computations
################################################################################

# NumPy is the go-to package for fast (vector/matrix) computations in Python. It
# provides a broad range of tools and mathematical functions. The module "numpy"
# is typically imported as the abbreviation "np". Useful links:
# Homepage: http://www.numpy.org/
# Quick tutorial: https://numpy.org/doc/stable/user/quickstart.html
# API documentation: https://numpy.org/doc/stable/reference/index.html

import numpy as np  # (not installed by default)

# The core data type of NumPy is the "np.ndarray". These arrays can hold any
# data type (it will fall back to Python's object data type in the worst case).
# A "np.ndarray" can have multiple dimensions. NumPy arrays are supported by the
# PyCharm debugger, so you may right-click and select "View as Array" on a
# NumPy array in the variable viewer to get a color-coded view of your array.

#
# Creation of arrays from Python sequences
#

# Creation of a NumPy array from a Python list
my_list = [1, 2, 3, 4]
my_array = np.array(my_list)  # Python list is converted to NumPy array
print(my_array)
print(my_array.tolist())  # NumPy array is converted back to Python list

# NumPy arrays can have multiple dimensions, e.g., 2D matrices. In contrast to
# cumbersome/inefficient nested Python lists, NumPy arrays are fast, more memory
# efficient and provide a better interface to the arrays. However, NumPy arrays
# have fixed and consistent array sizes, i.e., all rows have the same length and
# the shape stays the same, a consistent data type, and are not as flexible as
# Python lists.
my_nested_list = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
my_2d_array = np.array(my_nested_list)  # Inspect this in the debugger

#
# Data types
#

# NumPy arrays are consistent in terms of their data type, i.e., all elements
# have the same data type. NumPy will do its best to convert your list values
# into an efficient data type. Take care that you do not end up with unwanted
# data types when converting Python lists to NumPy arrays. To be on the safe
# side, you can specify the data type explicitly. You can specify the number of
# bits in floats/integers using dedicated NumPy data types. For example: float16
# -> 16bit float; int64 -> 64bit integer. Fore more details, see
# https://numpy.org/doc/stable/user/basics.types.html
# https://numpy.org/doc/stable/reference/arrays.scalars.html

my_list = [1, 2, 3, 4]
int_array = np.array(my_list)
print(int_array, int_array.dtype)  # "dtype" attribute contains the data type
float32_array = np.array(my_list, dtype=np.float32)  # Force 32bit float
print(float32_array, float32_array.dtype)
int_array = float32_array.astype(dtype=int)  # Cast to default int (=np.int_)
print(int_array, int_array.dtype)

my_list = [1, 2.5, 3, 4]
my_array = np.array(my_list)  # Cast to default float (=np.float_)
print(my_array, my_array.dtype)

# NumPy also supports boolean data types:
my_list = [0.0, 2.5, -3, 0]
my_array = np.array(my_list, dtype=bool)  # Applies standard type conversion
print(my_array, my_array.dtype)

#
# Creation of arrays by shape
#

# Instead of creating NumPy arrays from Python lists, they can be created based
# on a tuple that specifies the desired shape of the array dimensions and some
# initial value for the elements in the array.

# 1D array, with 5 elements, initialized with 1
ones_1d = np.ones(shape=(5,), dtype=int)
print(ones_1d, ones_1d.shape, ones_1d.dtype)

# 2D array, with 5x2 elements, initialized with 1
ones_2d = np.ones(shape=(5, 2), dtype=int)
print(ones_2d, ones_2d.shape, ones_2d.dtype)

# 2D array, with 6x3 elements, initialized with 0
zeros_2d = np.zeros(shape=(6, 3), dtype=float)
print(zeros_2d, zeros_2d.shape, zeros_2d.dtype)

# 3D with dimensions (2, 3, 4), faster but not initialized (arbitrary data)!
empty_3d = np.empty(shape=(2, 3, 4), dtype=float)
print(empty_3d, empty_3d.shape, empty_3d.dtype)

# Getting the shape
print("empty_3d.shape", empty_3d.shape)  # Getting the shape as tuple
print("empty_3d.ndim", empty_3d.ndim)  # Getting the number of dimensions
print("empty_3d.size", empty_3d.size)  # Getting the number of all elements

# The more dimensions an array has, the more difficult it becomes to make a
# useful visualization and interpretation of the data (also depends on the
# context). For the example above, we could think of the three dimensions as:
#     dim 0: number of elements, where each element is a matrix
#     dim 1: number of elements, where each element is a row of one such matrix
#     dim 2: number of elements, where each element is an entry of one such row
#            = column of a matrix
# which would translate to: 2 matrices, each with 3 rows and 4 columns.

#
# Creation of arrays with ranges of values
#

# Naive and slow:
my_range = np.array(range(5))

# Faster: Creation of ranges of values via "np.arange(start, stop, step, ...)"
print("np.arange(5)")
print(np.arange(5))
print("np.arange(2, 5)")
print(np.arange(2, 5))
print("np.arange(5, 0, -1)")
print(np.arange(5, 0, -1))

# Creation of ranges of values via "np.linspace(start, stop, num, ...)"
print("np.linspace(0, 5, 10)")
print(np.linspace(0, 5, 10))
print("np.linspace(0, 5, 10, endpoint=False)")
print(np.linspace(0, 5, 10, endpoint=False))

#
# Accessing elements
#

# Indexing is similar to Python lists:
one_dim = np.arange(25)

print("one_dim")
print(one_dim)

# Indexing via integer index
print("one_dim[3]")
print(one_dim[3])
print("one_dim[-3]")
print(one_dim[-3])

# Indexing via slice
print("one_dim[3:10:2]")
print(one_dim[3:10:2])
print("one_dim[3:6]")
print(one_dim[3:6])
print("one_dim[:]")
print(one_dim[:])

# "Fancy indexing": indexing via list/array of indices
print("one_dim[[3, 4, 6, 15]]")
print(one_dim[[3, 4, 6, 15]])

# For more dimensions, you can separate indices by commas, e.g., [2, 3] instead
# of repeated index accesses like [2][3]
two_dim = np.array([[1,  2,  3,  4],
                    [5,  6,  7,  8],
                    [9, 10, 11, 12]])

print("two_dim[2, 3]")
print(two_dim[2, 3])  # Equal to: two_dim[(2, 3)]
print("two_dim[2]")
print(two_dim[2])  # 2D case: row access; equal to: two_dim[2, :]
print("two_dim[:, 3]")
print(two_dim[:, 3])  # 2D case: column access
print("two_dim[1:2, 3]")
print(two_dim[1:2, 3])
print("two_dim[1:3, 2:4]")
print(two_dim[1:3, 2:4])  # Extract "bottom right 2x2 corner"

# "Fancy indexing": indexing via boolean masks
mask = np.array([[True, False, False, True],
                 [False, True, False, True],
                 [False, False, True, True]])
print("two_dim[mask]")
print(two_dim[mask])  # Returns flat (1D) array of elements where mask = True

# This can be a powerful tool when accessing certain parts based on some boolean
# condition that must hold for the data elements:
mask = (two_dim % 2) == 0  # Only extract even numbers
print("two_dim[mask] = two_dim[two_dim % 2 == 0]")
print(two_dim[mask])

# Important: Use commas to separate dimensions when using slicing!
x1 = two_dim[:, 2]  # All elements along first axis and index 2 in second axis
x2 = two_dim[:][2]  # What will this do?
# It will first create a slice of the array (in this case, this is simply the
# entire array again) and then access index 2 in that sliced array, which
# corresponds to "two_dim[2]"

# Writing to an array follows the same rules:
print(two_dim)
two_dim[1, 1] = 0
print(two_dim)

# Replacing entire parts of an array:
one_dim = np.arange(15)
print(one_dim)
one_dim[2:5] = [0, 0, 0]  # Must match the shape (here: 3 elements)
print(one_dim)

# NumPy "broadcasts" values to the target shape, i.e., the specified value will
# be automatically applied/extended to match the shape:
one_dim = np.arange(15)
print(one_dim)
one_dim[2:5] = 0  # 0 is applied for all elements -> no need for [0, 0, 0]
print(one_dim)

# Broadcasting also works for multidimensional arrays:
two_dim = np.array([[1,  2,  3,  4],
                    [5,  6,  7,  8],
                    [9, 10, 11, 12]])
two_dim[:] = 0
print("two_dim[:] = 0")
print(two_dim)

two_dim = np.array([[1,  2,  3,  4],
                    [5,  6,  7,  8],
                    [9, 10, 11, 12]])
two_dim[1, 1:4] = 0
print("two_dim[1, 1:4] = 0")
print(two_dim)

# Broadcasting subarrays (more details further below):
two_dim = np.array([[1,  2,  3,  4],
                    [5,  6,  7,  8],
                    [9, 10, 11, 12]])
two_dim[:, 1:3] = [7, 8]  # Equal to: np.array([7, 8])
print("two_dim[:, 1:3] = [7, 8]")
print(two_dim)

# More on indexing and broadcasting:
# https://numpy.org/doc/stable/user/basics.indexing.html
# https://numpy.org/doc/stable/user/basics.broadcasting.html

#
# Reshaping arrays
#

# Arrays can be arbitrarily reshaped (change the dimensions) as long as the
# total number of elements stays the same.
a = np.arange(24)  # 1D shape: 24 integers elements
print(a, a.shape)

new_shape = (4, 6)  # 2D shape: 4 subarrays of size 6
a = a.reshape(new_shape)  # Can be called with a tuple or with positional args
print(a, a.shape)

a = a.reshape(2, 3, 4)  # 3D shape: 2 matrices, each of size 3x4
print(a, a.shape)

a = a.reshape(4, 6)  # Back to the above 2D shape
print(a, a.shape)

# Empty dimensions can be inserted via None or "np.newaxis", or you can use
# "reshape" with 1-entries. All the following examples will add an empty
# dimension between dimension 0 and 1:
a1 = a[:, None, :]
a2 = a[:, np.newaxis, :]
a3 = a.reshape(4, 1, 6)
assert a1.shape == a2.shape == a3.shape
print(a1.shape)

# We can let NumPy determine one dimension by specifying a shape value of -1
# (this will be the remaining factor to get to the total number of elements):
a = a.reshape(4, 6, -1)  # 24 elements in total: dim 1 = 6 -> dim 2 = -1 = rest = 24 / 6 = 4
print(a.shape)

# Note: Often, there are equivalent functions that are available on the array
# objects themselves (methods) or via the numpy module ("free functions"). For
# example here, "my_array.reshape(...)" is the object method, and the
# corresponding free function is "np.reshape(my_array, ...)". Typically, they
# can be treated analogously, but there might be some differences in some cases.
# Here, for instance, the method "my_array.reshape(...)" can take the shape as a
# tuple but also as individual positional arguments, i.e.,
# "my_array.reshape(2, 3, 4)" is the same as "my_array.reshape((2, 3, 4))". The
# free function "np.reshape(my_array, ...)", however, can only accept the shape
# as a tuple. Therefore, always refer to the official documentation.

#
# Broadcasting and shapes
#

# If we want to apply broadcasting, we can be more specific/explicit along which
# dimensions to broadcast to avoid hard-to-find errors. Same example as above:
# Insert the subarray [7, 8] in the second dimension at position [1:3] for all
# entries in the first dimension (=broadcast along first dimension). Solution:
two_dim = np.array([[1,  2,  3,  4],
                    [5,  6,  7,  8],
                    [9, 10, 11, 12]])
subarray = np.array([7, 8])
two_dim[:, 1:3] = subarray

# The first step is to extend this subarray with empty dimensions (going from
# right to left) to match the dimensionality of the array where we want to
# insert our subarray:
#
# shape of two_dim[:, 1:3]: (3, 2) -> two dimensions
# shape of subarray before: (2,) -> one dimension
# shape of subarray after extending it: (1, 2) -> two dimensions
#
# Now, both arrays have the same dimensionality, which makes it much more clear
# along which dimensions broadcasting is applied, because broadcast dimensions
# will have a 1-entry in the shape, and non-broadcast dimensions will have
# identical shape entries (if they are neither identical nor 1, an error occurs)
#
# two_dim[:, 1:3] -> (3, 2)
# subarray        -> (1, 2)
#                     ↑  ↑
#                     │  These entries are identical, no broadcasting is done
#                     │
#                     This dimension has a 1-entry (empty dimension), which
#                     means broadcasting is applied along this dimension
#
# The shape of the resulting array will have the same dimensionality, where each
# dimension will be the bigger (or identical) entry. For the above example, the
# result would be of shape (3, 2).
two_dim = np.array([[1,  2,  3,  4],
                    [5,  6,  7,  8],
                    [9, 10, 11, 12]])
subarray = np.array([7, 8]).reshape(1, 2)  # Extend to match dimensionality
two_dim[:, 1:3] = subarray

# Normally, you do not have to worry about this, i.e., you do not have to
# manually ensure matching shapes (this is done automatically by NumPy). It can,
# however, help you in understanding how broadcasting is done, especially for
# more complex scenarios.

#
# Appending, concatenating, repeating, tiling
#

a = np.arange(5)
print(np.append(a, np.array([1, 2, 3])))
print(np.concatenate([a, a, a]))
print(np.repeat(a, repeats=5))
print(np.tile(a, reps=5))

#
# Operations on arrays
#

# Common mathematical operators work element-wise on NumPy arrays:
two_dim = np.arange(1, 13).reshape(3, 4)
print("two_dim")
print(two_dim)

new_two_dim = two_dim + 2  # Creates a new array
print("new_two_dim")
print(new_two_dim)

two_dim[:] = two_dim + 2  # Overwrites content of old array (in-place)
print("two_dim + 2")
print(two_dim)

two_dim[:] += two_dim * 3  # Overwrites content of old array (in-place)
print("two_dim + two_dim * 3")
print(two_dim)

# This "element-wise" is actually another example of broadcasting. Consider the
# above example of "two_dim + 2" again. Let's analyze their shapes:
#
# shape of two_dim: (3, 4)
# shape of integer 2: () -> np.array(2) -> scalar -> dimensionality of 0
#
# Extend to match dimensionality:
#
# two_dim -> (3, 4)
# 2       -> (1, 1)
#             ↑  ↑
#             1-entries, so both dimensions are used for broadcasting here

# As mentioned earlier, this can help you in understanding more complex
# scenarios, e.g., when you try to figure out if some operation works when you
# are dealing with high-dimensional arrays. Some examples:
#
#     original shape -> extended shape
# arr1: (2, 3, 4, 5) -> (2, 3, 4, 5)
# arr2: (4, 5)       -> (1, 1, 4, 5) -> works; result shape: (2, 3, 4, 5)
#
# arr1: (2, 3, 4, 5) -> (2, 3, 4, 5)
# arr2: (3, 5)       -> (1, 1, 3, 5) -> does not work (3 != 4)
#
# arr1: (2, 3, 4, 5) -> (2, 3, 4, 5)
# arr2: (2,)         -> (1, 1, 1, 2) -> does not work (2 != 5)
#
# arr1: (2, 3, 4, 5) -> (2, 3, 4, 5)
# arr2: (3, 4)       -> (1, 1, 3, 4) -> does not work (3 != 4 and 4 != 5)
#
# arr1: (2, 3, 4, 5) -> (2, 3, 4, 5)
# arr2: (3, 4, 1)    -> (1, 3, 4, 5) -> works; result shape: (2, 3, 4, 5)
#
# arr1: (2, 3, 4, 5) -> (2, 3, 4, 5)
# arr2: (3, 1, 5)    -> (1, 3, 1, 5) -> works; result shape: (2, 3, 4, 5)
#
# arr1: (7, 1, 3, 4) -> (7, 1, 3, 4)
# arr2: (2, 1, 4)    -> (1, 2, 1, 4) -> works; result shape: (7, 2, 3, 4)

#
# Useful functions (only a select few, NumPy provides many more)
#

# Sum (similar functions available for: mean, standard deviation, etc.)
arr = np.arange(2 * 3 * 4).reshape((2, 3, 4)) * 3
print(arr.sum(axis=0))  # Sum along first axis/dimension (axis 0 will be "collapsed" -> shape: (3, 4))
print(arr.sum(axis=1))  # Sum along second axis/dimension (axis 1 will be "collapsed" -> shape: (2, 4))
print(arr.sum(axis=2))  # Sum along second axis/dimension (axis 2 will be "collapsed" -> shape: (2, 3))
print(arr.sum(axis=(1, 2)))  # Sum along second + third axis (axis 1 + 2 will be "collapsed" -> shape: (2,))
print(arr.sum())  # "Global" sum along all axes/dimensions

# Min, max
print(arr.min())  # "Global" minimum (minimum along all axes/dimensions)
print(arr.max())  # "Global" maximum (maximum along all axes/dimensions)
print(arr.max(axis=1))  # Maximum elements in the second axis/dimension (shape: (2, 4))

# Index of min, max
i_min_flat = arr.argmin()  # Index in the (flat!) array where the minimum is stored
print(i_min_flat)
i_min_shape = np.unravel_index(i_min_flat, shape=arr.shape)  # Transform flat index to "arr"-shaped index
print(i_min_shape)
print(arr.argmax())  # Index in the (flat!) array where the maximum is stored
print(arr.argmax(axis=1))  # Indices of maximum elements in the second axis/dimension (shape: (2, 4))

# Functions with conditionals
other = -arr
print(np.where(arr % 2 == 0, arr, other))  # If condition is True, take from "arr", else take from "other"
print((arr > 50).all())  # Check if all elements meet the condition
print((arr > 50).any())  # Check if at least one element meets the condition

# Random values
rng = np.random.default_rng(seed=1234)  # Get a random number generator (RNG) with a fixed seed for reproducibility
arr = rng.random((4, 3))  # 4x3 random float samples from a uniform distribution over [0, 1)
print(arr)
arr = rng.integers(low=1, high=25, size=5)  # Get 5 random integers from the interval [1, 25)
print(arr)
arr = rng.choice(np.arange(100), size=10, replace=False)  # Take random samples with/without replacement
print(arr)

# Sorting
sorted_arr = np.sort(arr)
print(sorted_arr)
arr.sort()  # In-place sorting, i.e., the array is changed directly
print(arr)

#
# Memory consumption and performance
#

# Reshaping and slicing does not copy the original array, it only creates
# another "view" on the data. For example:
a1 = np.arange(4)
a2 = a1.reshape(2, 2)  # Only creates a view
a1[0] = -5  # This will also lead to "a2[0, 0]" being -5
print(a2)

# Casting an array to another data type via "my_array.astype(dtype=...)" or
# "np.array(my_array, dtype=...)" or setting "np.array(my_array, copy=True)"
# will copy the data, as does the explicit copy function "np.copy(my_array)".
a1 = np.arange(4)
a2 = np.array(a1, copy=True)  # copy=True is the default behavior
a1[0] = -5
print(a2)
# The similarly looking function "np.asarray(my_array, dtype=...)" will not copy
# in all cases, e.g., if "dtype" does not change.

# NumPy functions are fast (in most cases), so use them over Python functions
# when dealing with (vector/matrix) computations. Copying arrays takes time and
# memory due to memory allocation. Reusing the allocated arrays instead of
# creating new ones can lead to large speedups when dealing with larger arrays.

#
# Saving NumPy arrays
#

# NumPy allows for saving NumPy arrays and some other objects to files,
# optionally while using compression. Storing NumPy arrays using compression:
# https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html
# Storing NumPy arrays without compression:
# https://numpy.org/doc/stable/reference/generated/numpy.savez.html
# Loading NumPy arrays:
# https://numpy.org/doc/stable/reference/generated/numpy.load.html
# You can also use the "dill" module.
