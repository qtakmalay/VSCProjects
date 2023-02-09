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

Example solutions for tasks in the provided tasks file.
"""

import numpy as np

#
# Task 1
#

# Assume we want to store 2D array data in a Python list, but we do not want to
# use a nested list but rather a flat list with all elements (1D). Create a 1D
# Python list "my_1d_list" from the 2D list "my_2d_list". Use row-major order,
# i.e., the elements should be collected from each row. Analogously, also create
# a 1D NumPy array "my_1d_array" from "my_2d_list".
my_2d_list = [
    [10, 11, 12],  # This is the first row
    [13, 14, 15],  # This is the second row
    [16, 17, 18],  # This is the third row
    [19, 20, 21]   # This is the fourth row
]

# Your code here #
# Naive and slow solution:
my_1d_list = []
for row in my_2d_list:
    for element in row:
        my_1d_list.append(element)
        
# List-comprehension solution:
# my_1d_list = [element for row in my_2d_list for element in row]

# Making use of the "+" operator for list concatenation:
# my_1d_list = []
# for row in my_2d_list:
#     my_1d_list += row

# Same as above but more compact:
# my_1d_list = sum(my_2d_list, start=[])

# NumPy array:
my_1d_array = np.array(my_2d_list).flatten()  # Equal to: .reshape(-1)


#
# Task 2
#

# Create a NumPy array "my_array" with shape (8, 5) and 32bit integer values
# that are initialized with value 1.

# Your code here #
my_array = np.ones(shape=(8, 5), dtype=np.int32)


#
# Task 3
#

# Given the array below, extract the values at index 0 of the first dimension
# (axis 0). Then, overwrite them with ascending integer values from 0 to 4
# inclusive. Afterwards, print the array. Next, select the "bottom right 3x3
# corner" and set their values to 0. Print the array to see the effect.
arr = np.ones(shape=(4, 5))

# Your code here #
arr[0] = np.arange(5)
print(arr)
arr[-3:, -3:] = 0
print(arr)


#
# Task 4
#

# Given the array below, multiply all values at index 2 of the second dimension
# (axis 1) by 3. Then, reshape the array to the 3D shape (5, 4, 2) and print it.
arr = np.arange(8 * 5).reshape(8, 5)

# Your code here #
arr[:, 2] *= 3
arr = arr.reshape(5, 4, 2)
print(arr)


#
# Task 5
#

# Create an array "values" that contains 64bit integer values from 0 to 10,
# including the value 10. Then, create an array "squared" that contains the
# squared values of array "values".

# Your code here #
values = np.arange(11, dtype=np.int64)
squared = values ** 2


#
# Task 6
#

# Get the values of array "long_array" that are divisible by 3.
long_array = np.arange(3 * 5 * 4).reshape((3, 5, 4))

# Your code here #
vals = long_array[long_array % 3 == 0]


#
# Task 7
#

# Broadcast "subarray" over the first and last dimension of "array" such that
# after broadcasting, any access array[n, :, m] returns the values [1, 2, 3].
# Hint: You will have to be explicit about which dimensions to broadcast.
subarray = np.array([1, 2, 3])
array = np.zeros(shape=(5, 3, 3))

# Your code here #
# "array" has shape (5, 3, 3). To broadcast along first and last dimension, we
# need an array (1, 3, 1) or at least (3, 1), since the extension (prepending
# empty dimensions) is done automatically. Reshape into the desired dimensions:
subarray = subarray.reshape((1, 3, 1))
# Alternative solution: add empty dimensions or new axes:
# subarray = subarray[None, :, None]
# subarray = subarray[np.newaxis, :, np.newaxis]
array[:] = subarray
print(f"array[0, :, 0] -> {array[0, :, 0]}")
print(f"array[0, :, 2] -> {array[0, :, 2]}")
print(f"array[2, :, 1] -> {array[2, :, 1]}")


#
# Task 8
#

# Sample 10 random values from a uniform distribution over [0, 4) and retrieve
# the maximum value. Afterwards, sort the array and get the last element, which
# should then be the same as your retrieved maximum value.

# Your code here #
rng = np.random.default_rng(seed=0)
rnd = rng.random(10) * 4
print(rnd.max())
rnd.sort()  # In-place, "np.sort" would have also been possible, of course
print(rnd[-1])
