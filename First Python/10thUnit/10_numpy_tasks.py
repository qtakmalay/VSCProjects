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

Tasks for self-study. Try to solve these tasks on your own and compare your
solutions to the provided solutions file.
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


#
# Task 2
#

# Create a NumPy array "my_array" with shape (8, 5) and 32bit integer values
# that are initialized with value 1.

# Your code here #


#
# Task 3
#

# Given the array below, extract the values at index 0 of the first dimension
# (axis 0). Then, overwrite them with ascending integer values from 0 to 4
# inclusive. Afterwards, print the array. Next, select the "bottom right 3x3
# corner" and set their values to 0. Print the array to see the effect.
arr = np.ones(shape=(4, 5))

# Your code here #


#
# Task 4
#

# Given the array below, multiply all values at index 2 of the second dimension
# (axis 1) by 3. Then, reshape the array to the 3D shape (5, 4, 2) and print it.
arr = np.arange(8 * 5).reshape(8, 5)

# Your code here #


#
# Task 5
#

# Create an array "values" that contains 64bit integer values from 0 to 10,
# including the value 10. Then, create an array "squared" that contains the
# squared values of array "values".

# Your code here #


#
# Task 6
#

# Get the values of array "long_array" that are divisible by 3.
long_array = np.arange(3 * 5 * 4).reshape((3, 5, 4))

# Your code here #


#
# Task 7
#

# Broadcast "subarray" over the first and last dimension of "array" such that
# after broadcasting, any access array[n, :, m] returns the values [1, 2, 3].
# Hint: You will have to be explicit about which dimensions to broadcast.
subarray = np.array([1, 2, 3])
array = np.zeros(shape=(5, 3, 3))

# Your code here #


#
# Task 8
#

# Sample 10 random values from a uniform distribution over [0, 4) and retrieve
# the maximum value. Afterwards, sort the array and get the last element, which
# should then be the same as your retrieved maximum value.

# Your code here #
