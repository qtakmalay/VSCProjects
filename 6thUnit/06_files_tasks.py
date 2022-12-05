# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Van Quoc Phuong Huynh, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 29.07.2022

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

import glob
import os

#
# Task 1
#

# Create a File "my_file.txt" and write some text to it. Then, open it again and
# append some text. Finally, open it only to read all the content and print it
# to the console. Use "with" blocks to open the files.

# Your code here #


#
# Task 2
#

# Use the "glob" module to print a list of all files ending in ".txt" in the
# current working directory and all subdirectories. Also use the "os" module
# when joining paths.

# Your code here #


#
# Task 3
#

# Write a function that can write 2D nested lists/matrices (parameter 1) to a
# CSV file (parameter 2). As delimiter (parameter 3), a comma should be used by
# default. Optionally, matrix column names can be specified (parameter 4, can be
# assumed to be a list of strings), which must then also be written to the CSV
# file as the first row. The character encoding should be UTF-8 by default
# (parameter 5). You do not have to perform any error checking in this task like
# checking if the number of column names matches the number of columns in the
# specified matrix.
example_matrix = [
    [1, 10, 9, 12],
    [3, 10, 3, 10],
    [7, 14, 5, 28]
]

# Your code here #


#
# Task 4
#

# Write a function that can read CSV files (parameter 1) as produced by task 3
# (see above) and returns the data as a 2D nested list. The list elements can
# stay strings, i.e., no data type conversion is needed. As delimiter (parameter
# 2), a comma should be used by default. A boolean flag should indicate whether
# the file contains column names in the first row, which should be False by
# default (parameter 3). If True, the column names (as list) should be returned
# in addition to the 2D nested list, i.e., a 2-tuple should be returned. The
# character encoding should be UTF-8 by default (parameter 4).

# Your code here #
