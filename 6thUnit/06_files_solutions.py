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

Example solutions for tasks in the provided tasks file.
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
with open("my_file.txt", "w") as wf:
    wf.write("This will create or overwrite the file!\n")
    print("Writing with print().", file=wf)

with open("my_file.txt", "a") as af:
    af.write("Append to the existing file without overwriting it.\n")

with open("my_file.txt", "r") as rf:
    # Read file at once (entire content will be stored in memory!)
    print(rf.read())


#
# Task 2# Task 2 (create new task where we manually recreate "glob" with, e.g., os.listdir)
#

# Use the "glob" module to print a list of all files ending in ".txt" in the
# current working directory and all subdirectories. Also use the "os" module
# when joining paths.

# Your code here #
print(glob.glob(os.path.join("**", "*.txt"), recursive=True))


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


def write_matrix(matrix: list[list], path: str, delimiter: str = ",", header: list[str] = None, encoding: str = "utf-8"):
    with open(path, "w", encoding=encoding) as f:
        if header is not None:
            print(delimiter.join(header), file=f)
        for row in matrix:
            print(delimiter.join(str(i) for i in row), file=f)


write_matrix(example_matrix, "matrix.csv")
write_matrix(example_matrix, "matrix_with_header.csv", header=["a", "b", "c", "d"])


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


def read_matrix(path: str, delimiter: str = ",", extract_header: bool = False, encoding: str = "utf-8"):
    with open(path, encoding=encoding) as f:
        if extract_header:
            header = f.readline()[:-1].split(delimiter)
        matrix = [line[:-1].split(delimiter) for line in f]
    return (matrix, header) if extract_header else matrix


matrix1 = read_matrix("matrix.csv")
matrix2, matrix2_header = read_matrix("matrix_with_header.csv", extract_header=True)
