# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 27.07.2022

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

#
# Task 1
#

# Write a function with a variable number of integer arguments that specify the
# shape of a multidimensional int matrix (use nested lists as implementation).
# For instance, if the user calls "matrix(2, 3)", a nested list is returned,
# where the outer list has 2 inner lists, and each inner list has 3 ints. By
# default, 0 should be used as int values, but the user can specify a different
# fill value via an optional parameter. Use recursion to solve this task.

# Your code here #


#
# Task 2
#

# Write a function that takes an arbitrary number of positional arguments as
# input. You can assume these arguments are either strings or nested lists of
# strings. Create a flat (=not nested) list from these arguments that holds only
# the strings. Use the built-in function "isinstance(a, list)" to check if some
# object "a" is of type "list". Use recursion to solve this task.
some_list = ["file2.txt", "file3.txt"]
some_nested_list = ["file4.png", "file5.txt", ["file6.txt", "file7.dat"]]
# Example function call:
# my_function("file0.txt", "file1.png", some_list, some_nested_list)

# Your code here #


#
# Task 3
#

# Write a generator function "next_element(my_sequence)" that yields the next
# element in "my_sequence". If the sequence is exhausted (i.e., if the last
# element has been reached), the next element should be the first element again.
# Write a loop that calls "next_element" 25 times and prints the currently
# returned value. Example function call:
# next_element("abcdefg")

# Your code here #


#
# Task 4
#

# Write a generator function "next_element(my_sequence, max_iter)" that yields
# the next element in "my_sequence". If the sequence is exhausted (i.e., if the
# last element has been reached), the next element should be the first element
# again. Only a maximum of "max_iter" elements should be returned. Write a loop
# that calls "next_element" until some specified "max_iter" is reached and
# prints the currently returned value. Example function call:
# next_element("abcdefg", max_iter=20)

# Your code here #


#
# Task 5
#

# Write a generator function that takes an iterable of any type as input and
# yields 2-tuples where the first tuple entry is the number of the current loop
# iteration and the second entry is the current loop element. Do not use the
# existing function "enumerate".

# Your code here #
