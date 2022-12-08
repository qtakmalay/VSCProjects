# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 01.08.2022

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

# Create a class "Face" with an attribute "orientation" and two methods
# "look(new_orientation)" and "show()". "new_orientation" is a string argument.
# If "new_orientation" is the string "left", let the attribute "orientation"
# point to string "left". If "new_orientation" is the string "right", let the
# attribute "orientation" point to string "right". "show" should print a face in
# "right" or "left" orientation, based on the current value of "orientation".
# The default initialization is "left". Example:
# face = Face()
# face.look("left")
# face.show()  # Prints "<.<"
# face.look("right")
# face.show()  # Prints ">.>"
# Also implement a method "get_info()" that returns a string of the form:
# "Face is looking <orientation>", where <orientation> is the value of the
# attribute "orientation".

# Your code here #


#
# Task 2
#

# Create a class "OwlFace" that is derived from class "Face" from task 1.
# Override the method "look" as follows: If "new_orientation" is the string
# "left", let the attribute "orientation" point to string "left". If
# "new_orientation" is the string "right", let the attribute "orientation" point
# to string "right". If "new_orientation" is the string "ahead", let the
# attribute "orientation" point to string "ahead". Make sure to reuse code from
# the base class "Face". "show" should print a face in "right", "left" or
# "ahead" orientation, based on the current value of "orientation". Example:
# face = OwlFace()
# face.look("left")
# face.show()  # Prints "(O<O)"
# face.look("right")
# face.show()  # Prints "(O>O)"
# face.look("ahead")
# face.show()  # Prints "(OvO)"
# Also implement a method "get_info()" that returns a string of the form:
# "OwlFace is looking <orientation>", where <orientation> is the value of the
# attribute "orientation". Try to reuse as much code as possible from the base
# class "Face". Ideally, you do not need to override this method at all!

# Your code here #


#
# Task 3
#

# Create a class "Transformer" with two methods that operate on numerical lists:
#   > _transform(self, list_): A "private" method that takes a list as input,
#     processes/transforms this list and returns the processed/transformed list.
#     Since "Transformer" does not convey which transform operation should be
#     performed, the method should be considered "abstract", i.e., without any
#     actual implementation: Use "raise NotImplementedError" to indicate that
#     subclasses must overwrite this method. For more details, see
#     https://docs.python.org/3/library/exceptions.html#NotImplementedError
#   > transform(self, lists): This method takes a list of lists as input and
#     calls the "_transform" method for each list, collecting the returned lists
#     into a new list which is then returned.
# Afterwards, create these concrete transformer classes (concrete meaning that
# you have to implement the abstract method "_transform"):
#   > "AbsTransformer": Extends "Transformer". Returns the absolute values in
#     the list.
#   > "ScaleTransformer": Extends "Transformer". Has an "__init__" method with
#     two arguments "min_" and "max_" that will be used to scale the values in
#     the list to the interval [min_, max_].
#   > "NormalizeTransformer": Extends "ScaleTransformer" by setting "min_" and
#     "max_" to 0 and 1, respectively (i.e., a "NormalizeTransformer" is simply
#     a "ScaleTransformer" with 0 and 1 as fixed values).
# Finally, create a transformer object of each of the three classes and apply
# their transformation on the given input "example_lists:
#
# example_lists = [[0, 2, 7], [-1, -2, 10, 13, -1, 5]]
# example_lists -> AbsTransformer -> ScaleTransformer (with min_=1 anx max_=10)
# -> NormalizeTransformer
#
# The final result after the normalization transformation should be the
# following list (minor precision differences due to floating point numbers
# might occur):
# [[0.0, 0.2857142857142857, 1.0],
#  [0.0, 0.09999999999999999, 0.8999999999999999, 1.0, 0.0, 0.39999999999999997]]
example_lists = [[0, 2, 7], [-1, -2, 10, 11, -1, 5]]

# Your code here #
