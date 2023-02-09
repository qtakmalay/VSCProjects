# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 02.08.2022

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

################################################################################

In this file, we will learn how to process script/program arguments (passed by
the user invoking your Python program).
"""

################################################################################
# sys - accessing system parameters
################################################################################

# The "sys" module allows you to access Python system parameters and functions
# related to them. For example, it allows you to access the command line
# arguments passed to your Python script. For more details, see
# https://docs.python.org/3/library/sys.html

import sys

# Here, we will get our command line arguments:
command_line_args = sys.argv
print(f"I received these arguments: {command_line_args}")
print(f"With types: {[type(a) for a in command_line_args]}")

# Note that first element in "sys.argv" is always the program name itself. The
# arguments are always read in as strings, if you want numbers or other types,
# you will have to convert them.


################################################################################
# argparse - getting command line arguments easier
################################################################################

# The "argparse" module lets you easily and safely access the command line
# arguments that were passed to your Python program. This module is recommended
# over using the "sys" module for getting the command line arguments, since it
# is much more flexible, powerful and convenient. Only parts are shown here, for
# more details, see:
# https://docs.python.org/3/howto/argparse.html#argparse-tutorial
# https://docs.python.org/3/library/argparse.html

import argparse

# Create a parser object. With this object, you can now handle all argument
# parsing in a convenient way.
parser = argparse.ArgumentParser()

# Specify arguments you want to receive. The help text will be displayed when
# your program is called with "-h" (e.g., python my_program.py -h). The argument
# "type" specifies the data type you want to accept (will be checked and
# converted automatically). You can also specify default values and program
# arguments with multiple values.

# Positional arguments are passed based on their position, they cannot be
# specified by their internal name. Positional arguments are always required.
# Positional argument (string)
parser.add_argument("filename", type=str, help="some filename")
# Positional argument (float)
parser.add_argument("number", type=float, help="some float number")

# Keyword arguments are passed based on their name, they must be specified by
# their internal name. Keyword arguments typically start with "-" (the short
# identifier) or "--" (the long/full identifier) and are optional per default.
# Keyword argument (int), optional, default value will be None
parser.add_argument("--int_number1", type=int, help="optional int")
# Keyword argument (int), optional, default value specified
parser.add_argument("--int_number2", type=int, default=1, help="optional int with default")
# Keyword argument (int), required, short version identifier synonym included
# (can now specify values with both "-n3 XYZ" and "--int_number3 XYZ")
parser.add_argument("-n3", "--int_number3", type=int, required=True, help="required int")

# Multiple arguments as list (exact amount, each of type int), specify with
# "--numbers 1 2 3"
parser.add_argument("--numbers", nargs=3, type=int)
# Multiple arguments as list (>= 0, each of default type string), specify with
# "--items 4 hi". With >= 1 requirement: use nargs="+" instead of nargs="*"
parser.add_argument("--items", nargs="*")

# Boolean argument/flag/toggle with action="store_true" (with False as default)
# or action="store_false" (with True as default). It then suffices to simply
# specify (or not specify) "--my_flag" without any additional arguments
parser.add_argument("--my_flag", action="store_true")

# There is also support for groups, e.g., mutual exclusion, where only one of
# several arguments can be specified. The created group has the same
# functionality, i.e., arguments (and other subgroups) can be added. Example
# with three mutually exclusive options (the group itself is optional (default),
# so either all are None, or all except a single one are None):
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument("--option_a", type=int)
group.add_argument("--option_b", type=float)
group.add_argument("--option_c", type=str)

# Parse the arguments
args = parser.parse_args()

# After the above call, you can now use "args.argument_name" to access your
# arguments, e.g.:
my_filename = args.filename
my_floatnumber = args.number
# "args.n3" is invalid (only the long identifier is valid if both are specified)
my_int_number3 = args.int_number3
