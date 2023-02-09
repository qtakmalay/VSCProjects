# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 03.08.2022

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

import argparse

#
# Task 1
#

# Create a Python script that expects an integer as positional command line
# argument. It should then print this integer.

# Your code here #
parser = argparse.ArgumentParser()
parser.add_argument("my_arg", type=int)
args = parser.parse_args()
print(args.my_arg)
