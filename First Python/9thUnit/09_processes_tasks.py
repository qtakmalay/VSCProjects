# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Van Quoc Phuong Huynh, Andreas Schörgenhumer
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

Tasks for self-study. Try to solve these tasks on your own and compare your
solutions to the provided solutions file.
"""

import subprocess
import sys
from multiprocessing import Pool
import numpy as np

#
# Task 1
#

# Create some Python program that accepts one integer command line argument and
# prints this value to the console. Store the script to a Python file. Call this
# program using the "subprocess" module with arguments ranging from 0 to 10
# (inclusive) and print its output (no decoding necessary).

with open("C:\\Users\\azatv\\VSCProjects\\9thUnit\\test.py", "w") as f:
    f.write("""import argparse

a = argparse.ArgumentParser()
a.add_argument("int_number", type=int, help="some int num")
args = a.parse_args()

my_int_number3 = args.int_number
print(args.int_number)""")
# for i in np.arange(10):
#     result = subprocess.Popen(["python", "c:\\Users\\azatv\\VSCProjects\\9thUnit\\test.py", "--int_number", i], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     outs, errs = result.communicate()
#     #print(result.returncode)
#     print(f"outs: {outs.decode('oem')}")
#     print(f"errs: {errs.decode('oem')}")

for i in range(11):
    result = subprocess.run([sys.executable, "C:\\Users\\azatv\\VSCProjects\\9thUnit\\test.py", str(i)], capture_output=True)
    print(result.stdout)
    print(result.stderr)
# Task 2
#

# Create a function "f" which takes a number as an argument, calls your above
# Python program with this argument (as string) via "subprocess.run" and finally
# returns the passed argument again. Use 8 parallel worker processes for
# invoking this function with inputs from 0 to 500 (inclusive) using the
# "Pool.map" method from the "multiprocessing" module and collect the results in
# a list. Look at the output: Since it is asynchronous, the values might be
# printed in a different order, but the returned list of results has the same
# order as the list of arguments.

# Your code here #


#
# Task 3
#

# You are given the following data as nested list "data":
data = [list(range(i)) for i in [10, 100, 1000, 2000, 3000, 4000, 5000, 6000]]
# Your task is to compute the sum of each row in parallel, by using
# "multiprocessing.Pool" to spawn a set of worker processes. Afterwards, your
# code should print the result. For the example above, the output should be:
# [45, 4950, 499500, 1999000, 4498500, 7998000, 12497500, 17997000]
# Hint: You will need a function that computes the sum of one row, i.e., the sum
# over one list. You can write your own function or use a built-in function.

# Your code here #
