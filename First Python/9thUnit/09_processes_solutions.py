# -*- coding: utf-8 -*-
"""

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

import subprocess
import sys
from multiprocessing import Pool

#
# Task 1
#

# Create some Python program that accepts one integer command line argument and
# prints this value to the console. Store the script to a Python file. Call this
# program using the "subprocess" module with arguments ranging from 0 to 10
# (inclusive) and print its output (no decoding necessary).

# Your code here #
# Creating the file programmatically is not necessary, it is just done here for
# the sake of convenience and that this script is self-contained
with open("my_prog.py", "w") as f:
    f.write("""import argparse

parser = argparse.ArgumentParser()
parser.add_argument("my_arg", type=int)
args = parser.parse_args()
print(args.my_arg)
""")

for i in range(11):
    result = subprocess.run([sys.executable, "test.py", str(i)], capture_output=True)
    print(result.stdout)


#
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
n_worker_processes = 8
values = list(range(501))


def f(x):
    subprocess.run([sys.executable, "my_prog.py", str(x)])
    return x


if __name__ == "__main__":
    with Pool(n_worker_processes) as p:
        results = p.map(f, values)
    print(results)


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
from os import getpid   # To get the current process ID (not required, but useful)


def row_sum(row):
    sum_ = 0
    for element in row:
        sum_ += element
    # With printing the current process ID, we get to see which process is
    # actually computing the sum. While our process pool has multiple worker
    # processes, it is not guaranteed that every process will actually be used.
    # Execute the program multiple times to see different process assignments!
    print(f"We are in process ID '{getpid()}', which computed sum {sum_}")
    return sum_


if __name__ == "__main__":
    print(f"Process ID: {getpid()} starts")
    with Pool(2) as p:
        results = p.map(row_sum, data)
    print(f"Process ID: {getpid()} continues")
    print(f"{results}")
