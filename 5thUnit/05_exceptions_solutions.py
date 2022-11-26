# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Van Quoc Phuong Huynh, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 28.07.2022

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

#
# Task 1
#

# Write a function that divides two numbers and returns the result. The numbers
# should be taken as keyword arguments with defaults of 1.0 for both values. If
# the division fails due to a "ValueError" or "TypeError", print a warning and
# return None instead.

# Your code here #


def my_divide(numerator=1.0, denominator=1.0):
    try:
        return numerator / denominator
    except (ValueError, TypeError) as ex:
        print(f"Warning: {ex}")
        return None


print(my_divide(5, 3))
print(my_divide(5, "string"))


#
# Task 2
#

# Write a function "add(a, b)" that returns the sum of "a" and "b". If the
# addition fails due to a "TypeError", convert the values to integers and return
# the sum of the two integers. If this fails due to a "ValueError", convert the
# values to float and return the sum of the two float values. If this fails due
# to a "ValueError", raise the first exception from the first attempt at adding
# the values.

# Your code here #


def add(a, b):
    try:
        c = a + b
    except TypeError as ex:
        print("a + b failed. Trying to convert to int ...")
        try:
            c = int(a) + int(b)
        except ValueError:
            print("a + b failed. Trying to convert to float ...")
            try:
                c = float(a) + float(b)
            except ValueError:
                print("Could not convert values to int or float ...")
                raise ex
    return c


print(add(5, 3))
print(add(5, "3"))
print(add(5., "3"))
print(add(5, "3."))
print(add(5, "3.x"))


#
# Task 3
#

# Run the below code. You will see two types of exception errors:
# "ZeroDivisionError" and "IndexError". Change the code such that the function
# "list_division" returns [0.25, "ZDE", "IE", 0.5, "IE"], where "ZDE" is added
# if a "ZeroDivisionError" and "IE" if an "IndexError" was encountered.

# Your code here #
def list_division(num_list1, num_list2):
    results = []

    for num in num_list1:
        try:
            results.append(num / num_list2[num])
        except ZeroDivisionError:
            results.append("ZDE")
        except IndexError:
            results.append("IE")

    return results


list1 = [1, 3, 8, 5, 9]
list2 = [2, 4, 6, 0, 3, 10]
print(list_division(list1, list2))


#
# Task 4
#

# Write a function with three input parameters, where the first is a dict, the
# second is a key and the third a default value (with a default argument value
# of None). The user can request a key from the passed dictionary, and in case
# the dictionary does not contain such a key, the default value should be
# returned instead. Solve this task by catching a "KeyError".

# Your code here #
def get(d: dict, key, default=None):
    try:
        return d[key]
    except KeyError:
        return default
# Note: This functionality already exists for dictionaries: d.get(key, default)


#
# Task 5
#

# Inspect the following code and try to determine the output of the function "f"
# with the given arguments without actually running the code (the goal is to
# understand the program flow).

def f(x):
    try:
        g(x)
        print("f1")
    except ValueError:
        print("f2")
    else:
        print("f3")
    finally:
        print("f4")
    print("f5")


def g(x):
    try:
        if x <= 0:
            raise ValueError
        if x == 1:
            raise TypeError
        h(x)
        print("g1")
    except ValueError:
        print("g2")
        if x < 0:
            raise
    except TypeError:
        print("g3")
        h(x)
        print("g4")
    finally:
        print("g5")
    print("g6")


def h(x):
    try:
        if x == 2:
            raise ValueError
        if x == 4:
            raise TypeError
        print("h1")
    except ValueError:
        print("h2")
    print("h3")


f(-1)  # Output: g2 g5 f2 f4 f5
f(0)  # Output: g2 g5 g6 f1 f3 f4 f5
f(1)  # Output: g3 h1 h3 g4 g5 g6 f1 f3 f4 f5
f(2)  # Output: h2 h3 g1 g5 g6 f1 f3 f4 f5
f(3)  # Output: h1 h3 g1 g5 g6 f1 f3 f4 f5
f(4)  # Output: g3 g5 f4 TypeError
