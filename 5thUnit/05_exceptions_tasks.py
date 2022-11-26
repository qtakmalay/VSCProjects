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

Tasks for self-study. Try to solve these tasks on your own and compare your
solutions to the provided solutions file.
"""

#
# Task 1
#

# Write a function that divides two numbers and returns the result. The numbers
# should be taken as keyword arguments with defaults of 1.0 for both values. If
# the division fails due to a "ValueError" or "TypeError", print a warning and
# return None instead.

def my_divide(numerator=1.0, denominator=1.0):
    try:
        return numerator/denominator
    except (ValueError, TypeError) as ex:
        print(f"There's a {ex}")
        return None

my_divide(1,"1")

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
        return sum(a,b)
    except (TypeError) as ex:
        try:
            print(ex)
            return sum(int(a), int(b))
        except (ValueError) as ex:
            print(ex)
            try:
                return sum(float(a), float(b))
            except ValueError:
                raise ex

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
        except ZeroDivisionError as ex:
            results.append("ZDE")
        except IndexError as ex1:
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


f(-1)  # Output:
f(0)  # Output:
f(1)  # Output:
f(2)  # Output:
f(3)  # Output:
f(4)  # Output:
