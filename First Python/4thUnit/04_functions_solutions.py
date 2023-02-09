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

Example solutions for tasks in the provided tasks file.
"""

#
# Task 1
#

# Write a function that divides two numbers and returns the result. The numbers
# should be taken as keyword arguments with defaults of 1.0 for both values.

# Your code here #


def my_divide(numerator=1.0, divisor=1.0):
    return numerator / divisor


print(my_divide(5, 3))


#
# Task 2
#

# Write a function that takes one argument that can be assumed to be a list. The
# function should add up the last two elements in the list, print the sum and 
# append it at the end of the list without any return value (in-place update).
a = [1, 2, 3, 5]

# Your code here #


def my_funct(lst):
    s = lst[-2] + lst[-1]
    print(s)
    lst.append(s)
    
    
my_funct(a)
print(a)


#
# Task 3
#

# Write a function that takes two lists as input and creates a list of 2-tuples,
# where always the i-th elements of both lists are combined. If one list is
# longer than the other, the remaining elements are ignored. Do not use the
# existing function "zip".
a1 = [1, 2, 3]
a2 = [4, 5]

# Your code here #


def zip_lists(list1, list2):
    zipped = []
    size = min(len(list1), len(list2))
    for i in range(size):
        zipped.append((list1[i], list2[i]))
    return zipped


print(zip_lists(a1, a2))


#
# Task 4
#

# Write a function that takes an iterable of integers as input and returns their
# sum. If the iterable is empty, 0 should be returned. Do not use the existing
# function "sum".

# Your code here #


def my_sum(integers):
    s = 0
    for i in integers:
        s += i
    return s


#
# Task 5
#

# Write a function that takes an iterable of any type as input and returns the
# length. If the iterable is empty, 0 should be returned. Do not use the
# existing function "len".

# Your code here #


def my_len(iterable):
    length = 0
    for _ in iterable:
        length += 1
    return length


#
# Task 6
#

# Write a function that takes a list of integers as input, replaces all negative
# values with their positive/absolute values and returns this list. The function
# also has a boolean parameter which specifies if the replacement of the
# negative numbers should be done in-place, i.e., the changes are directly made
# on the input list, or if the replacement should be done in a new list, i.e.,
# the input list remains unchanged. By default, a new list should be returned.
# Otherwise, i.e., using in-place changes, nothing should be returned (None).
# Furthermore, the function should have a second parameter that specifies a set
# of negative values which should NOT be changed to positive values but instead
# should be excluded from the input list. By default, the negative numbers -1,
# -2 and -3 should be ignored.
a = [1, 2, -1, -5, 3, -4, -2, 0]

# Your code here #


def filter_numbers(numbers: list, inplace: bool = False, ignored_values: set = None):
    if ignored_values is None:
        ignored_values = {-1, -2, -3}
    if inplace:
        i = 0
        while i < len(numbers):
            if numbers[i] in ignored_values:
                numbers.pop(i)
            else:
                numbers[i] = abs(numbers[i])
                i += 1
    else:
        return [abs(x) for x in numbers if x not in ignored_values]


# alternative solution
def filter_numbers2(numbers, inplace=False, ignored_values=None):
    if ignored_values is None:
        ignored_values = {-1, -2, -3}
    new_list = [abs(x) for x in numbers if x not in ignored_values]
    if inplace:
        numbers[:] = new_list
    else:
        return new_list


print(filter_numbers(a))
print(filter_numbers(a, ignored_values=set()))
print(a)
filter_numbers(a, inplace=True)
print(a)
