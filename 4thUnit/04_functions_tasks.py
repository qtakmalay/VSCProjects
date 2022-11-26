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

# Write a function that divides two numbers and returns the result. The numbers
# should be taken as keyword arguments with defaults of 1.0 for both values.

def division_nums(a : float = 1.0, b : float = 1.0) -> float:
    return a/b

print(division_nums(3.23, 4.0))
#
# Task 2
#

# Write a function that takes one argument that can be assumed to be a list. The
# function should add up the last two elements in the list, print the sum and
# append it at the end of the list without any return value (in-place update).
a = [1, 2, 3, 5]

def l2_sum(in_list : list):
    in_list.append(in_list[-1] + in_list[-2]) 

l2_sum(a)
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
def list_mixed(list_1 : list, list_2 : list):
    comb_tpl:tuple
    for le1, le2 in zip(list_1, list_2):
        comb_tpl.append(le1 + le2) 
    print(comb_tpl)

list_mixed(a1, a2)
# Your code here #


#
# Task 4
#

# Write a function that takes an iterable of integers as input and returns their
# sum. If the iterable is empty, 0 should be returned. Do not use the existing
# function "sum".

# Your code here #


#
# Task 5
#

# Write a function that takes an iterable of any type as input and returns the
# length. If the iterable is empty, 0 should be returned. Do not use the
# existing function "len".

# Your code here #


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
