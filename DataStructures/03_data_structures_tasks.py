# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Van Quoc Phuong Huynh, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 26.07.2022

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

# Create a list with values 5, "a", 7. Then create a variable that points to the
# same object as the second element in the list.

# Your code here #


#
# Task 2
#

# Append the integer 5 to the end of the given list (use the "+" operator) and
# store the new list in a variable. Afterwards, change the first element in this
# new list to integer 0. Print both lists.
a = [1, 2, 3, 4]

# Your code here #


#
# Task 3
#

# Assign the given list to a new variable and append the integer 5 to the end of
# this list (use the in-place method "append"). Afterwards, change the first
# element in this list to integer 0. Print both lists.
a = [1, 2, 3, 4]

# Your code here #


#
# Task 4
#

# Append the integer 5 to the end of the given tuple (use the "+" operator) and
# store the new tuple in a variable. Afterwards, try to change the first element
# in this tuple to integer 0. Why will it not work to overwrite the element in
# the tuple? What can you do to create a new tuple where the first element is 0?
a = (1, 2, 3, 4)

# Your code here #


#
# Task 5
#

# Add an entry with key "c" and value 3 to the given dictionary.
a = dict(a=1, b=2)  # Equal to: a = {"a": 1, "b": 2}

# Your code here #


#
# Task 6
#

# Create a list with numbers from 0 to 100, excluding 100 (use "range"). Then,
# extract every third element starting at index 50 until index 70 and store it
# in a new list. Use slicing to solve this task.

# Your code here #


#
# Task 7
#

# String "a" contains elements that are separated by either a comma "," or a
# semicolon ";" character. Reverse the order of elements and replace the "," and
# ";" characters by a colon ":".
a = "element1,element2;element3;element4;element5,element6"

# Your code here #


#
# Task 8
#

# Add two elements 7 and 9 to a nested list "nl" so that it becomes
# [1, 2, 3, [4, 5, [6, 7, 8, 9], 10, 11], 12, 13, 14]
# Afterwards, flatten it to become a one-dimensional list
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
nl = [1, 2, 3, [4, 5, [6, 8], 10, 11], 12, 13, 14]

# Your code here #


#
# Task 9
#

# Create a dictionary "d3" that combines the two given dictionaries "d1" and
# "d2". Then, replace the entry {6: "five"} by {5: "five"}. Desired result:
# {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
d1 = {1: "one", 2: "two", 3: "three"}
d2 = {4: "four", 6: "five"}

# Your code here #


#
# Task 10
#

# You are given a group of employees as nested dictionary "nd". Given this
# dictionary, get the supervisor's name of the employee with ID "e2".
nd = {
    "sale department": {
        "e1": {
            "name": "Alice",
            "role": "Sale Manager",
            "salary": 5000
        },
        "e2": {
            "name": "Bob",
            "role": "Sale Employee",
            "salary": 4000,
            "supervisor": "e1"
        }
    }
}

# Your code here #


#
# Task 11
#

# Create a set with the elements 1, 2, 3, 5. Create a second set with the
# elements 2, 4, 6. Now create three new sets with (1) elements that are only
# contained in the first set but not the second, (2) elements that are only
# contained in both sets, (3) elements that are either contained in the first
# set or the second set but not both.

# Your code here #


#
# Task 12
#

# Loop through list "fnames" and count how many elements in the list end with
# ".png". You can use some_str.endswith("substring") to check if a string ends
# with a certain substring.
fnames = ["file0.txt", "file1.png", "file2.txt", "file3.txt",
          "file4.png", "file5.txt", "file6.txt", "file7.dat"]

# Your code here #


#
# Task 13
#

# Create a counter "count" and loop through list "some_list". For each element
# "not good" add -1, for each element "okay" add 0, for each element "good" add
# 1, and for each element "very good" in the list add 2 to the counter.
some_list = ["okay", "not good", "okay", "okay", "good", "good", "okay",
             "very good", "not good", "very good"]

# Your code here #


#
# Task 14
#

# Create a new list from list "fnames" but add "image" to the beginning of the
# element if the string ends with ".png" ("file1.png" -> "imagefile1.png"),
# "text" if the string ends with ".txt", and "data" otherwise. You can use
# some_str.endswith("substring") to check if a string ends with a certain
# substring. Use standard list processing to solve this task.
fnames = ["file0.txt", "file1.png", "file2.txt", "file3.txt",
          "file4.png", "file5.txt", "file6.txt", "file7.dat"]

# Your code here #


#
# Task 15
#

# Create a new list from list "fnames" but add "image" to the beginning of the
# element if the string ends with ".png" ("file1.png" -> "imagefile1.png"),
# "text" if the string ends with ".txt", and "data" otherwise. You can use
# some_str.endswith("substring") to check if a string ends with a certain
# substring. Use a list comprehension to solve this task.
fnames = ["file0.txt", "file1.png", "file2.txt", "file3.txt",
          "file4.png", "file5.txt", "file6.txt", "file7.dat"]

# Your code here #


#
# Task 16
#

# Merge two sorted lists, list1 and list2, to create a new sorted list (you must
# not use existing sorting functionality).
list1 = [1, 4, 7, 8, 12, 30]
list2 = [3, 5, 8, 9, 18, 20, 21, 25]

# Your code here #


#
# Task 17
#

# Given lists of points of students, make a summarized points list of students
# following the scheme: [[<student name>, <points sum>, <tuple of all individual points>], ...]
exam1s = [6, 10, 9, 8]
exam2s = [10, 7, 8, 8]
assg1s = [20, 15, 18, 17]
assg2s = [25, 20, 30, 28]
assg3s = [26, 25, 30, 25]
students = ["Alice", "Malice", "Sam", "May"]

# Expected result:
# [["Alice", 87, (6, 10, 20, 25, 26)],
#  ["Malice", 77, (10, 7, 15, 20, 25)],
#  ["Sam", 95, (9, 8, 18, 30, 30)],
#  ["May", 86, (8, 8, 17, 28, 25)]]

# Your code here #
