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

Example solutions for tasks in the provided tasks file.
"""

#
# Task 1
#

# Create a list with values 5, "a", 7. Then create a variable that points to the
# same object as the second element in the list.

# Your code here #
a = [5, "a", 7]
x = a[1]


#
# Task 2
#

# Append the integer 5 to the end of the given list (use the "+" operator) and
# store the new list in a variable. Afterwards, change the first element in this
# new list to integer 0. Print both lists.
a = [1, 2, 3, 4]

# Your code here #
b = a + [5]
b[0] = 0
print(a)
print(b)


#
# Task 3
#

# Assign the given list to a new variable and append the integer 5 to the end of
# this list (use the in-place method "append"). Afterwards, change the first
# element in this list to integer 0. Print both lists.
a = [1, 2, 3, 4]

# Your code here #
b = a
b.append(5)
b[0] = 0
print(a)
print(b)


#
# Task 4
#

# Append the integer 5 to the end of the given tuple (use the "+" operator) and
# store the new tuple in a variable. Afterwards, try to change the first element
# in this tuple to integer 0. Why will it not work to overwrite the element in
# the tuple? What can you do to create a new tuple where the first element is 0?
a = (1, 2, 3, 4)

# Your code here #
b = a + (5,)
# b[0] = 0  # This will fail because tuples are immutable
# However, we can always stitch together a new tuple from other tuples:
b = (0,) + b[1:]


#
# Task 5
#

# Add an entry with key "c" and value 3 to the given dictionary.
a = dict(a=1, b=2)  # Equal to: a = {"a": 1, "b": 2}

# Your code here #
a["c"] = 3


#
# Task 6
#

# Create a list with numbers from 0 to 100, excluding 100 (use "range"). Then,
# extract every third element starting at index 50 until index 70 and store it
# in a new list. Use slicing to solve this task.

# Your code here #
a = list(range(100))
a2 = a[50:70:3]  # start=50, stop=70, step=3


#
# Task 7
#

# String "a" contains elements that are separated by either a comma "," or a
# semicolon ";" character. Reverse the order of elements and replace the "," and
# ";" characters by a colon ":".
a = "element1,element2;element3;element4;element5,element6"

# Your code here #
# One way of solving this is to split string "a" at characters "," and ";",
# which will give us a list with the elements we are interested in. Then, we can
# reverse the list and join the elements with a ":" character to a string. To
# split the string we need to find a common split character. We can replace ";"
# by ",", then all elements are separated by ",".
common_a = a.replace(";", ",")  # Now all elements are separated by ","
split_a = common_a.split(",")  # Get a list of all ","-separated elements
reversed_a = split_a[::-1]  # Now we reverse the order of elements in the list
a = ":".join(reversed_a)  # Join the list elements by ":" to new string


#
# Task 8
#

# Add two elements 7 and 9 to a nested list "nl" so that it becomes
# [1, 2, 3, [4, 5, [6, 7, 8, 9], 10, 11], 12, 13, 14]
# Afterwards, flatten it to become a one-dimensional list
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
nl = [1, 2, 3, [4, 5, [6, 8], 10, 11], 12, 13, 14]

# Your code here #
nl[3][2].append(9)
nl[3][2].insert(1, 7)
print(nl)

sub_list = nl[3]  # Get sub list at index 3
sub_sub_list = sub_list[2]  # Get sub list at index 2 of sub_list
sub_list[2:3] = sub_sub_list  # Replace element at index 2 of sub_list by elements of sub_sub_list
nl[3:4] = sub_list  # Replace element at index 3 of list a by elements of sub_list
print(nl)

# Alternative solution:
nl = [1, 2, 3, [4, 5, [6, 8], 10, 11], 12, 13, 14]
nl[3][2].append(9)
nl[3][2].insert(1, 7)
sub_list = nl[3]  # Get sub list at index 3
sub_sub_list = sub_list[2]  # Get sub list at index 2 of sub_list
nl = nl[:3] + sub_list[:2] + sub_sub_list + sub_list[3:] + nl[4:]
print(nl)


#
# Task 9
#

# Create a dictionary "d3" that combines the two given dictionaries "d1" and
# "d2". Then, replace the entry {6: "five"} by {5: "five"}. Desired result:
# {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
d1 = {1: "one", 2: "two", 3: "three"}
d2 = {4: "four", 6: "five"}

# Your code here #
d3 = d1.copy()
d3.update(d2)  # All items of "d2" are added to "d3"
d3[5] = d3.pop(6)  # Remove item of key 6 from "d3", the value "five" is returned and used as the value of new key 5
# We could also have done this instead:
# d3[5] = d3[6]
# del d3[6]
print(d3)


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
sale_dept = nd["sale department"]
sup_id = sale_dept["e2"]["supervisor"]
print(sale_dept[sup_id]["name"])


#
# Task 11
#

# Create a set with the elements 1, 2, 3, 5. Create a second set with the
# elements 2, 4, 6. Now create three new sets with (1) elements that are only
# contained in the first set but not the second, (2) elements that are only
# contained in both sets, (3) elements that are either contained in the first
# set or the second set but not both.

# Your code here #
set1 = {1, 2, 3, 5}
set2 = {2, 4, 6}
only_set1 = set1 - set2
only_both_sets = set1 & set2
either_but_not_both_v1 = only_set1 | (set2 - set1)
either_but_not_both_v2 = (set1 | set2) - only_both_sets
either_but_not_both_v3 = set1 ^ set2


#
# Task 12
#

# Loop through list "fnames" and count how many elements in the list end with
# ".png". You can use some_str.endswith("substring") to check if a string ends
# with a certain substring.
fnames = ["file0.txt", "file1.png", "file2.txt", "file3.txt",
          "file4.png", "file5.txt", "file6.txt", "file7.dat"]

# Your code here #
count = 0
for fname in fnames:
    if fname.endswith(".png"):
        count += 1


#
# Task 13
#

# Create a counter "count" and loop through list "some_list". For each element
# "not good" add -1, for each element "okay" add 0, for each element "good" add
# 1, and for each element "very good" in the list add 2 to the counter.
some_list = ["okay", "not good", "okay", "okay", "good", "good", "okay",
             "very good", "not good", "very good"]

# Your code here #
count = 0
for element in some_list:
    if element == "not good":
        count += -1
    elif element == "okay":
        count += 0
    elif element == "good":
        count += 1
    elif element == "very good":
        count += 2

# Alternative solution (ignoring "okay" since we would only add 0):
count = 0
for element in some_list:
    if element == "not good":
        count += -1
    elif element == "good":
        count += 1
    elif element == "very good":
        count += 2


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
new_list = []
for fname in fnames:
    if fname.endswith(".png"):
        new_list.append("image" + fname)
    elif fname.endswith(".txt"):
        new_list.append("text" + fname)
    else:
        new_list.append("data" + fname)


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
new_list = ["image" + fname if fname.endswith(".png") else
            "text" + fname if fname.endswith(".txt") else "data" + fname
            for fname in fnames]


#
# Task 16
#

# Merge two sorted lists, list1 and list2, to create a new sorted list (you must
# not use existing sorting functionality).
list1 = [1, 4, 7, 8, 12, 30]
list2 = [3, 5, 8, 9, 18, 20, 21, 25]

# Your code here #
len1 = len(list1)
len2 = len(list2)

i = 0
j = 0
merged_list = []

while i < len1 and j < len2:
    e1 = list1[i]
    e2 = list2[j]
    
    if e1 < e2:
        merged_list.append(e1)
        i += 1
    else:
        merged_list.append(e2)
        j += 1

if i == len1:
    merged_list.extend(list2[j:])  # Alternative: merged_list += list2[j:]
else:
    merged_list.extend(list1[i:])


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
points1 = [[record[0], record[1] + record[2] + record[3] + record[4] + record[5], record[1:]]
           for record in zip(students, exam1s, exam2s, assg1s, assg2s, assg3s)]

# Alternative solution:
points2 = [[record[0], sum(record[1:]), record[1:]]
           for record in zip(students, exam1s, exam2s, assg1s, assg2s, assg3s)]

# Alternative solution:
points3 = []
for i in range(len(students)):
    p_tuple = (exam1s[i], exam2s[i], assg1s[i], assg2s[i], assg3s[i])
    entry = [students[i], sum(p_tuple), p_tuple]
    points3.append(entry)
