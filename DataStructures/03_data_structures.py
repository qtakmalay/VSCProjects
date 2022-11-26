# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 25.07.2022

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

################################################################################

In this file, we will learn about various kinds of data structures like lists,
tuples, sets and dictionaries, as well as how to interact with and use them.
https://docs.python.org/3/tutorial/datastructures.html
"""

################################################################################
# Lists
################################################################################

# A list is a mutable sequence type containing an ordered list of
# values/objects.
# https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range
# https://docs.python.org/3/library/stdtypes.html#list
# https://docs.python.org/3/glossary.html#term-sequence

# You can create a list with square brackets []. Empty lists can be created
# either with empty square brackets or with the built-in magic "list".
empty_list = []
empty_list = list()

# Lists can contain arbitrary objects, even of different data types.
a_list = [1, 2.3456, 3, "abc"]

# The length of the list can be determined with "len".
length = len(a_list)

# Each list element is like a variable. You can access it via an integer index
# via "a_list[i]", where (just like string indexing) the index "i" must be in
# the range [0, len(a_list)).
first_element = a_list[0]  # Here, we get the first element from "a_list"

# Only integers (and slices, which we will learn about later) are allowed as
# list indices:
# not_possible = a_list[0.0]  # This will not work

# Lists can store all kinds of objects, even other lists. If a list contains
# another list as element, it is referred to as a "nested" list. Here, we create
# a list, containing a list, containing yet another list:
nested_list = ["eg", 3, [23, 5, 65, ["aeg", 35]]]
# Here, we return the element at index 2, which is another list:
inner_list = nested_list[2]
# We can index this list "inner_list" again to get an element:
inner_list_element_zero = inner_list[0]
# We can navigate through the nested lists using chained indices like this:
nested_list_element = nested_list[2][3][0]

# We can modify the list content by assigning an object to the indexed list:
a_list = [1, 2.3456, 3, "abc"]
a_list[0] = 5  # Change the first element to 5
a_list[0] = a_list[0] * 2  # Change first element to first element times 2
# We can also use the abbreviated notations for "in-place" operations:
a_list[0] *= 3  # Change first element to first element times 3

# You may also use negative indices to start from the end of the list (using -1
# for the last element, -2 for the 2nd last, etc.)
last_element = a_list[-1]
second_last_element = a_list[-2]

# List elements can be deleted by index using "del"
old_first_element = a_list[0]
del a_list[0]  # Remove first element from list
new_first_element = a_list[0]

# You can append and remove elements from a list, and you can use many
# operations you already know from strings:
a_list.append("new_element")  # Append object to the end of list
a_list.remove("abc")  # Remove first occurrence of object "abc" from list
a_list *= 3  # Repeat list 3 times
detect = "abc" in a_list  # Check if object "abc" is contained in list
a_list += [-1, -2]  # Extend with another list (elements are appended)

# Sorting can be done in-place (will not create a new list) with method "sort"
numeric_list = [1, 5, 3, -5]
numeric_list.sort()  # Directly changes "numeric_list" (in-place)

# Or via the "sorted" function to get a sorted copy of the list
numeric_list = [1, 5, 3, -5]
sorted_list = sorted(numeric_list)  # "numeric_list" is not changed

# Iterable objects, e.g., strings, can be converted to lists using "list":
string_as_list = list("hello")  # Create a list ["h", "e", "l", "l", "o"]

# Important: Unlike strings, lists are mutable objects. Modifying a string means
# creating and allocating a completely new string. In contrast, modifications to
# a list will only change the modified elements. Changing, appending or removing
# elements will not create an entirely new list but reuse the old one.
long_string = "abc" * 10
long_string = long_string.replace("a", "")  # This will create a new string
long_list = ["a", "b", "c"] * 10
long_list.remove("a")  # This will not create a new list

# The mutability is crucial to understand when multiple variables reference the
# same list object (or any mutable object for that matter).
x = [1, 2]
y = x  # Let "y" point to the same list as "x", i.e., there is only one list now
assert x is y  # Both refer to the same object in memory
x.append(3)  # Change the list "x" is referencing via an in-place operation
print(x)  # Prints "[1, 2, 3]"
print(y)  # Also prints "[1, 2, 3]", since "y" simply points to the same list!
# This is true for all in-place operations. Other operations, i.e., those that
# do not change the object directly, do not have this effect:
x = x + [4]  # "+" concatenates the two lists and creates a new list object!
# Since we overwrote "x" with this new list object, "x" now references a
# different object than "y":
assert x is not y
print(x)  # Prints "[1, 2, 3, 4]" (second, new list object)
print(y)  # Still prints "[1, 2, 3]" (first, old list object)


################################################################################
# Tuples
################################################################################

# A tuple is an immutable sequence type containing an ordered list of
# values/objects. Its content cannot be changed after creation. They are similar
# to strings (think of the individual string characters as tuple elements).
# https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range
# https://docs.python.org/3/library/stdtypes.html#tuple

# You can create a tuple via optional parentheses (). Empty tuples can be
# created either with empty parentheses or with the built-in magic "tuple".
empty_tuple = ()
empty_tuple = tuple()
single_element_tuple = (1,)  # Special case: Must include comma to make it a tuple
a_tuple = 1, "2", 3  # Possible tuple creation
a_tuple = (1, "2", 3)  # This is more readable and better

# Available operations are similar to strings and lists
joined_tuples = a_tuple + a_tuple  # Concatenate 2 tuples
multiple_tuples = a_tuple * 5  # Repeat tuple 5 times

# As with strings and lists, you can use an index to extract an element
first_tuple_element = a_tuple[0]

# Due to their immutability (similar to strings), assignment is not possible:
# a_tuple[0] = 2

# Tuples can be converted to lists and vice versa:
now_a_tuple = tuple(a_list)
now_a_list = list(a_tuple)


################################################################################
# Ranges
################################################################################

# Just as additional information: We already used the range object. This is
# actually another (immutable) sequence type. For more details, see:
# https://docs.python.org/3/library/stdtypes.html#ranges

# Sometimes, it might be useful to get a list of numbers instead of a range
# object. You can create such a list by passing a range object to "list":
numbers = list(range(10))  # Creates the list [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


################################################################################
# Sets
################################################################################

# A set is a mutable, unordered collection of unique values/objects. They are
# useful if you want to perform any kinds of common set operations such as
# union, intersection, difference, etc.
# https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset
# https://docs.python.org/3/library/stdtypes.html#set

# You can create a set via curly braces {}. Empty sets must be created either
# with the built-in magic "set" (empty curly braces would create an empty
# dictionary, see further below).
empty_set = set()
# Create a set with the specified content, duplicate elements are dropped
# automatically (here, the second 3):
my_set = {1, 2, 3, 3}

# Note: A set can only contain certain objects, namely those that are so-called
# "hashable" (https://docs.python.org/3/glossary.html#term-hashable). This is an
# advanced topic, which we presumably will not cover in this course. So for now,
# you can assume that most Python built-in immutable objects (e.g., strings,
# integers) are hashable, while mutable objects (e.g., lists) are not hashable.

# Add an element to a set
my_set.add(4)
# Again, if the set already contains the element, nothing changes
my_set.add(4)

# Remove an element from a set (if it contains this element)
my_set.remove(2)

# Check if the set contains a specified element
is_element_there = 3 in my_set

# Common set operations (for a full list, see
# https://docs.python.org/3/library/stdtypes.html#set)
# Union
new_set = my_set | {1, 4, 7}
new_set = my_set.union({1, 4, 7})  # Alternative, allows other input than sets
# Intersection
new_set = my_set & {1, 4, 7}
new_set = my_set.intersection({1, 4, 7})  # Alternative, allows other input than sets
# Difference
new_set = my_set - {1, 4, 7}
new_set = my_set.difference({1, 4, 7})  # Alternative, allows other input than sets

# Sets can be converted to lists. Since sets are unordered, there is no
# guarantee on the resulting list ordering. Lists can also be converted to sets,
# however, duplicate elements will be dropped and the list ordering is also
# discarded, so information is potentially lost in this step.
set_to_list = list(my_set)
list_to_set = set(a_list)


################################################################################
# Dictionaries
################################################################################

# A dictionary is a mutable, ordered mapping type of key-value pairs, where the
# keys are unique (similar to how the values must be unique within a set). The
# guarantee that the insertion order is preserved was added in Python 3.7.
# https://docs.python.org/3/library/stdtypes.html#mapping-types-dict
# https://docs.python.org/3/library/stdtypes.html#dict

# You can create a dictionary via the built-in magic "dict" by writing
# dict(key=value, key=value, ...) or via {key: value, key: value}. Empty
# dictionaries can be created either with empty curly braces or with "dict".
# Similar to sets, only hashable objects can be used as keys (see note on sets
# above).
empty_dict = {}
empty_dict = dict()

# There is a slight difference in the dict creation: Using "dict", the keys are
# automatically converted to string objects, while creation with the curly
# braces uses the key objects directly, thus allowing more control.
some_key = "abc"
dictionary = dict(some_key=3.24, other_key="twh")
dictionary2 = {"stringkey": 55, some_key: "someitem", 23: 4}

# Dictionaries can be indexed. This is done via the key objects
element = dictionary["some_key"]
element2 = dictionary2[some_key] * dictionary2[23]

# You can also add key-value pairs via indexing or via method "update"
dictionary["new_key"] = "hello"  # Add new key-value pair: "new_key" -> "hello"
dictionary["new_key"] = 123  # Overwrite already associated value "hello"
dictionary.update({"u1": 0, "u2": 12.7})  # Update with another dictionary

# Get all keys as list (order of keys follows the insertion order)
keys = list(dictionary.keys())

# Get all values as list (order of values follows the insertion order)
values = list(dictionary.values())

# Get key-value pairs as tuples (order of pairs follows the insertion order)
pair_tuples = list(dictionary.items())

# Check for key
is_key_there = "some_key" in dictionary

# Remove key and return its associated value
removed = dictionary2.pop("stringkey")

# Get a value and use a default value if key does not exist
value = dictionary.get("nonexistingkey", "defaultvalue")

# Example of a more complex key: tuple of integers (this works because both the
# tuple and the integers are immutable, and (most of) Python's built-in
# immutable objects are hashable)
other_dict = dict()
other_dict[(1, 2, 3)] = "test"
other_dict_element = other_dict[(1, 2, 3)]


################################################################################
# Iteration
################################################################################

# All of the above data structures are so-called iterables and can thus be used
# in the already known for loop. More details on iterables:
# https://docs.python.org/3/glossary.html#term-iterable

# List iteration (iteration order of elements is guaranteed):
for elem in a_list:
    print(elem)

# Tuple iteration (iteration order of elements is guaranteed):
for elem in a_tuple:
    print(elem)

# Set iteration (iteration order of elements is not guaranteed):
for elem in my_set:
    print(elem)

# Dictionary iteration for keys (iteration order is guaranteed):
for key in dictionary:  # Equal to: dictionary.keys()
    print(key)

# Dictionary iteration for values (iteration order is guaranteed):
for value in dictionary.values():
    print(value)

# Dictionary iteration for key-value pairs (iteration order is guaranteed):
for key_value in dictionary.items():
    print(key_value)

# Important: Changing the content of the iterable while you iterate over it
# is tricky and might lead to unintended results (concurrent modification). Use
# a copy of the iterable instead if you want to modify it during the loop.

# Concurrent modification:
some_iterable = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for elem in some_iterable:
    print(elem)
    del some_iterable[0]

# Solution with copy (if you write "list(x)", a new list will be created with
# the same elements as the iterable "x"):
some_iterable = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
copy = list(some_iterable)  # Create new list with content of "some_iterable"
for elem in copy:
    print(elem)
    del some_iterable[0]


################################################################################
# Identity, equality, order comparisons
################################################################################

# In the following, we will see how the different data structures behave when we
# apply the identity check with "is", the equality operator "==" and order
# comparison operators (e.g., "<", ">") on various objects.

#
# Identity
#

# None of the above data structures are identical (i.e., refer to the same
# object in memory), even if their content is equal. Example with list objects
# (the same is true for the other data structures):
list1 = [0, 1, 2]
list2 = [0, 1, 2]
print(f"list1={list1} is list2={list2}: {list1 is list2}")

#
# Equality
#

# Two sequence data objects are considered equal if they are of the same type,
# have the same length and all elements are equal for each index position.
list3 = [4, 5, 6]
list4 = [0, 1, 2, 3]
tup1 = (0, 1, 2)
tup2 = (0, 1, 2)
print(f"list1={list1} == list2={list2}: {list1 == list2}")
print(f"list1={list1} == list3={list3}: {list1 == list3}")
print(f"list1={list1} == list4={list4}: {list1 == list4}")
print(f"tup1={tup1} == list1={list1}: {tup1 == list1}")
print(f"tup1={tup1} == tup2={tup2}: {tup1 == tup2}")

# Checking the equality of sequences and sets/dicts always returns False.
set1 = {0, 1, 2}
dict1 = {0: 0, 1: 1, 2: 2}
print(f"set1={set1} == list1={list1}: {set1 == list1}")
print(f"dict1={dict1} == list1={list1}: {dict1 == list1}")

# Set equality checks consider only the unique elements, regardless of order
# (which makes sense since sets are unordered collections).
set2 = {0, 1, 2}
set3 = {2, 0, 1, 1, 1}
set4 = {0, 1, 2, 3}
print(f"set1={set1} == set2={set2}: {set1 == set2}")
print(f"set1={set1} == set3={set3}: {set1 == set3}")
print(f"set1={set1} == set4={set4}: {set1 == set4}")

# Dictionaries are considered equal if they have the same key-value pairs,
# regardless of order.
dict2 = {0: 0, 1: 1, 2: 2}
dict3 = {2: 2, 0: 0, 1: 1}
dict4 = {0: 0, 1: 1, 2: 2, 3: 3}
dict5 = {0: "a", 1: "b", 2: "c"}
print(f"dict1={dict1} == dict2={dict2}: {dict1 == dict2}")
print(f"dict1={dict1} == dict3={dict3}: {dict1 == dict3}")
print(f"dict1={dict1} == dict4={dict4}: {dict1 == dict4}")
print(f"dict1={dict1} == dict5={dict5}: {dict1 == dict5}")

#
# Order comparisons
#

# While dictionaries do not support comparisons other than "==" and "!=",
# sequences and sets do. The following examples are based on the "less than"
# operator "<", all other operators are analogous. Contrary to above, comparing
# different types will result in an error rather than False.

# For sequences, the comparison "s1 < s2" will perform the following: For each
# index position, the elements of "s1" and "s2" are compared. If two
# corresponding elements are equal, the next index position is checked until the
# two elements are unequal, i.e., "s1[i] != s2[i]". If "s1[i]" is less than
# "s2[i]", the comparison "s1 < s2" will return True, False otherwise (all
# remaining elements are skipped!). If all elements of "s1" are equal, then
# "s1 < s2" will return True if "s2" has more elements, i.e.,
# "len(s2) > len(s1)", False otherwise.
list5 = [1, 0, 0]
print(f"list1={list1} < list1={list1}: {list1 < list1}")
print(f"list1={list1} < list3={list3}: {list1 < list3}")
print(f"list3={list3} < list1={list1}: {list3 < list1}")
print(f"list1={list1} < list4={list4}: {list1 < list4}")
print(f"list4={list4} < list1={list1}: {list4 < list1}")
print(f"list1={list1} < list5={list5}: {list1 < list5}")

# For sets, the comparison "s1 < s2" checks if "s1" is a strict subset of "s2".
print(f"set1={set1} < set1={set1}: {set1 < set1}")
print(f"set1={set1} < set4={set4}: {set1 < set4}")
print(f"set4={set4} < set1={set1}: {set4 < set1}")


################################################################################
# List, set, dictionary comprehensions
################################################################################

# List comprehensions are a fast(er) and compact way to write loops and
# conditions applied to an iterable. The results are collected in a list.
old_list = [1, 2, 3, 4, 5]
new_list = [str(i) for i in old_list]
# This returns the same results as:
new_list = []
for i in old_list:
    new_list.append(str(i))

# You can use an if statement at the end of a list comprehension to execute the
# code-block in the loop conditionally. Note that there is no element appended
# to the list if the condition is not True:
new_list = [str(i) for i in old_list if i < 3]
# This returns the same results as:
new_list = []
for i in old_list:
    if i < 3:
        new_list.append(str(i))

# You can also use if-else expressions in the loop-code block:
new_list = [str(i) if i < 5 else "end" for i in old_list]
# This returns the same results as:
new_list = []
for i in old_list:
    if i < 5:
        new_list.append(str(i))
    else:
        new_list.append("end")

# Something fancier:
new_list = [str(i) if 1 < i < 5 else "start" if i == 1 else "end"
            for i in old_list if i != 3]
# This returns the same results as:
new_list = []
for i in old_list:
    if i != 3:
        if 1 < i < 5:
            new_list.append(str(i))
        else:
            if i == 1:
                new_list.append("start")
            else:
                new_list.append("end")

# You can iterate over indices and use multiple iterables in one list
# comprehension:
old_list1 = [1, 2, 3, 4, 5]
old_list2 = [5, 4, 3, 2, 1]
new_list = [old_list1[i] + old_list2[i] for i in range(len(old_list1))]

# You can even nest list comprehensions:
old_list = [(1, "a"), (2, 12.524), (3, "c"), (4, "d", 0, 1), (5, 1, "hello")]
new_list = [str(element)
            for current_tuple in old_list
            for element in current_tuple]
# This returns the same results as:
new_list = []
for current_tuple in old_list:
    for element in current_tuple:
        new_list.append(str(element))

# Besides list comprehensions, there are also comprehensions for sets and
# dictionaries that create sets and dictionaries, respectively.
some_iterable = [1, 1, 1, 2, 3]
new_set = {elem for elem in some_iterable}
new_dict = {str(elem): elem * 10 for elem in some_iterable}


################################################################################
# Slicing
################################################################################

# A slice is an index object that allows to select multiple entries based on a
# simple pattern. It can be created via the built-in magic "slice" or directly
# in place of the indices as "some_sequence[start:end:step]". Note that the
# element at index "end" is not included. Omitting "start", "end", or "step"
# defaults to the first and last sequence element and a step size of 1. You may
# omit the last colon if you do not want to specify a step size. Slicing works
# on sequence objects (e.g., lists, tuples, strings, etc.).
# https://docs.python.org/3/glossary.html#term-sequence
some_list = [0, 1, 2, 3, 4]
some_elements = some_list[2:3]  # Equal to: some_list[2:3:]
first_2_elements = some_list[:2]
last_2_elements = some_list[-2:]
reversed_list = some_list[::-1]
all_elements = some_list[:]
string_slice = "abcdefg"[::-1]

# Slice objects can be created explicitly (and reused later) with "slice"
some_slice = slice(2, 5, 2)  # Notation: slice(start, stop, step)
some_other_elements = some_list[some_slice]  # Equal to: some_list[2:5:2]

# You can use slices to replace the sliced elements by new elements (replacement
# value must be an iterable). This is an in-place operation, i.e., the original
# object is changed.
original_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
original_list[3:7] = [10, 10]  # Replace sub-list [3:6] by [10, 10]


################################################################################
# Unpacking
################################################################################

# Python does not only allow to assign, e.g., a list or a tuple to a single
# variable, but you can also provide multiple variables which will then be
# assigned the individual items of this sequence. This is called unpacking.
some_list = [1, 2, 3, 4]  # Regular assignment
a, b, c, d = [1, 2, 3, 4]  # Unpacking: a -> 1, b -> 2, c -> 3, d -> 4
# This is equal to: (a, b, c, d) = [1, 2, 3, 4], i.e., parentheses are optional.

# The number of variables must match the number of sequence elements.
# Example of not enough values to unpack:
# a, b, c, d, e = [1, 2, 3, 4]
# Example of too many values to unpack:
# a, b = [1, 2, 3, 4]
# However, for the latter case of too many values to unpack, there is a solution
# which puts all the remaining elements in a list, using an asterisk *:
a, b, *rest = [1, 2, 3, 4]  # a -> 1, b -> 2, rest -> [3, 4]
a, *rest, d = [1, 2, 3, 4]  # a -> 1, rest -> [2, 3], d -> 4
# If there are exactly equally many elements, the list will be empty:
a, b, *rest = [1, 2]  # a -> 1, b -> 2, rest -> []

# Unpacking is very useful, e.g, if you iterate through key-value pairs of a
# dictionary, you can use the much more readable version here instead:
for key, value in dictionary.items():  # Instead of: key_value tuple
    print(key, value)  # Instead of: key_value[0], key_value[1]

# Unpacking can also be arbitrarily nested (must use parentheses):
nested_seq = [1, 2, ("a", [4.5, 5.1, 3.4, 4.2], "b", "c")]
int1, int2, tup = nested_seq  # Without nested unpacking
int1, int2, (str1, inner_list, str2, str3) = nested_seq  # Unpack inner tuple
int1, int2, (str1, (float1, *floats), *strs) = nested_seq  # Multi-nested unpacking

# When unpacking, it can often happen that not all values are of interest. Per
# convention, the underscore _ can be used to ignore those values on purpose:
_, interesting, _ = (1, "a", 5.2)  # Ignore first and third element
# For the sake of completeness, it must be noted that the underscore itself is
# actually a valid variable name and will thus reference objects (e.g., in the
# above example, _ is first referencing the integer 1 and then, immediately
# afterwards (it is relinked in the same line), is referencing the float 5.2,
# which it will still be referencing afterwards, i.e., nothing is deleted!). See
# https://stackoverflow.com/a/5893946/8176827 for a longer discussion.


################################################################################
# Useful functions
################################################################################

# We already saw the useful magic "len" and "enumerate" previously. These can
# also be applied to all other data structures mentioned here.
# https://docs.python.org/3/library/functions.html#len
# https://docs.python.org/3/library/functions.html#enumerate
tuple_len = len(a_tuple)
set_len = len(my_set)
dict_len = len(dictionary)  # Returns the number of keys/key-value pairs
for i, elem in enumerate(a_tuple):  # We actually use unpacking here
    print(i, elem)

# The built-in function "zip" can be used to join multiple iterables, by
# combining elements from each iterable in a tuple.
# https://docs.python.org/3/library/functions.html#zip
list1 = ["a", "b", "c"]
list2 = [1, 2, 3]
for e1, e2 in zip(list1, list2):  # Use unpacking again
    print(e1, e2)
# Another example: Create dictionary from list with keys and list with values:
keys = list1
values = list2
zip_dictionary = dict(zip(keys, values))

# If you want so sum up elements of an iterable (normally, the elements are
# numbers), you can use "sum".
# https://docs.python.org/3/library/functions.html#sum
ints = [4, 6, 10, 10, 3, 14]
int_sum = sum(ints)

# The built-in functions "min" and "max" return the minimum and maximum element
# of an iterable, respectively.
# https://docs.python.org/3/library/functions.html#min
# https://docs.python.org/3/library/functions.html#max
int_min = min(ints)
int_max = max(ints)

# The function "all" checks if all elements of an iterable evaluate to True
# (mathematical all-quantifier).
# https://docs.python.org/3/library/functions.html#all
if all([i <= 10 for i in ints]):  # Use list comprehension to get a list of booleans
    print("all numbers are <= 10")

# Analogously, there is the function "any", which checks if at least on of the
# elements of an iterable evaluates to True (mathematical exists-quantifier).
# https://docs.python.org/3/library/functions.html#any
if any([i <= 10 for i in ints]):  # Use list comprehension to get a list of booleans
    print("at least one number is <= 10")

