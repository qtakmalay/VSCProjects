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

In this file, we will learn about recursive functions, generator functions and
how to import Python modules.
https://docs.python.org/3/tutorial/classes.html#generators
https://docs.python.org/3/tutorial/modules.html
"""

################################################################################
# Recursion
################################################################################

# Functions are allowed to call themselves recursively. Typically, you split the
# entire task into a smaller subtask that you can solve immediately and the
# rest. The (now smaller) rest is processed by the same function until the rest
# is simple enough to also solve it immediately, which serves as the recursion
# end (base case or recursion anchor). Remember that there must be an end of the
# recursion somewhere and that there must be progress within the recursion
# (e.g., when splitting something, the rest should be smaller than before
# splitting), or otherwise, you will have an endless recursion.


# Example: Power function that calculates x^y. Assumptions: y: int >= 1
def power(x, y):
    # Base case: x^1 = x
    if y == 1:
        return x
    # Split into subtask of a multiplication of one "x" with the rest (x^(y-1)),
    # i.e., x^y = x * x^(y - 1)
    return x * power(x, y - 1)


# There can be multiple ends of a recursion, i.e., multiple base cases, and
# multiple recursive calls (either in different branches or even subsequent
# calls). Example: Power function that calculates x^y. Assumptions: y: int
def power(x, y):
    # Base case 1: x^0 = 1
    if y == 0:
        return 1
    # Base case 2: x^1 = x
    if y == 1:
        return x
    # Recursion case 1: y > 0, which means x^y
    # Split into subtask of a multiplication of one "x" with the rest (x^(y-1)),
    # i.e., x^y = x * x^(y - 1)
    if y > 0:
        return x * power(x, y - 1)
    # Recursion case 2: y < 0, which means x^-y = 1 / (x^+y)
    # Same idea of splitting into subtask and rest as above, just with
    # appropriate changes for the negative case. Here, we already know x^+y, so
    # our subtask is simply the division (and the negation of "y"). While we do
    # not split something off here, we still achieve recursion progress: We
    # negate "y", which leads to case 1 of the recursion and ultimately its end.
    return 1 / power(x, -y)


# Example: Summation of arbitrary many arguments
def add(*args):
    # Base case 1: no args -> return 0
    if len(args) == 0:  # Equal to: if not args:
        return 0
    # Base case 2: one arg -> return this arg
    if len(args) == 1:
        return args[0]
    # Otherwise, split into subtask of sum of first arg and the rest (args[1:])
    return args[0] + add(*args[1:])


# Simpler solution (make use of the fact that slicing with a too large start
# index, i.e., seq[start:] with start >= len(seq), yields an empty sequence)
def add(*args):
    # Base case: no args -> return 0
    if not args:
        return 0
    # Split into subtask of sum of first arg and the rest (args[1:]). Here, the
    # rest might be empty, which is OK, as we have a matching base case.
    return args[0] + add(*args[1:])


################################################################################
# Generators: Using a function as an iterable with "yield"
################################################################################

# Instead of writing "return", we can also write "yield" within a function. This
# will make the function a so-called generator function that returns an iterator
# as a generator object, which can be used to iterate through elements yielded
# by this function. This generator object stores the state of the function.
# Everytime an element is requested (e.g, using a for loop), the code is
# executed until the yield statement is reached and the specified value is
# returned. The execution is then suspended at this point, until the next
# element is requested and the execution is then resumed again until the next
# yield is encountered or there are no more elements to return. This means the
# code is executed when needed, rather than processing all elements immediately
# and returning them as, e.g., a list.
# https://docs.python.org/3/glossary.html#term-generator

# Example with a fixed number of iterations:
def iterable_function(n_elems):
    # Code we write here will be executed only once
    print("executed once when the first element is requested")
    for x in range(n_elems):
        # Code we write here will be executed at every iteration
        print("In function:", x)
        # Variable "x" will be returned at every iteration and the current state
        # of the function is saved (e.g., local variables). When another value
        # is requested, the function is resumed after the yield statement,
        # thereby restoring its entire state exactly as it was before.
        yield x
    # No more elements to yield, function terminates
    print("code completed")


for i in iterable_function(5):
    print(f"Function yielded: {i}")


# Another example with infinitely many elements:
def infinite_random_numbers():
    import random  # See further below for more information on module imports
    while True:
        yield random.random()  # float in the range [0.0, 1.0)


# Would result in an endless loop:
# for r in infinite_random_numbers():
#     print(r)

# Solution: We can also use the built-in function "next" to access the next
# element of a so-called iterator object, and a generator function returns such
# an iterator (a generator iterator).
# https://docs.python.org/3/glossary.html#term-generator-iterator
# https://docs.python.org/3/library/functions.html#next
inf = infinite_random_numbers()
for _ in range(5):  # Use an underscore to ignore the result of "range"
    print(next(inf))


# You can still use "return" in conjunction with "yield". If encountered during
# the execution, this will simply end the generator function, i.e., no more
# values will be produced after a "return". Manually calling "next" on an
# iterator that has no more elements will lead to a "StopIteration" error.
def yield_with_return():
    for j in range(10):
        if j == 3:
            return
        yield j


gen = yield_with_return()
for i in gen:
    print(i)
# This would lead to an error since there are no more elements:
# next(gen)

#
# Generator expressions
#

# In the data structures unit, we introduced list comprehensions. When replacing
# the brackets [] with parentheses (), this creates a new construct that is
# called generator expression, which will be evaluated dynamically rather than
# creating a full list. The explanation behind this is that a generator iterator
# will be created. From the official tutorial
# https://docs.python.org/3/tutorial/classes.html#generator-expressions:
# "These expressions are designed for situations where the generator is used
# right away by an enclosing function. Generator expressions are more compact
# but less versatile than full generator definitions and tend to be more memory
# friendly than equivalent list comprehensions."
gen = (i**2 for i in range(10))  # If written like this, can only use it once!
print(sum(gen))  # Prints 285
print(sum(gen))  # Print 0 ("gen" was already used and does not yield any more elements)
print(sum(i**2 for i in range(10)))  # Better: Enclose directly (parentheses now optional)


################################################################################
# Modules
################################################################################

#
# Importing modules
#

# You can import existing modules (=Python files) via the "import ..." or
# "import ... as ..." statement. Afterwards, you can use its contents. Here, the
# full module will be imported:
import sys
python_executable = sys.executable

# Many Python modules have suggested nicknames. You can choose a different name
# of the imported module using the "as" keyword:
import sys as system
encoding_in_this_python = system.getdefaultencoding()

# Use "from ... import ..." to only import a specific part of a module (can be
# a submodule, function, global variable, class, etc.):
from os import path
correct_path = path.join("some", "directory")
# To be precise, this imports names from the module into our global/module
# namespace without importing the import module name (e.g., "os" is undefined).

# You can import multiple modules with one statement using commas
import os, sys
from os import path, makedirs
print(os, sys, path, makedirs)

# If you import a module multiple times, Python will, by default, perform the
# import only once.

#
# Creating and using custom modules
#

# If you want to reuse content from your own files, you can simply import it
# from there. Always make sure that the PYTHONPATH environment variable is set
# correctly or the module is within your working directory. Python will search
# within the PYTHONPATH and the working directory (and a installation-dependent
# default) for the module.

# The following line imports function "add" from file "my_module.py" and binds
# it to the name "my_add":
from my_module import add as my_add

argument_list = list(range(10))
print(my_add(*argument_list))

# When importing from a file, the content of the file will be executed. Often,
# we want to include code that should only be executed when using the file as
# main file (= execute only when calling the file but not when importing from
# it). This can be done using the following syntax:

print("This code will be executed when this file is imported")

if __name__ == "__main__":
    print("This code will not be executed when this file is imported")

# You can check the example in "my_module.py" by executing "my_module.py"
# directly, i.e., executing it as a script with "python my_module.py".

#
# Packages
#

# To better structure your code files, you can put them into directories. To
# make Python recognize files within directories, place an empty "__init__.py"
# file inside, which makes the directory a Python package. The following
# examples are directly taken from the official Python tutorial:
# https://docs.python.org/3/tutorial/modules.html#packages

# Suppose you have the following project structure:
# sound/                          Top-level package
#       __init__.py               Initialize the sound package
#       formats/                  Subpackage for file format conversions
#               __init__.py
#               wavread.py
#               wavwrite.py
#               aiffread.py
#               aiffwrite.py
#               auread.py
#               auwrite.py
#               ...
#       effects/                  Subpackage for sound effects
#               __init__.py
#               echo.py
#               surround.py
#               reverse.py
#               ...
#       filters/                  Subpackage for filters
#               __init__.py
#               equalizer.py
#               vocoder.py
#               karaoke.py
#               ...

# You can then import from your main "sound" package as follows (examples):
# import sound  # Import the main package
# from sound.effects import echo  # Import the submodule "echo"
# from sound.effects.echo import echofilter  # Import the function "echofilter"
