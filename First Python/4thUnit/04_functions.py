# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
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

In this file, we will learn the basics about functions.
https://docs.python.org/3/tutorial/controlflow.html#defining-functions
https://docs.python.org/3/tutorial/controlflow.html#more-on-defining-functions
https://docs.python.org/3/library/functions.html
"""

################################################################################
# Function definition and invocation
################################################################################

# Functions can be used to create reusable parts of code. Analogous to variable
# names, Python functions should have lowercase names (with underscores to
# separate words for increasing readability). They can have parameters, and they
# are created with the "def" keyword as shown in the following examples. They
# must be defined before their invocation (in terms of code location). Just like
# everything in Python, functions are also objects. We already used several
# built-in functions, such as "len" or "print".


# This would be a function that takes no arguments, computes 3*5, converts the
# result to a string and returns it:
def no_arguments():
    a = 3 * 5
    a = str(a)
    return a


# We can then call/invoke this function whenever we need it:
result = no_arguments()
another_result = no_arguments()
# Since functions are objects, we can simply assign a variable to them:
different_func_name = no_arguments  # References the same function object
yet_another_result = different_func_name()  # Calls "no_arguments()"


# A function can also be defined within another function, which is called a
# nested function definition. Inner functions are not visible outside the outer
# function (see namespaces and scopes below).
def outer_function(strings, max_len):  # Also called enclosing function
    def inner_function(s: str):  # Also called local function
        s = s.upper()
        if len(s) > max_len:
            s = s[:max_len] + "..."
        return s
    
    return [inner_function(s) for s in strings]


################################################################################
# Namespaces and scopes
################################################################################

# Note that variables created inside a function only exist inside the function.
# This is something called a "namespace", a kind of dictionary that maps names
# to their objects. In Python, there are different namespaces with different
# lifetimes: There is the built-in namespace (containing names such as "len",
# "dict", "print", etc.), which always exists throughout the entire program
# execution. Then, there is the global namespace, which contains the names that
# are defined at the top level of the script/module (e.g, "no_arguments" and
# "another_result" from above). It exists as long as the module is loaded. Next,
# there is the enclosing namespace of a function, which contains names that are
# only defined within this function. It exists as long as the function is
# executed. Lastly, there is the local namespace, which contains names that are
# defined at the innermost level and exists until this level is left (e.g., an
# inner function within an outer function, where the inner function would form
# the local namespace and the outer function one enclosing namespace). Which
# object is now ultimately used by Python is determined by the "scope" of the
# name. Python looks up a name in the reverse order of the above namespaces,
# i.e., it starts to look for the name in the local namespace (local scope),
# then moves on to any enclosing namespaces (enclosing scope), then to the
# global namespace (global scope) and ultimately to the built-in namespace
# (built-in scope), stopping as soon as the first match is found (even if outer
# namespaces also contain the same name).

# Example of a function with two parameters and a total of three entries in the
# local namespace (in this case, there are no enclosing namespaces, since there
# is no nesting level here; or, alternatively, one could say that the local
# namespace is identical to an enclosing namespace here):
def add(a, b):
    # This code is executed when calling the function. Variables created here
    # are not available outside the function. This includes both parameters
    c = a + b  # The (local) variable "c" exists only within this function
    # You can check that "a", "b" and "c" are locals with the built-in "locals":
    print(locals())
    return c


# The variable "c" in the function does only exist within the function, so we
# can use another, different variable with the same name outside the function:
c = 15
print(f"c before function: {c}")
# This will not alter the variable "c" outside the function:
result = add(1, 2)
print(f"c after function: {c}")
# The reason is, as described above, namespaces and scopes: Here, "c" is both
# contained in the global and local namespace, but due to how scopes work, when
# we are inside the function, the local "c" is used (since the local namespace
# is searched first, and it immediately matches), whereas outside the function,
# the global "c" is used (since the local function namespace is not "active").
# Using the same name in multiple namespaces is called "shadowing": The inner
# name "shadows" the name of the outer scope, i.e., the name of the outer scope
# is not visible/accessible because Python only sees the most inner name.

#
# Using variables from outer scopes
#

# As mentioned above, identical names that exist in multiple namespaces lead to
# shadowing. However, different names can be used without any side effects. For
# example, we specify a global name "x":
x = 0


def some_function(b):
    # We can access "x" just fine (local scope -> global scope -> found!)
    c = x + b
    # However, assignments to "x" are, by default, not allowed:
    # x = 1  # Does not work
    # The reason is that writing the above line, Python will then insert "x"
    # into the local namespace, and suddenly, the line "c = x + b" becomes
    # invalid because the (now local) "x" has not been defined yet (it was only
    # defined in the line afterwards, which is too late).
    return c


print(x, some_function(100), x)


def some_function(b):
    # Python allows us to change the above default behaviour. If we explicitly
    # write the keyword "global" followed by the name of interest, we can force
    # Python to treat this name as a global, regardless of whether there is an
    # assignment to this name.
    # https://docs.python.org/3/reference/simple_stmts.html#the-global-statement
    global x
    c = x + b
    x = 1  # Now works because "x" is treated as a global
    return c


print(x, some_function(100), x)

# Analogously, there exists the "nonlocal" statement, which can be used for a
# similar purpose, by forcing Python to treat a name as being part of the
# nearest enclosing scope (excluding the global scope). This is a rather niche
# application, so we will not go into further details.
# https://docs.python.org/3/reference/simple_stmts.html#the-nonlocal-statement

# Important: Using globals other than constants in your functions can lead to
# many unwanted side effects and hard-to-find bugs. Rethink your implementation
# and check if you really need them (maybe an additional parameter suffices).


################################################################################
# Function parameters and arguments
################################################################################

# You can assign default argument values to parameters and make them optional:
def add(a, b=1):  # Set a default value of 1 for parameter "b"
    c = a + b
    return c


# Arguments can be passed to functions using keyword arguments (order can be
# different from the one specified in the function definition):
result = add(b=2, a=1)

# Or using positional arguments (following the position of the arguments in the
# function definition):
result = add(1, 2)

# Or a mix of both, as long as the positional arguments are in front:
result = add(1, b=2)

# Recap from unpacking: If you do not want to make use of the returned values,
# use an underscore:
_ = add(1, 2)

# Or just call the function and ignore the return value:
add(1, 2)

# Since we specified a default value, "b" does not need to be set:
add(1)


# Be careful when using default arguments with mutable objects. Default
# arguments are only evaluated once in the beginning, so if you change the
# object afterwards, the default value changes as well!
def print_list(default_empty_list: list = []):
    print(default_empty_list)
    return default_empty_list


my_list = print_list()  # Prints "[]"
my_list.append(1)  # Changes the default value since this is the same list object
print_list()  # Now prints "[1]" since the list was changed


# Solution: Use None as default value and create a new list internally
def print_list(default_empty_list: list = None):
    if default_empty_list is None:
        default_empty_list = []  # Will always create a new list
    print(default_empty_list)
    return default_empty_list


my_list = print_list()  # Prints "[]"
my_list.append(1)  # New list object was returned, so no side effects here
print_list()  # Still prints "[]"


# If you do not know how many arguments you will get, you can use "*arg" (or any
# other valid name after *) to collect all positional arguments in a tuple and
# "**kwargs" (or any other valid name after **) to collect all keyword arguments
# in a dictionary.
def unknown_arguments(*args, **kwargs):
    print(f"Function got args: {args}")
    print(f"Function got kwargs: {kwargs}")


unknown_arguments(1, 2, 3, a=4, b=5, c=6)


# Variable arguments and normal parameters can be mixed. "args" and "kwargs"
# will only contain non-normal arguments (here: "a" and "b" will not be
# contained in the tuple "args" and neither in the dict "kwargs")
def mixed_arguments(a, *args, b, **kwargs):
    print(f"a = {a}, b = {b}")
    print(f"Function got args: {args}")
    print(f"Function got kwargs: {kwargs}")


# Here, "b" is called a keyword-only argument because it appears after "*args"
# and thus cannot be specified in any other way:
mixed_arguments(1, b=2)
# mixed_arguments(1, b=2, a=3)  # Would fail since "a" is already specified (a=1)
mixed_arguments(1, 2, 3, b=2, c=5)


# If you want to pass list/tuple or dictionary elements to a function as
# separated arguments, use "*" or "**" respectively to unpack the values.
def add(a, b=1):
    c = a + b
    return c


arg_list = [1, 2]
add(*arg_list)  # Equal to: add(1, 2)

kwarg_dict = dict(a=1, b=2)
add(**kwarg_dict)  # Equal to: add(a=1, b=2)

unknown_arguments(*arg_list, **kwarg_dict)
# The following would fail since "a" is already specified via "*arg_list"
# (positional results in a=1) and "kwarg_list" also contains "a":
# mixed_arguments(*arg_list, **kwarg_dict)
mixed_arguments(*arg_list, b=10)  # a=1, b=10, args=(2,), kwargs={}
mixed_arguments(5, *arg_list, b=10)  # a=5, b=10, args=(1, 2), kwargs={}
mixed_arguments(**kwarg_dict)  # a=1, b=2, args=(), kwargs={}
mixed_arguments(x=123, **kwarg_dict, c=7)  # a=1, b=2, args=(), kwargs={"x": 123, "c": 7}
# mixed_arguments(5, **kwarg_dict)  # Would fail since "a" is already specified (a=5)

# You can even enforce positional-only parameters and keyword-only arguments.
# We will not cover this special topic here, but if you want more details, see
# https://docs.python.org/3/tutorial/controlflow.html#special-parameters


################################################################################
# Function return values
################################################################################

# Returning a value is optional in Python functions. If no return statement is
# used, None is returned by default. If you return multiple values, they are
# automatically packed into a tuple:
def return_values():
    return 1, 2, 3, 4  # Equal to: return (1, 2, 3, 4)


# Here, we just leave the return values as a tuple, but we could apply unpacking
# as well (see the unpacking section in the earlier unit on data structures):
result_tuple = return_values()


# The return statement not only returns values but also exits the function:
def return_values(a: bool, b: int):
    """Returns ``b`` if ``a`` is True, otherwise returns ``b**2``."""
    if a:
        # If we are here, we return "b", and the rest of the code is not
        # executed, since the function is exited immediately after the return.
        return b
    b **= 2
    return b


################################################################################
# Mutable objects as function arguments
################################################################################

# If you pass a variable that references a list object to a function, the
# argument now references the same list object as the passed variable (same as
# with assigning a list to multiple variables) using
# a = [1,2,3]  # If we pass "a" as argument to a function parameter "b", ...
# b = a  # ... this is what effectively happens in the function call.
# The list is not copied, we still only have one list object! So if you modify
# this list object in the function, this also affects the variable that
# references the same list object outside the function. The same goes for
# dictionaries and, more generally, other mutable objects. Note how the example
# below changes the list elements also outside the function:
def modify_list(some_list):
    some_list[0] = 55


my_list = [1, 2, 3]
modify_list(my_list)  # This changes the first element in "my_list"!


################################################################################
# Type hints and documentation
################################################################################

# You may use colons ":" to indicate the type of the parameters (this is only
# for hints and does not force anything!). You can indicate the return type
# using "->". See the module "typing" for more type hints:
# https://docs.python.org/3/library/typing.html
def add(a: int, b: int = 1) -> int:
    c = a + b
    return c


# This still works, but PyCharm warns us about an integer being expected:
add(1, 2.123)


# Make use of docstrings to show what your function is about:
def add(a: int, b: int = 1) -> int:
    """Adds two variables (small description).

    This function adds two variables via the ``+`` operator (long description).

    Parameters
    -------------
    a : int
        First argument.
    b : int
        Optional second argument. Default: 1

    Returns
    -------------
    int
        Returns ``a + b``.
    """
    c = a + b
    return c


# Alternative docstring structure
def add(a: int, b: int = 1) -> int:
    """Adds two variables (small description).

    This function adds two variables via the ``+`` operator (long description).

    :param a: First argument.
    :param b: Optional second argument. Default: 1
    :return: Returns ``a + b``.
    """
    c = a + b
    return c


# Left-click on the function "add" and Ctrl+q or go to View->Quick Documentation
# to display the information in the docstring as documentation in PyCharm.
add(1, 2.123)
