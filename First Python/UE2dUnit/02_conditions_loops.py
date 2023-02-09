# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 22.07.2022

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

################################################################################

In this file, we will learn about if, elif and else statements, for loops and
while. They will give us control over which lines of code are executed and for
how many repetitions certain code is run.
https://docs.python.org/3/tutorial/controlflow.html
"""

################################################################################
# Code blocks and indentation
################################################################################

# Python uses indentation to form blocks of code and statements, in contrast to,
# e.g., {} in other languages. Indentation can be done via tabs or spaces. It is
# common to use 4 spaces (in most editors, this is the default, or you can set
# tab to equal 4 spaces). In certain cases, you might not want to execute any
# code, but Python still requires the indentation for correct syntax. If so, use
# the "pass" statement, which is a no-op (no operation) and does nothing.


################################################################################
# if, elif, else (1)
################################################################################

# if-elif-else statements are useful when conditionally executing code, i.e.,
# some code should only be executed if certain conditions are/are not fulfilled.
# Together with loops (see further below), they constitute the basic control
# flow structures, which are essential in programming. For more details, see
# https://docs.python.org/3/reference/compound_stmts.html#the-if-statement

# Code within an "if" statement is executed only if the condition is True.
condition = True
if condition:
    print("This code block is executed since 'condition' is 'True'!")

condition = False
if condition:
    print("This code block is not executed since 'condition' is 'False'!")
    
if not condition:  # Here we use "not False" as condition
    print("'not False' evaluates to 'True', so this code block is executed!")

# We can combine the "if" statement with an "else" statement. This will execute
# the "if" part only if the condition is True, otherwise, the "else" is run.
condition = True
if condition:
    print("This code block is executed!")
else:
    print("This code block is not executed since the 'if'-part was already "
          "executed!")

if not condition:
    print("This code block is not executed!")
else:
    print("This code block is executed since the 'if'-part was not executed!")

# We can also combine the "if" statement with "elif" statements. Doing this, the
# conditions in the if-elif statements are checked one after the other. Only the
# part where the first condition is True is executed and the rest is ignored.
condition = True
if not condition:
    print("This code block is not executed!")
elif not condition:
    print("This code block is not executed!")
elif condition:
    print("Only this code block is executed!")
elif condition:
    print("This code block is not executed since a previous if-part was "
          "already executed!")
elif not condition:
    print("This code block is not executed since a previous if-part was "
          "already executed!")

# We can also combine the "if", "elif" and "else" statements. The conditions in
# the if-elif-else statements are checked one after the other. Only the part
# where the first condition is True is executed and the rest is ignored. If none
# of the if-elif parts are executed, the "else" part will be run.
condition = True
if not condition:
    print("This code block is not executed!")
elif not condition:
    print("This code block is not executed!")
elif not condition:
    print("This code block is not executed!")
else:
    print("This code block is executed since no other part was executed!")

# if-else can also be used as part of an expression, which returns the result
# according to how the boolean condition evaluates. Syntax:
# return_value = x if condition else y
# where "return_value" would be "x" if "condition" evaluated to True, or "y"
# otherwise. Example:
number = -123
sign = "-" if number < 0 else "+"


################################################################################
# Comparisons and boolean operations
################################################################################

#
# Comparisons
#

# We already saw that we need boolean values as conditions to decide which parts
# of code to execute in the if-elif-else statements. Comparisons in Python allow
# us to compare two objects and return a boolean value. You can use comparisons
# on objects of different data types. For more details, see
# https://docs.python.org/3/reference/expressions.html#comparisons

# Let us assume we have two variables "a" and "b" that refer to float objects:
a = 1.0
b = 2.0

# There are 8 comparison operations in Python:
equal = a == b  # Check if "a" is equal to "b"
not_equal = a != b  # Check if "a" is not equal to "b"
greater = a > b  # Check if "a" is greater than "b"
greater_equal = a >= b  # Check if "a" is greater than or equal to "b"
less = a < b  # Check if "a" is less than "b"
less_equal = a <= b  # Check if "a" is less than or equal to "b"
same_object = a is b  # Check if "a" and "b" refer to same object (identity check)
not_same_object = a is not b  # Check if "a" and "b" do not refer to same object

# Integers and floats of the same value yield True in the equality comparison,
# despite being different data types (equality is based on the numerical value)
is_equal = 1 == 1.0
# However, int 1 and float 1.0 are not the same object (they are two different
# objects in memory)
is_not_same_object = 1 is 1.0

# All objects support the "==" and "!=" operator, which is, per default,
# equivalent to "is" and "is not". Often, special support is provided to change
# this behavior, e.g., to compare values instead of object references such as it
# is the case for integers, floats, strings, etc.

# Other comparison operators are only supported by non-numerical objects "if
# they make sense". For example, you can use them on strings:
is_true = "abc" == "abc"
is_true = "abc" >= ""
is_true = "abc" >= "a"
is_true = "abc" >= "ab"
is_true = "abc" >= "abc"
is_false = "abc" >= "abd"
is_false = "abc" >= "abd"
# Danger: Make sure you are using the data types you intend to use:
is_true = 100 > 20
is_false = "100" > "20"  # Because "1" is smaller than "2" as character

# Here, we combine comparisons and if-elif-else statements:
a = 2
b = 3
if a > b:
    print("a > b !")
elif a < b:
    print("a < b !")
elif a == b:
    print("a == b !")
else:
    print("How did I get here?")

#
# Boolean operations
#

# You can chain comparisons using boolean operations like "and" or "or".
# https://docs.python.org/3/reference/expressions.html#boolean-operations

# "and" will be True if and only if both of the two expressions that it combines
# evaluate to True:
is_true = True and True
is_false = True and False
is_false = False and True
is_false = False and False

# "or" will be True if and only if one or both of the two expressions that it
# combines evaluate to True:
is_true = True or True
is_true = True or False
is_true = False or True
is_false = False or False

# The priority of comparisons is higher than that of boolean operations but
# lower than mathematical operators (for an exhaustive list, see
# https://docs.python.org/3/reference/expressions.html#operator-precedence):
is_true = (1 > -4) and (1 == 1)  # Equal to no parentheses: 1 > -4 and 1 == 1
is_true = (1 > -4) and (1 == 1) and (1 < 4)  # Again, parentheses are optional here
is_false = (1 > -4) and not (1 < 4)  # Again, parentheses are optional here

# You can also chain comparisons without "and" or "or":
is_true = 1 < 4 <= 5  # Equal to: 1 < 4 and 4 <= 5

# Note: "and" and "or" are evaluated in a boolean context, but they ultimately
# return the actual value (the last value following the short-circuit evaluation
# rules) rather than True or False. Examples for "and" ("or" is analogous):
and_return = 0 and 1  # -> bool(val1) and bool(val2) -> False and True -> return val1
and_return = 1 and 0  # -> bool(val1) and bool(val2) -> True and False -> return val2


################################################################################
# while loops
################################################################################

# While loops continue as long as the loop condition is True. For more details,
# see https://docs.python.org/3/reference/compound_stmts.html#the-while-statement

# The code of this loop is run exactly 10 times:
i = 0
while i < 10:
    print(i)
    i += 1

# The code of this loop is not run, since the condition is immediately False:
i = 0
while i < 0:
    print(i)
    i += 1

# This would run forever (endless/infinite loop):
# i = 0
# while True:
#     print(i)
#     i += 1


################################################################################
# for loops
################################################################################

# For loops in Python iterate over a so-called iterable. This can be, e.g., a
# range object, a list, dictionary, tuple, string or function return/yield (we
# will learn all this later). For now, we will focus only on range objects and
# strings.

# Again, we rely on some Python built-in magic, which is called "range" in this
# case. "range" returns a range of integer numbers and can be called as follows:
# range(stop) -> returns numbers in the range [0, stop) with a step size of 1
# range(start, stop) -> returns numbers in the range [start, stop) with a step
#     size of 1
# range(start, stop, step) -> returns numbers in the range [start, stop) with a
#     step size of "step" (can be negative)
for x in range(5):  # [0, 5) -> 0, 1, 2, 3, 4
    print(x)
for x in range(2, 6):  # [2, 6) -> 2, 3, 4, 5
    print(x)
for x in range(0, 10, 2):  # [0, 10) step 2 -> 0, 2, 4, 6, 8
    print(x)

# Getting the sum of all integer numbers up to "n" (inclusive):
n = 100
sum_up_to_n = 0
for x in range(n + 1):  # "stop" is exclusive, so we need the +1
    sum_up_to_n += x

# A string is also an iterable object (sequence of characters):
for char in "hello":
    print(char)

# If you want to iterate over iterable elements and get the number of the
# current loop iteration, you can use the "enumerate" function (again, just
# treat this as magic for now, we will learn about functions later):
for i, char in enumerate("hello"):
    print(f"loop iteration: {i}, element: '{char}'")

# Using the already known magic "len" function and string indexing with the [i]
# syntax, the above code could also be written as:
hello = "hello"
for i in range(len(hello)):
    print(f"loop iteration: {i}, element: '{hello[i]}'")


################################################################################
# break, continue, else
################################################################################

# The break and continue keywords can be used to either escape the loop or to
# jump to the next element immediately (the rest of the loop code is ignored)
for i in range(10):
    if i == 4:
        continue  # Continue with next loop iteration
    elif i == 8:
        break  # Leave the loop, i.e., jump to the code after the loop
    print(i)

# The else keyword put after a loop has a special meaning in Python. It is only
# executed if the loop has finished properly, i.e., the for loop iterable was
# exhausted or the while condition was False.
for i in range(5):
    if i == 3:
        break
    print(i)
else:
    print("Made it to the end!")


################################################################################
# Nesting
################################################################################

# All of the above control flow structures can be arbitrarily nested:
action = ""
while action != "exit":
    action = input("Please enter an action (must be one of "
                   "'sum', 'count', 'combinations', 'exit'): ")
    
    if action == "sum":
        n = int(input("Enter integer number > 0: "))
        
        if n > 0:
            sum_up_to_n = 0
            for x in range(n + 1):
                sum_up_to_n += x
            # Here, we can see an assertion. Since we know that "n" is > 0, the
            # sum of all numbers up to "n" (inclusive) must also be > 0, which
            # is exactly what we integrate here as a short sanity check. If the
            # assertion fails, we immediately know that we made a mistake.
            assert sum_up_to_n > 0
            print(f"Sum up to {n} = {sum_up_to_n}")
        else:
            print(f"Action failed: {n} is not > 0")
    elif action == "count":
        word = input("Enter word: ")
        char = input(f"Enter character to count in '{word}': ")
        count = 0
        
        for wchar in word:
            if wchar == char:
                count += 1
        
        print(f"Character '{char}' occurred {count} times in '{word}'")
    elif action == "combinations":
        n1 = int(input("Enter first integer number: "))
        n2 = int(input("Enter second integer number: "))
        
        for i1 in range(n1):
            for i2 in range(n2):
                print(f"{i1:3d} {i2:3d}")
    else:
        print(f"Unknown action '{action}'")
