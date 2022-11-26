# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
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

In this file, we will learn how to raise and catch exceptions.
https://docs.python.org/3/tutorial/errors.html
"""

################################################################################
# Exceptions
################################################################################

# Python supports a trial-and-error mentality. You can dynamically try to
# execute code and if something goes wrong, you handle the problem by either
# applying different code or reporting the error to the user. So instead of
# checking for possible errors in if-else conditions before executing the code,
# the code is executed and if an error arises, the error can be handled
# dynamically. "Errors" during code execution are referred to as "exceptions"
# in Python. Exceptions can be of different types, and they can provide
# additional information about the error that occurred.

# Note that this does not mean to use exception handling everywhere, do not
# overuse this feature. In many cases, it is perfectly fine to check conditions
# rather than throwing exceptions around. Always consider the current task at
# hand and decide what makes more sense.

# Exceptions can be raised by using the "raise" keyword with this syntax:
# raise TypeOfError("Some error message as string")
# Python has a variety of built-in exceptions, for a full list, see
# https://docs.python.org/3/library/exceptions.html#bltin-exceptions
a = False  # Set this to True to raise the following exception
if a:
    raise ValueError(f"Variable 'a' was {a}")

# Providing an exception message is not required:
if a:
    raise ValueError

# Once an exception is raised, the execution of the program jumps to the end of
# the program. If we do not want this behavior, we can handle exceptions with
# the "try" statement. However, we never have to, Python does not enforce this.
# https://docs.python.org/3/reference/compound_stmts.html#try

# Here is an example of handling exceptions with a "try" statement:
try:
    # Here, we put our "normal" code. If an exception should occur here, we can
    # "catch" it in the following "except" blocks.
    a = False  # Set this to True to raise the exception
    if a:
        # If something goes wrong, an exception will be raised. We can also do
        # this ourselves by raising it directly:
        raise ValueError(f"Variable 'a' was {a}")
except ValueError as ex:  # Storing the exception in a variable is optional
    # We will land here if a ValueError was raised. We can use this to execute
    # code after the exception was raised.
    print(f"A ValueError brought us here. The error was '{ex}'")
    # Important: At this point, the above exception was handled. If we still
    # want the exception to "continue", we have to raise the exception again:
    raise ex  # Could also just write "raise" (reraises the occurred exception)
except TypeError:
    # We will land here if a "TypeError" was raised.
    print("A TypeError brought us here")
    # Since we do not raise any further exception here, the program execution
    # would continue normally from here.
else:
    # This part is only executed if no exception occurred above.
    print("No exception occurred within the try, so execute this code")
finally:
    # You can use the "finally" keyword instead of "except" to execute code
    # independently of raised exceptions. This can, e.g., be used for clean-up.
    print("This will be executed anyway")

# It is also possible to catch multiple exceptions at once:
try:
    a = 1 + "f"  # This will raise a "TypeError"
except (ValueError, TypeError) as ex:
    # We will land here if a "ValueError" or "TypeError" was raised.
    print(f"We caught the exception '{ex}'")

# Try-except blocks may be nested arbitrarily:
try:
    a = 1 + "f"  # This will raise a "TypeError"
except (ValueError, TypeError) as ex1:
    # We will land here if a "ValueError" or "TypeError" was raised.
    print(f"We caught the exception '{ex1}'")
    # We can add inner/nested try-except blocks here
    try:
        a = 1 + [4, 5]
    except (ValueError, TypeError) as ex2:
        print(f"We caught the exception '{ex2}'")
        # We could also reraise "ex1" or "ex2" here
        print(f"ex1: '{ex1}'")
        print(f"ex2: '{ex2}'")
    print(f"ex1: '{ex1}'")

# The same rules apply, so be careful, e.g., when you have a nested "try" in
# your outer "finally" where some uncaught exception occurs. Example:
try:
    print("try 1")
# This "finally" is always executed, but it does not mean that the entire code
# runs successfully!
finally:
    # This inner "try" raises an exception which we do not handle, so the
    # execution of the outer "finally" will fail
    try:
        print("try 2")
        raise ValueError
    # Just for demonstrating purposes, this inner "finally" will, of course, be
    # executed again (the same rules apply)
    finally:
        print("finally 2")
    # Since the exception above was not caught, we do not reach this line of
    # code here, i.e., the rest of the outer "finally" code
    print("finally 1")


#
# Exceptions as control flow
#

# Exceptions can be used to create smooth program flows and are common in
# Python, since they are comparatively cheap (compared to other programming
# languages). However, remember to not overuse them (as mentioned earlier).

# Example 1: Increase a counter in a dictionary, skip if the key does not exist
numbers = [1, 2, 3, 4, 5, 1, 2, 3, 5, 2, 5, 5, 5]
counts = {2: 0, 5: 0}  # We only want to count values 2 and 5
for i in numbers:
    try:
        counts[i] += 1  # Raises a "KeyError" if the key does not exist
    except KeyError:
        # The "pass" statement is used where something is syntactically required
        # by Python, but you do not want anything to be executed (no operation)
        pass


# Example 2: Make a more robust "add" function:
def add(x, y):
    try:
        z = x + y
    except TypeError:
        print("Could not calculate 'x + y'. Trying to convert to floats ...")
        z = float(x) + float(y)  # Might raise yet another exception
    return z


# This will work now because we added a try-except block in the function:
print(add(1, "2"))

#
# Special cases
#

# There are a few special rules when there is a "finally" in a function. If the
# "finally" clause contains a "return", a "break" or a "continue", exceptions
# are not reraised, i.e., they are essentially silently swallowed! Furthermore,
# if the function returns something in the "try" statement and there is also a
# "return" in the "finally" block, the return value of the "try" is overwritten!


# Example 1: Simple code execution in "finally"
def divide(x, y):
    try:
        return x / y
    finally:
        print("always executed")


# Prints "always executed" and then the result of the successful function call
print(divide(1, 2))
# Prints "always executed" and then a "ZeroDivisionError" is raised
print(divide(1, 0))


# Example 2: "return" in "finally"
def divide(x, y):
    try:
        return x / y
    finally:
        print("always executed")
        return 0


# Prints "always executed", followed by the result which is returned in the
# "finally" clause, i.e., the return value in the "try" clause is overwritten!
print(divide(1, 2))
# Prints "always executed", followed by the result which is returned in the
# "finally" clause, i.e., the "ZeroDivisionError" is not reraised in this case!
print(divide(1, 0))
