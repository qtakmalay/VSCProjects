# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Van Quoc Phuong Huynh, Andreas Schörgenhumer
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

Example solutions for tasks in the provided tasks file.
"""

#
# Task 1
#

# Run through values in the range [3, 20) and compute the sum of their squares.
# Use a for loop to solve this task.

# Your code here #
result = 0
for value in range(3, 20):
    result += value ** 2


#
# Task 2
#

# Run through values in the range [3, 20) and compute the sum of their squares.
# Use a while loop to solve this task.

# Your code here #
result = 0
i = 3
while i < 20:
    result += i ** 2
    i += 1


#
# Task 3
#

# Run through values in the range [3, 20) and print only those numbers that are
# divisible by 3 (without remainder).

# Your code here #
for i in range(3, 20):
    if i % 3 == 0:
        print(i)

# Shorter solution with step size
for i in range(3, 20, 3):
    print(i)


#
# Task 4
#

# Create an empty string "some_string" and append user (console) input to this
# string until the user types "end". This last word "end" should not be appended
# to "some_string".

# Your code here #
some_string = ""
while True:
    user_input = input("Input: ")
    if user_input == "end":
        break
    some_string += user_input

# Solution that does not require an endless loop + break:
some_string = ""
# Initialize the user input with an empty string so the first string
# concatenation below is essentially a no-op. Another solution would be to
# initialize it to None, and then do the string concatenation only if the user
# input is not None ("if user_input is not None"). Both versions work, each with
# their benefits and drawbacks (string concatenation with empty string vs.
# repeated conditional check).
user_input = ""
while user_input != "end":
    some_string += user_input
    user_input = input("Input: ")


#
# Task 5
#

# Use a while loop to implement a pseudo-login scenario in which a user is asked
# to enter a password (console input). If the password is correct, print "Login
# success". Otherwise, let the user try again. In case of entering three wrong
# passwords, print "Contact the administrator to recover password". The choice
# of the correct password is up to you.

# Your code here #
# Solution with while-else statement
secret_pwd = "secret"
try_count = 0
try_limit = 3

while try_count < try_limit:
    password = input("Password: ")
    if password == secret_pwd:
        print("Login success")
        break
    try_count += 1
else:
    print("Contact the administrator to recover password")

# Solution with normal while (a bit longer but much better readability)
secret_pwd = "secret"
try_count = 0
try_limit = 3
password = None

while try_count < try_limit and password != secret_pwd:
    password = input("Password: ")
    try_count += 1

if password == secret_pwd:
    assert try_count <= try_limit, "try_count is larger than try_limit"
    print("Login success")
else:
    print(f"Contact the administrator to recover password")


#
# Task 6
#

# Read a string from the console input. Iterate through this string and count
# the number of digits (0-9), the number of lowercase characters and the number
# of other characters (neither digits nor lowercase characters). You can use the
# string methods "c.isdigit()" and "c.islower()".

# Your code here #
s = input("Enter something: ")
n_digits = 0
n_lowercase = 0
n_other = 0
for char in s:
    if char.isdigit():
        n_digits += 1
    elif char.islower():
        n_lowercase += 1
    else:
        n_other += 1
assert len(s) == n_digits + n_lowercase + n_other


#
# Task 7
#

# Use a double-nested loop to iterate through both of the strings "text" and
# "chars_to_check". If the current character from "text" matches one in
# "chars_to_check", print it together with its index position in "text".
text = "some string"
chars_to_check = "aeiou"

# Your code here #
for i, char in enumerate(text):
    for char_to_check in chars_to_check:
        if char == char_to_check:
            print(f"[{i}] {char}")
