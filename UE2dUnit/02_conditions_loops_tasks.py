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

Tasks for self-study. Try to solve these tasks on your own and compare your
solutions to the provided solutions file.
"""

#
# Task 1
#

# Run through values in the range [3, 20) and compute the sum of their squares.
# Use a for loop to solve this task.

#Your code here #
# y = 0
# for x in range(3,20): 
#     y += x**2 
#     print("x=%s : %s sum of square" %(x, y))

#
# Task 2
#

# Run through values in the range [3, 20) and compute the sum of their squares.
# Use a while loop to solve this task.

# Your code here #
# x = 3
# y = 0
# while (x>=3 and x<=20):
#     y += x**2 
#     print("x=%s : %s sum of square" %(x, y))
#     x= x+1


#
# Task 3
#

# Run through values in the range [3, 20) and print only those numbers that are
# divisible by 3 (without remainder).

# Your code here #
# x = int(3)

# for x in range(3,20):
#     if(x % 3 == 0):
#         print("x=%s : %s sum of square" %(x, x/3))
    


#
# Task 4
#

# Create an empty string "some_string" and append user (console) input to this
# string until the user types "end". This last word "end" should not be appended
# to "some_string".

# Your code here #
# x = "some_string"
# y = input("Type a word to print: ")
# while(y != "end"):
#     x += y
#     print("Word: ",x)
#     y = input("Type a word to print: ")
# print("Word: ",x)
#
# Task 5
#

# Use a while loop to implement a pseudo-login scenario in which a user is asked
# to enter a password (console input). If the password is correct, print "Login
# success". Otherwise, let the user try again. In case of entering three wrong
# passwords, print "Contact the administrator to recover password". The choice
# of the correct password is up to you.

# Your code here #
# password = "pswd"
# in_pswd = input("Enter the password: ")
# while(in_pswd.__len__()>0):
#     if(in_pswd == password):
#         print("Login success")
#         break
#     else:
#         print("Contact the administrator to recover password")
#         in_pswd = input("Enter the password: ")
#
# Task 6
#

# Read a string from the console input. Iterate through this string and count
# the number of digits (0-9), the number of lowercase characters and the number
# of other characters (neither digits nor lowercase characters). You can use the
# string methods "c.isdigit()" and "c.islower()".

# Your code here #
in_string = input("Enter a string: ")
n_digits, nl_char, n_chars = 0, 0, 0
for x in in_string:
    if(x.isdigit()):
        n_digits += 1
    elif(x.islower()):
        nl_char += 1 
    elif(not x.isdigit() and not x.islower()):
        n_chars += 1

    
print("Sting %s, Digits: %s, Lowercase: %s, Other chars: %s" %(in_string, n_digits, nl_char, n_chars))
#     
#
# Task 7
#

# Use a double-nested loop to iterate through both of the strings "text" and
# "chars_to_check". If the current character from "text" matches one in
# "chars_to_check", print it together with its index position in "text".
text = "some string"
chars_to_check = "aeiou"

# Your code here #
