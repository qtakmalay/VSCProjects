# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 03.08.2022

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

import re

#
# Task 1
#

# You are given a string "some_header". Use a regex to extract the ID value,
# i.e., the content of the line starting with "# id: " but without the "# id: ".
# In addition, assume that there can be arbitrary many (also 0) space " "
# characters following the colon of "# id:". The ID itself consists of at least
# one of the following characters: A-Z, 0-9.
# Subtask: Assume that the ID consists of arbitrary characters (except a new
# line character), but at least one.
some_header = """
# alpha: 55
# beta: 62
# some stuff
# id: A523B
# some stuff
"""
# Result should be:
# "A523B"

# Your code here #


#
# Task 2
#

# You are given a string "some_string". In the string, there are 2 words that
# are separated by at least one whitespace character. Extract the two words in
# the string without the whitespace characters and put them in a list "words".
some_string = "first_word          second_word"
# Result should be:
# words = ["first_word", "second_word"]

# Your code here #


#
# Task 3
#

# You are given a list of strings "some_strings". In each string, there are two
# words that are separated by at least one whitespace character. Extract the two
# words for each string in the list. Create two lists, "first_words" and
# "second_words" that each contain the collected first and second words.
some_strings = ["first_word second_word",
                "first_other_word   second_other_word",
                "other_first_word          other_second_word"]
# Result should be:
# first_words = ["first_word", "first_other_word", "other_first_word"]
# second_words = ["second_word", "second_other_word", "other_second_word"]

# Your code here #


#
# Task 4
#

# The string "text" contains various words including e-mail addresses that you
# have to extract into a list. E-mail addresses are specified as follows:
#     firstname.lastname@company.com
# with the following definitions
#     firstname: Optional, must start with a lowercase letter and can have
#                arbitrary many digits and letters afterwards (following letters
#                can be both upper and lower case) if it is specified. If not,
#                then the following "." character is also discarded.
#     lastname:  Must start with a letter, must be at least two characters long
#                and must not contain any other characters than letters (every
#                letter can be both upper and lower case)
#     company:   Must only contain lowercase letters (at least one). It can
#                contain further sub-departments that must be separated by the
#                "." character, and each such sub-department again must only
#                contain lower case letters (at least one). Examples:
#                "mycomp.it" or "mycomp.hr"
#     Every valid e-mail address must end with ".com".
#
# Additional task for an even greater challenge: Every valid e-mail address must
# have either a whitespace character or any of the following characters before
# the address: "(", "[", ">"
text = """
Dear madam or sir,

I am writing to this mail here (office@mycomp.com) to let you know
about some organizational shifts (also let @Micheal know). Here are
the mail addresses that need to be handled asap:
  > gabe.new_man@mycomp.com ---> rename to gabe.newman@mycomp.com
  > li_ly.mcNeil@mycomp.hr.com ---> let her know the IT-fix is delayed
  > andi12@mycomp.it.com ---> rename to andi12.chillings@mycomp.it.com!!
  > nancy.f@mycomp.org ---> change to nancy.f@mycomp.com (and maybe ask her if
    nancy.farlain@mycomp.com would also be OK for her)
  >support@mycomp.com (is this already setup? is support@mycomp.IT.com the same?)
Since we have a special HR-department for management-related personnel, can
we setup an e-mail address like tickets@mycomp.hr.man.com or something like
this (at least the x@mycomp.hr.com part is important, I think, you can name
"x" whatever you wish.)

Best regards!
"""
# Result for the standard task should be:
# addresses = ["office@mycomp.com", "man@mycomp.com", "gabe.newman@mycomp.com",
#              "ly.mcNeil@mycomp.hr.com", "andi12.chillings@mycomp.it.com",
#              "nancy.farlain@mycomp.com", "support@mycomp.com",
#              "tickets@mycomp.hr.man.com"]
#
# Result for the additional task should be:
# addresses = ["office@mycomp.com", "gabe.newman@mycomp.com",
#              "andi12.chillings@mycomp.it.com", "nancy.farlain@mycomp.com",
#              "support@mycomp.com", "tickets@mycomp.hr.man.com"]

# Your code here #
