# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 02.08.2022

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

################################################################################

In this file, we will look into how to use the "re" module to search for more
complex patterns in strings via regular expressions ("regex").
"""

################################################################################
# re - searching for complicated patterns in text via regular expressions
################################################################################

# The "re" module allows you to search for complex patterns in text. If you are
# only looking for simple patterns, the native Python string functions, e.g.,
# "string_with_substring".find("substring"), are simpler and faster and should
# be preferred. Regex syntax can become quite complex. For more details, see
# https://docs.python.org/3/howto/regex.html#regex-howto
# https://docs.python.org/3/library/re.html
# You can use tools like https://www.debuggex.com/ or https://regex101.com/ to
# analyze and debug a regex. If you are struggling to build a specific pattern,
# it might pay off to search for it (or similar patterns) first. Chances are
# that someone already created a similar pattern.

import re

#
# Searching for one occurrence of a pattern
#

# There are two basic ways of applying pattern match with the "re" module: The
# first is to create a "Pattern" object through the "re.compile" function and
# subsequently use this object to perform pattern matching with its various
# methods. The other way is to use the top-level convenience functions of the
# "re" module, which cover the same functionality as the object methods. If you
# plan on reusing patterns, the object-oriented approach is more efficient. But
# in the following examples, we will only use the function-based approach.

# There are two important functions for finding a pattern or a group of patterns
# in a string: "re.search" and "re.match".

# "re.search" will search for the first occurrence of a pattern and return a
# "Match" object if it found a pattern. If no pattern is found, None will be
# returned instead. Strings are scanned from start (left) to end (right).
pattern = "Elm Street"
text = "Ross McFluff: 155 Elm Street"
match_object = re.search(pattern, text)
print(f"{text} + {pattern} -> {match_object}")

# "re.match" will search for patterns only at the beginning of the string (even
# if it is a multi-line string).
pattern = "Ross Mc"
text = "Ross McFluff: 155 Elm Street"
match_object = re.match(pattern, text)
print(f"{text} + {pattern} -> {match_object}")

pattern = "Elm Street"
text = "Ross McFluff: 155 Elm Street"
match_object = re.match(pattern, text)
# Note that this will return None because no pattern was found
print(f"{text} + {pattern} -> {match_object}")

#
# Returning groups within patterns
#

# You can use groups to only return sub-patterns within a search pattern. Groups
# are created using parentheses ().

# This will match the string "Elm Street" and return "Elm" and "Str" separately
# in two groups:
pattern = "(Elm) (Str)eet"
text = "Ross McFluff: 155 Elm Street"
match_object = re.search(pattern, text)
print(f"{text} + {pattern} -> {match_object}")

# You can access the found pattern(s) with "match_object.group(i)", where "i" is
# the group number of the found pattern you want to get. "match_object.group()"
# or "match_object.group(0)" will return the complete pattern (=the full match),
# "match_object.group(1)" the first group, "match_object.group(2)" the second
# group and so on. "match_object.groups()" will return all found pattern groups.
print(f"{text} + {pattern} -> .groups() -> {match_object.groups()}")
print(f"{text} + {pattern} -> .group()  -> {match_object.group()}")
print(f"{text} + {pattern} -> .group(0) -> {match_object.group(0)}")
print(f"{text} + {pattern} -> .group(1) -> {match_object.group(1)}")
print(f"{text} + {pattern} -> .group(2) -> {match_object.group(2)}")

# You can also nest groups:
pattern = "Elm ((Str)eet)"
text = "Ross McFluff: 155 Elm Street"
match_object = re.search(pattern, text)
print(f"{text} + {pattern} -> .groups() -> {match_object.groups()}")
print(f"{text} + {pattern} -> .group()  -> {match_object.group()}")
print(f"{text} + {pattern} -> .group(0) -> {match_object.group(0)}")
print(f"{text} + {pattern} -> .group(1) -> {match_object.group(1)}")
print(f"{text} + {pattern} -> .group(2) -> {match_object.group(2)}")

#
# Getting additional data from "Match" objects
#

# "Match" objects contain more than just the found pattern. They also let you 
# access information like the start and end position, the width, etc. of the
# individual groups:
print(f"{text} + {pattern} -> .group(1) -> {match_object.group(1)}\n"
      f"  start, group 1: {match_object.start(1)}\n"
      f"  end,   group 1: {match_object.end(1)}\n"
      f"  span,  group 1: {match_object.span(1)}")

# Note that the end position is exclusive (i.e., index + 1) to enable slicing:
print(f"{text[match_object.start(1):match_object.end(1)]}")
print(f"{text[match_object.start(1)]}")
print(f"{text[match_object.end(1) - 1]}")

#
# Searching for (non-overlapping) multiple occurrences of a pattern
#

# "re.findall" will search for all patterns in the text and return a list.
# "re.finditer" also does this but one item at a time, and it returns a "Match"
# object. Strings are scanned from start (left) to end (right).

pattern = "bla"
text = "blablablibla"
match_list = re.findall(pattern, text)
print(f"{text} + {pattern} -> {match_list}")

for i, m in enumerate(re.finditer(pattern, text)):
    print(f"{i + 1}. match: {m.group()} (Pos: {m.start()}-{m.end()})")

# You can again use groups to return sub-patterns:
pattern = "(bl)(a)"
text = "blablablibla"
match_list = re.findall(pattern, text)
print(f"{text} + {pattern} -> {match_list}")

#
# Making a pattern flexible
#

# Patterns can include meta characters in regex syntax to search for flexible
# patterns. The regex syntax uses the special characters {}[]()^$.|*+?. If you
# want to use them as normal characters in a string, you need to escape their
# special function with a preceding backslash "\". For example: "\?".

# [] will specify a set of characters to match, here: "c" or "b" or "r"
# followed by "at"
pattern = "[cbr]at"
text = "cat bat rat dog sat"
match_list = re.findall(pattern, text)
print(f"{text} + {pattern} -> {match_list}")

# You can use the dash character "-" to specify ranges, for example, you can use
# [0-2] to match integers from 0 to 2 (inclusive). [0-9a-fA-F] will, e.g., match
# a characters of a hexadecimal number (it will match all integers from 0 to 9
# and all characters from "a" to "f", both lower and upper case).
pattern = "[0-5a-c]at"
text = "cat bat rat dog 3at 7at"
match_list = re.findall(pattern, text)
print(f"{text} + {pattern} -> {match_list}")

# You can use "^" to negate character patterns. [^0-9] will match all those
# characters that are not 0 to 9, i.e., non-numerical characters.
pattern = "[^0-9]"
text = "a1b2c3"
match_list = re.findall(pattern, text)
print(f"{text} + {pattern} -> {match_list}")

# There exist predefined groups of characters, such as "\d" for any digits or
# "\D" for any non-digit characters. Important: You need to write "\" in the
# string, meaning you need to escape the "\" in Python itself ("\\") or use a
# raw string. Other useful predefined groups of characters are "\s" for
# whitespace characters, "\S" for non-whitespace characters, "\w" for word
# characters (equivalent to [a-zA-Z0-9_]) or "\W" for non-word characters.
pattern = r"\d"  # equivalent to "\\d"
text = "a1b2c3"
match_list = re.findall(pattern, text)
print(f"{text} + {pattern} -> {match_list}")

# The dot character "." matches any character (except newline characters):
pattern = ".m."
text = "some sample string\nmy new line"
match_list = re.findall(pattern, text)
print(f"{text} + {pattern} -> {match_list}")

#
# Searching for alternative patterns
#

# The "|" character can be used to search for alternative patterns:
pattern = "[bcr]at|dog"
text = "cat bat rat dog sat"
match_list = re.findall(pattern, text)
print(f"{text} + {pattern} -> {match_list}")

# This can be combined with the parentheses () to group search patterns:
pattern = "(ai|ml) student"
text = "this matches ai students and ml students and returns 'ai' or 'ml'"
match_list = re.findall(pattern, text)
print(f"{text} + {pattern} -> {match_list}")

#
# Repetitions in patterns
#

# "*" will match any number of repetitions (also 0) and is by default greedy,
# i.e., it searches for the largest pattern (=match as many times as possible).
# "+" will match 1 or more repetitions and is also by default greedy. "?" will
# match 0 or 1 times and is also by default greedy.
pattern = "([bcr]at|dog)*"
text = "catcat batbat ratratrat dog"
match_object = re.search(pattern, text)
print(f"{text} + {pattern} -> {match_object.group()}")

# To be non-greedy, you must add the suffix "?" (in addition to "*", "+" or "?")
pattern = "([bcr]at|dog)+?"
text = "catcat batbat ratratrat dog"
match_object = re.search(pattern, text)
print(f"{text} + {pattern} -> {match_object.group()}")

# Note that "*" and "?" can also mean 0 repetitions, which, in the example here,
# also means that our pattern can lead to empty matches if the text starts with
# something different from what we specified in our pattern:
pattern = "([bcr]at|dog)*"
text = "Xcatcat batbat ratratrat dog"  # Additional "X" at the start
match_object = re.search(pattern, text)  # Will be an empty match (but not None)
print(f"{text} + {pattern} -> {match_object.group()}")

# "findall" will only report the list of captured groups in the last iteration:
pattern = "([bcr]at|dog)+"
text = "catcat batbat ratratrat dog"
match_list = re.findall(pattern, text)
print(f"{text} + {pattern} -> {match_list}")

# If you need to get all groups, you can wrap the repeated group within another
# capturing group:
pattern = "(([bcr]at|dog)+)"
text = "catcat batbat ratratrat dog"
match_list = re.findall(pattern, text)
print(f"{text} + {pattern} -> {match_list}")

# Or you can specify a non-capturing group with "(?:XYZ)" if you are not
# interested in the individual iteration captures but only in the entire match:
pattern = "(?:[bcr]at|dog)+"
text = "catcat batbat ratratrat dog"
match_list = re.findall(pattern, text)
print(f"{text} + {pattern} -> {match_list}")

# Or you use "re.finditer" to get "Match" objects to allow full control.

# You can use curly braces {} to specify the exact number of occurrences:
# {x}   -> Match exactly x times
# {,x}  -> Match between 0 and x (inclusive) times
# {x,}  -> Match x times or more
# {x,y} -> Match between x and y (inclusive) times
pattern = "(ab){3}"
text = "ab abab ababab abababab"
for i, m in enumerate(re.finditer(pattern, text)):
    print(f"{i + 1}. match: {m.group()} (Pos: {m.start()}-{m.end()})")

#
# Lookahead and lookbehind
#

# You can also specify a lookahead token in a regular expression which does not
# consume the input but just "looks" at it. There can be a positive lookahead or
# a negative lookahead. They are useful when creating patters like "only match
# if the following is XYZ" or "only match if it is not followed by XYZ".
# Positive lookahead syntax is "(?=XYZ)" and negative lookahead syntax is
# "(?!XYZ)". Analogously, there also exists a positive lookbehind and a negative
# lookbehind, which look at the characters preceding a potential match. They are
# useful when creating patterns like "only match if the preceding is XYZ" (or
# "is not XYZ", respectively). Positive lookbehind syntax is "(?<=XYZ)" and
# negative lookbehind syntax is "(?<!XYZ)".

# Negative lookahead: Only match "ab" if it is not followed by a digit
pattern = r"ab(?!\d)"
text = "ab12 abc 13ab ab1"
for i, m in enumerate(re.finditer(pattern, text)):
    print(f"{i + 1}. match: {m.group()} (Pos: {m.start()}-{m.end()})")

# Positive lookbehind: Only match "ab" if it is preceded by a digit
pattern = r"(?<=\d)ab"
text = "ab12 abc 13ab ab1"
for i, m in enumerate(re.finditer(pattern, text)):
    print(f"{i + 1}. match: {m.group()} (Pos: {m.start()}-{m.end()})")

#
# Larger example
#

# Extract dates where each date is expected to be structured in the following
# format:
#     day -> one or two digits
#     month -> one or two digits
#     year -> if specified (optional), four digits
#     separating characters are either "." or ";"
# Checking whether it is a semantically valid date (correct number of day in
# specified month, etc.) can be done afterwards. Here, we simply extract the
# dates with the corresponding regular expression.

text = "Four dates: 1) 01.12.2021, 2) 1;12;2021, 3) 24.01; 4) 2.2;1990"
# Pattern explanation:
pattern = r"\d{1,2}[\.;]\d{1,2}[\.;](\d{4})?"
#           ↑      ↑    ↑      ↑    ↑
#           │      │    │      │    Optional (0 or 1 times) group of 4 digits
#           │      │    │      │
#           │      │    Same pattern as before (up to 2 digits with separator)
#           │      │
#           │      Separation can either be a "." (must escape since it is a
#           │      special character in a regex string!) or a ";"
#           │
#           Look for 1 or 2 digits
for m in re.finditer(pattern, text):
    print(m.group())

#
# Substituting and splitting strings
#

# There are more functions available via the "re" module, such as "split" and
# "sub" (substitution). See the official documentation if you want to know more.
