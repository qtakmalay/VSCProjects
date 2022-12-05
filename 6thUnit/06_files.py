# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 29.07.2022

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

################################################################################

In this file, we will learn how to open, close and read from files. We will also
see that the glob module is handy for finding files in directories.
https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
"""

################################################################################
# Files
################################################################################

# Files can be opened with the "open" function, which returns a file handle. A
# file handle is an object representing the file but not the content of the
# file. The syntax of the "open" function is
# filehandle = open(filepath_and_name, mode)
# with "mode" being (only some are listed here)
#   "r" for read only (only reading from file, no write access),
#   "w" for (over-)write (write to file and overwrite file if it exists), and
#   "a" for append (write access where writing will append at the end of the
#   file if file already exists).
# If you want to handle a file not as text but as bytes, use "rb", "wb", "ab",
# otherwise, the file is interpreted as text and encoding/decoding is applied.
# By default, Python will open the file using the platform-dependent encoding
# (e.g., "cp1252" on Windows). You can explicitly specify which encoding to use
# with open(..., encoding=xyz), e.g., xyz = "utf-8". For more details, see
# https://docs.python.org/3/library/functions.html#open

# This will open the file "my_module.py" in read mode (default mode):
f = open("my_module.py", "r")  # Equal to: f = open("my_module")

# Now, we can use the file handle to, e.g., read its entire content as string
content = f.read()
print(content)

# And finally, we need to close the file, otherwise it might be blocked
f.close()

#
# Reading from files
#

# Python encourages you to write "safe" code. Needing to close the file by hand
# is problematic (e.g., if exceptions occur and close() is not executed).
# Therefore, it is strongly recommended to use the following "with" statement,
# which closes the file automatically, thereby also ensuring that any not yet
# written data (if we were writing to the file) is flushed:
with open("my_module.py", "r") as f:
    # We now have "f" as file handle available until the code block is left
    print(f.read())
# Here, the file handle "f" would be closed and not be available anymore for
# further file processing

# We can iterate over lines in a file using a for loop (using common line breaks
# such as "\n").
with open("my_module.py") as f:
    # The for loop automatically reads the file content line by line:
    for line in f:
        # "line" already contains a line break "\n" at the end, so no need to
        # print two line breaks "\n"
        print(line, end="")

# Iterating over a file line by line is slower than reading the whole content at
# once. But it allows us to use less memory since we only have to hold parts of
# the file in our memory. In case that even a single line is too big and does
# not fit into our memory, we can parameterize "read" to get even more control:
with open("my_module.py") as f:
    # This continuously reads at most 10 characters (in text mode). Repeated
    # calls read the data sequentially. The ":=" here is called an assignment
    # expression which makes this code very concise. For more details, see
    # https://docs.python.org/3/reference/expressions.html?#assignment-expressions
    while chars := f.read(10):
        print("Chunk:")
        print(chars)

#
# Writing to files
#

# The file handle allows us to write to a file using the syntax
# f.write("some_string\n")
# which will write a string, e.g., "some_string", to the file. "write" does not
# add a newline character at the end by default.

# If we use "w" mode, we will either overwrite or create a file and write to it.
with open("some_file.txt", "w") as f:
    f.write("This overwrites an existing file\nor creates a new one with this text!\n")
    f.write("This adds another line to the file without a newline character at the end.")
    f.write(" ... but now we add a newline character.\n")

# If we use "a" mode, we will either append to an existing file or create a new
# file and write to it.
with open("some_file.txt", "a") as f:
    f.write("And here this line is appended.\n")
    f.write("And another line is appended.\n")

# We can also use the "print" function to write to files by passing the
# file handle as argument "file". Keep in mind that the "print" function does
# add a newline character at the end by default (however, this can be changed by
# setting the keyword argument "end").
with open("some_file.txt", "a") as f:
    print("And another line via print()!", file=f)
    print("Note that print adds a newline by default.", file=f)

#
# Increasing compatibility
#

# Depending on your operating system (OS), the standard used to interpret your
# text file contents might be different. For example, on Unix-based systems, the
# Line-Feed (LF) character "\n" will be by default interpreted as a line break.
# On Windows, the Carriage-Return (CR) "\r" character followed by the LF
# character are used as default (="\r\n"). Python will automatically use the
# universal newlines mode, which translates all "\r", "\n" and "\r\n" to "\n"
# when reading and uses the operating system default to replace the "\n"
# character when writing. However, we can parameterize the "open" function to
# treat newline characters exactly as we want (although the default, the
# universal newlines mode, will suffice in the majority of the cases).

# Write file in universal newlines mode and translate "\n" to the OS default:
with open("some_file.txt", "w") as f:  # Equal to argument: newline=None
    print("Here we \r\n use different \n newline \r characters", file=f)

# Write file without translating "\n":
with open("some_file.txt", "w", newline="") as f:  # Equal to: newline="\n"
    print("Here we \r\n use different \n newline \r characters", file=f)

# Read file in universal newlines mode and translate line breaks (can be any of
# "\r", "\n", or "\r\n") to "\n":
with open("some_file.txt") as f:  # Equal to argument: newline=None
    fc = f.read()  # Take a look at "fc" in the debugger

# Read file without translating line breaks:
with open("some_file.txt", newline="") as f:
    fc = f.read()  # Take a look at "fc" in the debugger

# Write file in Windows standard (replace "\n" by "\r\n" when writing):
# Note: If you are already on Windows, there is, of course, no difference.
with open("some_file.txt", "w", newline="\r\n") as f:
    print("Here we \r\n use different \n newline \r characters", file=f)
    
with open("some_file.txt", newline="") as f:
    fc = f.read()  # Take a look at "fc" in the debugger


#
# Useful methods
#

# Reading lines into a list of strings (every line ends with "\n", except
# potentially for the last line if the file ends with this line and there
# is no empty new line at the bottom)
with open("some_file.txt", "r") as f:
    lines = f.readlines()

# Writing an iterable (e.g., a list) of strings. This is the same as consecutive
# calls to "f.write(...)", which also means that no "\n" characters will be
# written by default.
with open("some_file.txt", "w") as f:
    f.writelines([
        "This is line1",
        "---which directly continues here\n"
        "This will be on a new line"
    ])

#
# Character encoding/decoding
#

# A string/text in Python is represented with Unicode characters (each character
# is assigned a unique code point, see https://unicode-table.com/en/ for a fancy
# look-up table). An encoding specifies how such code points should be
# translated into a sequence of bytes. In the above examples, we only used ASCII
# characters (see https://en.wikipedia.org/wiki/ASCII for more details), where
# we typically do not encounter any issues, since many character encodings like
# UTF-8 or cp1252 (the default on Windows) are compatible with ASCII. However,
# if you intend to use more special characters, choosing (and sticking with) an
# encoding is important, since otherwise, you might end up with incorrect data.
# UTF-8 is highly recommended, as it is one of the most often used encodings,
# can encode any Unicode code point and has several other benefits. For more
# details, see
# https://docs.python.org/3/howto/unicode.html

# This is a string that contains characters where different encodings will yield
# different results. Since the current Python file here uses UTF-8 encoding, we
# can write these special characters literally (alternatively, we can provide
# the Unicode code point with an escape sequence "\u" or "\U" followed by the
# code point as a 16bit or 32bit hexadecimal value, respectively):
special_chars = "Special characters: ä é ô µ シ ᚠ 🐈 \u2205 \U0001F44D"

# Store the above string using the UTF-8 encoding:
with open("some_file.txt", "w", encoding="utf-8") as f:
    f.write(special_chars)

# If we use the same encoding to read the data, everything works as expected:
# (Depending on the console you are using to run this code, you might have to
# change its encoding to UTF-8 beforehand)
with open("some_file.txt", "r", encoding="utf-8") as f:
    print(f.read())

# If we use a different encoding (one that interprets these special characters
# in a different way), we get an incorrect output or potentially even an error.
# Here, we use "cp1252", which is the default on Windows:
with open("some_file.txt", "r", encoding="cp1252", errors="replace") as f:
    print(f.read())


################################################################################
# os - using the operating system in Python
################################################################################

# The "os" module allows you to use the operating system terminal/command line
# and call operating system functions independently of the operating system. It
# also provides functions to make path handling independent of the operating
# system via the submodule "os.path". Both modules contain much more functions
# than shown here, especially "os", since we only focus on a few selected
# functions for dealing with files and directories. For more details, see
# https://docs.python.org/3/library/os.html
# https://docs.python.org/3/library/os.html#files-and-directories
# https://docs.python.org/3/library/os.path.html

import os

# Get the current working directory:
print(os.getcwd())

# Create directories (if only a relative path (and not an absolute path) is
# specified here, the current working directory is used):
os.makedirs("new_directory", exist_ok=True)

# Rename files or directories:
os.rename("new_directory", "my_new_new_directory")

# Remove directories (must be empty to be removed):
os.removedirs("my_new_new_directory")

# Remove files:
os.remove("some_file.txt")

# List the contents of a directory (if no path is specified, the current working
# directory is used again by default):
print(os.listdir())

#
# os.path
#

# Using functions from "os.path", we can handle and process file paths and names
# independently of the operating systems.

# Join multiple paths together:
path = os.path.join("some", "directory", "filename.py")
print(path)  # Might look different on different OS

# Split the path into a 2-tuple, where the second tuple entry is the last path
# component and the first tuple entry the remainder, i.e., everything up to this
# last path component:
head, tail = os.path.split(path)
print(head, tail)

# Get the (parent) directory name:
print(os.path.dirname(path))  # Equal to: os.path.split(path)[0]

# Get the name of file itself:
print(os.path.basename(path))  # Equal to: os.path.split(path)[1]

# Get the extension of a file:
root, ext = os.path.splitext(path)
print(root, ext)

# Check if a file/directory exists:
print(os.path.exists(path))

# Check if a path refers to an existing file or an existing directory:
print(os.path.isfile(path))  # False, since "filename" does not exist
print(os.path.isfile("my_module.py"))
print(os.path.isdir("."))  # "." refers to the current working directory

# Get the absolute path from a given path, using the current working directory:
abs_filename = os.path.abspath(path)
print(abs_filename)

# Get the relative path from a given path, using the current working directory
# by default, or some other directory (parameter "start") as start directory:
print(os.path.relpath(abs_filename))  # Use current working directory as start


################################################################################
# glob - searching for files in (sub)directories
################################################################################

# The "glob" module allows you to (recursively) search for files and folders in
# directories and subdirectories, including some pattern matching, which can be
# convenient if you do not want to use "os.listdir" and manual file handling.
# For more details, see
# https://docs.python.org/3/library/glob.html

import glob

# The syntax for using glob for searching for files is
# found_files = glob.glob("search_pattern")
# where "search_pattern" is a string that decides what file patterns should be
# searched for. We can, for example, use the "*" character, which matches
# anything within a folder (excluding files in subfolders).

# Return a list of all files and directories in the current working directory
# (files/directories starting with "." are not included in this pattern, this
# would have to be explicitly specified, e.g., ".*" to search for such files):
print(glob.glob("*"))

# Search the current working directory for files that end in ".py":
print(glob.glob("*.py"))

# To search for files and folders recursively (=in all subdirectories), you can
# use the string "**" and set the argument recursive=True. "**" will then match
# all subdirectory names:
print(glob.glob("**", recursive=True))

# Search in a directory and in all subdirectories for files ending in ".py":
print(glob.glob(os.path.join("**", "*.py"), recursive=True))

# Search in folder "some_folder" and in all its subdirectories for files ending
# in ".py":
print(glob.glob(os.path.join("some_folder", "**", "*.py"), recursive=True))

# The list of files returned by the function "glob" does not have any specific
# order. You need to sort the list yourself afterwards:
found_files = glob.glob("*", recursive=True)
found_files.sort()


################################################################################
# pickle/dill - saving Python objects to files
################################################################################

# The "pickle" module allows you to save Python objects to files. The "dill"
# module provides additional functions to "pickle" and can handle more object
# types. It works for many Python objects but can lead to errors when trying to
# pickle Python classes or exotic objects (e.g., frame, generator, traceback).
# In doubt, always verify if your object type was saved correctly by loading the
# object in a different Python session and checking it. For more details, see
# https://docs.python.org/3/library/pickle.html
# https://dill.readthedocs.io/en/latest/ (not installed by default)

# import pickle  # This is the original "pickle" module, but we will use "dill"
import dill as pickle  # "dill" has the same interface as "pickle"

some_dict = dict(a=1, b=2, c=3)
some_list = [1, 2, 3]
some_tuple = (4, 5, 6)


def some_function(x):
    return x * 5


# If we want to store (=to pickle) multiple objects, it is recommended to use a
# dictionary containing our objects:
my_objects = dict(
    some_dict=some_dict,
    some_list=some_list,
    some_tuple=some_tuple,
    some_function=some_function
)

# To write our pickled objects to a file (serialization), we need to open it in
# byte mode "b" (i.e., no encoding is used, just the raw bytes are stored):
with open("my_objects.pkl", "wb") as f:
    # This will store our pickled "my_objects" dictionary o the specified file
    pickle.dump(my_objects, f)

# Now, we can load data from this file (deserialization), again using byte mode:
with open("my_objects.pkl", "rb") as f:
    # This will load the objects from the file into the dictionary "data"
    data = pickle.load(f)

print("data['some_dict']", data["some_dict"])
print("data['some_list']", data["some_list"])
print("data['some_function'](data['some_tuple'])", data["some_function"](data["some_tuple"]))


################################################################################
# CSV files
################################################################################

# Comma separated values (csv) files are a common data file format if the data
# shall be stored in text format. In CSV files a special character, typically a
# comma ",", semicolon ";", space " " or tab "\t", is used as separator between
# columns in the file. A CSV file is still just a text file, so using the
# separator is not enforced, and we could still print normal text to it.

# There exist dedicated modules to handle such files, e.g., the "csv" module,
# which allows you to read from and write to CSV files in a flexible and
# convenient manner. For more details, see
# https://docs.python.org/3/library/csv.html

# Modules such as "numpy" or "pandas" have their own built-in functions for
# reading/writing CSV-like files. For example, pandas uses a C-backend and is
# much faster than pure Python code. If you are dealing with bigger data files,
# this can provide a large speed-up. Link to pandas CSV reading function:
# https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
