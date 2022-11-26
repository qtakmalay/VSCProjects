# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 26.07.2022

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This  material,  no  matter  whether  in  printed  or  electronic  form,
may  be  used  for personal  and non-commercial educational use only.
Any reproduction of this manuscript, no matter whether as a whole or in parts,
no matter whether in printed or in electronic form, requires explicit prior
acceptance of the authors.

################################################################################

Example module.
"""


def add(*args):
    """This function adds up any arbitrary number of arguments recursively."""
    if not args:
        return 0
    return args[0] + add(*args[1:])


print("This code will be executed when this file is imported")

if __name__ == "__main__":
    print("This code will not be executed when this file is imported")
    print(f"Example: add(1, 2, 3, 4) -> {add(1, 2, 3, 4)}")
