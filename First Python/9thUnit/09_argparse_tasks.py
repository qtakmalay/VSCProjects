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

import argparse
import numpy as np
# a = argparse.ArgumentParser()
# a.add_argument("--int_number", type=int, help="required int")
# args = a.parse_args()
# my_int_number3 = args.int_number
# print(my_int_number3)
class C:
    a = 2
    b = 3
    c = 1
    def fun(self, a,b):
        a = C.c
        c = a * 2 + b
        return C.c
d = C()
print(d.fun(2,3))