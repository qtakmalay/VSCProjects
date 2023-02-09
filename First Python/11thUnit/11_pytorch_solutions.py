# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 09.08.2022

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

import numpy as np
import torch

#
# Task 1
#

# Use PyTorch to compute the derivative of "e" in
# e = a * d + b * d ** 2.0 + c * d ** 3.0
# w.r.t. input variable "a" and "d" for values of:
a = 2.0
b = -3.0
c = 4.0
d = -5.0

# Your code here #
a = torch.tensor(a, requires_grad=True)
b = torch.tensor(b, requires_grad=False)
c = torch.tensor(c, requires_grad=False)
d = torch.tensor(d, requires_grad=True)

e = a * d + b * d ** 2.0 + c * d ** 3.0
e.backward()
print(f"a.grad = {a.grad}")
print(f"d.grad = {d.grad}")


#
# Task 2
#

# Use PyTorch to compute the derivative of "e" in
# e = sum(a * d + b * d ** 2.0 + c * d ** 3.0)
# w.r.t. input variable "a" and "d". The input variables are arrays and the
# computation of the formula should be done element-wise, resulting in an output
# array of shape 100, which should then be summed up, resulting in a scalar. For
# computing the sum, you can use function "torch.sum".
a = np.linspace(-5.0, 5.0, num=100)
b = np.linspace(0.0, 1.0, num=100) ** 2
c = np.ones_like(a)
d = np.linspace(5.0, -5.0, num=100)

# Your code here #
a = torch.tensor(a, requires_grad=True)
b = torch.tensor(b, requires_grad=False)
c = torch.tensor(c, requires_grad=False)
d = torch.tensor(d, requires_grad=True)

e = torch.sum(a * d + b * d ** 2.0 + c * d ** 3.0)
e.backward()
print(f"a.grad = {a.grad}")
print(f"d.grad = {d.grad}")


#
# Task 3
#

# Perform Task 2 on the GPU and with the data type "torch.float32".
a = np.linspace(-5.0, 5.0, num=100)
b = np.linspace(0.0, 1.0, num=100) ** 2
c = np.ones_like(a)
d = np.linspace(5.0, -5.0, num=100)

# Your code here #
a = torch.tensor(a, requires_grad=True, dtype=torch.float32, device="cuda:0")
b = torch.tensor(b, requires_grad=False, dtype=torch.float32, device="cuda:0")
c = torch.tensor(c, requires_grad=False, dtype=torch.float32, device="cuda:0")
d = torch.tensor(d, requires_grad=True, dtype=torch.float32, device="cuda:0")

e = torch.sum(a * d + b * d ** 2.0 + c * d ** 3.0)
e.backward()
print(f"a.grad = {a.grad}")
print(f"d.grad = {d.grad}")
