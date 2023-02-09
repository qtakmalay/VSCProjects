# -*- coding: utf-8 -*-
"""
Author -- Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 08.08.2022

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

import matplotlib.pyplot as plt
import numpy as np

# Especially for matplotlib, make sure you have the online documentation ready
# that will help you immensely when completing these tasks, or, more generally,
# when creating your plots. Often, matplotlib functions can be very complex with
# a huge number of settings that control various aspects of the resulting
# output. Looking up the corresponding documentation (and examples) is thus
# highly encouraged and recommended.

#
# Task 1
#

# Using the provided data "x" and "y" below, create the following plot:
#   > Red dashed line plot with star symbol markers of size 10
#     https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
#   > x-ticks same as "x"
#   > 10 evenly separated y-ticks ranging from the minimum of "y" to the maximum
#     of "y"
#   > y-limits are the minimum of "y" and the maximum of "y"
rng = np.random.default_rng(seed=0)
x = np.arange(1, 11)
y = rng.random(size=len(x))

# Your code here #


#
# Task 2
#

# Using the provided data "class1", "class2" and "class3" below, create the
# following plot:
#   > 3-part scatter plot where each part shows the data from one of the classes
#     https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html
#   > A legend shows these three parts as "class 1", "class 2" and "class 3"
#   > Axis title is set to "2D data showing three classes"
rng = np.random.default_rng(seed=0)
class1 = rng.normal(loc=1, scale=0.2, size=(20, 2))
class2 = rng.normal(loc=2, scale=0.5, size=(20, 2))
class3 = rng.normal(loc=1, scale=2.5, size=(30, 2))

# Your code here #


#
# Task 3
#

# Using the provided data "speed" and "threshold" below, create the following
# plot:
#   > Line plot showing the speed (= y-axis, x-axis is determined automatically)
#   > Horizontal line at y=threshold that stretches along the entire x-axis
#     https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hlines.html
#     or even simpler:
#     https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axhline.html
#   > Area above this line and below the speed curve must be filled with
#     color="red" and with alpha=0.2
#     https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.fill_between.html
rng = np.random.default_rng(seed=0)
speed = np.sin(np.linspace(0, np.pi, 100)) + rng.uniform(-0.4, 0.4, 100)
threshold = 0.6

# Your code here #


#
# Task 4
#

# Using the provided data "size" and "Z" below, create the following plot:
#   > Figure of size (5, 5)
#   > Show "Z" as image (choose interpolation and colormap as you see fit).
#     Note: By default, the x-range and y-range will be set based on the size
#     of the displayed image. You can change this with the parameter
#     extent=[left, right, bottom, top] to assign new value ranges (you will
#     need this to correctly assign the x-ticks and y-ticks below). Also, set
#     the parameter origin="lower", since the "Z" values will be incorrectly
#     displayed otherwise. The default is "upper", meaning that the starting
#     coordinate will be the upper left corner (this matches indexing
#     conventions in math and computer graphics), but we want the starting
#     coordinate to be the lower left corner (since we are dealing with a
#     Cartesian coordinate system here).
#     https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html
#     https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
#     https://matplotlib.org/stable/tutorials/colors/colormaps.html
#     Additional note: For this particular case, the method
#     "ax.pcolormesh(x, y, Z)" would have also been a possibility, which
#     directly allows us to specify the coordinates in conjunction with the "Z"
#     values (i.e., no messing around with extent and origin):
#     https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pcolormesh.html
#     Here is a good post which discusses the pros and cons of each method:
#     https://stackoverflow.com/questions/21166679/when-to-use-imshow-over-pcolormesh
#   > 11 evenly separated x-ticks and y-ticks ranging from "size" to "-size"
#   > Rotation of x-tick labels by 90 degrees
#     https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xticks.html
#   > Tight figure layout (to avoid that the rotated labels are cut off)
# Save the figure as a PDF (vector graphics).
size = 2
x = y = np.linspace(-size, size, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(Y + 0.8) ** 2 - (X + 0.4) ** 2)

# Your code here #


#
# Task 5
#

# Using the provided data "student_ids", "exercise_ids", "min_points",
# "max_points", "points" and "nbins", create the following plot:
#   > Figure of size (9, 4) with two columns, i.e., two axis subplots (left and
#     right)
#   > Figure title is set to "Student exercise statistics"
#   > Tight figure layout
#   > Left subplot:
#       - Show points as image with "min_points" as minimum image value and
#         "max_points" as maximum image value. Set the color map to "RdYlGn"
#       - x-tick labels are the "exercise_ids" (hint: use a range from 0 to
#         len(exercise_ids) as x-ticks). Rotation by 90 degrees.
#       - y-tick labels are the "student_ids" (hint: use a range from 0 to
#         len(student_ids) as y-ticks)
#       - Optional: Add a color bar to the right of the image
#         https://matplotlib.org/stable/gallery/axes_grid1/simple_colorbar.html
#         https://matplotlib.org/stable/gallery/axes_grid1/demo_colorbar_with_axes_divider.html
#   > Right subplot:
#       - Histogram showing the point distribution of all exercises (combined
#         from all students). Number of bins is "nbins", the value range is from
#         "min_points" to "max_points". Color of bars is "#7C67D4", edge color
#         of bars is "black" with a line width of 0.5
#         https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html
#       - x-tick labels are the bin widths of the format "[from, to)" except for
#         the last bin, which has the format "[from, to]". Rotation of 45
#         degrees. Hints: Use the return value of "ax.hist" to grab the limits
#         of the created bins. Make use of the "Text" keyword args
#         "horizontalalignment" (or "ha") and "rotation_mode" when specifying
#         the x-ticks and labels
#         https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xticks.html
#         https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text
#       - Grid on y-axis
#       - Label on y-axis is set to "Number of exercises"
rng = np.random.default_rng(seed=0)
student_ids = [f"k{i:02}" for i in range(1, 11)]
exercise_ids = [f"ex{i}" for i in range(1, 16)]
min_points = 0
max_points = 100
points = np.empty((len(student_ids), len(exercise_ids)), dtype=np.int8)
means_and_stds = [(75, 20), (95, 5), (50, 15), (80, 10), (80, 15),
                  (70, 40), (60, 1), (50, 10), (65, 15), (75, 10)]
for i, (mean, std) in enumerate(means_and_stds):
    student_points = rng.normal(mean, std, len(exercise_ids)).astype(np.int8)
    points[i] = np.clip(student_points, min_points, max_points)
nbins = 10

# Your code here #

