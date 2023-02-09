# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
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

In this file, we will learn how to create plots in Python using "matplotlib".
"""

################################################################################
# matplotlib - plotting all kinds of stuff in Python
################################################################################

# The module "matplotlib" includes plotting functions for all kinds of plotting.
# The large functionality and flexibility of this module unfortunately comes at
# the price of partly over-complicated function calls and parameters. Plotting
# with matplotlib without Internet connection is either brave or foolish. Note
# that this file only captures a tiny part of the functionality of matplotlib.
# There is much more that you can do, especially considering all the fine-tuning
# which can, however, quickly get a bit tricky. Useful links:
# Homepage: https://matplotlib.org/
# Tutorials: https://matplotlib.org/stable/users/index.html
# Main tutorial: https://matplotlib.org/stable/tutorials/introductory/usage.html
# API documentation: https://matplotlib.org/stable/api/index.html

# Important: matplotlib uses system backends to do the plotting. Different OS
# offer different backends, some of which are specialized for certain tasks. If
# you run into performance issues (e.g., 3D plots or creating videos), you may
# want to switch to a more suitable backend.
# https://matplotlib.org/stable/users/explain/backends.html

# We will now work through parts of the tutorial together. Depending on your
# needs or interests, consult the matplotlib plot types and (example) gallery
# https://matplotlib.org/stable/plot_types/index.html
# https://matplotlib.org/stable/gallery/index.html
# for examples and code snippets that you can use.

import matplotlib.pyplot as plt
import numpy as np

# Choose interactive mode off or on
plt.ioff()  # Show figures when explicitly stated ("plt.show")
# plt.ion()  # Show figures immediately (no "plt.show" needed)

# Data for plotting
t = np.arange(0, 100)

#
# Basic line plot
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
#

fig, ax = plt.subplots()  # This creates a figure and axis/axes handle
# The axis is the object where we can plot something. The figure is the window
# that contains axes. There are many methods that you can call on an axis object
ax.plot(t)  # Add a plotted line to the axis (x-values are assumed to go from 0 to len(t) - 1)
ax.set_xlabel("Label for x")  # Set labels for x-axis and y-axis (don't confuse with the plot axis "ax")
ax.set_ylabel("Label for y")
ax.set_title("Title of the axis")  # Set a title for this axis
ax.grid()  # Add a grid to our axis (default: x-axis grid and y-axis grid)
fig.suptitle("Title of the figure", fontsize=16)  # Set a super-title for the figure
fig.savefig("test.pdf")  # We can save the figure to a file
fig.savefig("test.png")  # The file extension is interpreted by matplotlib
# Visualizes the created plot (and consumes it, i.e., cannot call "show" twice).
# If block=True, the execution of the program is paused until the window is
# closed. In non-interactive mode ("plt.ioff"), blocking is the default
# behavior. In interactive mode ("plt.ion"), block=False is the default behavior
plt.show(block=True)  # Can be manually set independently of the (non-)interactive mode
# We can close the figure window programmatically as well. If the above "show"
# is blocking, then this line here has no (visual) effect, since the code
# execution only continues after the user closed the window. However, there is
# still a program state effect, since simply closing the (GUI) figure window
# does not automatically release all resources (there are still internal
# references kept). Only explicitly calling this "close" function will release
# all resources (memory). See the last paragraph at
# https://matplotlib.org/stable/tutorials/introductory/pyplot.html#working-with-multiple-figures-and-axes
plt.close(fig)

#
# Overlapping/Joined/Combined plots
#

x1 = range(1, 11)
y11 = [i if i % 2 == 0 else -i for i in x1]
y12 = [i ** 2 for i in x1]
x2 = [2, 5, 6, 13]
y2 = [0, 10, 11, 5]
_, ax = plt.subplots(figsize=(10, 3))  # Many settings available (see documentation)
ax.plot(x1, y11)
ax.plot(x1, y12, linestyle="--")  # Many settings available (see documentation)
ax.plot(x2, y2, color="red", marker="o")
plt.show()

#
# Multiple subplots
#

fig, ax = plt.subplots(2, 3)  # "ax" is a 2x3 array containing the axes
ax[1, 1].plot(t)  # Select an axis and plot data "t"
ax[0, 1].plot(t)  # Select another axis and plot data "t" again
ax[0, 1].plot(-t)  # Select another axis and plot data "t * (-1)"
ax[0, 0].plot(t, label="data t")  # Plot "t" again and add a label (will add a legend entry)
ax[0, 0].plot(-t, label="data -t")  # Plot "-t" again and add a label (will add a legend entry)
ax[0, 0].legend()  # Create a legend for the labeled plots
fig.tight_layout()  # Automatically tweak spacings to prevent clipping of labels
fig.savefig("subplots.png")
plt.show()

#
# Histogram plot
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html
#

rng = np.random.default_rng(seed=0)
a = rng.normal(size=500, loc=100, scale=15)  # Some random data
fig, ax = plt.subplots()
n, bins, patches = ax.hist(a, bins=10, density=True)  # The histogram of the data (showing density instead of counts)
ax.set_xlabel("Random data")
ax.set_ylabel("Probability density")
ax.set_title(r"Histogram example: $\mu=100$, $\sigma=15$")
fig.savefig("histogram.png")
plt.show()

#
# Scatter plot
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html
#

rng = np.random.default_rng(seed=0)
x, y = rng.uniform(size=50), rng.uniform(size=50)
ticks = np.linspace(0, 1, 11, endpoint=True)
_, ax = plt.subplots()
ax.scatter(x, y)
ax.scatter(x, x*y, marker="x")
ax.set_xticks(ticks)  # Explicit x-ticks setting
ax.set_yticks(ticks)
ax.set_xlim((0, 1))  # Explicit x-limit setting (range that is shown in the plot)
ax.set_ylim((0, 1))
plt.show()

#
# Bar plot
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html
#
import matplotlib.pyplot as plt
import numpy as np
grades = ["very good", "good", "satisfactory", "sufficient", "not sufficient", "not graded"]
counts = [12, 14, 10, 9, 3, 5]
# Colors can be defined in multiple ways:
# https://matplotlib.org/stable/gallery/color/color_demo.html
# https://matplotlib.org/stable/gallery/color/named_colors.html
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
cmap = plt.get_cmap("viridis")
bar_colors = ["g", "gold", "tab:cyan", (1.0, 0.5, 0.25), "#FF0000", cmap(0.5)]
#             ↑    ↑       ↑           ↑                 ↑          ↑
#             │    │       │           │                 │          % of colormap range
#             │    │       │           │                 RGB(A) hex string
#             │    │       │           RGB(A) tuple of floats (%)
#             │    │       standard tableau colors
#             │    CSS colors
#             base colors

_, ax = plt.subplots(figsize=(8, 4))
bar = ax.bar(grades, counts, color=bar_colors)
ax.bar_label(bar, counts, rotation=45)  # Add text labels above the bars
ax.grid(axis="y", linestyle="--")
ax.set_ylabel("Counts")
ax.set_ylim(0, max(counts) + 1.5)
plt.show()

#
# Box plots
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.boxplot.html
#

rng = np.random.default_rng(seed=26)
n_boxes = 4
data = [rng.normal(0, 3, size=50) for _ in range(n_boxes)]
_, ax = plt.subplots()
ax.boxplot(
    data,
    labels=[f"$r_{i}$" for i in range(n_boxes)],  # TeX support for fancy labels
    medianprops=dict(linestyle=":", color="b")  # Settings for the median line
)
plt.show()

#
# Plotting images
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html
#

rng = np.random.default_rng(seed=0)
pseudo_image = rng.uniform(size=(150, 150, 3))  # Create random RGB data
_, ax = plt.subplots()
ax.imshow(pseudo_image)
ax.set_xticks([])  # Remove x-axis ticks (also would remove an existing grid)
ax.set_yticks([])  # Remove y-axis ticks (also would remove an existing grid)
ax.set_title("Some pseudo image data")
plt.show()

#
# Reading image data
#

read_image_data = plt.imread("histogram.png")
# "read_image_data" is read-only and can be stored as a normal NumPy array as
# read_image_data = np.array(read_image_data)
_, ax = plt.subplots()
ax.imshow(read_image_data)
# Here is an alternative way of handling ticks and labels (would NOT remove any
# existing grids):
ax.tick_params(
    axis="both",  # Changes affect both x-axis and y-axis
    bottom=False,  # Hide ticks at the bottom edge
    left=False,  # Hide ticks at the left edge
    labelbottom=False,  # Hide labels at the bottom edge
    labelleft=False,  # Hide labels at the left edge
)
ax.set_title("The image data we just read")
plt.show()

#
# Scalar image data (1D data)
#

# Scalar image data (1D data, i.e., no 3D RGB or 4D RGBA image data) is by
# default visualized using a colormap (=mapping of value to color). There are
# many colormaps available: https://matplotlib.org/stable/tutorials/colors/colormaps.html
# By default, the values of the scalar image data are normalized to [0, 1],
# i.e., the lowest value will be assigned the darkest color and the highest
# value will be assigned the lightest color of the corresponding colormap.

# Let's create an 8-bit grayscale array, i.e., values in the range [0, 255],
# where 0 represents the lowest brightness (black) and 255 the highest (white)
grayscale = np.zeros((49, 49), dtype=np.uint8)  # 0 = lowest brightness (black)
grayscale[::2, ::2] = 130  # Set every 2nd pixel to medium brightness (medium gray)
_, ax = plt.subplots()
# Choose an appropriate colormap. Also, we set the minimum and maximum values
# ourselves, so matplotlib does not do the automatic normalization described
# above (this would normalize the lowest value 0 to 0 and the highest value 130
# to 1, but we know that the highest value is actually 255, so the normalization
# should be from 255 to 1 to get the correct grayscale image visualization).
ax.imshow(grayscale, cmap="gray", vmin=0, vmax=255)
ax.axis(False)  # Or axis("off"); disables all axis ticks, labels, grids and the entire border
ax.set_title("The image data we just created")
plt.show()


################################################################################
# Other modules for vision tasks (optional, in case of interest)
################################################################################

#
# seaborn - statistical data visualization
#

# Data visualization framework based on matplotlib which provides a rich
# collection of various statistical data graphs with a high-level interface.
# Works well with NumPy array and "DataFrame" objects from "pandas".
# https://seaborn.pydata.org/

#
# OpenCV - fast image processing
#

# Advanced module specialized on fast image/video frame processing. Setup
# might not be trivial but its performance is great.
# https://docs.opencv.org/master/d6/d00/tutorial_py_root.html

#
# Datashader - optimized large-scale plotting in Python
#

# Allows for fast large-scale plotting, such as scatter plots with millions
# of points or plotting pipelines.
# https://datashader.org/

#
# Videos/Animations
#

# Matplotlib provides a submodule for simple animations:
# https://matplotlib.org/stable/api/animation_api.html
# https://matplotlib.org/stable/gallery/index.html#animation
# Depending on the task, it might be faster/more stable to use the "ffmpeg"
# package (not a Python package!), which supports fast video- and audio
# editing on different OS: https://www.ffmpeg.org/

#
# Web interfaces for visualization (like Shiny in R)
#

# https://bokeh.pydata.org/en/latest/
# https://plotly.com/dash/
