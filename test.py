import matplotlib.pyplot as plt
import numpy as np

# Suppose you have the following data
x = np.array([1, 2, 3, 4, 5])
y1 = np.array([1.1, 2.2, 3.3, 4.4, 5.5])  # First set of y-values
y2 = np.array([5.5, 4.4, 3.3, 2.2, 1.1])  # Second set of y-values

# Create a figure and an axis
fig, ax1 = plt.subplots()

# Plot the first set of data on ax1
ax1.plot(x, y1, color='blue')
ax1.set_ylabel('Y1', color='blue')
ax1.tick_params('y', colors='blue')

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

# Plot the second set of data on ax2
ax2.plot(x, y2, color='red')
ax2.set_ylabel('Y2', color='red')
ax2.tick_params('y', colors='red')

plt.show()
