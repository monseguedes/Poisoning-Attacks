"""
Example of code for animated plot.
"""

import random


def improve_points(points):
    new_points = []
    for point in points:
        new_point = point + random.uniform(-0.1, 0.1)
        if new_point < 0:
            new_point = 0
        elif new_point > 1:
            new_point = 1
        new_points.append(new_point)
    return new_points

import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.style.use("seaborn")

# Generate some initial points
points = [random.random() for i in range(20)]

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Define a function to update the plot with new data
def update(frame):
    global points
    points = improve_points(points)
    ax.clear()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] * ((len(points) // 5) + 1)
    for i in range(0, len(points), 5):
        ax.scatter([i]*5, points[i:i+5], c=colors[i//5])
    mean = sum(points) / len(points)
    ax.axhline(mean, color='black', linestyle='--', label=f"Mean = {mean:.2f}")
    ax.legend(loc= "upper right")
    ax.set_ylim(0, 1)
    ax.set_title(f"Iteration {frame}")

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=range(50), interval=100)

# Show the plot
plt.show()
