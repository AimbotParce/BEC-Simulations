import jax.numpy as np

# Space and time steps [m] and [s]
dx = 0.1
dt = 0.001
r = dt / (dx**2)

# Space interval [m]
xMin = -10
xMax = 10

# Time interval [s]
tMin = 0
tMax = 3

# Number of space and time steps
xCount = int((xMax - xMin) / dx)
tCount = int((tMax - tMin) / dt)

# Constants
g = 10
ns = 1
mu = g * ns
hbar = 1
m = 1
velocity = 1


# Plot constants
plotStep = 20  # Plot every plotStep time steps
plotYMax = 2
plotYMin = -2
plotPause = 0.001
