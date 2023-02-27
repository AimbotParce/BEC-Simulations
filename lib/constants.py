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
tMax = 10

# Number of space and time steps
xCount = int((xMax - xMin) / dx)
tCount = int((tMax - tMin) / dt)

# Constants
g = -100
ns = 1
mu = g * ns
hbar = 1
m = 1
velocity = 0


# Plot constants
plotPause = 0.001
plotFPS = 1 / plotPause
plotStep = 100
plotYMax = 2
plotYMin = -2
