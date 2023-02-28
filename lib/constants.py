import jax.numpy as jnp

# Space and time steps [m] and [s]
dx = 0.1
dt = 0.005
r = dt / (dx**2)

# Space interval [m]
xMin = -15
xMax = 15

# Time interval [s]
tMin = 0
tMax = 20

# Number of space and time steps
xCount = int((xMax - xMin) / dx)
tCount = int((tMax - tMin) / dt)

# Constants
g = -1000
baseDensity = 1
chemicalPotential = g * baseDensity
hbar = 1
mass = 1
velocity = 0
healingLength = hbar / jnp.sqrt(2 * mass * chemicalPotential)


# Plot constants
plotPause = 0.001
plotFPS = 1 / plotPause
plotStep = 10
plotYMax = 2
plotYMin = -2
