import jax.numpy as np

Lx = 30
dim = 500
dx = 2 * Lx / dim
dt = 0.0001
Lt = 3
dimT = int(Lt / dt)

xMin = -10
xMax = 10

g = 1
ns = 1
hbar = 1
m = 1
