import jax.numpy as jnp

from lib.waveFunctions import brightSoliton as waveFunction

# Override constants
g = -10


def V(x, t, constants):
    return -1 / 2 * x**2
