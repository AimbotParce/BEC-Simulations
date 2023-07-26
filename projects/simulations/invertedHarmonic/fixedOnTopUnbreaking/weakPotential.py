import jax.numpy as jnp

from lib.waveFunctions import brightSoliton as waveFunction

g = -100


def V(x, t, constants):
    return -1 / 2 * x**2
