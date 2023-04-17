import jax.numpy as jnp

from lib.waveFunctions import brightSoliton as waveFunction

velocity = 0
x0 = -1


def V(x, t, constants):
    return -1 / 2 * x**2
