# Random frequency
import jax.numpy as jnp

from lib.waveFunctions import brightSoliton as waveFunction

w0 = 1.0


def V(x, t, constants):
    x0 = constants["x0"] * jnp.cos(t * w0)
    return -((x - x0) ** 2) / 2
