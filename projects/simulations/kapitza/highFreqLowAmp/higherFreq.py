# Random frequency
import jax.numpy as jnp

from lib.waveFunctions import brightSoliton as waveFunction

w0 = 5.0
amplitude = 1.0


def V(x, t, constants):
    x0 = constants["amplitude"] * jnp.cos(t * constants["w0"])
    return -((x - x0) ** 2) / 2
