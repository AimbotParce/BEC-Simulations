# Random frequency
import jax.numpy as jnp

from lib.waveFunctions import brightSoliton as waveFunction

# w0 = 5.0
# w0 must be set on the command line.
amplitude = 1.0


def V(x, t, constants):
    x0 = constants["amplitude"] * jnp.cos(t * constants["w0"])
    return -((x - x0) ** 2) / 2
