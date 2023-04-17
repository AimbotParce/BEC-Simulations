import jax.numpy as jnp

from lib.waveFunctions import brightSoliton as waveFunction

velocity = 1


def V(x, t, constants):
    return jnp.zeros_like(x)
