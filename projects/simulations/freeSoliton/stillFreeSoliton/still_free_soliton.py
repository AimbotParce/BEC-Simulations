import jax.numpy as jnp

from lib.waveFunctions import brightSoliton as waveFunction

velocity = 0


def V(x, t, constants):
    return jnp.zeros_like(x)
