from lib.waveFunctions import brightSoliton as waveFunction
import jax.numpy as jnp

velocity = 1


def V(x, t, constants):
    return jnp.zeros_like(x)
