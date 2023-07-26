import jax.numpy as jnp

from lib.waveFunctions import brightSolitonWiki


def waveFunction(x, t):
    return brightSolitonWiki(x, t) * jnp.exp(-1j * x)


def V(x, t, constants):
    return jnp.zeros_like(x)
