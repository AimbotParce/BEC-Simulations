import jax.numpy as jnp

from lib.waveFunctions import brightSolitonWiki

waveFunction = brightSolitonWiki


def V(x, t, constants):
    return jnp.zeros_like(x)
