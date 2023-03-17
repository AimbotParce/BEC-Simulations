import jax.numpy as jnp

from lib.waveFunctions import brightSolitonWiki

waveFunction = brightSolitonWiki


def V(x, t, constants):
    return -((x - 0.01) ** 2) / 2
