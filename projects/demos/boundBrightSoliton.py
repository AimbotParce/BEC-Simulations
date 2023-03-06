import jax.numpy as jnp

from lib.waveFunctions import brightSolitonMalo

waveFunction = brightSolitonMalo


def V(x, t):
    return (x) ** 2
