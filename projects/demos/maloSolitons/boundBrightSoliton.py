import jax.numpy as jnp

from lib.waveFunctions import brightSolitonMalo

waveFunction = brightSolitonMalo


def V(x, t, constants):
    return x**2 / 2
