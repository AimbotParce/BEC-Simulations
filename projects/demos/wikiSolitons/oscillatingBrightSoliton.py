import jax.numpy as jnp

import lib.constants as constants
from lib.waveFunctions import brightSolitonWiki


def waveFunction(x, t, constants):
    return brightSolitonWiki(x - 2, t)


def V(x, t, constants):
    return x**2 / 2
