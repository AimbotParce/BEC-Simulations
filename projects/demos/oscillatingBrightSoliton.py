import jax.numpy as jnp

import lib.constants as constants

B = (constants.g / jnp.abs(constants.g) / 2 / constants.baseDensity) ** 1 / 3


def waveFunction(x, t):
    timeIndependent = (jnp.tanh(B * x) + 1) / 10
    return timeIndependent


def V(x, t):
    return x**2 / constants.chemicalPotential
