import jax.numpy as jnp

import lib.constants as constants

constants.g = 0


def V(x, t, constants):
    return x**2 / 2


def waveFunction(x, t, constants):
    return jnp.exp(-((2 * x) ** 2) / 2) / (jnp.pi) ** (1 / 4)
