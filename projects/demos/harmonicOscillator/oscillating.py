import jax.numpy as jnp

import lib.constants as constants

constants.g = 0
constants.x0 = 3
w = 1


def V(x, t):
    return 1 / 2 * constants.mass * w**2 * x**2


def waveFunction(x, t):
    const = (constants.mass * w / (constants.hbar * jnp.pi)) ** (1 / 4)
    exponential = jnp.exp(-constants.mass * w * (x - constants.x0) ** 2 / (2 * constants.hbar))
    return const * exponential
