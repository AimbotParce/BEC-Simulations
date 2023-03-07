import jax.numpy as jnp

import lib.constants as constants


def V(x, t):
    return x**2 / 2


def waveFunction(x, t):
    return jnp.exp(-((x - constants.x0) ** 2) / 2) / (jnp.pi) ** (1 / 4)
