import jax.numpy as jnp

import lib.constants as constants


def waveFunction(x, t):
    return jnp.exp(-((x - 7) ** 2) / 4 - 1j * constants.velocity * x) / (2 * jnp.pi) ** (1 / 4)


def V(x, t):
    return jnp.zeros_like(x)
