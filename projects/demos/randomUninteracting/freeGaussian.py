import jax.numpy as jnp

import lib.constants as constants

constants.g = 0


def waveFunction(x, t):
    timeIndependent = (
        jnp.sqrt(constants.baseDensity)
        / jnp.cosh(x - 1 / jnp.sqrt(2))
        * jnp.exp(1j * x - 1 * constants.velocity / jnp.sqrt(2))
    )

    return timeIndependent


def V(x, t):
    return jnp.zeros_like(x)
