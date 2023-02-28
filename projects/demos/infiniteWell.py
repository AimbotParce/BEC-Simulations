import jax.numpy as jnp

import lib.constants as constants


def waveFunction(x, t):
    timeIndependent = (
        jnp.sqrt(constants.baseDensity)
        / jnp.cosh(x - 1 / jnp.sqrt(2))
        * jnp.exp(1j * x - 1 * constants.velocity / jnp.sqrt(2))
    )

    return timeIndependent


def V(x, t):
    return 999 * (jnp.abs(x) >= 15)
