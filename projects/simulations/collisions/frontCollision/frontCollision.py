import jax.numpy as jnp


def waveFunction(x, t, constants):
    eta = 1

    kappa1 = 1
    omega1 = (kappa1**2 + eta**2) / 2
    v1 = kappa1

    kappa2 = -1
    omega2 = (kappa2**2 + eta**2) / 2
    v2 = kappa2

    x1 = -3
    x2 = 3

    wf1 = eta * jnp.exp(1j * kappa1 * (x - x1) - 1j * omega1 * t) / jnp.cosh(eta * ((x - x1) - v1 * t))
    wf2 = eta * jnp.exp(1j * kappa2 * (x - x2) - 1j * omega2 * t) / jnp.cosh(eta * ((x - x2) - v2 * t))
    return wf1 + wf2


def V(x, t, constants):
    return jnp.zeros_like(x)
