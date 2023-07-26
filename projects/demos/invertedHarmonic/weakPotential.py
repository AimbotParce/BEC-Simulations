import jax.numpy as jnp

# Override constants
g = -10


def waveFunction(x, t, constants):
    """Fix g to a higher value"""
    eta = 1 / constants["healingLength"]

    kappa1 = 0
    omega1 = (kappa1**2 + eta**2) / 2
    v1 = kappa1

    x1 = 0

    wf1 = eta * jnp.exp(1j * kappa1 * (x - x1) - 1j * omega1 * t) / jnp.cosh(eta * ((x - x1) - v1 * t))
    return wf1


def V(x, t, constants):
    return -1 / 2 * x**2
