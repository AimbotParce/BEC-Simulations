import jax.numpy as jnp

eta = 1
kappa = 0
omega = (kappa**2 + eta**2) / 2
v = kappa


def waveFunction(x, t, constants):
    A = eta * jnp.exp(1j * kappa * x - 1j * omega * t) / jnp.cosh(eta * (x - 3 - v * t))
    B = eta * jnp.exp(1j * kappa * x - 1j * omega * t) / jnp.cosh(eta * (x + 3 + v * t))
    return A + B


def V(x, t, constants):
    return x**2 / 2
