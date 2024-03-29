import jax.numpy as jnp

eta = 1
kappa = 1
omega = (kappa**2 + eta**2) / 2
v = kappa


def waveFunction(x, t, constants):
    return eta * jnp.exp(1j * kappa * x - 1j * omega * t) / jnp.cosh(eta * (x - v * t))


def V(x, t, constants):
    return jnp.zeros_like(x)
