import jax.numpy as jnp

eta = 1
kappa = 0
v = kappa
omega = (kappa**2 + eta**2) / 2


def waveFunction(x, t, constants):
    kappa = constants["kappa"]
    eta = constants["eta"]
    v = constants["v"]
    omega = constants["omega"]
    return eta * jnp.exp(1j * kappa * x - 1j * omega * t) / jnp.cosh(eta * (x - v * t))


def V(x, t, constants):
    return jnp.zeros_like(x)
