import jax.numpy as jnp


x0 = 0
tMax = 10


def waveFunction(x, t, constants):
    """Displaced the soliton 0.01 a0 to the right."""
    psi0 = constants["psi0"]
    xi = constants["healingLength"]
    x0 = constants["x0"]

    return psi0 / jnp.cosh((x - x0) / jnp.sqrt(2) / xi)


def V(x, t, constants):
    return 1 / 2 * x**2
