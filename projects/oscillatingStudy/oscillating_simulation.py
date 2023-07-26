import jax.numpy as jnp


x0 = 1
xMin = -2
xMax = 2
tMax = float(2 * jnp.pi)
healingLength = 0.1
dx = healingLength / 5
dt = 0.3 * dx**2  # r = 0.3


def waveFunction(x, t, constants):
    """Displaced the soliton 0.01 a0 to the right."""
    psi0 = constants["psi0"]
    xi = constants["healingLength"]
    x0 = constants["x0"]

    return psi0 / jnp.cosh((x - x0) / jnp.sqrt(2) / xi)


def V(x, t, constants):
    return 1 / 2 * x**2
