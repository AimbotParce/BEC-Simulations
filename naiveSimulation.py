import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from lib import constants

# j = jax.lax.complex(0.0, 1.0)
# @jax.jit
# def hamiltonian(psi, dx):
#     return 1j * jnp.gradient(jnp.gradient(psi, dx), dx) - 1j * jnp.abs(psi) ** 2 * psi


@jax.jit
def hamiltonian(psi, dx):
    return -constants.hbar**2 / (2 * constants.m) * jnp.gradient(jnp.gradient(psi, dx), dx)


# @jax.jit
def nextPsi(psi, dx, dt):
    return psi - dt * 1j * hamiltonian(psi, dx) / constants.hbar


p = 1
x = jnp.arange(-constants.Lx / 2, constants.Lx / 2, constants.dx)
# psi = jnp.exp(-((x / 2) ** 2) - 1j * p * x) / jnp.sqrt(2 * jnp.pi)
psi = jnp.zeros((constants.dimT, *x.shape), dtype=jnp.complex64)
psi.at[0].set(jnp.exp(-((x - 7) ** 2) / 4 - 1j * p * x) / (2 * jnp.pi) ** (1 / 4))


for i in range(constants.dimT - 1):
    psi.at[i + 1].set(nextPsi(psi[i], constants.dx, constants.dt))


fig, ax = plt.subplots()
(line,) = ax.plot([], [], lw=2)
(lineReal,) = ax.plot([], [], lw=2)
(lineImag,) = ax.plot([], [], lw=2)


def init():
    ax.set_xlim(0, 2 * jnp.pi)
    ax.set_ylim(-1, 1)
    return line, lineReal, lineImag


def update(frame):
    line.set_data(x, jnp.abs(psi[frame]) ** 2)
    lineReal.set_data(x, jnp.real(psi[frame]))
    lineImag.set_data(x, jnp.imag(psi[frame]))
    return line, lineReal, lineImag


animation = FuncAnimation(
    fig,
    update,
    frames=range(constants.dimT - 1),
    init_func=init,
    blit=True,
)

plt.show()
