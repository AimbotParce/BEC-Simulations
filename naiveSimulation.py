import logging as log

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from lib import constants
from lib.managers.logging import setupLog

# j = jax.lax.complex(0.0, 1.0)
# @jax.jit
# def hamiltonian(psi, dx):
#     return 1j * jnp.gradient(jnp.gradient(psi, dx), dx) - 1j * jnp.abs(psi) ** 2 * psi

# def potential(x):


@jax.jit
def hamiltonian(psi, dx):
    return -constants.hbar**2 / (2 * constants.m) * jnp.gradient(jnp.gradient(psi, dx), dx)


# @jax.jit
def nextPsi(psi, dx, dt):
    return psi - dt * 1j * hamiltonian(psi, dx) / constants.hbar


setupLog()
log.info("Starting simulation (%s frames)", constants.dimT)
p = 1
x = jnp.arange(-constants.Lx / 2, constants.Lx / 2, constants.dx)
psi = np.zeros((constants.dimT, *x.shape), dtype=np.complex64)
log.info("x shape = %s, psi shape = %s", x.shape, psi.shape)
psi0 = jnp.exp(-((x - 7) ** 2) / 4 - 1j * p * x) / (2 * jnp.pi) ** (1 / 4)
psi[0] = psi0


for i in range(constants.dimT - 1):
    psi[i + 1] = nextPsi(psi[i], constants.dx, constants.dt)
log.info("Finished simulation")


log.info("Starting animation")
plt.ion()
plt.figure()
axis = plt.axes(xlim=(-10, 10), ylim=(-1, 1))
(probability,) = axis.plot([], [], lw=2)
(real,) = axis.plot([], [], lw=2)
(imag,) = axis.plot([], [], lw=2)
probability.set_data(x, jnp.abs(psi) ** 2)
real.set_data(x, jnp.real(psi))
imag.set_data(x, jnp.imag(psi))

title = axis.text(0.5, 1.05, "", bbox={"facecolor": "w", "alpha": 0.5, "pad": 5}, transform=axis.transAxes, ha="center")

plt.show()
for i in range(0, constants.dimT, 100):
    title.set_text(f"t = {i * constants.dt:.2f}s")

    probability.set_data(x, jnp.abs(psi[i]) ** 2)
    real.set_data(x, jnp.real(psi[i]))
    imag.set_data(x, jnp.imag(psi[i]))
    plt.pause(0.001)
