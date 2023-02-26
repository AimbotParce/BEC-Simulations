import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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
psi = jnp.exp(-((x - 7) ** 2) / 4 - 1j * p * x) / (2 * jnp.pi) ** (1 / 4)

# plot
plt.ion()
plt.figure()
axis = plt.axes(xlim=(-10, 10), ylim=(-1, 1))
(probability,) = axis.plot([], [], lw=2)
(real,) = axis.plot([], [], lw=2)
(imag,) = axis.plot([], [], lw=2)
probability.set_data(x, jnp.abs(psi) ** 2)
real.set_data(x, jnp.real(psi))
imag.set_data(x, jnp.imag(psi))


plt.show()
t = 0.0
timeText = axis.text(0.02, 0.95, "", transform=axis.transAxes)
while True:
    t += constants.dt
    timeText.set_text("time = %.3f" % t)
    nxt = nextPsi(psi, constants.dx, constants.dt)
    psi = nxt
    probability.set_data(x, jnp.abs(psi) ** 2)
    real.set_data(x, jnp.real(psi))
    imag.set_data(x, jnp.imag(psi))
    plt.pause(0.001)
