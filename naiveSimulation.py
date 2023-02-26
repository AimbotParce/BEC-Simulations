import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# j = jax.lax.complex(0.0, 1.0)
# @jax.jit
# def hamiltonian(psi, dx):
#     return 1j * jnp.gradient(jnp.gradient(psi, dx), dx) - 1j * jnp.abs(psi) ** 2 * psi


@jax.jit
def hamiltonian(psi, dx):
    return -jnp.gradient(jnp.gradient(psi, dx), dx)


# @jax.jit
def nextPsi(psi, dx, dt):
    return psi - dt * 1j * hamiltonian(psi, dx)


p = 1
x = jnp.arange(-100, 100, 0.01)
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

while True:
    nxt = nextPsi(psi, 0.01, 0.00005)
    psi = nxt
    probability.set_data(x, jnp.abs(psi) ** 2)
    real.set_data(x, jnp.real(psi))
    imag.set_data(x, jnp.imag(psi))
    plt.pause(0.001)
