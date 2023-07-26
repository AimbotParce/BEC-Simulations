# from lib.waveFunctions import stillBrightSoliton as waveFunction
import jax.numpy as jnp

x0 = 7  # Position of the center of the wave packet.
a0_over_healingLength = 1  # Potential length in units of healing length
hbar = 1
mass = 1


def V(x, t, constants):
    """
    The potential energy function.
    """
    # The width of the harmonic oscillator potential.
    w = constants["potentialW"]
    # The mass of the particle.
    m = constants["mass"]
    # The potential energy at the given position and time.
    return m * w**2 * x**2 / 2


def waveFunction(x, t, constants):
    w = constants["potentialW"]
    m = constants["mass"]
    hbar = constants["hbar"]
    x0 = constants["x0"]

    const = (m * w / (hbar * jnp.pi)) ** (1 / 4)
    exponential = jnp.exp(-m * w * (x - x0) ** 2 / (2 * hbar))
    return const * exponential
