import logging as log
from typing import Callable

import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import lib.constants as constants
from lib.managers.integrals import computeEnergy, integrateProbability


def animate(
    x: jnp.ndarray,
    t: jnp.ndarray,
    psi: jnp.ndarray,
    V: Callable,
    theoreticalWaveFunction: Callable,
    energyFunction: Callable = computeEnergy,
    integratingFunction: Callable = integrateProbability,
):
    """
    Parameters
    ----------
    x : jax.numpy.ndarray
        The space steps (shape: (xCount,))
    t : jax.numpy.ndarray
        The time steps (shape: (tCount,))
    psi : jax.numpy.ndarray
        The wave function at each time step (shape: (tCount, xCount))
    V : Callable
        The potential function at each time step (shape: (tCount, xCount))
    theoreticalWaveFunction : Callable
        The wave function generator (signature: waveFunctionGenerator(x, t))
    energyFunction : Callable
        The energy function (signature: energy(x, psi, V))
    integratingFunction : Callable
        The cumulative probability function (signature: cumulativeProbability(x, psi))

    Returns
    -------
    None
    """
    log.info("Initializing animation...")
    # Animation
    fig, ax = plt.subplots()
    ax.set_ylim(constants.plotYMin, constants.plotYMax)
    ax.set_xlim(constants.xMin, constants.xMax)
    ax.set_xlabel("x")
    ax.set_ylabel("psi(x), V(x)")
    ax.set_title("Simulation of the Gross-Pitaevskii equation")

    # Lines
    (potential,) = ax.plot(x, V(x, 0), color="red")
    (probability,) = ax.plot(x, jnp.abs(psi[0]) ** 2)
    (realPart,) = ax.plot(x, jnp.real(psi[0]))
    (imaginaryPart,) = ax.plot(x, jnp.imag(psi[0]))
    (theoretical,) = ax.plot(x, jnp.abs(theoreticalWaveFunction(x, 0)) ** 2, color="black")

    # Legends
    ax.legend(
        (potential, probability, realPart, imaginaryPart, theoretical),
        ("V(x)", "Probability", "Real part", "Imaginary part", "Theoretical"),
        loc="lower right",
    )

    # Texts
    timeText = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    cumulativeProbabilityText = ax.text(0.02, 0.90, "", transform=ax.transAxes)
    energyText = ax.text(0.02, 0.85, "", transform=ax.transAxes)

    def update(iteration):
        time = t[iteration]
        # Update texts
        timeText.set_text("t = %.2f" % time)
        cumulativeProbabilityText.set_text("Cumulative probability = %.2f" % integratingFunction(x, psi[iteration]))
        energyText.set_text("Energy = %.8f" % energyFunction(x, time, psi[iteration], V(x, time)))

        # Update lines
        potential.set_ydata(V(x, time))
        probability.set_ydata(jnp.abs(psi[iteration]) ** 2)
        realPart.set_ydata(jnp.real(psi[iteration]))
        imaginaryPart.set_ydata(jnp.imag(psi[iteration]))
        theoretical.set_ydata(jnp.abs(theoreticalWaveFunction(x, time)) ** 2)

    log.info("Loading animation...")

    anim = animation.FuncAnimation(
        fig, update, frames=range(0, constants.tCount, constants.plotStep), interval=1, repeat=True
    )
    plt.show()

    log.info("Animation finished.")
