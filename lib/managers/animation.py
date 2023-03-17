import logging as log
from argparse import Namespace
from typing import Callable, Union

import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt

log = log.getLogger("BECsimulations")


def animate(
    x: jnp.ndarray,
    t: jnp.ndarray,
    psi: jnp.ndarray,
    V: Callable,
    args: Union[Namespace, dict],
    constants: dict,
    energyFunction: Callable,
    integratingFunction: Callable,
    theoreticalPsi: jnp.ndarray = None,
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
    arguments : Namespace | dict
        The arguments passed to the program
    constants : dict
        The constants used in the simulation
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
    ax.set_ylim(constants["plotYMin"], constants["plotYMax"])
    ax.set_xlim(constants["xMin"], constants["xMax"])
    ax.set_xlabel("x")
    ax.set_ylabel("psi(x), V(x)")
    ax.set_title("Simulation of the Gross-Pitaevskii equation")

    # Lines
    (potential,) = ax.plot(x, V(x, 0), color="red", label="V(x)")
    (probability,) = ax.plot(x, jnp.abs(psi[0]) ** 2, label="Probability")

    if args.showParts:
        (realPart,) = ax.plot(x, jnp.real(psi[0]), label="Real part")
        (imaginaryPart,) = ax.plot(x, jnp.imag(psi[0]), label="Imaginary part")

    if args.theoretical:
        (theoreticalProbability,) = ax.plot(
            x, jnp.abs(theoreticalPsi[0]) ** 2, color="green", label="Theoretical probability"
        )

    # Legends
    ax.legend(loc="lower right")

    # Texts
    timeText = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    cumulativeProbabilityText = ax.text(0.02, 0.90, "", transform=ax.transAxes)
    energyText = ax.text(0.02, 0.85, "", transform=ax.transAxes)

    if args.theoretical:
        similarityText = ax.text(0.02, 0.80, "", transform=ax.transAxes)

    def update(iteration):
        time = t[iteration]
        # Update texts
        timeText.set_text("t = %.2f" % time)
        norm = integratingFunction(x, jnp.abs(psi[iteration]) ** 2, constants["dx"])
        cumulativeProbabilityText.set_text("Norm = %.10f" % norm)
        energyText.set_text(
            "Energy = %.8f"
            % energyFunction(
                x,
                time,
                psi[iteration],
                V(x, time),
                constants["dx"],
                constants["g"],
                constants["mass"],
                constants["hbar"],
            )
        )
        if args.theoretical:
            normTeo = integratingFunction(x, jnp.abs(theoreticalPsi[iteration]) ** 2, constants["dx"])
            similarity = integratingFunction(
                x,
                jnp.abs(psi[iteration] * jnp.conj(theoreticalPsi[iteration])) / jnp.sqrt(normTeo * norm),
                constants["dx"],
            )
            similarityText.set_text("Similarity = %.8f" % similarity)

        # Update lines
        potential.set_ydata(V(x, time))
        probability.set_ydata(jnp.abs(psi[iteration]) ** 2)
        if args.showParts:
            realPart.set_ydata(jnp.real(psi[iteration]))
            imaginaryPart.set_ydata(jnp.imag(psi[iteration]))
        if args.theoretical:
            theoreticalProbability.set_ydata(jnp.abs(theoreticalPsi[iteration]) ** 2)

    log.info("Loading animation...")

    anim = animation.FuncAnimation(
        fig, update, frames=range(0, constants["tCount"], constants["plotStep"]), interval=1, repeat=True
    )
    plt.show()

    log.info("Animation finished.")
