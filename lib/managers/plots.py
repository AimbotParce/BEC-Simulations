"""
Simple python script with functions to plot results from the simulation.
"""
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def plotEvolutionImage(path: str, output: str = None):
    psi = np.load(path)
    plt.figure()
    plt.imshow(np.abs(psi) ** 2, aspect="auto")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.colorbar()
    if output:
        plt.savefig(output)
    else:
        plt.show()


def plotEvolutionFunction(path: str, function: Callable, constants: dict, output: str = None):
    """
    Plots the evolution of the wave function stored in the file, with a function (of time)
    overlaid on the image.

    Parameters
    ----------
    path : str
        path to the file containing the wave function (.npy)
    function : Callable
        function of time to plot on the image
    output : str, optional
        path to save the image to, by default None. If None, the image is shown.
    """
    psi = np.load(path)
    plt.figure()
    plt.imshow(np.abs(psi) ** 2, aspect="auto")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.colorbar()

    # Add sinus wave sin(t)
    t = np.linspace(constants["tMin"], constants["tMax"], psi.shape[0] - 1)
    x = function(t) - constants["xMin"]
    plt.plot(
        x * psi.shape[1] / (constants["xMax"] - constants["xMin"]),
        t * (psi.shape[0] - 1) / (constants["tMax"] - constants["tMin"]),
        color="red",
    )

    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    plotEvolutionImage("projects/demos/dimensionlessSolitons/boundBrightSoliton.npy")
    plotEvolutionFunction(
        "projects/demos/dimensionlessSolitons/boundBrightSoliton.npy",
        lambda t: np.sin(t),
        {"tMin": 0, "tMax": 15, "xMin": -10, "xMax": 10},
    )
