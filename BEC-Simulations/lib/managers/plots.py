"""
Simple python script with functions to plot results from the simulation.
"""
import json
import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from lib.managers.integrals import computeIntegral


def plotEvolutionImage(path: str, output: str = None):
    psi = np.load(os.path.join(path, "evolution.npy"))
    plt.figure()
    plt.imshow(np.abs(psi) ** 2, aspect="auto")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.colorbar()
    if output:
        plt.savefig(output)
    else:
        plt.show()


def plotEvolutionFunction(path: str, function: Callable, output: str = None):
    """
    Plots the evolution of the wave function stored in the file, with a function (of time)
    overlaid on the image.

    Parameters
    ----------
    path : str
        path to the folder containing the wave function solutions
    function : Callable
        function of time to plot on the image
    output : str, optional
        path to save the image to, by default None. If None, the image is shown.
    """
    psi = np.load(os.path.join(path, "evolution.npy"))
    plt.figure()
    plt.imshow(np.abs(psi) ** 2, aspect="auto")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.colorbar()

    with open(os.path.join(path, "metadata.json"), "r") as f:
        constants = json.load(f)["constants"]

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


def plotTheoreticalSimilarity(path: str, output: str = None):
    """
    Plots the evolution of the wave function stored in the file, with a function (of time)
    overlaid on the image.

    Parameters
    ----------
    path : str
        path to the folder containing the wave function solutions
    constants : dict
        dictionary containing the constants used in the simulation
    output : str, optional
        path to save the image to, by default None. If None, the image is shown.
    """
    psi = np.load(os.path.join(path, "evolution.npy"))
    teo = np.load(os.path.join(path, "theoretical.npy"))

    with open(os.path.join(path, "metadata.json"), "r") as f:
        constants = json.load(f)["constants"]

    # Add sinus wave sin(t)
    J = np.arange(psi.shape[0])
    t = np.linspace(constants["tMin"], constants["tMax"], psi.shape[0])
    x = np.linspace(constants["xMin"], constants["xMax"], psi.shape[1])

    similarity = np.zeros_like(t)
    for j in J:
        norm = computeIntegral(x, np.abs(psi[j]) ** 2, constants["dx"])
        normTeo = computeIntegral(x, np.abs(teo[j]) ** 2, constants["dx"])
        similarity[j] = computeIntegral(x, np.abs(psi[j] * np.conj(teo[j]) / np.sqrt(norm * normTeo)), constants["dx"])

    plt.figure()
    plt.xlabel("Time")
    plt.ylabel("Similarity (%)")
    plt.plot(t, similarity * 100, color="red")

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
