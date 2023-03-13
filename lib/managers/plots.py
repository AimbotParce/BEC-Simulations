"""
Simple python script with functions to plot results from the simulation.
"""
import matplotlib.pyplot as plt
import numpy as np


def plotEvolutionImage(path, output=None):
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


def plotEvolutionImageSinusWave(path, output=None):
    psi = np.load(path)
    plt.figure()
    plt.imshow(np.abs(psi) ** 2, aspect="auto")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.colorbar()

    # Add sinus wave sin(t)
    t = np.arange(0, psi.shape[0])
    x = 5 * np.sin(t * 15 / psi.shape[0]) + psi.shape[1] / 2
    plt.plot(x, t, color="red")

    if output:
        plt.savefig(output)
    else:
        plt.show()
