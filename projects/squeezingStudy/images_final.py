"""
Load each of the evolutions in this direcory and plot the sigma at each healing length
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


tau_max = 10
xmax = 2
xmin = -xmax

os.chdir(os.path.dirname(__file__))
folders = ["0.35", "0.45", "0.80", "1.00"]

# Join all the evolutions in a single plot
main_fig, main_ax = plt.subplots(2, 2, figsize=(6, 5), dpi=200)
main_ax = main_ax.flatten()

for j, folder in enumerate(sorted(folders)):
    healing_length = float(folder)
    # Load the data
    evolution = np.load(os.path.join(folder, "evolution.npy"))

    # This is a wavefunction (complex) -> the density is the absolute value squared
    density = np.abs(evolution) ** 2

    # Load the parameters
    with open(os.path.join(folder, "metadata.json"), "r") as f:
        constants = json.load(f)["constants"]
    # Find the index of tau=3
    max_index = min(int((tau_max - constants["tMin"]) / constants["dt"]), density.shape[0] - 1)

    X = np.linspace(constants["xMin"], constants["xMax"], density.shape[1])
    T = np.linspace(constants["tMin"], constants["tMax"], density.shape[0])

    # imshow the density
    main_ax[j].imshow(density, aspect="auto", extent=[X[0], X[-1], constants["tMax"], constants["tMin"]])
    main_ax[j].set_title(r"$\xi = {} a_0$".format(healing_length))
    main_ax[j].set_xlabel(r"$x [a_0]$")
    main_ax[j].set_ylabel(r"$t [\tau]$")
    main_ax[j].set_xlim([xmin, xmax])
    main_ax[j].set_ylim([tau_max, constants["tMin"]])


main_fig.tight_layout()
main_fig.savefig("final_evolution_over_healing_length.png")

plt.show()
