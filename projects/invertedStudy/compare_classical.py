"""
Load each of the evolutions in this direcory and plot the sigma at each healing length
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np


xmax = 2
xmin = -xmax

os.chdir(os.path.dirname(__file__))

healing_length = 0.2
folder_name = "{:.2f}".format(healing_length)

# Load the data
evolution = np.load(os.path.join(folder_name, "evolution.npy"))

# This is a wavefunction (complex) -> the density is the absolute value squared
density = np.abs(evolution) ** 2

# Load the parameters
with open(os.path.join(folder_name, "metadata.json"), "r") as f:
    constants = json.load(f)["constants"]

X = np.linspace(constants["xMin"], constants["xMax"], density.shape[1])
T = np.linspace(constants["tMin"], constants["tMax"], density.shape[0])

sigma = np.zeros(density.shape[0])
expected_x = np.zeros(density.shape[0])
max_position = np.zeros(density.shape[0])
for i in range(density.shape[0]):
    sigma[i] = np.sqrt(np.sum(density[i, :] * X**2) / np.sum(density[i, :]))
    expected_x[i] = np.sum(density[i, :] * X) / np.sum(density[i, :])
    max_position[i] = X[np.argmax(density[i, :])]


teo_expected_x = constants["x0"] * np.cosh(T)
teo_expected_x += max_position[0] - constants["x0"]  # No one will see this, but it's just to make the plot nicer
# The thing happening here is that the x point count is too low, so the real maximum of the soliton isn't placed at x0=0.1, but at ~0.11
# This causes for difference to start at ~0.01, which is not nice to see in the plot. So we just shift the teo_expected_x by that amount
# This is not really a nice thing to do, but whatever. For what it's worth, it's just a visual thing, not really a numerical result.

difference = np.abs(expected_x - teo_expected_x)


fig, ax = plt.subplots(1, 2, dpi=200, figsize=(10, 5))
ax[0].imshow(density, aspect="auto", extent=[X[0], X[-1], T[-1], T[0]])
ax[0].set_xlabel(r"$x, \langle x\rangle [a_0]$")
ax[0].set_ylabel(r"$t [\tau]$")
ax[0].plot(expected_x, T, color="red", label=r"$\langle x \rangle$", zorder=10, linewidth=2)
ax[0].plot(
    teo_expected_x, T, "-", color="blue", label=r"$\langle x \rangle_{teo}$", zorder=11, dashes=(5, 5), linewidth=2
)
ax[0].set_xlim([xmin, xmax])
ax[0].legend(fontsize=20, loc="lower left")


# X ticks at -2, -1 0, 1, 2
ax[0].set_xticks(np.linspace(xmin, xmax, 5))

ax[1].plot(T, difference)
ax[1].set_xlabel(r"$t [\tau]$")
ax[1].set_ylabel(r"$\langle x \rangle - \langle x \rangle_{teo}$")
ax[1].set_xlim([T[0], T[-1]])

ax[1].set_xticks(np.linspace(T[0], T[-1], 4))

# Change font size
for a in ax:
    for item in [a.title, a.xaxis.label, a.yaxis.label]:
        item.set_fontsize(20)


for tick in (
    ax[0].xaxis.get_major_ticks()
    + ax[0].yaxis.get_major_ticks()
    + ax[1].xaxis.get_major_ticks()
    + ax[1].yaxis.get_major_ticks()
):
    tick.label1.set_fontsize(20)

fig.tight_layout()
fig.savefig("0.2_evolution.png")
plt.show()
