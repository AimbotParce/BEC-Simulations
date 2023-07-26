"""
Load each of the evolutions in this direcory and plot the sigma at each healing length
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np


tau_max = 3
xmax = 2
xmin = -xmax

os.chdir(os.path.dirname(__file__))
files = os.listdir(".")
folders = [f for f in files if os.path.isdir(f) and f != "__pycache__"]

plot_folders = [0.1, 0.3, 0.4, 0.6]


sigmas = []
sigmas_start = []
end_xs = []
healing_lengths = []

# Join all the evolutions in a single plot
main_fig, main_ax = plt.subplots(2, 2, figsize=(6, 5), dpi=200)
main_ax = main_ax.flatten()

sigma_fig, sigma_ax = plt.subplots(1, 1, dpi=200)
sigma_ax.set_xlabel(r"$t [\tau]$")
sigma_ax.set_ylabel(r"$\sigma [a_0]$")

expected_x_fig, expected_x_ax = plt.subplots(1, 1, dpi=200)
expected_x_ax.set_xlabel(r"$t [\tau]$")
expected_x_ax.set_ylabel(r"$\langle x \rangle [a_0]$")
expected_x_ax.set_ylim([0, xmax])

for j, folder in enumerate(sorted(folders)):
    if not os.path.exists(os.path.join(folder, "evolution.npy")):
        continue
    healing_length = float(folder)
    healing_lengths.append(healing_length)

    # Load the data
    evolution = np.load(os.path.join(folder, "evolution.npy"))

    # This is a wavefunction (complex) -> the density is the absolute value squared
    density = np.abs(evolution) ** 2

    # Load the parameters
    with open(os.path.join(folder, "metadata.json"), "r") as f:
        constants = json.load(f)["constants"]

    X = np.linspace(constants["xMin"], constants["xMax"], density.shape[1])
    T = np.linspace(constants["tMin"], constants["tMax"], density.shape[0])

    sigma = np.zeros(density.shape[0])
    expected_x = np.zeros(density.shape[0])
    for i in range(density.shape[0]):
        expected_x[i] = np.sum(density[i, :] * X) / np.sum(density[i, :])
        sigma[i] = np.sqrt(np.sum(density[i, :] * (X - expected_x[i]) ** 2) / np.sum(density[i, :]))

    # Find the index of tau=3
    max_index = min(int((tau_max - constants["tMin"]) / constants["dt"]), density.shape[0] - 1)

    if healing_length in plot_folders:
        i = plot_folders.index(healing_length)
        # imshow the density
        main_ax[i].imshow(density, aspect="auto", extent=[X[0], X[-1], constants["tMax"], constants["tMin"]])
        main_ax[i].set_title(r"$\xi = {} a_0$".format(healing_length))
        main_ax[i].set_xlabel(r"$x [a_0]$")
        main_ax[i].set_ylabel(r"$t [\tau]$")
        main_ax[i].set_xlim([xmin, xmax])
        main_ax[i].set_ylim([tau_max, constants["tMin"]])

        # for tick in main_ax[i].xaxis.get_major_ticks() + main_ax[i].yaxis.get_major_ticks():
        #     tick.label1.set_fontsize(20)

    # Plot the sigma over time
    sigma_ax.plot(T, sigma, label=r"$\xi = {} a_0$".format(healing_length))

    # Plot the expected x over time
    expected_x_ax.plot(T, expected_x, label=r"$\xi = {} a_0$".format(healing_length))

    # Save the sigma and expected x at tau=3
    sigmas.append(sigma[max_index])
    sigmas_start.append(sigma[0])
    end_xs.append(expected_x[max_index])


healing_lengths = np.array(healing_lengths)
sigmas = np.array(sigmas)
sigmas_start = np.array(sigmas_start)

main_fig.tight_layout()
main_fig.savefig("evolution_over_healing_length.png")

sigma_ax.legend()
sigma_fig.tight_layout()
sigma_fig.savefig("sigma_over_time.png")

expected_x_ax.legend()
expected_x_fig.tight_layout()
expected_x_fig.savefig("expected_x_over_time.png")


plt.figure()
plt.xlabel(r"$\xi [a_0]$", fontsize=20)
plt.ylabel(r"$\left(\frac{\sigma_{x}}{\sigma_0}\right)_{t=%.0f\tau}$" % tau_max, fontsize=24)
plt.plot(healing_lengths, sigmas / sigmas_start, "o", color="black")
ax = plt.gca()
for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(20)

plt.tight_layout()

plt.savefig("sigma_over_healing_length.png")


chemical_potential = 1 / (4 * healing_lengths**2)
plt.figure()
plt.xlabel(r"$\mu [\hbar\omega]$", fontsize=16)  # chemical potential
plt.ylabel(r"$\left(\frac{\sigma_{x}}{\sigma_0}\right)_{t=%.0f\tau}$" % tau_max, fontsize=18)
plt.plot(chemical_potential, sigmas / sigmas_start, "o", color="black")
plt.tight_layout()
fig = plt.gcf()
ax = plt.gca()
plt.savefig("sigma_over_chemical_potential.png")

# Make it loglog
ax.loglog()
fig.tight_layout()
fig.savefig("log_sigma_over_chemical_potential.png")


plt.figure()
plt.xlabel(r"$\xi [a_0]$", fontsize=16)
plt.ylabel("$<x>_{t=%.0f\\tau}$ $[a_0]$" % tau_max, fontsize=16)
plt.ylim([0, 1.1 * np.max(end_xs)])
plt.axhline(1, color="red", linestyle="--")
plt.plot(healing_lengths, end_xs, "o", color="black")
plt.savefig("end_x_over_healing_length.png")

print("DONE!")
# plt.show()
