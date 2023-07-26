"""
Load each of the evolutions in this direcory and plot the sigma at each healing length
"""

import os
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


tau_max = 2 * np.pi
xmax = 2
xmin = -xmax

os.chdir(os.path.dirname(__file__))
files = os.listdir(".")
folders = [f for f in files if os.path.isdir(f) and f != "__pycache__"]

omegas = []
healing_lengths = []

# Join all the evolutions in a single plot
main_fig, main_ax = plt.subplots(2, 4, figsize=(10, 5), dpi=200)
main_ax = main_ax.flatten()


final_fig = plt.figure(dpi=200, figsize=(12, 6))
gs = GridSpec(2, 4, figure=final_fig)
# Add a big axis on the right side of the figure
ax_plot = final_fig.add_subplot(gs[:, 2:])
idxs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ax_imgs = [final_fig.add_subplot(gs[idx[0], idx[1]]) for idx in idxs]
final_healing_lengths = [0.2, 0.3, 0.5, 0.8]

sigma_fig, sigma_ax = plt.subplots(1, 1, dpi=200)
sigma_ax.set_xlabel(r"$t [\tau]$")
sigma_ax.set_ylabel(r"$\sigma [a_0]$")

expected_x_fig, expected_x_ax = plt.subplots(1, 1, dpi=200)
expected_x_ax.set_xlabel(r"$t [\tau]$")
expected_x_ax.set_ylabel(r"$\langle x \rangle [a_0]$")
expected_x_ax.set_ylim([xmin, xmax])

for j, folder in enumerate(sorted(folders)):
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

    # Compute the frequency of oscillation
    #  Find all the maxima
    delta = 5
    rolled_expected_x = np.zeros((2 * delta + 1, expected_x.shape[0]))
    tmp_expected_x = np.pad(expected_x, delta, mode="edge")
    for i in range(-delta, delta + 1):
        rolled_expected_x[i + delta, :] = np.roll(tmp_expected_x, i)[delta:-delta]

    # Find the maxima
    maxima = expected_x == np.max(rolled_expected_x, axis=0)
    maxima_indices = np.where(maxima)[0]

    # Suppose the error is exactly delta.

    # Find the average distance between maxima
    if len(maxima_indices) > 1:
        period = np.mean(np.diff(T[maxima_indices]))  # [tau]
        freq = 1 / period  # [tau^-1]
        omega = 2 * np.pi * freq  # [tau^-1]
        omegas.append(omega)
    else:
        omegas.append(0)

    # imshow the density
    main_ax[j].imshow(density, aspect="auto", extent=[X[0], X[-1], constants["tMax"], constants["tMin"]])
    main_ax[j].set_title(r"$\xi = {} a_0$".format(healing_length))
    main_ax[j].set_xlabel(r"$x [a_0]$")
    main_ax[j].set_ylabel(r"$t [\tau]$")
    main_ax[j].set_xlim([xmin, xmax])
    main_ax[j].set_ylim([tau_max, constants["tMin"]])

    if healing_length in final_healing_lengths:
        i = final_healing_lengths.index(healing_length)
        ax_imgs[i].imshow(density, aspect="auto", extent=[X[0], X[-1], constants["tMax"], constants["tMin"]])
        ax_imgs[i].set_title(r"$\xi = {} a_0$".format(healing_length), fontsize=20)
        ax_imgs[i].set_xlabel(r"$x [a_0]$", fontsize=20)
        ax_imgs[i].set_ylabel(r"$t [\tau]$", fontsize=20)
        ax_imgs[i].set_xlim([xmin, xmax])
        ax_imgs[i].set_ylim([tau_max, constants["tMin"]])

        for tick in ax_imgs[i].xaxis.get_major_ticks() + ax_imgs[i].yaxis.get_major_ticks():
            tick.label1.set_fontsize(20)

        ax_imgs[i].plot(expected_x, T, "-", color="red", linewidth=2, zorder=10)

    # Plot the sigma over time
    sigma_ax.plot(T, sigma, label=r"$\xi = {} a_0$".format(healing_length))

    # Plot the expected x over time
    expected_x_ax.plot(T, expected_x, label=r"$\xi = {} a_0$".format(healing_length))


main_fig.tight_layout()
main_fig.savefig("evolution_over_healing_length.png")

sigma_ax.legend()
sigma_fig.tight_layout()
sigma_fig.savefig("sigma_over_time.png")

expected_x_ax.legend()
expected_x_fig.tight_layout()
expected_x_fig.savefig("expected_x_over_time.png")

# Plot the frequency over healing length
omega_fig, omega_ax = plt.subplots(1, 1, dpi=200)
omega_ax.set_xlabel(r"$\xi [a_0]$")
omega_ax.set_ylabel(r"$\omega_{osc} [\omega]$")
omega_ax.plot(healing_lengths, omegas, "o", color="black")
omega_ax.axhline(1, color="red", linestyle="--", zorder=-1)
omega_fig.tight_layout()
omega_fig.savefig("omega_over_healing_length.png")


ax_plot.plot(healing_lengths, omegas, "o", color="black", markersize=10, label=r"$\omega_{osc}$")
ax_plot.axhline(1, color="red", linestyle="--", zorder=-1, label=r"$\omega_{expected} = 1 \omega$", linewidth=2)
ax_plot.set_xlabel(r"$\xi [a_0]$", fontsize=20)
ax_plot.set_ylabel(r"$\omega_{osc} [\omega]$", fontsize=20)

# Legend
ax_plot.legend(fontsize=20)

# Ax ticks font size
ticks = (
    ax_plot.xaxis.get_minor_ticks()
    + ax_plot.xaxis.get_major_ticks()
    + ax_plot.yaxis.get_major_ticks()
    + ax_plot.yaxis.get_minor_ticks()
)
for tick in ticks:
    tick.label1.set_fontsize(20)


final_fig.tight_layout(pad=0, w_pad=0, h_pad=0)
final_fig.savefig("oscillating_final_image.png")

print("DONE!")
