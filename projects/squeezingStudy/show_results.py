"""
Load each of the evolutions in this direcory and plot the sigma at each healing length
"""

import os
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter
import numpy as np
from sklearn.linear_model import LinearRegression


tau_max = 10
xmax = 2
xmin = -xmax

os.chdir(os.path.dirname(__file__))
files = os.listdir(".")
folders = [f for f in files if os.path.isdir(f) and f != "__pycache__"]


sigma_evolutions = []
healing_lengths = []
omegas = []

# Join all the evolutions in a single plot
main_fig, main_ax = plt.subplots(2, 4, figsize=(10, 5), dpi=200)
main_ax = main_ax.flatten()

sigma_fig, sigma_ax = plt.subplots(1, 1, dpi=200)
sigma_ax.set_xlabel(r"$t [\tau]$")
sigma_ax.set_ylabel(r"$\sigma [a_0]$")


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

    sigma = np.zeros(density.shape[0])

    for i in range(density.shape[0]):
        sigma[i] = np.sqrt(np.sum(density[i, :] * X**2) / np.sum(density[i, :]))

    sigma_evolutions.append(sigma)
    healing_lengths.append(healing_length)

    # Compute the frequency of oscillation
    #  Find all the maxima
    delta = 5
    rolled_sigmas = np.zeros((2 * delta + 1, sigma.shape[0]))
    tmp_sigma = np.pad(sigma, delta, mode="edge")
    for i in range(-delta, delta + 1):
        rolled_sigmas[i + delta, :] = np.roll(tmp_sigma, i)[delta:-delta]

    # Find the maxima
    maxima = sigma == np.max(rolled_sigmas, axis=0)
    maxima_indices = np.where(maxima)[0][:-1]  # The last one is not correct

    # Suppose the error is exactly delta.

    # Find the average distance between maxima
    if len(maxima_indices) > 1:
        period = np.mean(np.diff(T[maxima_indices])) * 2  # [tau] (factor of 2 because we only count half a period)
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

    # Plot the evolution of sigma
    sigma_ax.plot(T, sigma, label=r"$\xi = {} a_0$".format(healing_length))
    # Mark the maxima
    sigma_ax.scatter(T[maxima_indices], sigma[maxima_indices], color="black", zorder=10)


main_fig.tight_layout()
main_fig.savefig("evolution_over_healing_length.png")

sigma_ax.legend(loc="upper right")
sigma_fig.tight_layout()
sigma_fig.savefig("sigma_evolutions.png")


# Plot the frequency over the healing length
omega_fig, omega_ax = plt.subplots(1, 1, dpi=200)
omega_ax.set_xlabel(r"$\xi [a_0]$", fontsize=16)
omega_ax.set_ylabel(r"$\omega_{sq} [\omega]$", fontsize=16)
omega_ax.set_ylim([0, 1.1 * np.max(omegas)])
omega_ax.plot(healing_lengths, omegas, "o", color="black")


omega_fig.tight_layout()
omega_fig.savefig("frequency_over_healing_length.png")

omegas = np.array(omegas)
healing_lengths = np.array(healing_lengths)

log_omegas = np.log(omegas - 1)
log_healing_lengths = np.log(healing_lengths)

fit = LinearRegression().fit(log_healing_lengths.reshape(-1, 1), log_omegas)
slope, intercept = fit.coef_[0], fit.intercept_
R_sq = fit.score(log_healing_lengths.reshape(-1, 1), log_omegas)


w_pred = 1 + healing_lengths**slope * np.exp(intercept)

omgea_log_fig, omega_log_ax = plt.subplots(1, 1, dpi=200)
omega_log_ax.set_xlabel(r"$\xi [a_0]$ (logscale)", fontsize=16)
omega_log_ax.set_ylabel(r"$\omega_{sq} -1 [\omega]$ (logscale)", fontsize=16)
omega_log_ax.plot(healing_lengths, omegas - 1, "o", color="black")
omega_log_ax.plot(
    healing_lengths,
    w_pred - 1,
    color="red",
    label="$\\omega_{{sq}} - 1 = %.3f\\times\\xi^{{%.2f}}$\n$R^2=%.3f$" % (np.exp(intercept), slope, R_sq),
)
omega_log_ax.loglog()
omega_log_ax.legend(loc="upper right", fontsize=16)


# Ax ticks font size
ticks = (
    omega_log_ax.xaxis.get_minor_ticks() + omega_log_ax.xaxis.get_major_ticks() + omega_log_ax.yaxis.get_major_ticks()
)
for tick in ticks:
    tick.label1.set_fontsize(16)


class LogFormatterFloatMode(Formatter):
    def __call__(self, x, pos=None):
        # The %g formatter will choose the most compact representation of the
        # value.  Use isfinite to guard against log(0) or log(-ve) cases.
        if np.isfinite(x):
            return "%g" % x
        else:
            return ""


omega_log_ax.xaxis.set_major_formatter(LogFormatterFloatMode())
omega_log_ax.xaxis.set_minor_formatter(LogFormatterFloatMode())
omega_log_ax.yaxis.set_major_formatter(LogFormatterFloatMode())


omgea_log_fig.tight_layout()
omgea_log_fig.savefig("log_frequency_over_log_healing_length.png")

print("DONE!")
# plt.show()
