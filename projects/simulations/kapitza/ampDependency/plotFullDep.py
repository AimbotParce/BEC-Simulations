# Plot a single figure with all the frequency dependencies

import json
import os

import matplotlib.pyplot as plt
import numpy as np

amps = [0.1, 0.5, 1.0, 2.0]

fig, ax = plt.subplots(1, 4, figsize=(20, 4), sharey=True)

idxs = [0, 1, 2, 3]

for amp, idx in zip(amps, idxs):
    path = lambda x: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "amp" + str(amp), x))
    # Load the solution
    psi = np.load(path("evolution.npy"))
    if os.path.exists(path("theoretical.npy")):
        teo = np.load(path("theoretical.npy"))
    # Load the parameters
    with open(path("metadata.json"), "r") as f:
        metadata = json.load(f)
    constants = metadata["constants"]

    I = np.arange(psi.shape[1])
    J = np.arange(psi.shape[0])
    T = np.linspace(constants["tMin"], constants["tMax"], psi.shape[0])
    X = np.linspace(constants["xMin"], constants["xMax"], psi.shape[1])

    expectedX = np.sum(np.abs(psi) ** 2 * X, axis=1) / np.sum(np.abs(psi) ** 2, axis=1)

    # theoretical_X = -np.cosh(T) + 1

    ax[idx].imshow(np.abs(psi), aspect="auto", extent=[X[0], X[-1], T[-1], T[0]])
    # plt.plot(theoretical_X, T, color="red")
    ax[idx].plot(expectedX, T, color="white")

    ax[idx].set_xlabel("Position/$a_0$")
    ax[idx].set_ylabel("Time/$\\tau$")

    ax[idx].set_xlim(X[0], X[-1])
    ax[idx].set_ylim(T[-1], T[0])

    # # Change the ticks to show the real values
    # xTicks = X[:: int(len(X) / 10)]
    # xTicksLabels = [f"{x:.2f}" for x in xTicks]
    # plt.xticks(I[:: int(len(I) / 10)], xTicksLabels)

    # tTicks = T[:: int(len(T) / 10)]
    # tTicksLabels = [f"{t:.2f}" for t in tTicks]
    # plt.yticks(J[:: int(len(J) / 10)], tTicksLabels)

    ax[idx].set_title("$A_V = $" + str(amp) + "$a_0$")

plt.savefig(os.path.join(os.path.dirname(__file__), "AmplitudeDependency.png"))
