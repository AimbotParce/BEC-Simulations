# %%
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

fpath = os.path.join(os.path.dirname(__file__), "projects", "simulations", "invertedHarmonic", "fixedOnTopUnbreaking")


def path(*args):
    return os.path.join(fpath, *args)


# Load the solution
psi = np.load(path("evolution.npy"))

# %%
with open(path("metadata.json"), "r") as f:
    metadata = json.load(f)
constants = metadata["constants"]

# %%


I = np.arange(psi.shape[1])
J = np.arange(psi.shape[0])
T = np.linspace(constants["tMin"], constants["tMax"], psi.shape[0])
X = np.linspace(constants["xMin"], constants["xMax"], psi.shape[1])

# %%


# #################################
# #             Plots             #
# #################################

# Plot the evolution
evolution = plt.figure()
plt.imshow(np.abs(psi) ** 2, aspect="auto")
plt.xlabel("Position/$a_0$")
plt.ylabel(r"Time/$\tau$")


# Change the ticks to show the real values
xTicks = X[:: int(len(X) / 10)]
xTicksLabels = [f"{x:.2f}" for x in xTicks]
plt.xticks(I[:: int(len(I) / 10)], xTicksLabels)

tTicks = T[:: int(len(T) / 10)]
tTicksLabels = [f"{t:.2f}" for t in tTicks]
plt.yticks(J[:: int(len(J) / 10)], tTicksLabels)


plt.title("Evolution of the wave function")
plt.colorbar()

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)


for plot, j in enumerate([0, 250, 500]):
    # Plot the evolution
    ax[plot].plot(X, np.abs(psi[j]) ** 2)
    ax[plot].set_xlabel("Position/$a_0$")
    ax[plot].set_ylabel(r"Probability Density ($|\tilde{\psi}|^2$)")
    ax[plot].set_title("$\\frac{{t}}{\\tau}=%.2f$" % (T[j]))
