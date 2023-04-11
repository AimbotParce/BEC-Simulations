# %%
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# Load the solution
psi = np.load("evolution.npy")[:400]

# %%
with open("metadata.json", "r") as f:
    metadata = json.load(f)
constants = metadata["constants"]

# %%


I = np.arange(psi.shape[1])
J = np.arange(psi.shape[0])
T = np.linspace(constants["tMin"], constants["tMax"], psi.shape[0])
X = np.linspace(constants["xMin"], constants["xMax"], psi.shape[1])

imgFolder = "report_images"

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
