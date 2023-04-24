"""
Create full report of a solution
"""
# %%
import argparse
import json
import logging
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import types

import matplotlib.pyplot as plt
import numpy as np

# Load the solution
psi = np.load("evolution.npy")
if os.path.exists("theoretical.npy"):
    teo = np.load("theoretical.npy")
print(".", end="", flush=True)
# Load the parameters
with open("metadata.json", "r") as f:
    metadata = json.load(f)
constants = metadata["constants"]
print(".", end="", flush=True)


I = np.arange(psi.shape[1])
J = np.arange(psi.shape[0])
T = np.linspace(constants["tMin"], constants["tMax"], psi.shape[0])
X = np.linspace(constants["xMin"], constants["xMax"], psi.shape[1])

imgFolder = os.path.abspath("report_images")
if not os.path.exists(imgFolder):
    os.mkdir(imgFolder)


# %%

expectedX = np.sum(np.abs(psi) ** 2 * X, axis=1) / np.sum(np.abs(psi) ** 2, axis=1)

theoretical_X = -np.cosh(T) + 1

plt.imshow(np.abs(psi), aspect="auto", extent=[X[0], X[-1], T[-1], T[0]])
plt.plot(theoretical_X, T, color="red")
plt.plot(expectedX, T, color="white")


plt.xlabel("Position/$a_0$")
plt.ylabel("Time/$\\tau$")

plt.xlim(X[0], X[-1])
plt.ylim(T[-1], T[0])

# # Change the ticks to show the real values
# xTicks = X[:: int(len(X) / 10)]
# xTicksLabels = [f"{x:.2f}" for x in xTicks]
# plt.xticks(I[:: int(len(I) / 10)], xTicksLabels)

# tTicks = T[:: int(len(T) / 10)]
# tTicksLabels = [f"{t:.2f}" for t in tTicks]
# plt.yticks(J[:: int(len(J) / 10)], tTicksLabels)


plt.title("Evolution of the wave function")
plt.colorbar()
plt.savefig(os.path.join(imgFolder, "evolutionExpected.png"))
# %%

difference = np.abs(expectedX - theoretical_X)

plt.plot(T, difference)
plt.xlabel("Time/$\\tau$")
plt.ylabel("Difference/$a_0$")
plt.title("Difference between the expected value and the theoretical one")
plt.savefig(os.path.join(imgFolder, "difference.png"))


# %%
