"""
Create full report of a solution
"""
import argparse
import json
import os
from lib.utils.units import get_unit
from lib.interface.logging import setupLog

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np

font = "Times"
# plt.rcParams["font.family"] = "Times"
codeFont = "Courier"


args = argparse.ArgumentParser()
args.add_argument("-i", "--input", help="Path to the folder containing the solution", required=True, dest="path")
args.add_argument("-v", "--verbose", help="Verbose Level", default="INFO", dest="verbose")
args.add_argument("-t", "--titles", help="Add titles on plots.", action="store_true", dest="titles")
args = args.parse_args()

setupLog(args.verbose)

# Load the solution
psi = np.load(os.path.join(args.path, "evolution.npy"))
if os.path.exists(os.path.join(args.path, "theoretical.npy")):
    teo = np.load(os.path.join(args.path, "theoretical.npy"))

# Load the parameters
with open(os.path.join(args.path, "metadata.json"), "r") as f:
    metadata = json.load(f)
constants = metadata["constants"]

unit = get_unit(metadata["simulator"])


I = np.arange(psi.shape[1])
J = np.arange(psi.shape[0])
T = np.linspace(constants["tMin"], constants["tMax"], psi.shape[0])
X = np.linspace(constants["xMin"], constants["xMax"], psi.shape[1])

imgFolder = os.path.join(args.path, "report_images")
if not os.path.exists(imgFolder):
    os.mkdir(imgFolder)


# Plot the evolution
evolution = plt.figure()
expectedX = np.sum(np.abs(psi) ** 2 * X, axis=1) / np.sum(np.abs(psi) ** 2, axis=1)

x0 = constants["x0"]

theoreticalX = x0 * np.cos(T * constants["potentialW"])

plt.plot(expectedX, T, color="white")
plt.plot(theoreticalX, T, color="red", linestyle="--")
plt.imshow(np.abs(psi) ** 2, aspect="auto", extent=[X[0], X[-1], T[-1], T[0]])
plt.xlabel(unit.x("Position"))
plt.ylabel(unit.t("Time"))
plt.xlim(X[0], X[-1])
plt.ylim(T[-1], T[0])

if args.titles:
    plt.title("Comparison to classical evolution")
plt.colorbar()
plt.savefig(os.path.join(imgFolder, "compared_to_classical.png"))
