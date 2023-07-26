"""
Load the given solution and create a gif of the evolution
"""

import argparse
from tqdm import tqdm
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "BEC-Simulations"))
from lib.utils.units import get_unit

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Path to the folder containing the solution", required=True, dest="path")

args = parser.parse_args()


os.chdir(args.path)

# Load the solution
psi = np.load("evolution.npy")

# Load the parameters
with open("metadata.json", "r") as f:
    metadata = json.load(f)

constants = metadata["constants"]
simulator = metadata["simulator"]

u = get_unit(simulator)

psi = np.abs(psi) ** 2
ylim = np.max(psi) * 1.1


# Create the animation
fig, ax = plt.subplots()
ax.set_xlabel(u.x("Position"), fontsize=14)
ax.set_ylabel(r"$\psi^2$", fontsize=16)
ax.set_xlim([-2, 2])
ax.set_ylim([0, ylim])

X = np.linspace(constants["xMin"], constants["xMax"], 1000)  # No need to plot all the points
orig_X = np.linspace(constants["xMin"], constants["xMax"], psi.shape[1])

(line,) = ax.plot([], [], lw=2, color="black")
text = ax.text(0.02, 0.90, "", transform=ax.transAxes, fontsize=14)


max_frames = 1000
delta = int(psi.shape[0] / max_frames)


def frame(i):
    line.set_data(X, np.interp(X, orig_X, psi[i * delta, :]))
    text.set_text(r"$t = {:.2f} \tau$".format(i * delta * constants["dt"]))
    return (line,)


anim = animation.FuncAnimation(
    fig,
    frame,
    frames=tqdm(range(max_frames)),
    interval=1000 / 60,
    blit=True,
)

# Show the animation


anim.save("evolution.gif", fps=30)
