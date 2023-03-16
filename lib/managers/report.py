"""
Create full report of a solution
"""
import argparse
import json
import os
import types

import matplotlib.pyplot as plt
import numpy as np

font = "Times"
# plt.rcParams["font.family"] = "Times"
codeFont = "Courier"

args = argparse.ArgumentParser()
args.add_argument("-p", "--path", help="Path to the folder containing the solution", required=True)
args = args.parse_args()

# Load the solution
psi = np.load(os.path.join(args.path, "evolution.npy"))
if os.path.exists(os.path.join(args.path, "theoretical.npy")):
    teo = np.load(os.path.join(args.path, "theoretical.npy"))

# Load the parameters
with open(os.path.join(args.path, "metadata.json"), "r") as f:
    metadata = json.load(f)
constants = metadata["constants"]

# Plot the evolution
evolution = plt.figure()
plt.imshow(np.abs(psi) ** 2, aspect="auto")
plt.xlabel("Position [a.u.]")
plt.ylabel("Time [a.u.]")
plt.title("Evolution of the wave function")
plt.colorbar()
plt.savefig(os.path.join(args.path, "evolution.png"))

# Plot the theoretical solution
theoretical = plt.figure()
plt.imshow(np.abs(teo) ** 2, aspect="auto")
plt.xlabel("Position [a.u.]")
plt.ylabel("Time [a.u.]")
plt.title("Theoretical evolution of the wave function")
plt.colorbar()
plt.savefig(os.path.join(args.path, "theoretical.png"))

# Plot the evolution of the energy

I = np.arange(psi.shape[1])
J = np.arange(psi.shape[0])
T = np.linspace(constants["tMin"], constants["tMax"], psi.shape[0])
X = np.linspace(constants["xMin"], constants["xMax"], psi.shape[1])

# Potential function:

import base64
import marshal

import jax.numpy as jnp

# Recreate the potential function
decodedPotential = base64.b64decode(metadata["potential"])
potentialCode = marshal.loads(decodedPotential)
potential = types.FunctionType(potentialCode, globals())

# Recreate the wave function
decodedWaveFunction = base64.b64decode(metadata["wave_function"])
waveFunctionCode = marshal.loads(decodedWaveFunction)
waveFunction = types.FunctionType(waveFunctionCode, globals())

V = np.zeros_like(psi)
for j in J:
    V[j] = potential(X, T[j])

potentialPlot0 = plt.figure()
plt.plot(X, V[0])
plt.xlabel("Position [a.u.]")
plt.ylabel("Potential [a.u.]")
plt.title("Potential at time t=0")
plt.savefig(os.path.join(args.path, "potential0.png"))

waveFunctionPlot0 = plt.figure()
plt.plot(X, np.abs(psi[0]) ** 2)
plt.xlabel("Position [a.u.]")
plt.ylabel("Wave function [a.u.]")
plt.title("Wave function at time t=0")
plt.savefig(os.path.join(args.path, "waveFunction0.png"))


from integrals import computeEnergy, computeIntegral

nrg = np.zeros_like(T)
for j in J:
    nrg[j] = computeEnergy(X, T[j], psi[j], V[j], constants["dx"], constants["g"], constants["mass"], constants["hbar"])

energy = plt.figure()
plt.plot(T, nrg)
plt.xlabel("Time [a.u.]")
plt.ylabel("Energy [a.u.]")
plt.title("Evolution of the energy")
plt.savefig(os.path.join(args.path, "energy.png"))

# Plot the evolution of the norm
norm = np.zeros_like(T)
for j in J:
    norm[j] = computeIntegral(X, np.abs(psi[j]) ** 2, constants["dx"])

normPlot = plt.figure()
plt.plot(T, norm)
plt.xlabel("Time [a.u.]")
plt.ylabel("Norm [a.u.]")
plt.title("Evolution of the norm")
plt.savefig(os.path.join(args.path, "norm.png"))


# Norm theoretical
normTeo = np.zeros_like(T)
for j in J:
    normTeo[j] = computeIntegral(X, np.abs(teo[j]) ** 2, constants["dx"])

normTeoPlot = plt.figure()
plt.plot(T, normTeo)
plt.xlabel("Time [a.u.]")
plt.ylabel("Norm [a.u.]")
plt.title("Evolution of the norm (theoretical)")


# Plot the evolution of the similarity
similarity = np.zeros_like(T)
for j in J:
    similarity[j] = computeIntegral(
        X, np.abs(psi[j] * np.conj(teo[j]) / np.sqrt(norm[j] * normTeo[j])), constants["dx"]
    )

similarityPlot = plt.figure()
plt.plot(T, similarity)
plt.xlabel("Time [a.u.]")
plt.ylabel("Similarity [a.u.]")
plt.title("Evolution of the similarity")
plt.savefig(os.path.join(args.path, "similarity.png"))

# Make the report (pdf)

import inspect
import io
import textwrap

from fpdf import FPDF

pdf = FPDF()

# Add the title
pdf.add_page()
pdf.set_font(font, size=20)
pdf.cell(200, 10, txt="Report for %s" % os.path.abspath(args.path).split(os.sep)[-1], ln=1, align="C")

# Add information about the simulation
pdf.set_font(font, size=12)
pdf.cell(200, 10, txt="Simulated with:  " + metadata["simulator"], ln=1, align="C")

# Add the simulation constants
pdf.cell(200, 10, txt="Simulation constants:", ln=1, align="L")
pdf.set_font(codeFont, size=10)
# Wrap into three columns
for i, (key, value) in enumerate(constants.items()):
    if i % 3 == 0 and not i == 0:
        pdf.cell(0, 10, txt="", ln=1, align="L")
    if isinstance(value, float):
        value = "%.3f" % value
    pdf.cell(60, 10, txt="%s: %s" % (key, value), ln=0, align="L")
pdf.cell(0, 10, txt="", ln=1, align="L")

# Add the functions used
pdf.set_font(font, size=12)
pdf.cell(200, 10, txt="Wave function:", ln=1, align="L")
pdf.set_font(codeFont, size=10)

text = inspect.getsource(waveFunction)
text = textwrap.fill(text, 80).split("\n")
for t in text:
    pdf.cell(200, 10, txt=t, ln=1, align="L")

pdf.set_font(font, size=12)
pdf.cell(200, 10, txt="Potential function:", ln=1, align="L")
pdf.set_font(codeFont, size=10)

text = inspect.getsource(potential)
text = textwrap.fill(text, 80).split("\n")
for t in text:
    pdf.cell(200, 10, txt=t, ln=1, align="L")


# Add the plots


# Potential
pdf.image(os.path.join(args.path, "potential0.png"), w=140)

# Wave function
pdf.image(os.path.join(args.path, "waveFunction0.png"), w=140)


pdf.set_font(font, size=20)
pdf.cell(200, 10, txt="Results", ln=1, align="L")

# Evolution
pdf.image(os.path.join(args.path, "evolution.png"), w=140)

# Theoretical
pdf.image(os.path.join(args.path, "theoretical.png"), w=140)

# Similarity
pdf.image(os.path.join(args.path, "similarity.png"), w=140)

# Energy
pdf.image(os.path.join(args.path, "energy.png"), w=140)

# Norm
pdf.image(os.path.join(args.path, "norm.png"), w=140)


# Save the pdf
pdf.output(os.path.join(args.path, "report.pdf"))
