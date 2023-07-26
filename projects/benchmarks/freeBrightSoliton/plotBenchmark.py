# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
benchmark = pd.read_csv("benchmark.txt", sep="\t", comment="#", header=None, names=["dx", "time", "args"])

# %%
benchmark["xSize"] = (20 / benchmark["dx"].astype(float)).astype(int)

# %%
# Plot the time vs xSize
plt.figure(figsize=(10, 6))

plt.plot(benchmark["xSize"][:5], benchmark["time"][:5], label="CPU")
plt.plot(benchmark["xSize"][5:], benchmark["time"][5:], label="GPU")


# plt.yscale("log")
plt.xlabel("xSize")
plt.ylabel("Simulation Time (s)")
plt.title("Time vs xSize")
plt.legend()
plt.savefig("time_vs_xSize.png")

# %%
