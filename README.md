<img src="https://user-images.githubusercontent.com/40344474/223997006-4c4ae49e-3a96-43cb-95ef-9a9727b502d5.png#gh-light-mode-only" width=30% align="center">
<img src="https://user-images.githubusercontent.com/40344474/223997775-5516082f-edf9-4c87-9c30-a5734f2d2321.png#gh-dark-mode-only" width=30% align="center">

## Simulation of a Bose Einstein Condensate

A simulation program for Bose Einstein Condensates (BECs) that uses the **Crank-Nicolson method** is a powerful tool for investigating the properties of these exotic quantum systems. This program allows the user to specify any wave function and potential for the BEC, and then use the Crank-Nicolson method to simulate the dynamics of the system over time. The Crank-Nicolson method is a numerical technique for solving partial differential equations that is especially well-suited for simulating quantum systems like BECs, where the wave function is described by a complex-valued function that satisfies the **Gross-Pitaevskii equation**.

To accelerate the simulation process, this program uses **JAX** instead of simple numpy to perform calculations on the GPU. JAX is a machine learning library that provides automatic differentiation and high-performance computing features. It is particularly useful for accelerating scientific computing tasks that involve large arrays and complex mathematical operations, making it an ideal choice for simulating BECs. By using JAX, this simulation program can run simulations faster and more efficiently than if it were relying solely on numpy, allowing researchers to explore a wider range of BEC scenarios and investigate more complex systems.

This program is being used to simulate Bose-Einstein Condensate solutions such as **bright and dark solitons** on a Kapitza pendulum-like potential. This potential creates a periodic modulation in space and time, which can be used to study the behavior of BECs in non-trivial geometries.

To create a new simulation, one can just make a new **.py** file anywhere, onto which one must add two functions: `waveFunction(x,t) -> jax.numpy.ndarray`, which takes `x : jax.numpy.ndarray` and `t : float` as the space grid (in meters) and the time (in seconds) and returns the complex wave function evaluated over all the position lattice x at time t; and `V(x, t) -> jax.numpy.ndarray`, which also takes `x` and `t` and shall return the real function for the potential evaluated over all x and at time t.

One can then run the simulation by executing the following command:

```bash
python3 run.py -i <path> [-v <LEVEL>] [-t] [-inan] [-oc <const=value>]
```
Where `-v` sets the verbosity level, `-t/--show-theoretical` shows the theoretical wave function over time (one can define only the wave function for t=0, which is the one used for the simulation, or define how it should evolve over time, for comparison reasons on the animation.), `-inan/--ignore-nan` ignores the error for NaN encountered and `-oc/--override-constants` allows the user to change the value for a global constant by providing it in the form of `constName=value`.
