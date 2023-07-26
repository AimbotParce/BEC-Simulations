This is a study on how a soliton falls on the side of an inverted potential depending on the ratio between the potential's length and the soliton's healing length.

Keep in mind that in the dimensionless equation, the potential's length is always 1, and the soliton's length is the following:

$$
	\tilde{\xi} = \frac{\hbar}{a_0\sqrt{2m n_i gN}} = \frac{\xi}{a_0}
$$

This means, the $\xi$ itself is in terms of the potential's length. In this units, the interaction strength reads as follows:

$$
	\tilde{U_0} = - \frac{1}{2\tilde{n_i}\tilde{\xi}^2}
$$

Each of the files in this directory is named after the relation $\xi/a_0$ on the simulation they are holding. All of the simulations come from the same python document, and it must be called with the **override constants** parameter, setting its $\xi$ value.

All the simulations share something in common: $\tilde{n_i}$ is taken equal to 1, and the soliton is displaced from the center of the potential by $0.1a_0$.