# ODIL-ALM
Augmented Lagrangian Method (ALM) is applied in Optimizing a DIscrete Loss (ODIL) framework.
Considering wave equation, $$\frac{\delta^2 u}{\delta t^2} - 4 \frac{\delta^2 u}{\delta x^2} = 0,$$
The exact solution is $u(x,t) = \sin(\pi x) \cos(2\pi t) + \frac{1}{2} \sin(4\pi x)\cos(8\pi t)$, where $x \in [0, 1]$ and $t \in [0, 1]$.
The boundary condition is $u(0,t) = u(1,t) = 0$, and the initial condition is $u(x,0) = \sin(\pi x) + \frac{1}{2} \sin(4\pi x)$.
Central difference scheme is applied to discretize temporal and spatial terms. The mesh grid is (25, 25).
