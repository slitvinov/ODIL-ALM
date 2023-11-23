import torch
import numpy as np
import math
import torch.optim
import matplotlib.pyplot as plt


def u_exact(x, t):
    return np.sin(math.pi * x) * math.cos(2 * math.pi * t) + np.sin(
        4 * math.pi * x) * math.cos(8 * math.pi * t) / 2

def _closure():
    u_tt = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) * (nt - 1)**2
    u_xx = (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) * (nx - 1)**2
    pde = torch.mean((u_tt - 4 * u_xx)**2)
    ic = torch.mean((u[0, :] - u_e)**2)
    bc = torch.mean(u[:, 0]**2 + u[:, -1]**2)
    loss = pde + li * ic + lb * bc + 0.5 * (mi * ic**2 + mb * bc**2)
    return loss, pde, ic, bc

def closure():
    opt.zero_grad()
    loss, *rest = _closure()
    loss.backward()
    return loss

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)
nx, nt = 26, 26
b = math.sqrt(6 / (nt * nx))
u = torch.zeros((nt, nx), requires_grad=True)
opt = torch.optim.LBFGS([u], tolerance_grad=0, tolerance_change=0)
li, lb = 0, 0
mi, mb = 1, 1
epsilon = 1e-16
mu_max = 10000
eta = 0
u_e = torch.tensor(u_exact(np.linspace(0, 1, nx), 0))
for epoch in range(1, 500):
    opt.step(closure)
    loss, pde, ic, bc = _closure()
    with torch.no_grad():
        penalty = ic**2 + bc**2
        if penalty >= eta/16 and penalty > epsilon:
            mi = min(2 * mi, mu_max)
            mb = min(2 * mb, mu_max)
            li += mi * ic
            lb += mb * bc
        eta = penalty
    if epoch % 100 == 1:
        print(f": {epoch:3d} {pde.detach().item():2.3e}")

x = np.fromiter((i / (nx - 1) for i in range(nx)), dtype=float)
for ti in 0, nt//2, nt - 1:
    time = ti / (nt - 1)
    plt.plot(x, u[ti, :].detach().numpy(), "ko", markerfacecolor="none")
    plt.plot(x, u_exact(x, time), "k-")
plt.show()
