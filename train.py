import torch
import numpy as np
import math
from torch.optim import LBFGS


def u_exact(x, t):
    return np.sin(math.pi * x) * math.cos(2 * math.pi * t) + np.sin(
        4 * math.pi * x) * math.cos(8 * math.pi * t) / 2


torch.set_default_dtype(torch.float64)
nx, nt = 26, 26
b = math.sqrt(6 / (nt * nx))
u = torch.tensor(np.random.uniform(-b, b, (nt, nx)), requires_grad=True)
opt = LBFGS([u], line_search_fn="strong_wolfe")
Mu = torch.ones(2, 1)
Lambda = torch.zeros(2, 1)
epsilon = 1e-8
mu_max = 1e4
eta = 0
x = np.linspace(0, 1, nx)
u_e = torch.tensor(u_exact(x, 0))
for epoch in range(1, 501):

    def _closure():
        u_xx = (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) * nx**2
        u_tt = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) * nt**2
        pde_loss = torch.mean((u_tt - 4 * u_xx)**2)
        bc_loss = torch.cat((u[:, 0], u[:, -1]))**2
        ic_loss = (u[0, :] - u_e)**2
        avg_bc_loss = torch.mean(bc_loss).reshape(1, 1)
        avg_ic_loss = torch.mean(ic_loss).reshape(1, 1)
        constr = torch.cat((, avg_bc_loss), 0)
        penalty = torch.sum(constr**2)
        loss = pde_loss + torch.sum(Lambda * constr + 0.5 * (Mu * constr**2))
        return pde_loss, constr, penalty, loss

    def closure():
        opt.zero_grad()
        pde_loss, constr, penalty, loss = _closure()
        loss.backward()
        return loss

    opt.step(closure)
    pde_loss, constr, penalty, loss = _closure()
    with torch.no_grad():
        if torch.sqrt(penalty) >= eta / 4 and torch.sqrt(penalty) > epsilon:
            Mu = 2 * Mu
            Mu[Mu > mu_max] = mu_max
            Lambda += Mu * constr
        eta = torch.sqrt(penalty)
    if epoch % 20 == 1:
        print(f": {epoch:3d} {pde_loss.detach().item():2.3e}")
