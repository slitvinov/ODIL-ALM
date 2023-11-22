import torch
import numpy as np
import math
from torch.optim import LBFGS

def u_exact(x, t):
    return torch.sin(math.pi * x) * torch.cos(2 * math.pi * t) + torch.sin(
        4 * math.pi * x) * torch.cos(8 * math.pi * t) / 2

torch.set_default_dtype(torch.float64)
nx, nt = 26, 26
x0 = torch.linspace(0, 1, nx).reshape(-1, 1)
x = x0.repeat(nt, 1).reshape(-1, 1)
t0 = torch.linspace(0, 1, nt).reshape(-1, 1)
t = t0.repeat(1, nx).reshape(-1, 1)
b = math.sqrt(6 / (nt * nx))
u = torch.tensor(np.random.uniform(-b, b, (nt, nx)), requires_grad=True)
opt_lbfgs = LBFGS([u], line_search_fn="strong_wolfe")
Mu = torch.ones(2, 1)
Lambda = torch.zeros(2, 1)
epsilon = 1e-8
mu_max = 1e4
eta = 0
x_ic = x.reshape(nt, nx)[0, :]
t_ic = t.reshape(nt, nx)[0, :]
u_e = u_exact(x_ic, t_ic)
for epoch in range(1, 501):
    def _closure():
        u_xx = (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) * nx**2
        u_tt = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) * nt**2
        pde_loss = torch.mean((u_tt - 4 * u_xx)**2)
        u_bc = torch.cat((u[:, 0], u[:, -1]))
        bc_loss = u_bc**2
        ic_loss = (u[0, :] - u_e)**2
        avg_bc_loss = torch.mean(bc_loss).reshape(1, 1)
        avg_ic_loss = torch.mean(ic_loss).reshape(1, 1)
        constr = torch.cat((avg_ic_loss, avg_bc_loss), 0)
        penalty = constr.pow(2).sum()
        loss = pde_loss + (
            Lambda * constr).sum() + 0.5 * (Mu * constr.pow(2)).sum()
        return pde_loss, constr, penalty, loss
    def closure():
        if torch.is_grad_enabled():
            opt_lbfgs.zero_grad()
        pde_loss, constr, penalty, loss = _closure()
        if loss.requires_grad:
            loss.backward()
        return loss
    opt_lbfgs.step(closure)
    pde_loss, constr, penalty, loss = _closure()
    with torch.no_grad():
        if (torch.sqrt(penalty) >= 0.25 * eta) and (torch.sqrt(penalty)
                                                    > epsilon):
            Mu = 2 * Mu
            Mu[Mu > mu_max] = mu_max
            Lambda += Mu * constr
        eta = torch.sqrt(penalty)
    if epoch % 20 == 1:
        print(
            f": {epoch:3d} {pde_loss.detach().item():2.3e}"
        )
