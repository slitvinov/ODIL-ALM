import time
import os
import torch
import numpy as np
from torch.optim import LBFGS
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from scipy.interpolate import griddata


def fetch_grid_data(domain, dom_grd):
    x_min = domain[0][0]
    x_max = domain[1][0]

    t_min = domain[0][1]
    t_max = domain[1][1]

    x0 = torch.linspace(x_min, x_max, dom_grd[0] + 1).reshape(-1, 1)
    x = x0.repeat(dom_grd[1] + 1, 1).reshape(-1, 1)
    t0 = torch.linspace(t_min, t_max, dom_grd[1] + 1).reshape(-1, 1)
    t = t0.repeat(1, dom_grd[0] + 1).reshape(-1, 1)
    return x, t


def u_exact(x, t):
    return torch.sin(torch.pi * x) * torch.cos(2 * torch.pi * t) + torch.sin(
        4 * torch.pi * x) * torch.cos(8 * torch.pi * t) / 2


def discretized_physics_loss(u, x, y, domain, dom_grd):
    deltax = (domain[1][0] - domain[0][0]) / (dom_grd[0] + 1)
    deltat = (domain[1][1] - domain[0][1]) / (dom_grd[1] + 1)
    u_mat = u.reshape(dom_grd[1] + 1, dom_grd[0] + 1)

    u_xx = (u_mat[1:-1, 2:] - 2 * u_mat[1:-1, 1:-1] +
            u_mat[1:-1, :-2]) / deltax**2
    u_tt = (u_mat[2:, 1:-1] - 2 * u_mat[1:-1, 1:-1] +
            u_mat[:-2, 1:-1]) / deltat**2

    loss = (u_tt - 4 * u_xx).reshape(-1, 1).pow(2)
    return loss


def discretized_boundary_loss(u, dom_grd):
    u_mat = u.reshape(dom_grd[1] + 1, dom_grd[0] + 1)
    u_bc = torch.cat((u_mat[:, 0], u_mat[:, -1])).reshape(-1, 1)
    loss = (u_bc - 0.).pow(2)
    return loss


def discretized_initial_loss(u, x, t, dom_grd):
    u_mat = u.reshape(dom_grd[1] + 1, dom_grd[0] + 1)
    u_ic = u_mat[0, :].reshape(-1, 1)

    x_ic = x.reshape(dom_grd[1] + 1, dom_grd[0] + 1)[0, :].reshape(-1, 1)
    t_ic = t.reshape(dom_grd[1] + 1, dom_grd[0] + 1)[0, :].reshape(-1, 1)
    u_e = u_exact(x_ic, t_ic)
    loss = (u_ic - u_e).pow(2)
    return loss


def evaluate(u, x, y):
    u_star = u_exact(x, y)

    l2 = np.linalg.norm(u_star - u.detach(), 2) / np.linalg.norm(u_star, 2)
    linf = max(abs(u_star - u.detach().numpy())).item()
    return l2, linf


methodname = 'odil_alm'
torch.set_default_dtype(torch.float64)
domain = [[0., 0.], [1., 1.]]
dom_grd = [25, 25]
x_min = domain[0][0]
x_max = domain[1][0]

t_min = domain[0][1]
t_max = domain[1][1]

x0 = torch.linspace(x_min, x_max, dom_grd[0] + 1).reshape(-1, 1)
x = x0.repeat(dom_grd[1] + 1, 1).reshape(-1, 1)
t0 = torch.linspace(t_min, t_max, dom_grd[1] + 1).reshape(-1, 1)
t = t0.repeat(1, dom_grd[0] + 1).reshape(-1, 1)


b = torch.tensor(np.sqrt(6 / x_dm.shape[0]))
u_dm = (-b - b) * torch.rand(x_dm.shape) + b
u_dm = u_dm.requires_grad_(True)
opt_lbfgs = LBFGS([u_dm], line_search_fn="strong_wolfe")
num_lambda = 2
Mu = torch.ones(num_lambda, 1)
Lambda = torch.zeros(num_lambda, 1)
epsilon = torch.tensor(1e-8)
mu_max = torch.tensor(1e+4)
eta = torch.tensor(0.0)
for epoch in range(1, 501):
    def _closure():
        pde_loss = discretized_physics_loss(u_dm, x_dm, t_dm, domain,
                                            dom_grd)
        avg_pde_loss = torch.mean(pde_loss)

        bc_loss = discretized_boundary_loss(u_dm, dom_grd)
        ic_loss = discretized_initial_loss(u_dm, x_dm, t_dm, dom_grd)
        avg_bc_loss = torch.mean(bc_loss).reshape(1, 1)
        avg_ic_loss = torch.mean(ic_loss).reshape(1, 1)

        constr = torch.cat((avg_ic_loss, avg_bc_loss), 0)
        penalty = constr.pow(2).sum()
        loss = avg_pde_loss + (
            Lambda * constr).sum() + 0.5 * (Mu * constr.pow(2)).sum()
        return avg_pde_loss, constr, penalty, loss
    def closure():
        if torch.is_grad_enabled():
            opt_lbfgs.zero_grad()
        avg_pde_loss, constr, penalty, loss = _closure()
        if loss.requires_grad:
            loss.backward()
        return loss
    opt_lbfgs.step(closure)
    avg_pde_loss, constr, penalty, loss = _closure()
    with torch.no_grad():
        if (torch.sqrt(penalty) >= 0.25 * eta) and (torch.sqrt(penalty)
                                                    > epsilon):
            Mu = 2 * Mu
            Mu[Mu > mu_max] = mu_max
            Lambda += Mu * constr
        eta = torch.sqrt(penalty)
    if epoch % 20 == 1:
        print(
            f": {epoch:3d} {avg_pde_loss.detach().item():2.3e}"
        )
