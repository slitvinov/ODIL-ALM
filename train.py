import torch
import numpy as np
import math
import torch.optim
import matplotlib.pyplot as plt


def exact(t, x):
    return math.sin(math.pi * x) * math.cos(2 * math.pi * t) + math.sin(
        4 * math.pi * x) * math.cos(8 * math.pi * t) / 2


class Closure:

    def __call__(self):
        opt.zero_grad()
        u_tt = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) * (nt - 1)**2
        u_xx = (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) * (nx - 1)**2
        self.pde = torch.mean((u_tt - 4 * u_xx)**2)
        self.ic = torch.mean((u[0, :] - u_ic)**2)
        self.bc = torch.mean(u[:, 0]**2 + u[:, -1]**2)
        loss = self.pde + li * self.ic + lb * self.bc + 0.5 * (
            mi * self.ic**2 + mb * self.bc**2)
        loss.backward()
        return loss


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)
nt, nx = 30, 15
b = math.sqrt(6 / (nt * nx))
u = torch.zeros((nt, nx), requires_grad=True)
opt = torch.optim.LBFGS([u],
                        tolerance_grad=0,
                        tolerance_change=0,
                        history_size=15)
li, lb = 0, 0
mi, mb = 1, 1
epsilon = 1e-16
mu_max = 10000
eta = 0
x = np.linspace(0, 1, nx)
u_ic = torch.tensor([exact(0, x) for x in x])
c = Closure()
for epoch in range(51):
    opt.step(c)
    with torch.no_grad():
        penalty = c.ic**2 + c.bc**2
        if penalty >= eta / 16 and penalty > epsilon:
            mi = min(2 * mi, mu_max)
            mb = min(2 * mb, mu_max)
            li += mi * c.ic
            lb += mb * c.bc
        eta = penalty
    if epoch % 10 == 0:
        print(f": {epoch + 1:3d} {c.pde.detach().item():2.3e}")

u = u.detach().numpy()
t = np.linspace(0, 1, nt)
u_e = [[exact(t, x) for x in x] for t in t]
print("relative l2 error: %.6e" %
      (np.linalg.norm(u - u_e) / np.linalg.norm(u_e)))
for ti in 0, nt // 2, 3 * nt // 4, nt - 1:
    time = ti / (nt - 1)
    plt.plot(x, u[ti], "ko", markerfacecolor="none")
    plt.plot(x, [exact(time, x) for x in x], "k-")
plt.savefig("train.png")
