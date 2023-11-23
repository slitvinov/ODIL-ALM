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
        self.pde = torch.sum((u_tt - 4 * u_xx)**2)
        self.g = [(u[0] - u_ic), u[:, 0], u[:, -1]]
        self.g2 = sum(torch.sum(g**2) for g in self.g)
        loss = self.pde + sum(z @ g for z, g in zip(z, self.g)) + mu * self.g2
        loss.backward()
        return loss


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)
nt, nx = 30, 15
u = torch.zeros((nt, nx), requires_grad=True)
opt = torch.optim.LBFGS([u],
                        tolerance_grad=0,
                        tolerance_change=0,
                        history_size=100)
z = [torch.zeros(nx), torch.zeros(nt), torch.zeros(nt)]
mu = 1
x = np.linspace(0, 1, nx)
u_ic = torch.tensor([exact(0, x) for x in x])
c = Closure()
prev = None
for epoch in range(101):
    opt.step(c)
    with torch.no_grad():
        z = [z + 2 * mu * g for z, g in zip(z, c.g)]
        if prev != None and 16 * c.g2 < prev:
            mu *= 2
            print("mu <- ", mu)
        prev = c.g2
    if epoch % 10 == 0:
        print(f"{epoch + 1:6d} {c.pde.detach().item():2.3e}")

u = u.detach().numpy()
t = np.linspace(0, 1, nt)
u_e = [[exact(t, x) for x in x] for t in t]
print("relative l2 error: %.6e" %
      (np.linalg.norm(u - u_e) / np.linalg.norm(u_e)))
for ti in 0, nt // 2, 3 * nt // 4, nt - 1:
    time = ti / (nt - 1)
    plt.plot(x, u[ti], "ko", markerfacecolor="none")
    plt.plot(x, [exact(time, x) for x in x], "k-")
plt.savefig("alm.png")
