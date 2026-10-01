import math
import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)
trials = 200
beta = 5.0

def success_rate(N, d):
    flips = d // 4
    ok = 0
    for _ in range(trials):
        X = torch.sign(torch.randn(N, d))
        xi = X[0].clone()
        idx = torch.randperm(d)[:flips]
        xi[idx] = -xi[idx]
        w = torch.softmax(beta * (X @ xi), dim=0)
        out = torch.sign(w @ X)
        if (out == X[0]).all():
            ok += 1
    return ok / trials

def theory(N, d):
    flips = d // 4
    p = sum(math.comb(d, k) for k in range(flips + 1)) / 2**d
    return (1 - p) ** (N - 1)

Ns = [10, 30, 100, 300, 1000]
for d, color in [(16, "tab:red"), (32, "tab:blue"), (64, "tab:green")]:
    measured = [success_rate(N, d) for N in Ns]
    predicted = [theory(N, d) for N in Ns]
    plt.plot(Ns, measured, "o", color=color, label=f"measured d={d}")
    plt.plot(Ns, predicted, "-", color=color, label=f"theory d={d}")

plt.xscale("log")
plt.xlabel("number of stored patterns N")
plt.ylabel("fraction perfectly retrieved")
plt.title("Hopfield retrieval: experiment vs simple theory")
plt.legend()
plt.savefig("capacity_plot.png", dpi=150)
plt.show()