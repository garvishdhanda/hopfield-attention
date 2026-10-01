import torch

torch.manual_seed(0)
d = 32
trials = 200
flips = 8

def success_rate(N, beta):
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

print("columns: beta = 0.01, 0.1, 0.5, 1.0")
for N in [5, 20, 100, 500]:
    row = [success_rate(N, beta) for beta in [0.01, 0.1, 0.5, 1.0]]
    print("N =", N, row)