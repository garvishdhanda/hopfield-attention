import torch

torch.manual_seed(0)
trials = 200

def success_rate(N, beta, d):
    flips = d // 4          # corrupt 25% of entries, same noise level for every d
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

betas = [1.0, 2.0, 5.0, 10.0]
for d in [32, 64]:
    print("d =", d, " columns: beta =", betas)
    for N in [20, 100, 500]:
        print("  N =", N, [success_rate(N, b, d) for b in betas])
        