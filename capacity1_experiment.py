import torch

torch.manual_seed(0)
d = 32
trials = 200
flips = 8

def success_rate(N, beta, steps):
    ok = 0
    for _ in range(trials):
        X = torch.sign(torch.randn(N, d))
        xi = X[0].clone()
        idx = torch.randperm(d)[:flips]
        xi[idx] = -xi[idx]
        for _ in range(steps):                      # repeat the update
            w = torch.softmax(beta * (X @ xi), dim=0)
            xi = torch.sign(w @ X)                  # output becomes the next query
        if (xi == X[0]).all():
            ok += 1
    return ok / trials

betas = [0.01, 0.1, 0.5, 1.0]
print("columns: beta =", betas)
for steps in [1, 2, 3, 5]:
    print("\nsteps =", steps)
    for N in [5, 20, 100, 500]:
        row = [success_rate(N, beta, steps) for beta in betas]
        print("N =", N, row)