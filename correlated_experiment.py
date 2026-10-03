import torch

torch.manual_seed(0)
d = 32
trials = 200
flips = 8
beta = 1.0

def make_patterns(N, p):
    # every pattern is a copy of one shared base, with each bit flipped with probability p
    # p = 0.5 gives independent random patterns (your earlier case); smaller p = more alike
    base = torch.sign(torch.randn(d))
    noise = torch.where(torch.rand(N, d) < p, -1.0, 1.0)
    return base * noise

def success_rate(N, p, steps):
    ok = 0
    for _ in range(trials):
        X = make_patterns(N, p)
        xi = X[0].clone()
        idx = torch.randperm(d)[:flips]
        xi[idx] = -xi[idx]
        for _ in range(steps):
            w = torch.softmax(beta * (X @ xi), dim=0)
            xi = torch.sign(w @ X)
        if (xi == X[0]).all():
            ok += 1
    return ok / trials

ps = [0.5, 0.4, 0.3, 0.2]
print("columns: p =", ps, " (smaller p = more correlated)")
for steps in [1, 3]:
    print("\nsteps =", steps)
    for N in [5, 20, 100]:
        row = [success_rate(N, p, steps) for p in ps]
        print("N =", N, row)