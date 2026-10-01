import torch

torch.manual_seed(0)
N, d = 5, 32
X = torch.sign(torch.randn(N, d))   # 5 stored patterns, 32 entries each, all +1 or -1

xi = X[0].clone()                   # start from pattern 0
flip = torch.randperm(d)[:8]
xi[flip] = -xi[flip]                # corrupt 8 of the 32 entries

print("before: matches pattern 0 on", (xi == X[0]).sum().item(), "of", d)

beta = 1.0
weights = torch.softmax(beta * (X @ xi), dim=0)
print("weights:", weights)

xi_new = weights @ X
retrieved = torch.sign(xi_new)
print("after one step: matches pattern 0 on", (retrieved == X[0]).sum().item(), "of", d)