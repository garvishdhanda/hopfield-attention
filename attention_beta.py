import torch

torch.manual_seed(0)
X = torch.rand(3, 4)
scores = X @ X.T

for beta in [0.1, 1, 5, 20]:
    weights = torch.softmax(beta * scores, dim=1)
    print("beta =", beta)
    print(weights)
    print()