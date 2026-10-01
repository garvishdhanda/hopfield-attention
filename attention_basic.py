import torch

torch.manual_seed(0)
X = torch.rand(3, 4)

scores = X @ X.T
weights = torch.softmax(scores, dim=1)
print("weights:")
print(weights)
print("row sums:", weights.sum(dim=1))

output = weights @ X
print("output shape:", output.shape)
print(output)