import torch

# A 2D tensor: think of it as 3 "words", each represented by 4 numbers
X = torch.rand(3, 4)
print("X shape:", X.shape)
print(X)

# Matrix multiply X with its own transpose
scores = X @ X.T
print("scores shape:", scores.shape)
print(scores)