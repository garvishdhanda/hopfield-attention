import torch
import torch.nn.functional as F

torch.manual_seed(0)
n, d = 4, 8
X = torch.randn(1, n, d)   # PyTorch's attention expects a batch dimension
q = torch.randn(1, 1, d)

beta = 1.0

# your version
scores = beta * (q @ X.transpose(-2, -1))
weights = torch.softmax(scores, dim=-1)
out_mine = weights @ X

# PyTorch's built-in scaled dot-product attention, with scaling disabled
# so we can supply our own beta and compare like-for-like
out_torch = F.scaled_dot_product_attention(q, X, X, scale=beta)

print("mine: ", out_mine)
print("torch:", out_torch)
print("max difference:", (out_mine - out_torch).abs().max().item())