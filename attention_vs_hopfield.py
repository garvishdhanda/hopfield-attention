import torch

torch.manual_seed(0)
n, d = 4, 8          # 4 tokens, 8-dim embeddings

X = torch.randn(n, d)   # stored patterns / keys & values
q = torch.randn(1, d)   # a single query

def self_attention(q, K, V, beta):
    scores = beta * (q @ K.T)          # (1, n)
    weights = torch.softmax(scores, dim=1)
    return weights @ V                 # (1, d)

def hopfield_step(xi, X, beta):
    scores = beta * (X @ xi.T).T       # (1, n)
    weights = torch.softmax(scores, dim=1)
    return weights @ X                 # (1, d)

beta = 1.0
out_attn = self_attention(q, X, X, beta)
out_hop = hopfield_step(q, X, beta)

print("attention output:", out_attn)
print("hopfield output: ", out_hop)
print("max difference:  ", (out_attn - out_hop).abs().max().item())