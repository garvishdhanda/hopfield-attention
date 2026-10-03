# Hopfield retrieval and attention

Small experiments on the link between Modern Hopfield Networks and Transformer attention.

## Key idea
One retrieval step, softmax(beta * X @ xi) @ X, has the same form as one self-attention step. Here beta is the inverse temperature from statistical mechanics.

## Verification
I implemented self-attention and a Hopfield retrieval step independently, then compared both against PyTorch's own `scaled_dot_product_attention`. All three agree to within floating-point rounding (~1e-8), confirming the equivalence directly rather than assuming it.

## Capacity experiment
I stored random +1/-1 patterns, corrupted 25% of one stored pattern, and checked for exact recovery after one step (200 trials per point, beta = 5).

![capacity plot](capacity_plot.png)

- Longer patterns store far more memories: at N = 1000, d = 64 recovers almost every time, d = 32 rarely, d = 16 never.
- Measured success follows a simple theory curve: the chance that no other stored pattern scores at least as well as the target.
- Low beta blurs everything into an average, and high beta behaves like picking the best match.

## Files
- attention_basic.py, attention_beta.py: self-attention from scratch, effect of beta
- hopfield_retrieval.py: first Hopfield retrieval demo
- capacity_experiment.py, capacity_experiment2.py, capacity_plot.py: capacity vs N, beta, d, with theory comparison
- attention_vs_hopfield.py: attention and Hopfield step shown numerically identical
- vs_pytorch_attention.py: verified against PyTorch's real attention implementation