import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

torch.manual_seed(0)
digits = load_digits()
images = digits.images          # (1797, 8, 8), pixel values 0..16
labels = digits.target

def binarize(img):
    # turn the 8x8 image into a 64-long vector of +1 / -1
    return torch.tensor(np.where(img.flatten() > 8, 1.0, -1.0), dtype=torch.float32)

# store one example of each digit 0-9
X = torch.stack([binarize(images[np.where(labels == k)[0][0]]) for k in range(10)])
d = X.shape[1]
beta = 1.0
steps = 3

def retrieve(xi):
    for _ in range(steps):
        w = torch.softmax(beta * (X @ xi), dim=0)
        xi = torch.sign(w @ X)
    return xi

def corrupt(x, flips):
    xi = x.clone()
    idx = torch.randperm(d)[:flips]
    xi[idx] = -xi[idx]
    return xi

# 1) success rate vs number of corrupted pixels
print("flips -> fraction of exact recoveries (10 stored digits, 100 trials each)")
for flips in [4, 8, 12, 16, 20]:
    ok = 0
    for k in range(10):
        for _ in range(100):
            if (retrieve(corrupt(X[k], flips)) == X[k]).all():
                ok += 1
    print(flips, ok / 1000)

# 2) picture: stored / corrupted / restored
flips = 12
fig, axes = plt.subplots(3, 10, figsize=(14, 4.5))
for k in range(10):
    c = corrupt(X[k], flips)
    r = retrieve(c)
    for row, img in enumerate([X[k], c, r]):
        axes[row, k].imshow(img.reshape(8, 8), cmap="gray_r")
        axes[row, k].axis("off")
axes[0, 0].set_title("stored", loc="left")
axes[1, 0].set_title("corrupted", loc="left")
axes[2, 0].set_title("restored", loc="left")
plt.tight_layout()
plt.savefig("digits_demo.png", dpi=150)