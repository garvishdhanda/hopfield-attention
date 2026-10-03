import matplotlib.pyplot as plt

steps = [1, 2, 3, 5]
data = {
    0.5: {5: [0.985, 0.99, 0.995, 0.985],
          20: [0.945, 0.94, 0.97, 0.93],
          100: [0.66, 0.775, 0.82, 0.79],
          500: [0.0, 0.15, 0.245, 0.265]},
    1.0: {5: [0.98, 0.995, 0.99, 0.995],
          20: [0.955, 0.945, 0.97, 0.965],
          100: [0.72, 0.74, 0.775, 0.795],
          500: [0.17, 0.29, 0.28, 0.30]},
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (beta, rows) in zip(axes, data.items()):
    for N, ys in rows.items():
        ax.plot(steps, ys, marker="o", label=f"N = {N}")
    ax.set_title(f"beta = {beta}")
    ax.set_xlabel("update steps")
    ax.set_xticks(steps)
axes[0].set_ylabel("exact recovery rate")
axes[0].legend()
plt.tight_layout()
plt.savefig("capacity1_plot.png", dpi=150)