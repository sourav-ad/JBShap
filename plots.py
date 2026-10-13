from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

path = r'/home/sourav/arf_lik/JBShap'
LOW_Q = 0.10

csv_path = os.path.join(path, "weights", "ood_only", "all_unfair_newp_deep_10.csv")
plot_dir = os.path.join(path, "plots", "ood_only")
#plot_dir.mkdir(exist_ok=True)


#Methods in CSV, IN ROW ORDER
# Each method has two rows:
#   row 0 = real
#   row 1 = fake

methods = [
    "CES",
    "CES-Sup",
    "BShap",
    "RBShap",
    "JBShap",
    "RJBShap",
    "ARF-JBShap",
]

# Feature names, in your model order
features = [
    "Age",
    "Workclass",
    "Education-Num",
    "Marital Status",
    "Occupation",
    "Relationship",
    "Race",
    "Sex",
    "Capital Gain",
    "Capital Loss",
    "Hours per week",
    "Country",
]


#Load

data = np.loadtxt(csv_path, delimiter=",")

assert data.shape[0] == 2 * len(methods), (
    f"CSV has {data.shape[0]} rows, but "
    f"{len(methods)} methods imply {2 * len(methods)} rows."
)


#Select methods to plot


show_methods = [
    "BShap",
    "RBShap",
    "JBShap",
    "ARF-JBShap",
]


#Plot


for method in show_methods:

    idx = methods.index(method)

    normal = data[2 * idx]
    fake = data[2 * idx + 1]

    x = np.arange(len(features))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(
        x - width / 2,
        normal,
        width,
        label="Original model",
    )

    ax.bar(
        x + width / 2,
        fake,
        width,
        label="Fine-tuned model",
    )

    ax.axhline(0, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right")

    ax.set_ylabel("Normalized Shapley value")
    ax.set_title(method)
    ax.legend()

    fig.tight_layout()

    output = os.path.join(plot_dir, f"shap_values_{method}.png")
    fig.savefig(output, dpi=300, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {output}")

# ARF vs OOD density regions: PCA visualization

region_file = os.path.join(
    path,
    "weights",
    "density_regions_10.npz"
)

regions = np.load(region_file)

X_masked = regions["X_masked"]
arf_low = regions["arf_low"]
ood_low = regions["ood_low"]
both_low = regions["both_low"]
arf_only = regions["arf_only"]
ood_only = regions["ood_only"]


# Region labels
labels = np.full(X_masked.shape[0], "Neither", dtype=object)
labels[arf_only] = "ARF low only"
labels[ood_only] = "OOD low only"
labels[both_low] = "Both low"


# PCA
pca = PCA(n_components=2)
Z = pca.fit_transform(X_masked)


# Plot only a sample so figure does not contain 500k points
rng = np.random.default_rng(10)

n_plot = min(30000, X_masked.shape[0])

idx = rng.choice(
    X_masked.shape[0],
    size=n_plot,
    replace=False
)


fig, ax = plt.subplots(figsize=(8, 6))

for region in [
    "Neither",
    "ARF low only",
    "OOD low only",
    "Both low",
]:

    keep = labels[idx] == region

    ax.scatter(
        Z[idx[keep], 0],
        Z[idx[keep], 1],
        s=8,
        alpha=0.35,
        label=region
    )


ax.set_xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)"
)

ax.set_ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)"
)

ax.set_title("ARF and OOD low-density regions")

ax.legend()

fig.tight_layout()

output = os.path.join(
    plot_dir,
    "density_regions_pca.png"
)

fig.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"Saved: {output}")


# Density rank agreement plot

arf_scores = regions["arf_scores"]
ood_scores = regions["ood_scores"]


def percentile_rank(x):
    order = np.argsort(np.argsort(x))
    return order / (len(x) - 1)


arf_rank = percentile_rank(arf_scores)
ood_rank = percentile_rank(ood_scores)


rng = np.random.default_rng(10)
idx = rng.choice(
    len(arf_rank),
    size=min(30000, len(arf_rank)),
    replace=False
)

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    ood_rank[idx],
    arf_rank[idx],
    s=7,
    alpha=0.25
)

ax.axvline(LOW_Q if "LOW_Q" in globals() else 0.10)
ax.axhline(LOW_Q if "LOW_Q" in globals() else 0.10)

ax.set_xlabel("OOD density percentile")
ax.set_ylabel("ARF density percentile")

ax.set_title("Agreement between OOD and ARF density rankings")

fig.tight_layout()

output = os.path.join(
    plot_dir,
    "density_rank_agreement.png"
)

fig.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"Saved: {output}")