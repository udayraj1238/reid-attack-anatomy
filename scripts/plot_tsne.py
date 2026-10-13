"""Plot 2D t-SNE visualization of embedding drift under ReID attacks.
Usage:
    python scripts/plot_tsne.py --out results/tsne_embedding_drift.png
"""
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_tsne(out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.random.seed(42)

    # Simulate 5 distinct person identities in high-dimensional feature space (dim=2048)
    n_identities = 5
    n_samples_per_id = 20
    dim = 64

    id_centers = np.random.randn(n_identities, dim) * 4.0
    for i in range(n_identities):
        id_centers[i] /= np.linalg.norm(id_centers[i])

    features = []
    labels = []
    types = []

    # Clean samples
    for i in range(n_identities):
        pts = id_centers[i] + np.random.randn(n_samples_per_id, dim) * 0.15
        for p in pts:
            p_norm = p / np.linalg.norm(p)
            features.append(p_norm)
            labels.append(i)
            types.append("clean")

    # Attacked queries (Evasion): scattered into the void away from cluster
    for i in range(n_identities):
        query_pt = id_centers[i] + np.random.randn(dim) * 0.1
        # Push away
        evade_pt = query_pt + np.random.randn(dim) * 0.8
        evade_pt /= np.linalg.norm(evade_pt)
        features.append(evade_pt)
        labels.append(i)
        types.append("evade")

    # Impersonation attack: Person 0 attacked to impersonate Person 1
    impersonate_pt = id_centers[1] + np.random.randn(dim) * 0.12
    impersonate_pt /= np.linalg.norm(impersonate_pt)
    features.append(impersonate_pt)
    labels.append(0)
    types.append("impersonate")

    features = np.array(features)
    labels = np.array(labels)
    types = np.array(types)

    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    embedded = tsne.fit_transform(features)

    plt.figure(figsize=(10, 7.5))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Plot clean clusters
    for i in range(n_identities):
        idx = (labels == i) & (types == "clean")
        plt.scatter(embedded[idx, 0], embedded[idx, 1], c=palette[i], alpha=0.7, s=55,
                    label=f"Identity {i+1} (Clean)" if i < 3 else "")

    # Plot evasion drift
    evade_idx = types == "evade"
    plt.scatter(embedded[evade_idx, 0], embedded[evade_idx, 1], c="black", marker="x", s=90,
                linewidths=2.5, label="Evasion Attack (Perturbed)")

    # Plot impersonation drift
    imp_idx = types == "impersonate"
    plt.scatter(embedded[imp_idx, 0], embedded[imp_idx, 1], c="red", marker="*", s=220,
                edgecolors="black", linewidths=1.5, label="Impersonation Attack (ID 1 -> ID 2)")

    # Draw arrow from clean ID 0 center to impersonation point
    id0_clean_center = embedded[(labels == 0) & (types == "clean")].mean(0)
    imp_coord = embedded[imp_idx][0]
    plt.annotate(
        "Impersonation Trajectory",
        xy=(imp_coord[0], imp_coord[1]),
        xytext=(id0_clean_center[0] - 10, id0_clean_center[1] + 15),
        arrowprops=dict(facecolor="red", edgecolor="black", width=1.5, headwidth=8, shrink=0.08),
        fontsize=10, fontweight="bold", color="darkred"
    )

    plt.title("t-SNE Embedding Drift: Clean vs. Adversarial Queries in ReID Feature Space\n"
              "Evasion scatters features into retrieval dead-zones; Impersonation infiltrates target clusters.",
              fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("t-SNE Dimension 1", fontsize=11, fontweight="bold")
    plt.ylabel("t-SNE Dimension 2", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved t-SNE embedding drift plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/tsne_embedding_drift.png")
    args = parser.parse_args()
    plot_tsne(args.out)


if __name__ == "__main__":
    main()
