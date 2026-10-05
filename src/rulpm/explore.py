"""Unsupervised exploration: PCA, DBSCAN operating regimes, t-SNE.

Usage:
    python -m rulpm.explore --subset FD002 --data-dir data/raw
Writes results/figs/<subset>_*.png and results/<subset>_explore.json
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import DBSCAN  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

from .data import OP_COLS, add_labels, load_train  # noqa: E402
from .features import RegimeNormalizer, informative_sensors  # noqa: E402
from .train_baseline import N_REGIMES  # noqa: E402


DBSCAN_EPS = 1.0  # in raw setting units (altitude, Mach number, throttle angle)


def count_regimes(train, eps: float = DBSCAN_EPS, min_samples: int = 50):
    """Count operating regimes with DBSCAN on the raw operating settings.

    Raw units (not standardized) on purpose: in FD001/FD003 the settings are one condition
    plus tiny quantized noise; standardizing blows that noise up to unit scale and DBSCAN
    then reports spurious clusters. In raw units the noise is far below eps, while the real
    FD002/FD004 conditions are many units apart.
    """
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(train[OP_COLS].to_numpy())
    return int(len(set(labels)) - (1 if -1 in labels else 0)), float(np.mean(labels == -1))


def explore(data_dir: str, subset: str, out_dir: str = "results", tsne_n: int = 3000, seed: int = 0):
    train = add_labels(load_train(data_dir, subset))
    sensors = informative_sensors(train)
    figs = Path(out_dir, "figs")
    figs.mkdir(parents=True, exist_ok=True)
    summary = {"subset": subset, "n_sensors_kept": len(sensors)}

    # 1) DBSCAN on operating settings: how many distinct regimes are there?
    n_reg, noise = count_regimes(train)
    summary["dbscan_regimes"] = n_reg
    summary["dbscan_noise_fraction"] = noise
    summary["dbscan_eps_raw_units"] = DBSCAN_EPS

    # 2) PCA on regime-normalized sensors
    norm = RegimeNormalizer(N_REGIMES.get(subset, 1)).fit(train, sensors)
    Z = norm.transform(train)[sensors].to_numpy()
    pca = PCA(n_components=min(10, Z.shape[1])).fit(Z)
    summary["pca_explained_variance"] = [float(v) for v in pca.explained_variance_ratio_]
    P = pca.transform(Z)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    ax[0].set(xlabel="components", ylabel="cumulative explained variance", title="PCA")
    sc = ax[1].scatter(P[:, 0], P[:, 1], c=train["rul"], s=2, cmap="viridis")
    ax[1].set(xlabel="PC1", ylabel="PC2", title="Sensors in PCA space, colored by RUL")
    fig.colorbar(sc, ax=ax[1], label="RUL")
    fig.tight_layout()
    fig.savefig(figs / f"{subset}_pca.png", dpi=130)
    plt.close(fig)

    # 3) t-SNE on a subsample of the PCA-reduced data
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(P), size=min(tsne_n, len(P)), replace=False)
    T = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(P[idx])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(T[:, 0], T[:, 1], c=train["rul"].to_numpy()[idx], s=4, cmap="viridis")
    ax.set(title="t-SNE of sensor state (colored by RUL)")
    fig.colorbar(sc, ax=ax, label="RUL")
    fig.tight_layout()
    fig.savefig(figs / f"{subset}_tsne.png", dpi=130)
    plt.close(fig)

    Path(out_dir, f"{subset}_explore.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=list(N_REGIMES))
    ap.add_argument("--data-dir", default="data/raw")
    a = ap.parse_args()
    explore(a.data_dir, a.subset)
