"""Corner plot do ajuste MCMC — marginais 1D + pares 2D com contornos 1σ/2σ.

Lê o resultado do fit (JSON com `samples`+`parameterNames`) ou um checkpoint
(.npz), descarta os parâmetros fixos (colunas constantes), e desenha o triângulo.
Sem dependência de `corner`: matplotlib puro, mas com **suavização gaussiana**
(2D e 1D) — sem ela, com ~mil amostras em 40×40 células, os contornos saem
granulados (ruído de Poisson do histograma). Menos células + suavização dão o
aspecto liso dos corner plots de referência.

Uso:
    python3 scripts/cornerplot.py fit_joint.json saida.png
    python3 scripts/cornerplot.py joint_ckpt.npz saida.png   # preview do checkpoint
"""
from __future__ import annotations

import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import gaussian_filter, gaussian_filter1d
    _SMOOTH = True
except Exception:                                   # pragma: no cover
    _SMOOTH = False

BINS = 26          # células por eixo — casadas ao nº típico de amostras
SIGMA = 1.1        # suavização, em células


def load(path: str):
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        samples = d["samples"].reshape(-1, d["samples"].shape[-1])
        names = [str(n) for n in d["names"]]
        return samples, names
    d = json.loads(open(path).read())
    return np.asarray(d["samples"], dtype=float), list(d["parameterNames"])


def _smooth2d(counts):
    return gaussian_filter(counts, SIGMA) if _SMOOTH else counts


def _smooth1d(counts):
    return gaussian_filter1d(counts, SIGMA) if _SMOOTH else counts


def hist2d(ax, x, y):
    counts, xe, ye = np.histogram2d(x, y, bins=BINS)
    counts = _smooth2d(counts.T)
    total = counts.sum()
    if total <= 0:
        return
    flat = np.sort(counts.ravel())[::-1]
    csum = np.cumsum(flat) / flat.sum()
    lv1 = flat[np.searchsorted(csum, 0.683)]
    lv2 = flat[np.searchsorted(csum, 0.954)]
    xc = 0.5 * (xe[1:] + xe[:-1])
    yc = 0.5 * (ye[1:] + ye[:-1])
    levels = sorted(set([lv2, lv1, counts.max() * 1.001]))
    if len(levels) >= 2:
        colors = ["#cfe0ee", "#7fb3d5"][-(len(levels) - 1):]
        ax.contourf(xc, yc, counts, levels=levels, colors=colors, alpha=0.9)
    ax.contour(xc, yc, counts, levels=[lv2, lv1], colors="#1f4e79", linewidths=0.7)


def hist1d(ax, x):
    counts, edges = np.histogram(x, bins=BINS)
    counts = _smooth1d(counts.astype(float))
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax.fill_between(centers, counts, color="#7fb3d5", alpha=0.55, linewidth=0)
    ax.plot(centers, counts, color="#1f4e79", lw=1.2)
    ax.set_ylim(bottom=0)


def main() -> None:
    path, out = sys.argv[1], sys.argv[2]
    samples, names = load(path)
    keep = [i for i in range(samples.shape[1]) if np.ptp(samples[:, i]) > 1e-9]
    S = samples[:, keep]; labels = [names[i] for i in keep]
    n = len(labels)
    fig, axes = plt.subplots(n, n, figsize=(1.7 * n, 1.7 * n))
    if n == 1:
        axes = np.array([[axes]])
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off"); continue
            if i == j:
                hist1d(ax, S[:, i])
                q = np.percentile(S[:, i], [16, 50, 84])
                for v in q:
                    ax.axvline(v, color="#bc4749", ls="--", lw=0.8)
                ax.set_title(r"%s = %.3g$^{+%.2g}_{-%.2g}$" % (
                    labels[i], q[1], q[2] - q[1], q[1] - q[0]), fontsize=8)
                ax.set_yticks([])
            else:
                hist2d(ax, S[:, j], S[:, i])
            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(labels[j], fontsize=8); ax.tick_params(labelsize=6)
            if j > 0 or i == 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(labels[i], fontsize=8); ax.tick_params(labelsize=6)
    smooth = "suavizado" if _SMOOTH else "sem scipy — não suavizado"
    fig.suptitle(f"Corner — ajuste MAGNUS (RBS 1223) · contornos 1σ e 2σ ({smooth})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(out, dpi=110)
    print(f"corner salvo: {out}  ({len(S)} amostras, {n} parâmetros livres)")


if __name__ == "__main__":
    main()
