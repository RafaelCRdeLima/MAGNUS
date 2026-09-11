#!/usr/bin/env python3
"""Corner plot suave (contornos de densidade preenchidos, 1sigma/2sigma) da
corrida z-ciclotron (continuacao). Sem aparencia granulada."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

GM = 1.4766250385
FITDIR = "/home/rafael/Codes/MAGNUS/exploracoes/ajuste_beta_1keV_2026-09-11"
d = np.load(f"{FITDIR}/ckpt.npz", allow_pickle=True)
S = d["samples"]; nm = [str(x) for x in d["names"]]
flat = S[:, S.shape[1]//2:, :].reshape(-1, len(nm))
col = lambda k: flat[:, nm.index(k)]
u = col("compactness"); R = col("radius")
z = 1/np.sqrt(1-u) - 1
B13 = 10**col("logMagneticField") / 1e13
M = u * R / (2*GM)
data = [z, B13, M, R, col("atmFraction"), col("inclination"), col("poleTilt2")]
labs = [r"$z$", r"$B\,[10^{13}\,\mathrm{G}]$", r"$M\,[M_\odot]$", r"$R$ [km]", r"$f$",
        r"$i$ [$^\circ$]", r"$\beta$ [$^\circ$]"]
N = len(data)
rng = [(np.percentile(x, 0.5), np.percentile(x, 99.5)) for x in data]

BASE = "#1d3557"
cmap = LinearSegmentedColormap.from_list("c", ["white", "#a8c4dd", "#4f7cae", BASE])

def levels_2d(H):
    """limiares de densidade que encerram 0.393 (1s), 0.865 (2s) da massa."""
    Hf = H.flatten(); Hf = Hf[np.argsort(Hf)][::-1]
    cs = np.cumsum(Hf); cs /= cs[-1]
    out = []
    for frac in (0.865, 0.393):
        out.append(Hf[np.searchsorted(cs, frac)])
    return sorted(out)

fig, axes = plt.subplots(N, N, figsize=(12.5, 12.5))
for i in range(N):
    for j in range(N):
        ax = axes[i, j]
        if j > i:
            ax.axis("off"); continue
        if i == j:
            h, e = np.histogram(data[i], bins=60, range=rng[i], density=True)
            hc = gaussian_filter(h, 1.4); xc = 0.5*(e[1:]+e[:-1])
            ax.fill_between(xc, hc, color="#a8c4dd", alpha=0.7)
            ax.plot(xc, hc, color=BASE, lw=1.4)
            q16, q50, q84 = np.percentile(data[i], [16, 50, 84])
            for q in (q16, q50, q84):
                ax.axvline(q, color=BASE, ls="--", lw=0.9)
            ax.set_title(fr"{labs[i]} $= {q50:.3g}_{{-{q50-q16:.2g}}}^{{+{q84-q50:.2g}}}$",
                         fontsize=9.5)
            ax.set_yticks([]); ax.set_xlim(rng[i]); ax.set_ylim(0, hc.max()*1.15)
        else:
            H, xe, ye = np.histogram2d(data[j], data[i], bins=70, range=[rng[j], rng[i]])
            Hs = gaussian_filter(H.T, 2.2)
            xc = 0.5*(xe[1:]+xe[:-1]); yc = 0.5*(ye[1:]+ye[:-1])
            lv = levels_2d(Hs)
            ax.contourf(xc, yc, Hs, levels=[lv[0], lv[1], Hs.max()],
                        colors=["#c9dcec", "#5f8bbd"], alpha=0.9)
            ax.contour(xc, yc, Hs, levels=lv, colors=BASE, linewidths=1.0)
            ax.set_xlim(rng[j]); ax.set_ylim(rng[i])
        if i == N-1: ax.set_xlabel(labs[j], fontsize=11)
        else: ax.set_xticklabels([])
        if j == 0 and i > 0: ax.set_ylabel(labs[i], fontsize=11)
        elif i != j: ax.set_yticklabels([])
        ax.tick_params(labelsize=7)

fig.tight_layout()
fig.savefig(f"{FITDIR}/fig_corner.png", dpi=140, bbox_inches="tight")
fig.savefig(f"{FITDIR}/fig_corner.pdf", bbox_inches="tight")
print("corner suave salvo (7 params, com beta).")
