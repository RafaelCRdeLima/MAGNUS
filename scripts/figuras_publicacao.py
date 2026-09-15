"""Figuras de publicação a partir de um ajuste do MAGNUS.

Estilo de periódico: em inglês, SEM grade, ticks para dentro nos quatro lados,
fonte serif, saída vetorial (PDF) + PNG. Amostra a posteriori pelo motor e
desenha as bandas 1σ/2σ do pulso e do espectro sobre os dados.

Uso:
    python3 scripts/figuras_publicacao.py <dir_do_ajuste> [n_amostras]

O diretório precisa ter resultado.json e pedido.json. As figuras saem nele.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcmc_fit as m                                                  # noqa: E402

# ---- estilo de publicação --------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 13,
    "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.minor.size": 3, "ytick.minor.size": 3,
    "axes.grid": False,
    "legend.frameon": False,
    "figure.dpi": 120,
})

DATA = "#111111"          # data points
MED = "#1a3d6e"           # posterior median
S1 = "#7fa8d0"            # 1 sigma
S2 = "#cfe0ee"            # 2 sigma


def sample_models(result_dir: Path, n: int):
    d = json.loads((result_dir / "resultado.json").read_text())
    req = json.loads((result_dir / "pedido.json").read_text())
    S = np.asarray(d["samples"], float)
    L = np.asarray(d["sampleLogLikelihood"], float)
    fin = S[np.isfinite(L)]
    # dados: do preparedDataNpz se houver, senão dos eventos
    ev = req.get("_events") or _guess_events(req)
    meta, events = m.read_events(Path(ev))
    bg = None
    problem = m.FitProblem(req, events, meta, m.read_background(Path(req["_background"]))
                           if req.get("_background") else None)
    nph, nen = problem.phase_bins, problem.energy_bins
    emin = req.get("energyMin", 0.15); emax = req.get("energyMax", 1.2)
    # Recorte da banda dura: plota (e soma o pulso) apenas nos bins de energia
    # cujo CENTRO fica abaixo de energyFitMax, coerente com a mascara do ajuste.
    efm = float(req.get("energyFitMax", emax))
    width = (emax - emin) / nen
    centers = emin + (np.arange(nen) + 0.5) * width
    emask = centers < efm
    rng = np.random.default_rng(11)
    draw = fin[rng.choice(len(fin), size=min(n, len(fin)), replace=False)]
    w = m.EngineWorker(problem.worker_command())
    LC, SP = [], []
    for v in draw:
        _, exp = problem.evaluate(w, list(v), return_model=True)
        if exp is None:
            continue
        g = np.array(exp, float).reshape(nph, nen)
        LC.append(g[:, emask].sum(1)); SP.append(g[:, emask].sum(0))
    w.close()
    obs = np.array(problem.observed, float).reshape(nph, nen)
    return (np.array(LC), np.array(SP), obs[:, emask].sum(1), obs[:, emask].sum(0),
            nph, centers[emask])


def _guess_events(req):
    import os
    ev = req.get("events") or os.environ.get("MAGNUS_EVENTS")
    if not ev:
        raise SystemExit("arquivo de eventos nao informado: use --events, a chave "
                         "'events' do pedido ou a variavel MAGNUS_EVENTS")
    return ev


def _finish(ax):
    for s in ax.spines.values():
        s.set_linewidth(1.0)


def lightcurve(LC, lo, nph, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ph = (np.arange(nph) + 0.5) / nph
    ph2 = np.concatenate([ph, ph + 1]); dbl = lambda x: np.concatenate([x, x])
    P = np.percentile(LC, [2.5, 16, 50, 84, 97.5], axis=0)
    ax.fill_between(ph2, dbl(P[0]), dbl(P[4]), color=S2, label=r"$2\sigma$")
    ax.fill_between(ph2, dbl(P[1]), dbl(P[3]), color=S1, label=r"$1\sigma$")
    ax.plot(ph2, dbl(P[2]), color=MED, lw=1.8, label="Posterior median")
    ax.errorbar(ph2, dbl(lo), yerr=np.sqrt(dbl(lo)), fmt="o", ms=4, color=DATA,
                elinewidth=0.9, capsize=0, label="RBS 1223", zorder=5)
    ax.set_xlabel("Rotational phase")
    ax.set_ylabel("Counts")
    ax.set_xlim(0, 2)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True, framealpha=0.9,
              facecolor="white", edgecolor="none", borderpad=0.6,
              labelspacing=0.35, handlelength=1.6)
    _finish(ax)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200)


def spectrum(SP, so, energy, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    Q = np.percentile(SP, [2.5, 16, 50, 84, 97.5], axis=0)
    ax.fill_between(energy, Q[0], Q[4], color=S2, label=r"$2\sigma$")
    ax.fill_between(energy, Q[1], Q[3], color=S1, label=r"$1\sigma$")
    ax.plot(energy, Q[2], color=MED, lw=1.8, label="Posterior median")
    ax.errorbar(energy, so, yerr=np.sqrt(np.maximum(so, 1)), fmt="o", ms=4,
                color=DATA, elinewidth=0.9, capsize=0, label="RBS 1223", zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts (phase-summed)")
    ax.set_xlim(energy[0], energy[-1])
    ax.legend(loc="upper right", fontsize=11)
    _finish(ax)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200)


def main():
    d = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 140
    LC, SP, lo, so, nph, energy = sample_models(d, n)
    lightcurve(LC, lo, nph, d / "fig_lightcurve")
    spectrum(SP, so, energy, d / "fig_spectrum")
    print(f"figuras de publicação salvas em {d}/ (fig_lightcurve, fig_spectrum; .pdf + .png)")


if __name__ == "__main__":
    main()
