"""Bandas 1σ/2σ do ESPECTRO a posteriori vs os dados.

Amostra a posteriori do ajuste, avalia o modelo de cada amostra pelo motor,
soma em fase para o espectro (contagens por bin de energia), e desenha os
percentis (2,5/16/50/84/97,5) contra o espectro observado. As bandas do fit
(posteriorPredictive) são do PULSO em fase; estas são do ESPECTRO.

Uso:
    python3 scripts/bandas_espectro.py req.json fit.json eventos.csv fundo.csv saida.png
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
import mcmc_fit as m                                                 # noqa: E402


def main() -> None:
    req_path, fit_path, ev_path, bg_path, out = sys.argv[1:6]
    request = json.loads(Path(req_path).read_text())
    metadata, events = m.read_events(Path(ev_path))
    background = m.read_background(Path(bg_path))
    problem = m.FitProblem(request, events, metadata, background)
    result = json.loads(Path(fit_path).read_text())
    samples = np.asarray(result["samples"], dtype=float)
    logps = np.asarray(result["sampleLogLikelihood"], dtype=float)
    finite = samples[np.isfinite(logps)]
    rng = np.random.default_rng(1)
    draw = finite[rng.choice(len(finite), size=min(400, len(finite)), replace=False)]

    nphase, nenergy = problem.phase_bins, problem.energy_bins
    worker = m.EngineWorker(problem.worker_command())
    spectra = []
    for values in draw:
        logp, expected = problem.evaluate(worker, list(values), return_model=True)
        if expected is None:
            continue
        spec = np.asarray(expected).reshape(nphase, nenergy).sum(axis=0)
        spectra.append(spec)
    worker.close()
    spectra = np.array(spectra)
    observed = np.asarray(problem.observed, dtype=float).reshape(nphase, nenergy).sum(axis=0)
    energy = np.linspace(problem.energy_min, problem.energy_max, nenergy + 1)
    ec = 0.5 * (energy[1:] + energy[:-1])

    pcts = np.percentile(spectra, [2.5, 16, 50, 84, 97.5], axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.fill_between(ec, pcts[0], pcts[4], color="#cfe0ee", label="2σ")
    ax.fill_between(ec, pcts[1], pcts[3], color="#7fb3d5", label="1σ")
    ax.plot(ec, pcts[2], color="#1f4e79", lw=1.5, label="mediana da posteriori")
    ax.errorbar(ec, observed, yerr=np.sqrt(np.maximum(observed, 1.0)), fmt="o",
                ms=4, color="#bc4749", label="dados (RBS 1223)", zorder=5, lw=1)
    ax.set_xlabel("energia (keV)"); ax.set_ylabel("contagens no espectro (soma em fase)")
    ax.set_title("Espectro a posteriori do ajuste conjunto MAGNUS — bandas 1σ/2σ")
    ax.legend(); ax.set_yscale("log"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"bandas espectrais salvas: {out}  ({len(spectra)} amostras da posteriori)")


if __name__ == "__main__":
    main()
