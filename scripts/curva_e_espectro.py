"""Duas leituras do mesmo ajuste: a curva de luz dobrada e o espectro, ambas
modelo vs dados. Lê o resultado do MCMC (com `observed`/`expected`) e desenha.

Uso:
    python3 scripts/curva_e_espectro.py resultado.json saida.png [energyMin] [energyMax]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    res, out = sys.argv[1], sys.argv[2]
    emin = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    emax = float(sys.argv[4]) if len(sys.argv) > 4 else 1.2
    d = json.loads(Path(res).read_text())
    nph, nen = d["phaseBins"], d["energyBins"]
    obs = np.asarray(d["observed"], float).reshape(nph, nen)
    exp = np.asarray(d["expected"], float).reshape(nph, nen)

    lc_o, lc_e = obs.sum(1), exp.sum(1)
    sp_o, sp_e = obs.sum(0), exp.sum(0)
    phase = (np.arange(nph) + 0.5) / nph
    en = np.linspace(emin, emax, nen + 1)
    ec = 0.5 * (en[1:] + en[:-1])

    def amp(x):
        return (x.max() - x.min()) / (x.max() + x.min())

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 8.2))

    # curva de luz dobrada — duas voltas, para ler o pulso
    ph2 = np.concatenate([phase, phase + 1.0])
    a1.errorbar(ph2, np.concatenate([lc_o, lc_o]),
                yerr=np.sqrt(np.concatenate([lc_o, lc_o])), fmt="o", ms=4,
                color="#bc4749", label=f"dados (fp={amp(lc_o)*100:.0f}%)", zorder=5)
    a1.plot(ph2, np.concatenate([lc_e, lc_e]), color="#1f4e79", lw=2,
            label=f"modelo (fp={amp(lc_e)*100:.0f}%)")
    a1.set_xlabel("fase"); a1.set_ylabel("contagens (soma em energia)")
    a1.set_title("Curva de luz dobrada — MAGNUS fundo+spots vs RBS 1223")
    a1.legend(); a1.grid(alpha=.2)

    # espectro somado em fase
    a2.errorbar(ec, sp_o, yerr=np.sqrt(np.maximum(sp_o, 1)), fmt="o", ms=4,
                color="#bc4749", label="dados", zorder=5)
    a2.plot(ec, sp_e, color="#1f4e79", lw=2, label="modelo")
    a2.set_xlabel("energia (keV)"); a2.set_ylabel("contagens (soma em fase)")
    a2.set_title("Espectro — modelo excede os dados acima de ~0,8 keV")
    a2.set_yscale("log"); a2.legend(); a2.grid(alpha=.2)

    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"figura salva: {out}")
    print(f"pulso modelo={amp(lc_e)*100:.1f}%  dados={amp(lc_o)*100:.1f}%")


if __name__ == "__main__":
    main()
