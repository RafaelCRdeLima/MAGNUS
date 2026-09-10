"""Diagnostico do ajuste z-ciclotron: (1) resumo z/B/M/R do posterior e (2)
espectro medio modelo x dados com painel de residuo (pull), marcando a energia
observada da feicao 0.63*B14/(1+z). Responde: o z esta ancorado e a feicao do
modelo casa com a feicao dos NOSSOS dados?

Uso: python3 scripts/diag_zcyclotron.py resultado.json saida.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GM = 1.4766250385  # GM_sun/c^2 em km


def main() -> None:
    res, out = sys.argv[1], sys.argv[2]
    d = json.loads(Path(res).read_text())
    names = list(d["parameterNames"])
    smp = np.asarray(d["samples"], float)
    if smp.ndim == 1:
        smp = smp.reshape(-1, len(names))
    col = {n: smp[:, i] for i, n in enumerate(names)}

    u = col["compactness"]
    R = col["radius"]
    lgB = col["logMagneticField"]
    z = 1.0 / np.sqrt(1.0 - u) - 1.0
    B14 = 10 ** lgB / 1e14
    M = u * R / (2 * GM)
    feat = 0.63 * B14 / (1.0 + z)  # energia observada prevista da feicao (keV)

    def q(x):
        return np.percentile(x, [16, 50, 84])

    print("=== posterior (16/50/84) ===")
    for nm, x in [("z", z), ("B14", B14), ("M[Msun]", M), ("R[km]", R),
                  ("u", u), ("feic_obs[keV]", feat),
                  ("f", col.get("atmFraction", np.array([np.nan]))),
                  ("lineWidth", col.get("lineWidth", np.array([np.nan]))),
                  ("lineDepth", col.get("lineDepth", np.array([np.nan])))]:
        a, b, c = q(x)
        print(f"  {nm:14s} {a:.4g} / {b:.4g} / {c:.4g}   (larg {c-a:.4g})")
    print(f"  corr(z,lgB) = {np.corrcoef(z,lgB)[0,1]:.3f}   corr(M,R) = {np.corrcoef(M,R)[0,1]:.3f}")
    # R-hat, se o resultado trouxer no summary/diagnostics
    for key in ("summary",):
        s = d.get(key)
        if isinstance(s, list) and s and isinstance(s[0], dict):
            rh = {e.get("name"): e.get("rHat", e.get("rhat")) for e in s}
            print("  R-hat:", {k: (round(v, 3) if isinstance(v, (int, float)) else v)
                               for k, v in rh.items() if k in ("compactness", "radius", "logMagneticField")})

    # ---- espectro medio modelo x dados + residuo ----
    nph, nen = d["phaseBins"], d["energyBins"]
    obs = np.asarray(d["observed"], float).reshape(nph, nen).sum(0)
    exp = np.asarray(d["expected"], float).reshape(nph, nen).sum(0)
    emin = float(d.get("metadata", {}).get("energy_min_keV", 0.15))
    emax = float(d.get("metadata", {}).get("energy_max_keV", 1.2))
    edges = np.linspace(emin, emax, nen + 1)
    ec = 0.5 * (edges[1:] + edges[:-1])
    err = np.sqrt(np.maximum(obs, 1))
    pull = (obs - exp) / err
    fmed = float(np.median(feat))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 7.6), sharex=True,
                                 gridspec_kw={"height_ratios": [2.4, 1]})
    a1.errorbar(ec, obs, yerr=err, fmt="o", ms=4, color="#bc4749",
                label="dados (6 obs co-add)", zorder=5)
    a1.plot(ec, exp, color="#1f4e79", lw=2, label="modelo (mediana)")
    a1.axvline(fmed, color="#2a9d8f", ls="--", lw=1.6,
               label=f"feicao ciclotron {fmed:.2f} keV")
    a1.axvspan(0.28, 0.32, color="0.6", alpha=0.25, label="~0.3 keV (literatura)")
    a1.set_ylabel("contagens (soma em fase)"); a1.set_yscale("log")
    a1.legend(fontsize=9); a1.grid(alpha=.2)
    a2.axhline(0, color="0.4", lw=1)
    a2.axhline(2, color="0.7", lw=.7, ls=":"); a2.axhline(-2, color="0.7", lw=.7, ls=":")
    a2.bar(ec, pull, width=(edges[1] - edges[0]) * 0.9, color="#457b9d")
    a2.axvline(fmed, color="#2a9d8f", ls="--", lw=1.6)
    a2.set_xlabel("energia (keV)"); a2.set_ylabel(r"(dados$-$modelo)/$\sigma$")
    a2.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("figura salva:", out)
    # onde esta o maior deficit nos dados relativo a um continuum suave?
    print(f"maior |pull| em {ec[np.argmax(np.abs(pull))]:.2f} keV (pull={pull[np.argmax(np.abs(pull))]:.1f})")


if __name__ == "__main__":
    main()
