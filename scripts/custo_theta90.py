#!/usr/bin/env python3
"""Quanto custou a tabela parar em 60 graus, medido no modelo publicado.

Avalia o MESMO vetor de parametros, o de maxima verossimilhanca da bacia
fiducial, com a tabela v4, que para em theta_B = 60 e e grampeada pelo motor
acima disso, e com a v5, que vai ate 90. A diferenca isola o efeito do
grampeamento, sem refazer ajuste nenhum.

Isto e uma medida CONDICIONAL, a parametros fixos. Nao e uma deteccao
marginalizada, e parte dela seria absorvida pelos parametros livres da linha num
reajuste, como aconteceu com o vacuo. Achado 14 da auditoria de 29/09/2026.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mcmc_fit as m  # noqa: E402
import figuras_publicacao as F  # noqa: E402

FIDUCIAL = ROOT / "exploracoes/convergencia_bacias_2026-09-25/B_i95"
V4 = "tabelas/magnus_campoBg_vacuo_v4.magnus"
V5 = "build/magnus_campoBg_vacuo_v5.magnus"


def avalia(req, vetor):
    ev = req.get("_events") or F._guess_events(req)
    meta, events = m.read_events(Path(ev))
    bg = m.read_background(Path(req["_background"])) if req.get("_background") else None
    problem = m.FitProblem(req, events, meta, bg)
    w = m.EngineWorker(problem.worker_command())
    lnL, esperado = problem.evaluate(w, list(vetor), return_model=True)
    w.close()
    g = np.array(esperado, float).reshape(problem.phase_bins, problem.energy_bins)
    return lnL, g, problem


def main():
    req = json.loads((FIDUCIAL / "pedido.json").read_text())
    d = np.load(FIDUCIAL / "checkpoint.npz", allow_pickle=True)
    lp = d["logps"].astype(np.float64)
    i, j = np.unravel_index(int(np.argmax(lp)), lp.shape)
    vetor = d["samples"][i, j].astype(np.float64)
    print("vetor de maxima verossimilhanca: caminhante %d, passo %d, lnL gravado %.3f"
          % (i, j, lp[i, j]))

    saida = {}
    for rot, tab in (("v4 (para em 60 graus)", V4), ("v5 (vai ate 90 graus)", V5)):
        r = dict(req); r["atmosphereTable"] = tab
        lnL, g, problem = avalia(r, vetor)
        saida[tab] = (lnL, g, problem)
        print("  %-24s lnL = %.3f" % (rot, lnL))

    (l4, g4, p4), (l5, g5, _) = saida[V4], saida[V5]
    print()
    print("Delta lnL (v5 - v4) = %+.2f" % (l5 - l4))

    emin = req.get("energyMin", 0.15); emax = req.get("energyMax", 1.2)
    nen = p4.energy_bins
    centros = emin + (np.arange(nen) + 0.5) * (emax - emin) / nen
    mask = centros < float(req.get("energyFitMax", emax))

    s4, s5 = g4[:, mask].sum(0), g5[:, mask].sum(0)
    rel = (s5 - s4) / np.maximum(s4, 1e-30)
    print("espectro somado em fase, nos %d bins do ajuste:" % mask.sum())
    print("   mudanca relativa: mediana %+.3f %%, maior %+.2f %% em %.3f keV"
          % (100 * np.median(rel), 100 * rel[np.argmax(abs(rel))],
             centros[mask][np.argmax(abs(rel))]))
    print("   contagem total : %+.3f %%" % (100 * (s5.sum() - s4.sum()) / s4.sum()))

    def pf(g):
        c = g[:, mask].sum(1)
        return (c.max() - c.min()) / (c.max() + c.min())
    print("fracao pulsada : v4 %.4f -> v5 %.4f  (%+.3f pontos percentuais)"
          % (pf(g4), pf(g5), 100 * (pf(g5) - pf(g4))))

    obs = np.array(p4.observed, float).reshape(p4.phase_bins, nen)
    def chi2(g):
        e = g[:, mask]; o = obs[:, mask]
        return float(np.sum((o - e) ** 2 / np.maximum(e, 1e-9)) / e.size)
    print("chi2/bin       : v4 %.4f -> v5 %.4f" % (chi2(g4), chi2(g5)))

    Path(ROOT / "exploracoes/theta90_diagnostico").mkdir(parents=True, exist_ok=True)
    json.dump({"lnL_v4": l4, "lnL_v5": l5, "delta_lnL": l5 - l4,
               "mudanca_contagem_total_pct": 100 * (s5.sum() - s4.sum()) / s4.sum(),
               "pf_v4": pf(g4), "pf_v5": pf(g5),
               "chi2_bin_v4": chi2(g4), "chi2_bin_v5": chi2(g5),
               "nota": "condicional, a parametros fixos; nao e deteccao marginalizada"},
              open(ROOT / "exploracoes/theta90_diagnostico/custo_theta90.json", "w"),
              indent=2)


if __name__ == "__main__":
    main()
