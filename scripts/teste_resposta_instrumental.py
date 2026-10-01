#!/usr/bin/env python3
"""Quanto a resposta instrumental varia entre apontamentos EPN Thin1.

A coadicao soma contagens de 13 observacoes e ajusta tudo com a resposta de UMA
delas, a de 2019, que contribui 6123 s de 131978 s. O manuscrito justifica isso
pelo modo de leitura comum, o que e plausibilidade e nao verificacao; achados 8
de 29/09 e 7 de 01/10.

As respostas individuais das 12 observacoes de 2003-2007 nao estao no acervo,
entao a soma ponderada completa nao e possivel aqui. O que da para fazer, e e
muito mais do que o argumento atual, e medir quanto o modelo previsto muda ao
trocar por OUTRA resposta EPN Thin1 do mesmo instrumento, a de 0852980201. Ela
nao esta nas 13; serve de afericao da variacao entre apontamentos.

A resolucao temporal e sobrescrita com a da nossa observacao, senao a comparacao
misturaria resposta com borramento de fase.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mcmc_fit as m  # noqa: E402

FIDUCIAL = ROOT / "exploracoes/convergencia_bacias_2026-09-25/B_i95"
NOSSA, OUTRA = "xmm_epn_imaging_0844140101", "xmm_epn_imaging_0852980201"


def comando_com_tempo_de(cmd, ref):
    """Troca --time-resolution-us e --dead-time-us pelos valores de ref."""
    out = list(cmd)
    for chave in ("--time-resolution-us", "--dead-time-us"):
        i, j = out.index(chave), ref.index(chave)
        out[i + 1] = ref[j + 1]
    return out


def main():
    import os
    ev = os.environ.get("MAGNUS_EVENTS", str(ROOT / "dados/eventos.csv"))
    base = json.loads((FIDUCIAL / "pedido.json").read_text())
    meta, eventos = m.read_events(Path(ev))
    d = np.load(FIDUCIAL / "checkpoint.npz", allow_pickle=True)
    lp = d["logps"]
    i, j = np.unravel_index(int(np.argmax(lp)), lp.shape)
    v = d["samples"][i, j].astype(float)

    p_nossa = m.FitProblem(dict(base, instrument=NOSSA), eventos, meta, None)
    cmd_nossa = p_nossa.worker_command()
    p_outra = m.FitProblem(dict(base, instrument=OUTRA), eventos, meta, None)
    cmd_outra = comando_com_tempo_de(p_outra.worker_command(), cmd_nossa)

    res = {}
    for rot, prob, cmd in (("0844140101 (a usada)", p_nossa, cmd_nossa),
                           ("0852980201 (outra)", p_outra, cmd_outra)):
        w = m.EngineWorker(cmd)
        lnL, esperado = prob.evaluate(w, list(v), return_model=True)
        w.close()
        g = np.array(esperado, float).reshape(prob.phase_bins, prob.energy_bins)
        res[rot] = (lnL, g)
        print("  %-22s lnL = %.3f" % (rot, lnL))

    (l1, g1), (l2, g2) = res["0844140101 (a usada)"], res["0852980201 (outra)"]
    emin = base.get("energyMin", 0.15); emax = base.get("energyMax", 1.2)
    nen = p_nossa.energy_bins
    centros = emin + (np.arange(nen) + 0.5) * (emax - emin) / nen
    mask = centros < float(base.get("energyFitMax", emax))
    s1, s2 = g1[:, mask].sum(0), g2[:, mask].sum(0)
    rel = (s2 - s1) / np.maximum(s1, 1e-30)

    print()
    print("Delta lnL = %+.1f" % (l2 - l1))
    print("espectro somado em fase, %d bins do ajuste:" % mask.sum())
    print("   mediana |mudanca| = %.2f %% | p90 = %.2f %% | maior = %.2f %% em %.3f keV"
          % (100 * np.median(abs(rel)), 100 * np.percentile(abs(rel), 90),
             100 * abs(rel).max(), centros[mask][np.argmax(abs(rel))]))
    print("   contagem total: %+.2f %%" % (100 * (s2.sum() - s1.sum()) / s1.sum()))
    saida = dict(delta_lnL=float(l2 - l1),
                 mudanca_mediana_pct=float(100 * np.median(abs(rel))),
                 mudanca_p90_pct=float(100 * np.percentile(abs(rel), 90)),
                 mudanca_maxima_pct=float(100 * abs(rel).max()),
                 contagem_total_pct=float(100 * (s2.sum() - s1.sum()) / s1.sum()),
                 nota="0852980201 nao esta nas 13; afere a variacao entre "
                      "apontamentos EPN Thin1. Resolucao temporal igualada.")
    (ROOT / "exploracoes/resposta_instrumental.json").write_text(
        json.dumps(saida, indent=2) + "\n")


if __name__ == "__main__":
    main()
