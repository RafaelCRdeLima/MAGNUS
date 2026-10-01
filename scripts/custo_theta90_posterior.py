#!/usr/bin/env python3
"""O efeito da tabela de 90 graus SOBRE A POSTERIOR, nao num vetor so.

A primeira versao disto avaliou um unico vetor, o de maxima verossimilhanca, e o
artigo concluiu dali que o grampeamento "nao afeta a inferencia", chamando o
Delta lnL de teto. A auditoria de 01/10/2026 apontou, com razao, que uma
diferenca pontual nao limita deslocamento de quantis nem Delta z.

Aqui: sorteia amostras da segunda metade da cadeia fiducial, avalia cada uma com
a v4 (para em 60 graus) e com a v5 (vai ate 90), e usa os Delta lnL para
REPONDERAR a posterior por amostragem de importancia. Isso da o deslocamento dos
quantis sem refazer o ajuste. O peso e proporcional a exp(Delta lnL), e o ESS
dos pesos diz se a reponderacao e confiavel.
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def problema(tab, req):
    r = dict(req); r["atmosphereTable"] = tab
    ev = r.get("_events") or F._guess_events(r)
    meta, events = m.read_events(Path(ev))
    bg = m.read_background(Path(r["_background"])) if r.get("_background") else None
    return m.FitProblem(r, events, meta, bg)


def main():
    req = json.loads((FIDUCIAL / "pedido.json").read_text())
    d = np.load(FIDUCIAL / "checkpoint.npz", allow_pickle=True)
    nomes = [str(v) for v in d["names"]]
    s = d["samples"]; lp = d["logps"]
    meia = s.shape[1] // 2
    S = s[:, meia:].reshape(-1, s.shape[2])
    L = lp[:, meia:].ravel()
    rng = np.random.default_rng(7)
    idx = rng.choice(len(S), size=N, replace=False)
    V = S[idx].astype(np.float64)
    print("amostras: %d da segunda metade (%d disponiveis)" % (N, len(S)), flush=True)

    res = {}
    for tab in (V4, V5):
        p = problema(tab, req)
        w = m.EngineWorker(p.worker_command())
        vals = np.array([p.evaluate(w, list(v)) for v in V], float)
        w.close()
        res[tab] = vals
        print("  %-44s avaliado" % tab, flush=True)

    delta = res[V5] - res[V4]
    bom = np.isfinite(delta)
    delta = delta[bom]; Vb = V[bom]
    print()
    print("Delta lnL (v5 - v4) sobre %d amostras da posterior:" % delta.size)
    print("   mediana %+.3f | media %+.3f | d.p. %.3f | min %+.3f | max %+.3f"
          % (np.median(delta), delta.mean(), delta.std(), delta.min(), delta.max()))

    # reponderacao por importancia
    lw = delta - delta.max()
    pesos = np.exp(lw); pesos /= pesos.sum()
    ess = 1.0 / np.sum(pesos ** 2)
    print("   ESS dos pesos: %.0f de %d (%.0f %%)" % (ess, delta.size, 100 * ess / delta.size))

    iu = nomes.index("compactness")
    u = Vb[:, iu]; z = 1.0 / np.sqrt(1.0 - u) - 1.0

    def qw(x, p, w=None):
        if w is None:
            return np.percentile(x, p)
        o = np.argsort(x); xs, ws = x[o], w[o]
        c = np.cumsum(ws) - 0.5 * ws; c /= ws.sum()
        return np.interp(np.array(p) / 100.0, c, xs)

    qs = [16, 50, 84]
    a = qw(z, qs); b = qw(z, qs, pesos)
    print()
    print("z, quantis 16/50/84:")
    print("   com a v4 (as amostras como estao) : %.5f  %.5f  %.5f" % tuple(a))
    print("   reponderado para a v5             : %.5f  %.5f  %.5f" % tuple(b))
    print("   deslocamento                      : %+.5f %+.5f %+.5f" % tuple(b - a))
    print("   deslocamento da mediana em unidades de sigma: %.3f"
          % ((b[1] - a[1]) / ((a[2] - a[0]) / 2)))

    saida = dict(n=int(delta.size), delta_mediana=float(np.median(delta)),
                 delta_media=float(delta.mean()), delta_dp=float(delta.std()),
                 delta_min=float(delta.min()), delta_max=float(delta.max()),
                 ess_pesos=float(ess),
                 z_v4=[float(x) for x in a], z_v5_reponderado=[float(x) for x in b],
                 deslocamento_mediana_sigma=float((b[1]-a[1])/((a[2]-a[0])/2)),
                 nota="reponderacao por importancia; valida enquanto o ESS dos pesos for alto")
    (ROOT / "exploracoes/theta90_diagnostico/custo_theta90_posterior.json").write_text(
        json.dumps(saida, indent=2) + "\n")


if __name__ == "__main__":
    main()
