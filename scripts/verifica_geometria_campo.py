#!/usr/bin/env python3
"""Verifica o conserto da geometria do campo, e mede o que ele custa.

O motor calculava a DIRECAO do campo em |cmag|, relativo ao eixo magnetico, e a
INTENSIDADE em max(c1, c2), que inclui o eixo do segundo polo TERMICO. Nao ha
dipolo do segundo polo: axis2 aparece num unico ponto do motor e so alimenta o
lobo de temperatura e esse max. O max existia porque local_log_b grampeia
cosseno negativo em zero, e usar cmag cru daria campo equatorial em todo o
hemisferio distante; o certo e |cmag|.

Com beta = 0 os dois caminhos TEM de coincidir, porque axis2 = -m e
max(c, -c) = |c|. Essa e a verificacao. Com beta != 0 o remendo erra a
colatitude efetiva por beta.

Uso:  MAGNUS_EVENTS=dados/eventos.csv python3 scripts/verifica_geometria_campo.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mcmc_fit as m  # noqa: E402

FIDUCIAL = ROOT / "exploracoes/convergencia_bacias_2026-09-25/B_i95"


def main():
    import os
    ev = os.environ.get("MAGNUS_EVENTS", str(ROOT / "dados/eventos.csv"))
    base = json.loads((FIDUCIAL / "pedido.json").read_text())
    meta, eventos = m.read_events(Path(ev))
    d = np.load(FIDUCIAL / "checkpoint.npz", allow_pickle=True)
    nomes = [str(x) for x in d["names"]]
    lp = d["logps"]
    i, j = np.unravel_index(int(np.argmax(lp)), lp.shape)
    v0 = d["samples"][i, j].astype(float)
    it = nomes.index("poleTilt2")

    saida = {}
    for rotulo, beta in (("beta_zero", 0.0), ("beta_fiducial", float(v0[it]))):
        v = v0.copy(); v[it] = beta
        par = {}
        for flag, nome in ((True, "polo_mais_proximo"), (False, "dipolo_coerente")):
            r = dict(base); r["fieldNearestPole"] = flag
            p = m.FitProblem(r, eventos, meta, None)
            w = m.EngineWorker(p.worker_command())
            par[nome] = float(p.evaluate(w, list(v)))
            w.close()
        par["delta"] = par["dipolo_coerente"] - par["polo_mais_proximo"]
        par["beta_graus"] = beta
        saida[rotulo] = par
        print("%-16s beta = %+6.2f  antigo %.6f  novo %.6f  delta %+.3f"
              % (rotulo, beta, par["polo_mais_proximo"],
                 par["dipolo_coerente"], par["delta"]))

    ok = abs(saida["beta_zero"]["delta"]) < 1e-6
    print()
    print("EQUIVALENCIA EM beta = 0: %s" % ("ok" if ok else "FALHOU"))
    if not ok:
        raise SystemExit("os dois caminhos deveriam coincidir em beta = 0")
    (ROOT / "exploracoes/theta90_diagnostico").mkdir(parents=True, exist_ok=True)
    (ROOT / "exploracoes/geometria_campo.json").write_text(
        json.dumps(saida, indent=2) + "\n")


if __name__ == "__main__":
    main()
