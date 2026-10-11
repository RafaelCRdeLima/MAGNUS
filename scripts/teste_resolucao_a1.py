#!/usr/bin/env python3
"""O modo a1 -> 0 esta resolvido pela amostragem da superficie?

A lei de temperatura usa lobe(c,a,T^4) = T^4 c^2/(c^2 + a s^2) para a > 0, e
T^4 para a <= 0, com zero no hemisferio oposto (engine/main.cpp:2150). No limite
a -> 0 isso vira um DEGRAU no equador magnetico: T^4 salta de T^4 para zero ao
cruzar c = 0. A largura da transicao vai como raiz de a.

A superficie e amostrada em base_bands = clamp(2*rings, 16, 60) faixas, com
rings = 13 por padrao, ou seja 26 faixas, ~7 graus. Um degrau nessa grade pode
ser mal representado, e nesse caso a vantagem de Delta lnL ~ 21 do modo a1 ~ 0
seria em parte artefato numerico.

Este teste avalia o MESMO vetor com varias resolucoes. Se o lnL do modo colado
se mover com a resolucao e o do modo com a1 > 0 nao, a vantagem e numerica.

Uso: MAGNUS_EVENTS=... python3 scripts/teste_resolucao_a1.py vetor.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mcmc_fit as m  # noqa: E402

FIDUCIAL = ROOT / "exploracoes/convergencia_bacias_2026-09-25/B_i95"
# o pedido define a FAIXA DE INCLINACAO da bacia: avaliar o vetor de uma bacia
# com o pedido da outra devolve -inf, porque a priori rejeita.
RINGS = (13, 18, 24, 30)


def main():
    import os
    dados = json.loads(Path(sys.argv[1]).read_text())
    nomes = dados["nomes"]
    ev = os.environ.get("MAGNUS_EVENTS", str(ROOT / "dados/eventos.csv"))
    pedido = Path(sys.argv[2]) if len(sys.argv) > 2 else FIDUCIAL / "pedido.json"
    base = json.loads(pedido.read_text())
    base["fieldNearestPole"] = False      # geometria corrigida, como a corrida nova
    base["atmosphereTable"] = "tabelas/magnus_campoBg_vacuo_v5.magnus"
    base["instrument"] = "xmm_epn_imaging_combinado13"
    meta, eventos = m.read_events(Path(ev))
    prob = m.FitProblem(base, eventos, meta, None)
    # o MELHOR vetor de cada modo, nao o mesmo vetor com a1 trocado: os dois
    # modos tem geometrias e temperaturas diferentes, e trocar so o a1 avalia um
    # ponto que nao pertence a nenhum deles.
    casos = {"modo colado  (a1 = %.3f)" % dados["colado"]["a1"]: dados["colado"]["vetor"],
             "modo aberto  (a1 = %.3f)" % dados["aberto"]["a1"]: dados["aberto"]["vetor"]}
    print("%-26s %s" % ("", "  ".join("rings=%-2d" % r for r in RINGS)))
    ref = {}
    for rot, v in casos.items():
        linha = []
        for r in RINGS:
            cmd = list(prob.worker_command()) + ["--rings", str(r)]
            w = m.EngineWorker(cmd)
            L = prob.evaluate(w, v)
            w.close()
            linha.append(L)
        ref[rot] = linha
        print("%-26s %s" % (rot, "  ".join("%.2f" % x for x in linha)))

    print()
    for rot, linha in ref.items():
        print("%-26s variacao com a resolucao: %+.2f (de rings=13 a 30)"
              % (rot, linha[-1] - linha[0]))
    a, b = list(ref.values())
    print()
    print("vantagem do modo colado, por resolucao:")
    for r, x, y in zip(RINGS, a, b):
        print("   rings=%-2d  Delta lnL = %+.2f" % (r, x - y))


if __name__ == "__main__":
    main()
