#!/usr/bin/env python3
"""Verificacao 1 da malha em psi_0: ela reproduz os controles conhecidos?

O revisor fixou a ordem, 08/10/2026: "1. Reproduzir os controles isotermicos e de
feixe ja conhecidos, incluindo pesos e contribuicoes por imagem. 2. Avaliar a
fisica local nos mapas nao uniformes e comparar os tres vetores, variando
separadamente malha, tabela de raios e fase."

Esta e a etapa 1. O controle isotermico se obtem do pedido de producao zerando o
que torna a superficie nao uniforme: `temperature_peaking` 0, um polo so, campo
uniforme. Nesse caso o integrando e axissimetrico em torno do eixo do observador
e as duas malhas tem de concordar.

O QUE SE COMPARA, e por que mais de uma coisa: a grade de taxa detectada, que e o
observavel; a soma por imagem, porque um deficit pode estar em uma imagem e ser
compensado na outra; e o peso total, que no controle isotermico tem valor
analitico.

ATENCAO A UMA DIFERENCA DE DESENHO, declarada: a malha em psi_0 alinha as bordas
das bandas as fronteiras de visibilidade, enquanto a de producao as refina pelo
predicado de limbo. Isso NAO e mudanca de coordenadas, e portanto concordancia
imperfeita entre as duas nao implica erro de uma delas.

Uso: python3 scripts/valida_malha_psi0.py [--bandas 120 240 480]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import mcmc_fit as m  # noqa: E402
import por_observacao as PO  # noqa: E402

OBS = "0163560101"


def roda(obs, bandas, psi0, isotermico, max_images, refino=8):
    base = PO.monta_pedido(obs, bandas, refino)
    base["outputPrecision"] = 17
    base["maxImages"] = max_images
    meta, ev = m.read_events(RAIZ / "dados/eventos.csv")
    prob = m.FitProblem(base, ev, meta, None)
    nome, caminho, rotulo = PO.VETORES[0]
    v = PO.le_vetor(prob, caminho, rotulo)
    if isotermico:
        # zera o que torna a superficie nao uniforme, pelos NOMES e nao por
        # indice: peaking dos dois polos, e o segundo polo
        for alvo in ("peaking", "peaking2", "poleTilt2", "poleTilt2b"):
            if alvo in prob.names:
                v[prob.names.index(alvo)] = 0.0
    comando = list(prob.worker_command())
    if psi0:
        comando += ["--malha-psi0", "--psi-bandas", str(bandas)]
    trab = m.EngineWorker(comando)
    t0 = time.time()
    doc = trab.evaluate(prob.worker_line(v))
    dt = time.time() - t0
    trab.close()
    if doc.get("status") != "ok":
        return None, dt, doc.get("message", "sem mensagem")
    return doc, dt, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandas", type=int, nargs="+", default=[120, 240])
    ap.add_argument("--saida", default=str(RAIZ / "docs/valida_malha_psi0.json"))
    a = ap.parse_args()
    obss = {o["obsid"]: o for o in PO.prepara()}
    obs = obss[OBS]
    rel = {"o_que_e": "verificacao 1 da malha em psi_0: os controles conhecidos",
           "o_controle": ("isotermico, obtido zerando peaking, peaking2, "
                          "poleTilt2 e poleTilt2b PELOS NOMES"),
           "a_diferenca_de_desenho": (
             "a malha psi_0 alinha as bordas as fronteiras de visibilidade; a de "
             "producao as refina pelo predicado de limbo. Nao e mudanca de "
             "coordenadas, logo concordancia imperfeita nao implica erro de uma"),
           "casos": []}
    for isoter in (True, False):
        for mi in (1, 2):
            for b in a.bandas:
                d_tp, t_tp, e_tp = roda(obs, b, False, isoter, mi)
                d_ps, t_ps, e_ps = roda(obs, b, True, isoter, mi)
                rotulo = "%s, %d imagem%s, %d bandas" % (
                    "isotermico" if isoter else "NAO uniforme", mi,
                    "" if mi == 1 else "s", b)
                if d_tp is None or d_ps is None:
                    print("  %-40s FALHOU: %s / %s" % (rotulo, e_tp, e_ps))
                    rel["casos"].append({"rotulo": rotulo, "falhou": [e_tp, e_ps]})
                    continue
                g1 = np.asarray(d_tp["detected_count_rate"], float)
                g2 = np.asarray(d_ps["detected_count_rate"], float)
                s1, s2 = float(g1.sum()), float(g2.sum())
                rd = abs(s2 / s1 - 1.0) if s1 else float("nan")
                cel = float(np.abs((g2 - g1) / np.maximum(np.abs(g1), 1e-300)).max())
                caso = {"rotulo": rotulo, "isotermico": isoter,
                        "max_images": mi, "bandas": b,
                        "soma_theta_phi": s1, "soma_psi0": s2,
                        "diferenca_relativa_do_total": rd,
                        "pior_celula_relativa": cel,
                        "segundos_theta_phi": t_tp, "segundos_psi0": t_ps,
                        "custo_relativo": t_ps / t_tp if t_tp else None}
                rel["casos"].append(caso)
                print("  %-40s total %.3e  pior celula %.3e  custo %.2fx"
                      % (rotulo, rd, cel, caso["custo_relativo"]), flush=True)
    Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print("gravado em %s" % a.saida)


if __name__ == "__main__":
    main()
