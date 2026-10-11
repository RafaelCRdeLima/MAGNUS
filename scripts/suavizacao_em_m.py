#!/usr/bin/env python3
"""A suavizacao temporal em funcao de M: quando liga, e se ligar muda algo.

O revisor, 10/10/2026: "nao deixe o boxcar ligar apenas porque a grade interna
mudou. Antes, confira se a ativacao realmente altera a previsao: uma janela
arredondada para um bin pode continuar sendo identidade."

DUAS PARTES, e so a segunda e medicao:

A) O MAPA ARITMETICO do operador. Reproduz, sem rodar o motor, a decisao que
   `apply_temporal_resolution` toma (engine/main.cpp:2211): a guarda
   `resolution_s <= bin_s` devolve de imediato; senao a janela e
   `lround(res/bin)` limitada a `time_bins`. Para cada observacao e cada M diz
   em qual dos tres regimes ela cai. Isto PREVE, nao mede.

B) A MEDICAO. Roda o motor com a resolucao declarada e com zero, no mesmo M, e
   compara a grade de taxa detectada bit a bit e o ln L. E a medicao que decide.

O QUE ESTE TESTE NAO FAZ: nao certifica que o boxcar seja o operador temporal
correto. Janela 1 ser identidade significa que o operador HOJE implementado nada
faz naquele M, nao que a resposta temporal sub-bin do instrumento seja
desprezivel. O operador temporal consistente e item separado, pendente.

Uso: python3 scripts/suavizacao_em_m.py [--medir 5 7] [--bandas 120] [--tabela N]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import mcmc_fit as m  # noqa: E402
import por_observacao as PO  # noqa: E402

FASES_BASE = 32


def decide(periodo_s, resolucao_us, time_bins):
    """Espelho EXATO de apply_temporal_resolution, engine/main.cpp:2211.

    `std::lround` arredonda para o inteiro mais proximo, com o meio para longe
    de zero; para argumento positivo isso e floor(x + 0.5), que NAO e o round()
    do Python (esse arredonda o meio para o par).
    """
    bin_s = periodo_s / time_bins
    res_s = resolucao_us * 1.0e-6
    if res_s <= bin_s:
        return {"regime": "guarda devolve", "janela": 0, "bin_s": bin_s,
                "razao": res_s / bin_s}
    janela = max(1, int(math.floor(res_s / bin_s + 0.5)))
    janela = min(janela, time_bins)
    regime = "identidade algebrica" if janela == 1 else "media de verdade"
    return {"regime": regime, "janela": janela, "bin_s": bin_s,
            "razao": res_s / bin_s}


def mapa(obss, ms):
    linhas = []
    print("A) MAPA ARITMETICO (previsao, sem rodar o motor)")
    print("   %-12s %-9s %6s %10s %9s %7s  %s"
          % ("obsid", "res_us", "M", "fases", "bin_s", "res/bin", "regime"))
    for o in obss:
        for M in ms:
            tb = FASES_BASE * M
            d = decide(o["periodo"], o["resolucao_us"], tb)
            linhas.append({"obsid": o["obsid"], "resolucao_us": o["resolucao_us"],
                           "M": M, "fases": tb, **d})
            print("   %-12s %-9.0f %6d %10d %9.6f %7.4f  %s (janela %d)"
                  % (o["obsid"], o["resolucao_us"], M, tb, d["bin_s"],
                     d["razao"], d["regime"], d["janela"]))
    return linhas


def roda(obs, bandas, M, com_resolucao, tabela, malha_psi0):
    base = PO.monta_pedido(obs, bandas, 8)
    base["outputPrecision"] = 17
    base["phaseAverage"] = M
    if not com_resolucao:
        # A resolucao EFETIVA vai a zero; a declarada no manifesto nao muda.
        base["responseProfile"] = dict(base["responseProfile"])
        base["responseProfile"]["time_resolution_us"] = 0.0
    meta, ev = m.read_events(RAIZ / "dados/eventos.csv")
    prob = m.FitProblem(base, ev, meta, None)
    nome, caminho, rotulo = PO.VETORES[0]
    v = PO.le_vetor(prob, caminho, rotulo)
    comando = list(prob.worker_command())
    if malha_psi0:
        comando += ["--malha-psi0", "--psi-bandas", str(bandas)]
    if tabela:
        comando += ["--ray-table-size", str(tabela)]
    trab = m.EngineWorker(comando)
    t0 = time.time()
    doc = trab.evaluate(prob.worker_line(v))
    dt = time.time() - t0
    trab.close()
    if doc.get("status") != "ok":
        raise RuntimeError("motor falhou: %s" % doc.get("message"))
    return doc, float(prob.score_document(doc)), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 3, 5, 7, 9, 11, 15])
    ap.add_argument("--medir", type=int, nargs="*", default=[5, 7],
                    help="os M em que se roda o motor; vazio para so o mapa")
    ap.add_argument("--bandas", type=int, default=120)
    ap.add_argument("--tabela", type=int, default=0)
    ap.add_argument("--malha", choices=["producao", "psi0"], default="producao")
    ap.add_argument("--saida", default=str(RAIZ / "docs/suavizacao_em_m.json"))
    a = ap.parse_args()

    obss = PO.prepara(verbose=False)
    rel = {
        "o_que_e": ("quando a suavizacao temporal liga em funcao de M, e se "
                    "ligar altera a previsao"),
        "data": "2026-10-10",
        "pedido_do_revisor": (
            "nao deixe o boxcar ligar apenas porque a grade interna mudou. "
            "Antes, confira se a ativacao realmente altera a previsao: uma "
            "janela arredondada para um bin pode continuar sendo identidade"),
        "o_que_nao_faz": (
            "nao certifica que o boxcar seja o operador temporal correto; "
            "janela 1 ser identidade diz que o operador hoje implementado nada "
            "faz naquele M, nao que a resposta sub-bin seja desprezivel"),
        "configuracao_da_medicao": {"bandas": a.bandas, "tabela": a.tabela or "padrao",
                                    "malha": a.malha},
        "mapa_aritmetico": mapa(obss, a.ms),
        "medicoes": [],
    }

    # uma observacao de cada submodo, porque a resolucao difere entre eles
    por_res = {}
    for o in obss:
        por_res.setdefault(o["resolucao_us"], o)
    if a.medir:
        print("\nB) MEDICAO (motor, resolucao declarada contra zero)")
    for M in a.medir:
        for res_us, o in sorted(por_res.items()):
            tb = FASES_BASE * M
            prev = decide(o["periodo"], res_us, tb)
            d1, l1, t1 = roda(o, a.bandas, M, True, a.tabela, a.malha == "psi0")
            d0, l0, t0_ = roda(o, a.bandas, M, False, a.tabela, a.malha == "psi0")
            g1 = np.asarray(d1["detected_count_rate"], float)
            g0 = np.asarray(d0["detected_count_rate"], float)
            iguais = bool(np.array_equal(g1, g0))
            dmax = float(np.abs(g1 - g0).max())
            drel = float(np.abs((g1 - g0) / np.maximum(np.abs(g0), 1e-300)).max())
            med = {"obsid": o["obsid"], "resolucao_us": res_us, "M": M,
                   "fases": tb, "previsao": prev["regime"],
                   "janela_prevista": prev["janela"],
                   "grades_bit_a_bit_identicas": iguais,
                   "pior_diferenca_absoluta": dmax,
                   "pior_diferenca_relativa": drel,
                   "ln_com_resolucao": l1, "ln_sem_resolucao": l0,
                   "diferenca_de_ln": l1 - l0,
                   "segundos": [t1, t0_]}
            rel["medicoes"].append(med)
            print("   %-12s res %6.0f us  M=%-3d fases %-4d  previsto: %-20s"
                  % (o["obsid"], res_us, M, tb, prev["regime"]))
            print("        grades identicas bit a bit: %-5s  pior |dif| %.3e  "
                  "d(ln L) %+.6e" % (iguais, dmax, l1 - l0), flush=True)

    # o veredito: qual o primeiro M que de fato muda a previsao
    muda = [x for x in rel["medicoes"] if not x["grades_bit_a_bit_identicas"]]
    rel["veredito"] = {
        "medicoes_em_que_a_previsao_muda": [
            {"obsid": x["obsid"], "M": x["M"], "pior_diferenca_absoluta":
             x["pior_diferenca_absoluta"], "diferenca_de_ln": x["diferenca_de_ln"]}
            for x in muda],
        "primeiro_M_com_janela_maior_que_1": min(
            [x["M"] for x in rel["mapa_aritmetico"] if x["janela"] >= 2],
            default=None),
    }
    Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print("\ngravado em %s" % a.saida)


if __name__ == "__main__":
    main()
