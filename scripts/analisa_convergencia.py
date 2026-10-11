#!/usr/bin/env python3
"""O que a discretizacao faz as DIFERENCAS de ln L, nao ao absoluto.

POR QUE ESTE SCRIPT EXISTE. O `convergencia_producao.py` compara |D ln L|
ABSOLUTO contra a meta de 0,1, e isso esta errado pelo mesmo motivo que a
constante de agregacao de -770650 nao dizia nada na comparacao das formulacoes:
um deslocamento comum a todos os vetores sai inteiro das diferencas e nao afeta
conclusao nenhuma sobre parametros. O que decide e

    mudanca na diferenca = [lnL_i(prod) - lnL_j(prod)]
                           - [lnL_i(ref) - lnL_j(ref)]

Os valores por banda ja estao salvos, logo isto e analise e nao nova corrida.

Uso: python3 scripts/analisa_convergencia.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
BANDAS = (120, 240, 480, 960)


def richardson(hs, ys):
    d12, d23 = ys[0] - ys[1], ys[1] - ys[2]
    if d23 == 0.0 or d12 / d23 <= 0.0:
        return None, None
    razao = d12 / d23
    p = float(np.log(razao) / np.log(hs[0] / hs[1]))
    return p, float(ys[2] + d23 / (razao - 1.0))


def main():
    casos = {}
    for f in sorted(glob.glob(str(RAIZ / "docs/convergencia_producao_*.json"))):
        d = json.loads(Path(f).read_text())
        for c in d["casos"]:
            if c.get("falhou") is None and len(c["ln_L"]) == len(BANDAS):
                casos[c["rotulo"]] = c
    if not casos:
        raise SystemExit("nenhum caso completo ainda")
    hs = [1.0 / b for b in BANDAS]
    print("=== por caso: ordem, referencia e o deslocamento ABSOLUTO ===")
    print("%-26s %7s %7s %14s %10s" % ("caso", "ord a", "ord b", "referencia",
                                       "D abs"))
    for r, c in casos.items():
        pa, ya = richardson(hs[0:3], c["ln_L"][0:3])
        pb, yb = richardson(hs[1:4], c["ln_L"][1:4])
        c["_ref"], c["_pa"], c["_pb"], c["_ref_a"] = yb, pa, pb, ya
        c["_dabs"] = (c["ln_L"][0] - yb) if yb else None
        print("%-26s %7s %7s %14s %10s"
              % (r, "%.2f" % pa if pa else "-", "%.2f" % pb if pb else "-",
                 "%.4f" % yb if yb else "-",
                 "%+.4f" % c["_dabs"] if c["_dabs"] is not None else "-"))
    bons = {r: c for r, c in casos.items() if c.get("_ref")}
    print()
    print("=== o que decide: a mudanca nas DIFERENCAS entre casos ===")
    print("%-26s %-26s %12s %12s %10s"
          % ("caso i", "caso j", "dif producao", "dif referencia", "mudanca"))
    pares, piores = [], 0.0
    nomes = list(bons)
    for i in range(len(nomes)):
        for j in range(i + 1, len(nomes)):
            a, b = bons[nomes[i]], bons[nomes[j]]
            dp = a["ln_L"][0] - b["ln_L"][0]
            dr = a["_ref"] - b["_ref"]
            mud = dp - dr
            pares.append({"i": nomes[i], "j": nomes[j], "dif_producao": dp,
                          "dif_referencia": dr, "mudanca": mud})
            piores = max(piores, abs(mud))
            print("%-26s %-26s %12.4f %12.4f %+10.4f"
                  % (nomes[i], nomes[j], dp, dr, mud))
    # o erro da propria referencia, propagado as diferencas
    erro_ref = max(abs(c["_ref"] - c["_ref_a"]) for c in bons.values()
                   if c.get("_ref_a"))
    print()
    print("pior mudanca nas diferencas: %+.4f em ln L" % piores)
    print("erro da propria referencia, por caso: ate %.4f" % erro_ref)
    print("meta conservadora: 0,1")
    if erro_ref > piores:
        veredito = ("INCONCLUSIVO: o erro da referencia, %.4f, e maior que o "
                    "efeito nas diferencas, %.4f" % (erro_ref, piores))
    elif piores <= 0.1:
        veredito = ("PASSA: a discretizacao do integrador muda as diferencas de "
                    "ln L por no maximo %.4f, dentro de 0,1" % piores)
    else:
        veredito = ("EXIGE DECISAO: a discretizacao muda as diferencas por "
                    "%.4f, acima de 0,1" % piores)
    print(veredito)
    rel = {"o_que_e": "o efeito da discretizacao do integrador nas DIFERENCAS",
           "por_que_nao_o_absoluto": (
             "um deslocamento comum a todos os vetores sai inteiro das "
             "diferencas; foi o que a constante de agregacao de -770649 mostrou "
             "na comparacao das formulacoes"),
           "casos": {r: {"ordem_120_480": c["_pa"], "ordem_240_960": c["_pb"],
                         "referencia": c["_ref"], "deslocamento_absoluto": c["_dabs"],
                         "ln_L": c["ln_L"]} for r, c in bons.items()},
           "pares": pares, "pior_mudanca_nas_diferencas": piores,
           "erro_da_referencia": erro_ref, "meta": 0.1, "veredito": veredito}
    Path(RAIZ / "docs/convergencia_nas_diferencas.json").write_text(
        json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print("gravado em docs/convergencia_nas_diferencas.json")


if __name__ == "__main__":
    main()
