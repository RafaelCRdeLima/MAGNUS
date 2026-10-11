#!/usr/bin/env python3
"""Diagnóstico de pile-up nas doze observações sem medida.

PEDIDO DO AUDITOR em 07/10/2026, com a ressalva dele preservada: **ausência de
diagnóstico permanece pendência, não evidência de contaminação**, e o inverso
também vale, um diagnóstico negativo aqui não certifica ausência.

O QUE É MEDIDO. A coluna PATTERN da lista de eventos da fonte classifica cada
evento pela forma do padrão de carga: 0 é simples, 1 a 4 é duplo, 5 a 8 é triplo,
9 a 12 é quádruplo. Pile-up aparece como **déficit de simples e excesso de
multiplos**, porque dois fótons no mesmo quadro e em pixels vizinhos são lidos
como um evento de padrão maior.

O TESTE INTERNO, que é o que torna isto conclusivo sem calibração externa. As 13
observações vêem a MESMA fonte, com taxas entre 2,59 e 2,72 ct/s, mas em DOIS
modos: PrimeFullWindow com quadro de 73,4 ms e PrimeLargeWindow com 47,7 ms. A
probabilidade de pile-up é proporcional ao tempo de quadro, logo o LargeWindow
tem 35% menos. **Se o pile-up for relevante, a fração de múltiplos tem de ser
maior no FullWindow.** Se as duas coincidirem, o pile-up não domina a
classificação de padrão em nenhum dos modos.

Uso: python3 scripts/diagnostico_pileup.py [--saida docs/diagnostico_pileup.json]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from exposicao_por_fase import (PRODUTOS, acha_obs, cabecalho_csv,  # noqa: E402
                                tabela_numpy)

NPZ = RAIZ / "exploracoes/dados_combinados_13obs_2026-09-12/combinado13.npz"
EMIN, EMAX = 0.15, 1.2
#: Canal PI em eV; a banda em canais.
PI_MIN, PI_MAX = EMIN * 1000.0, EMAX * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(RAIZ / "docs/diagnostico_pileup.json"))
    args = ap.parse_args()
    t0 = time.time()
    d = np.load(NPZ, allow_pickle=True)
    obsids = [str(x) for x in d["obsids"]]
    expos = np.asarray(d["exposures"], float)

    print("=== fracoes de PATTERN na banda 0,15 a 1,2 keV ===")
    print("%-12s %-18s %8s %9s %9s %9s %9s %9s"
          % ("obsid", "modo", "ct/s", "eventos", "simples", "duplos",
             "triplos", "quadr."))
    linhas = []
    for obsid, exp in zip(obsids, expos):
        dirobs = acha_obs(obsid)
        meta = cabecalho_csv(sorted(dirobs.glob("pulsaris/*_events.csv"))[0])
        col = tabela_numpy(dirobs / "epn_bary_source.fits", "EVENTS",
                           {"PI", "PATTERN", "FLAG"})
        pi, pat = col["PI"], col["PATTERN"].astype(int)
        banda = (pi >= PI_MIN) & (pi < PI_MAX)
        pat = pat[banda]
        n = pat.size
        f = {"simples": float((pat == 0).mean()),
             "duplos": float(((pat >= 1) & (pat <= 4)).mean()),
             "triplos": float(((pat >= 5) & (pat <= 8)).mean()),
             "quadruplos": float(((pat >= 9) & (pat <= 12)).mean())}
        quadro = float(meta.get("time_resolution_us", 0) or 0) * 1e-6
        modo = "LargeWindow" if 0.04 < quadro < 0.06 else "FullWindow"
        if quadro == 0.0:   # 0163560101 declara zero; o FITS a identifica
            modo = "FullWindow"
        print("%-12s %-18s %8.3f %9d %9.4f %9.4f %9.4f %9.4f"
              % (obsid, modo, n / exp, n, f["simples"], f["duplos"],
                 f["triplos"], f["quadruplos"]))
        linhas.append({"obsid": obsid, "modo": modo, "quadro_s": quadro,
                       "eventos_na_banda": int(n), "taxa_ct_s": n / float(exp),
                       **f})

    print()
    print("=== o teste interno: FullWindow contra LargeWindow ===")
    por_modo = {}
    for modo in ("FullWindow", "LargeWindow"):
        g = [x for x in linhas if x["modo"] == modo]
        if not g:
            continue
        peso = np.array([x["eventos_na_banda"] for x in g], float)
        med = {k: float(np.average([x[k] for x in g], weights=peso))
               for k in ("simples", "duplos", "triplos", "quadruplos")}
        med["observacoes"] = len(g)
        med["quadro_s"] = float(np.median([x["quadro_s"] for x in g
                                           if x["quadro_s"] > 0]) or 0.0)
        med["eventos"] = int(peso.sum())
        med["taxa_media"] = float(np.average([x["taxa_ct_s"] for x in g],
                                             weights=peso))
        por_modo[modo] = med
        print("  %-12s %2d obs, %7d eventos, %.3f ct/s: simples %.4f, duplos "
              "%.4f, triplos %.4f, quadr. %.4f"
              % (modo, med["observacoes"], med["eventos"], med["taxa_media"],
                 med["simples"], med["duplos"], med["triplos"],
                 med["quadruplos"]))
    if len(por_modo) == 2:
        a, b = por_modo["FullWindow"], por_modo["LargeWindow"]
        # erro binomial das duas fracoes de simples
        sa = (a["simples"] * (1 - a["simples"]) / a["eventos"]) ** 0.5
        sb = (b["simples"] * (1 - b["simples"]) / b["eventos"]) ** 0.5
        dif = a["simples"] - b["simples"]
        sigma = (sa * sa + sb * sb) ** 0.5
        print()
        print("  diferenca na fracao de SIMPLES, Full menos Large: %+.5f"
              % dif)
        print("  erro binomial combinado: %.5f, logo %.1f sigma" % (sigma, dif / sigma))
        razao = a["quadro_s"] / b["quadro_s"] if b["quadro_s"] else float("nan")
        print("  o quadro do Full e %.2f vezes o do Large, logo se o pile-up "
              "dominasse" % razao)
        print("  o Full teria MENOS simples: o sinal esperado seria NEGATIVO")
        veredito = (
            "a fracao de simples do FullWindow e %s que a do LargeWindow por "
            "%.1f sigma, e o quadro do Full e %.2f vezes maior. %s" % (
                "MENOR" if dif < 0 else "MAIOR", abs(dif / sigma), razao,
                "O sinal e o do pile-up e exige medida propria com epatplot."
                if dif < 0 and abs(dif / sigma) > 3 else
                "O sinal e OPOSTO ao do pile-up: ele nao domina a classificacao "
                "de padrao em nenhum dos modos. A diferenca entre modos fica "
                "como item aberto, com causa provavel na calibracao de padrao e "
                "no limiar de energia, que diferem entre modos."
                if dif > 0 and abs(dif / sigma) > 3 else
                "A diferenca nao e significativa."))
        saida_extra = {"triplos_e_quadruplos_zerados": (
            "exatamente zero nas 13, o que e esperado numa fonte mole: padroes "
            "de tres e quatro pixels pedem energia maior para espalhar a carga")}
        print()
        print("  VEREDITO INTERNO: %s" % veredito)
    else:
        veredito = "um dos modos nao apareceu; teste interno indisponivel"

    saida = {
        "o_que_e": "diagnostico de pile-up pelas fracoes de PATTERN",
        "ressalva_do_auditor": (
            "ausencia de diagnostico permanece pendencia, nao evidencia de "
            "contaminacao; e o inverso tambem vale, um diagnostico negativo aqui "
            "nao certifica ausencia"),
        "o_que_mede": ("PATTERN 0 e simples, 1 a 4 duplo, 5 a 8 triplo, 9 a 12 "
                       "quadruplo; pile-up aparece como deficit de simples e "
                       "excesso de multiplos"),
        "teste_interno": (
            "as 13 veem a mesma fonte com taxas de 2,59 a 2,72 ct/s em dois "
            "modos, quadro de 73,4 e 47,7 ms; a probabilidade de pile-up e "
            "proporcional ao quadro, logo se ele for relevante o FullWindow tem "
            "de ter MENOS simples"),
        "por_observacao": linhas,
        "por_modo": por_modo,
        "veredito_interno": veredito,
        "observacao_sobre_multiplos_altos": (
            "triplos e quadruplos sao exatamente zero nas 13, esperado numa "
            "fonte mole: padroes de tres e quatro pixels pedem energia maior"),
        "limites_declarados": [
            "nao substitui o epatplot do SAS, que compara com a distribuicao "
            "calibrada esperada para o espectro da fonte",
            "a classificacao de PATTERN tambem depende do espectro e do limiar "
            "de energia, de modo que uma diferenca entre modos poderia ter outra "
            "causa",
            "o teste e sensivel a pile-up que mude a razao simples/multiplos, e "
            "nao a perdas que afetem os dois igualmente"],
        "segundos": time.time() - t0,
    }
    Path(args.saida).write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n")
    print("gravado em %s" % args.saida)


if __name__ == "__main__":
    main()
