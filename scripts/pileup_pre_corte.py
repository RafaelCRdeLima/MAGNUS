#!/usr/bin/env python3
"""Pile-up nos produtos ANTERIORES ao corte PATTERN <= 4.

PEDIDO DO AUDITOR: "para pile-up, use produtos anteriores ao corte PATTERN<=4 e
o diagnostico apropriado". O corte remove justamente os multiplos altos que o
pile-up criaria, de modo que o diagnostico sobre a lista cortada esta cego a
assinatura mais forte.

A lista do ODF, `*_EPN_S*_ImagingEvts.ds`, traz PATTERN ate 255, o codigo de
padrao nao reconhecido do EPIC-pn, e e anterior a qualquer corte de padrao ou de
FLAG. Aqui ela e restringida a regiao da fonte, a banda e a FLAG == 0, e o
histograma COMPLETO de padroes e medido.

O DIAGNOSTICO. Pile-up converte simples em duplos, duplos em triplos, e cria
padroes nao reconhecidos. Dois indicadores:
  - a fracao de multiplos ALTOS, PATTERN de 5 a 12, que o corte eliminava;
  - a fracao de padroes INVALIDOS, PATTERN acima de 12, que inclui o 255.
E o teste entre modos continua valendo: o quadro do FullWindow e 1,54 vez o do
LargeWindow, logo se o pile-up for relevante os multiplos altos tem de ser mais
frequentes nele.

LIMITE DECLARADO: isto nao e o `epatplot` do SAS, que compara a distribuicao
medida com a esperada para o espectro da fonte. E um diagnostico RELATIVO entre
modos e entre observacoes, com a vantagem de ver os padroes que o corte removia.

Uso: python3 scripts/pileup_pre_corte.py [--saida docs/pileup_pre_corte.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from exposicao_por_fase import acha_obs, cabecalho_csv, tabela_numpy  # noqa: E402

NPZ = RAIZ / "exploracoes/dados_combinados_13obs_2026-09-12/combinado13.npz"
PI_MIN, PI_MAX = 150.0, 1200.0
#: A regiao da fonte vem do cabecalho do CSV, em unidades de 0,05 arcsec.
UNIDADES_POR_ARCSEG = 20.0


def regiao_do_csv(meta):
    """Centro e raio da regiao, do texto 'circulo r=30.0" em (X,Y)'."""
    import re
    texto = meta.get("region", "")
    m = re.search(r"r=([0-9.]+)\"?\s*em\s*\((\d+),\s*(\d+)\)", texto)
    if not m:
        return None
    raio_arcseg = float(m.group(1))
    return (float(m.group(2)), float(m.group(3)),
            raio_arcseg * UNIDADES_POR_ARCSEG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(RAIZ / "docs/pileup_pre_corte.json"))
    args = ap.parse_args()
    t0 = time.time()
    d = np.load(NPZ, allow_pickle=True)
    obsids = [str(x) for x in d["obsids"]]
    expos = np.asarray(d["exposures"], float)

    print("=== padroes ANTES do corte, na regiao da fonte e na banda ===")
    print("%-12s %-12s %8s %9s %9s %9s %9s %9s"
          % ("obsid", "modo", "eventos", "simples", "duplos", "altos 5-12",
             "invalidos", ">12 %"))
    linhas = []
    for obsid, exp in zip(obsids, expos):
        dirobs = acha_obs(obsid)
        meta = cabecalho_csv(sorted(dirobs.glob("pulsaris/*_events.csv"))[0])
        reg = regiao_do_csv(meta)
        odf = sorted(dirobs.glob("*_EPN_S*_ImagingEvts.ds"))
        if not odf or reg is None:
            print("%-12s %s" % (obsid, "sem lista do ODF ou sem regiao no CSV"))
            continue
        col = tabela_numpy(odf[0], "EVENTS", {"X", "Y", "PI", "PATTERN", "FLAG"})
        cx, cy, raio = reg
        dentro = ((col["X"] - cx) ** 2 + (col["Y"] - cy) ** 2) <= raio ** 2
        sel = dentro & (col["PI"] >= PI_MIN) & (col["PI"] < PI_MAX) \
            & (col["FLAG"] == 0)
        pat = col["PATTERN"][sel].astype(int)
        n = pat.size
        if n == 0:
            print("%-12s %s" % (obsid, "nenhum evento na selecao"))
            continue
        f = {"simples": float((pat == 0).mean()),
             "duplos": float(((pat >= 1) & (pat <= 4)).mean()),
             "altos": float(((pat >= 5) & (pat <= 12)).mean()),
             "invalidos": float((pat > 12).mean())}
        q = float(meta.get("time_resolution_us", 0) or 0) * 1e-6
        modo = "LargeWindow" if 0.04 < q < 0.06 else "FullWindow"
        print("%-12s %-12s %8d %9.5f %9.5f %9.5f %9.5f %9.3f"
              % (obsid, modo, n, f["simples"], f["duplos"], f["altos"],
                 f["invalidos"], 100 * f["invalidos"]))
        linhas.append({"obsid": obsid, "modo": modo, "eventos": int(n),
                       "regiao": {"x": cx, "y": cy, "raio_unidades": raio},
                       "lista": str(odf[0].name), **f})

    print()
    print("=== o teste entre modos, com os padroes que o corte removia ===")
    resumo = {}
    for modo in ("FullWindow", "LargeWindow"):
        g = [x for x in linhas if x["modo"] == modo]
        if not g:
            continue
        peso = np.array([x["eventos"] for x in g], float)
        med = {k: float(np.average([x[k] for x in g], weights=peso))
               for k in ("simples", "duplos", "altos", "invalidos")}
        entre = {k: float(np.std([x[k] for x in g], ddof=1) / np.sqrt(len(g)))
                 for k in ("simples", "duplos", "altos", "invalidos")}
        resumo[modo] = {"n": len(g), "eventos": int(peso.sum()),
                        "fracoes": med, "erro_entre_observacoes": entre}
        print("  %-12s n=%d %8d eventos: simples %.5f, duplos %.5f, altos %.5f, "
              "invalidos %.5f" % (modo, len(g), int(peso.sum()), med["simples"],
                                  med["duplos"], med["altos"], med["invalidos"]))
    if len(resumo) == 2:
        a, b = resumo["FullWindow"], resumo["LargeWindow"]
        print()
        print("  %-12s %12s %12s %12s %10s"
              % ("grandeza", "Full", "Large", "Full-Large", "sigma"))
        vereditos = {}
        for k in ("simples", "duplos", "altos", "invalidos"):
            dif = a["fracoes"][k] - b["fracoes"][k]
            s = (a["erro_entre_observacoes"][k] ** 2
                 + b["erro_entre_observacoes"][k] ** 2) ** 0.5
            print("  %-12s %12.5f %12.5f %12.5f %10.1f"
                  % (k, a["fracoes"][k], b["fracoes"][k], dif,
                     dif / s if s else float("nan")))
            vereditos[k] = {"diferenca": dif, "sigma": dif / s if s else None}
        resumo["comparacao"] = vereditos
        altos = vereditos["altos"]
        print()
        print("  o quadro do Full e 1,54 vez o do Large; se o pile-up for")
        print("  relevante, os ALTOS tem de ser mais frequentes nele")
        if altos["sigma"] is not None and altos["sigma"] > 3:
            v = ("os multiplos altos SAO mais frequentes no FullWindow, por %.1f "
                 "sigma: o sinal e o do pile-up e exige epatplot" % altos["sigma"])
        elif altos["sigma"] is not None and altos["sigma"] < -3:
            v = ("os multiplos altos sao MENOS frequentes no FullWindow, por %.1f "
                 "sigma: o sinal e oposto ao do pile-up" % abs(altos["sigma"]))
        else:
            v = ("a diferenca nos multiplos altos nao e significativa, %.1f sigma"
                 % (altos["sigma"] if altos["sigma"] is not None else float("nan")))
        print("  VEREDITO: %s" % v)
        resumo["veredito"] = v

    saida = {
        "o_que_e": "pile-up nos produtos anteriores ao corte PATTERN <= 4",
        "por_que": ("o corte remove justamente os multiplos altos que o pile-up "
                    "criaria, de modo que o diagnostico na lista cortada esta "
                    "cego a assinatura mais forte"),
        "lista_usada": "*_EPN_S*_ImagingEvts.ds do ODF, com PATTERN ate 255",
        "selecao": ("regiao da fonte do cabecalho do CSV, banda 0,15 a 1,2 keV, "
                    "FLAG == 0"),
        "por_observacao": linhas, "por_modo": resumo,
        "limites_declarados": [
            "nao e o epatplot do SAS, que compara a distribuicao medida com a "
            "esperada para o espectro da fonte",
            "e diagnostico RELATIVO entre modos e entre observacoes",
            "a regiao e reconstruida do texto do cabecalho e nao da expressao de "
            "selecao do SAS, logo pode diferir nas bordas"],
        "segundos": time.time() - t0,
    }
    Path(args.saida).write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n")
    print()
    print("gravado em %s" % args.saida)


if __name__ == "__main__":
    main()
