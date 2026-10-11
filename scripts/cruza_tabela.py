#!/usr/bin/env python3
"""Teste CRUZADO da tabela de raios, com malha e fases FIXAS.

O revisor, 08/10/2026: "eu faria agora um teste cruzado da tabela de raios,
mantendo malha e fases fixas: nos casos envolvendo A, que dominam a
instabilidade, compare a tabela atual com uma mais fina. Isso separa um
componente que ja apareceu como piso anteriormente."

POR QUE A TABELA E CANDIDATA. O padrao do motor e 1800 nos e os pedidos nao o
alteram, logo a producao usa 1800. Medido antes: a tabela so e inerte enquanto o
erro da MALHA domina; com a malha fina ela e o PISO. A candidata a 480 e 960
bandas tem malha fina, e portanto e o regime em que a tabela passa a aparecer.

O DESENHO SEPARA UM EIXO SO: malha fixa em 480 bandas, 32 fases, e apenas a
tabela varia, de 1800 para 115200 nos. Se a mudanca de ln L entre niveis de malha
for explicada pela tabela, corrige-se esse componente e repetem-se so os casos
afetados. Se nao for, a discrepancia da candidata tem de ser localizada antes de
refinar de novo.

O que isto NAO mede: erro contra a solucao continua. Mede discordancia entre duas
configuracoes de tabela.

Uso: python3 scripts/cruza_tabela.py [--bandas 480] [--tabelas 1800 115200]
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
import compara_malhas as CM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandas", type=int, default=480)
    ap.add_argument("--tabelas", type=int, nargs="+", default=[1800, 115200])
    ap.add_argument("--saida", default=str(RAIZ / "docs/cruza_tabela.json"))
    a = ap.parse_args()
    obss = {o["obsid"]: o for o in PO.prepara()}
    ordem = sorted(obss)
    meta, ev = m.read_events(RAIZ / "dados/eventos.csv")
    probs = {}
    for obsid, o in obss.items():
        b = PO.monta_pedido(o, 120, 8)
        b["outputPrecision"] = 17
        probs[obsid] = m.FitProblem(b, ev, meta, None)

    rel = {"o_que_e": "teste cruzado da tabela de raios, malha e fases FIXAS",
           "malha": "psi_0, %d bandas" % a.bandas, "fases": 32,
           "tabelas": a.tabelas,
           "o_que_NAO_mede": ("erro contra a solucao continua. Mede "
                              "discordancia entre duas configuracoes de tabela"),
           "o_padrao_do_motor": "1800 nos, e os pedidos nao o alteram",
           "ln_L": {}}
    print("=== tabela de raios cruzada, malha psi_0 %d bandas, 32 fases ==="
          % a.bandas)
    print("%-12s %10s %18s %10s" % ("vetor", "tabela", "ln L conjunto", "segundos"))
    for vetor, _, _ in PO.VETORES:
        rel["ln_L"][vetor] = {}
        for tab in a.tabelas:
            grades, dt = CM.uma_rodada(obss, ordem, vetor, "psi0", a.bandas, 8,
                                       meta, ev,
                                       extra=["--ray-table-size", str(tab)])
            total, por_obs, _ = CM.pontua(probs, grades, ordem)
            rel["ln_L"][vetor][str(tab)] = {"ln_L_conjunto": total,
                                            "segundos": dt,
                                            "por_observacao": por_obs}
            print("%-12s %10d %18.6f %10.1f" % (vetor, tab, total, dt), flush=True)

    print()
    print("=== o efeito da TABELA nas diferencas entre vetores ===")
    vs = [v for v, _, _ in PO.VETORES]
    t0, t1 = str(a.tabelas[0]), str(a.tabelas[-1])
    print("%-24s %14s %14s %12s" % ("par", "tabela %s" % t0,
                                    "tabela %s" % t1, "mudanca"))
    pares, pior = [], 0.0
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            d0 = rel["ln_L"][vs[i]][t0]["ln_L_conjunto"] - rel["ln_L"][vs[j]][t0]["ln_L_conjunto"]
            d1 = rel["ln_L"][vs[i]][t1]["ln_L_conjunto"] - rel["ln_L"][vs[j]][t1]["ln_L_conjunto"]
            pares.append({"i": vs[i], "j": vs[j], "dif_tabela_grossa": d0,
                          "dif_tabela_fina": d1, "mudanca": d1 - d0})
            pior = max(pior, abs(d1 - d0))
            print("%-24s %14.4f %14.4f %+12.4f"
                  % ("%s - %s" % (vs[i], vs[j]), d0, d1, d1 - d0))
    print()
    #: A referencia de comparacao: quanto a MALHA movia entre 480 e 960, que foi
    #: 0,1077 no pior par (docs/compara_malhas_fino.json).
    DA_MALHA = 0.1077
    print("pior mudanca pela TABELA: %.4f" % pior)
    print("para comparar, a mudanca pela MALHA entre 480 e 960 foi %.4f" % DA_MALHA)
    frac = pior / DA_MALHA if DA_MALHA else float("nan")
    print("razao: %.2f" % frac)
    #: O VEREDITO ABAIXO SO VALE quando a lista de tabelas comeca no PADRAO do
    #: motor, 1800, porque a comparacao e contra o efeito da malha. Usado para
    #: medir CONVERGENCIA da tabela, entre dois niveis finos, ele rotula errado:
    #: uma mudanca pequena significaria tabela CONVERGIDA e nao "nao explica".
    #: Aconteceu, e eu reportei a leitura certa em vez de citar o rotulo.
    if a.tabelas[0] != 1800:
        veredito = ("CONVERGENCIA DA TABELA: entre %d e %d nos as diferencas "
                    "movem no maximo %.4f, contra a margem de 0,02. %s"
                    % (a.tabelas[0], a.tabelas[-1], pior,
                       "CONVERGIDA" if pior < 0.02 else "NAO convergida"))
        print(veredito)
    elif frac > 0.5:
        veredito = ("A TABELA EXPLICA BOA PARTE: %.0f por cento da mudanca que a "
                    "malha mostrava. Corrigir a tabela e repetir so os casos "
                    "afetados" % (100 * frac))
    elif frac < 0.2:
        veredito = ("A TABELA NAO EXPLICA: so %.0f por cento. A discrepancia da "
                    "candidata tem de ser localizada antes de refinar de novo"
                    % (100 * frac))
    else:
        veredito = ("PARCIAL: a tabela responde por %.0f por cento, logo nem "
                    "explica nem se descarta" % (100 * frac))
    print(veredito)
    rel["pares"] = pares
    rel["pior_mudanca_pela_tabela"] = pior
    rel["mudanca_pela_malha_480_960"] = DA_MALHA
    rel["razao"] = frac
    rel["veredito"] = veredito
    Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print("gravado em %s" % a.saida)


if __name__ == "__main__":
    main()
