#!/usr/bin/env python3
"""Onde estao os 0,7175: decomposicao EXATA de D ln L por bin observado.

O revisor, 08/10/2026: "O proximo teste deve localizar o efeito, sem refinar
tudo: decompor a mudanca psi_0 120 -> 240 de ln L por fase/energia, para cada
vetor e depois para cada par; identificar quais bins e contribuicoes espaciais
dominam os 0,7175."

A DECOMPOSICAO E EXATA, nao aproximada. Como

    ln L = soma_b [ n_b ln mu_b - mu_b ]

a mudanca entre duas previsoes separa por bin sem resto:

    D ln L|_b = n_b ln(mu'_b / mu_b) - (mu'_b - mu_b)

e a mudanca na DIFERENCA entre dois vetores e a diferenca das decomposicoes.

O QUE ISTO DECIDE: se os 0,7175 estiverem concentrados em poucos bins, vale
adaptacao no plano da imagem; se estiverem distribuidos, refinar seletivamente
nao resolve e o tratamento tem de ser outro.

E O QUE NAO DECIDE: qual malha esta correta. Isto localiza a MUDANCA entre dois
niveis da candidata, que e instabilidade dela, nao erro da producao.

Uso: python3 scripts/localiza_malha.py
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

FASES = PO.FASES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--niveis", type=int, nargs=2, default=[120, 240])
    ap.add_argument("--saida", default=str(RAIZ / "docs/localiza_malha.json"))
    a = ap.parse_args()
    obss = {o["obsid"]: o for o in PO.prepara()}
    ordem = sorted(obss)
    meta, ev = m.read_events(RAIZ / "dados/eventos.csv")
    probs, obs_n = {}, {}
    for obsid, o in obss.items():
        b = PO.monta_pedido(o, 120, 8)
        b["outputPrecision"] = 17
        probs[obsid] = m.FitProblem(b, ev, meta, None)
        obs_n[obsid] = np.asarray(probs[obsid].observed, float).reshape(FASES, -1)

    n1, n2 = a.niveis
    print("=== as seis chamadas da candidata, guardando as grades ===")
    mus = {}
    for vetor, _, _ in PO.VETORES:
        for nivel in (n1, n2):
            grades, dt = CM.uma_rodada(obss, ordem, vetor, "psi0", nivel, 8,
                                       meta, ev)
            _, _, mu = CM.pontua(probs, grades, ordem)
            mus[(vetor, nivel)] = mu
            print("  %-12s psi0 %3d  %.1f s" % (vetor, nivel, dt), flush=True)

    rel = {"o_que_e": "decomposicao EXATA de D ln L por bin observado",
           "a_identidade": "D ln L|_b = n_b ln(mu'_b/mu_b) - (mu'_b - mu_b)",
           "o_que_NAO_decide": (
             "qual malha esta correta. Isto localiza a MUDANCA entre dois niveis "
             "da candidata, que e instabilidade dela"),
           "niveis": [n1, n2], "por_vetor": {}, "por_par": {}}

    def decompoe(va, vb):
        """D ln L|_b entre dois conjuntos de mus, somado nas 13 observacoes."""
        acc = np.zeros((FASES, obs_n[ordem[0]].shape[1]))
        for obsid in ordem:
            mu, mu2 = va[obsid], vb[obsid]
            n = obs_n[obsid]
            acc += n * np.log(np.maximum(mu2, 1e-300) / np.maximum(mu, 1e-300)) \
                   - (mu2 - mu)
        return acc

    print()
    print("=== por VETOR: onde esta a mudanca de %d para %d bandas ===" % (n1, n2))
    print("%-12s %12s %12s %12s %10s" % ("vetor", "D lnL total", "top 1 bin",
                                         "top 10", "top 1%"))
    for vetor, _, _ in PO.VETORES:
        dd = decompoe(mus[(vetor, n1)], mus[(vetor, n2)])
        pl = dd.ravel()
        tot = float(pl.sum())
        ordenado = np.sort(np.abs(pl))[::-1]
        nb = pl.size
        top1 = float(ordenado[0])
        top10 = float(ordenado[:10].sum())
        top1pc = float(ordenado[:max(1, nb // 100)].sum())
        soma_abs = float(ordenado.sum())
        ip, ie = np.unravel_index(int(np.argmax(np.abs(pl))), dd.shape)
        rel["por_vetor"][vetor] = {
            "D_lnL_total": tot, "soma_dos_valores_absolutos": soma_abs,
            "maior_bin": top1, "maior_bin_fase": int(ip), "maior_bin_energia": int(ie),
            "top_10_absoluto": top10, "top_1_por_cento_absoluto": top1pc,
            "bins": nb,
            "fracao_do_absoluto_no_top_1_por_cento": top1pc / soma_abs if soma_abs else None,
            "por_fase": dd.sum(1).tolist(), "por_energia": dd.sum(0).tolist()}
        print("%-12s %12.4f %12.4f %12.4f %10.1f%%"
              % (vetor, tot, top1, top10,
                 100 * top1pc / soma_abs if soma_abs else float("nan")))

    print()
    print("=== por PAR: a mudanca na DIFERENCA, e quao concentrada ===")
    print("%-24s %12s %12s %10s" % ("par", "mudanca", "top 10", "top 1%"))
    vs = [v for v, _, _ in PO.VETORES]
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            da = decompoe(mus[(vs[i], n1)], mus[(vs[i], n2)])
            db = decompoe(mus[(vs[j], n1)], mus[(vs[j], n2)])
            dd = da - db
            pl = dd.ravel()
            ordenado = np.sort(np.abs(pl))[::-1]
            soma_abs = float(ordenado.sum())
            top1pc = float(ordenado[:max(1, pl.size // 100)].sum())
            rel["por_par"]["%s - %s" % (vs[i], vs[j])] = {
                "mudanca_na_diferenca": float(pl.sum()),
                "soma_dos_valores_absolutos": soma_abs,
                "top_10_absoluto": float(ordenado[:10].sum()),
                "top_1_por_cento_absoluto": top1pc,
                "fracao_do_absoluto_no_top_1_por_cento":
                    top1pc / soma_abs if soma_abs else None,
                "por_fase": dd.sum(1).tolist(),
                "por_energia": dd.sum(0).tolist()}
            print("%-24s %12.4f %12.4f %9.1f%%"
                  % ("%s - %s" % (vs[i], vs[j]), pl.sum(),
                     float(ordenado[:10].sum()),
                     100 * top1pc / soma_abs if soma_abs else float("nan")))
    pior = max(rel["por_par"].values(),
               key=lambda x: x["fracao_do_absoluto_no_top_1_por_cento"] or 0.0)
    f = pior["fracao_do_absoluto_no_top_1_por_cento"]
    print()
    if f and f > 0.5:
        print("CONCENTRADO: o 1%% dos bins carrega %.0f%% do absoluto, logo "
              "refino SELETIVO e o caminho" % (100 * f))
    else:
        print("DISTRIBUIDO: o 1%% dos bins carrega so %.0f%% do absoluto, logo "
              "refino seletivo nao resolve" % (100 * (f or 0)))
    rel["veredito_da_concentracao"] = (
        "concentrado" if f and f > 0.5 else "distribuido")
    Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print("gravado em %s" % a.saida)


if __name__ == "__main__":
    main()
