#!/usr/bin/env python3
"""Figura massa-raio: a compacidade medida contra equacoes de estado.

A faixa vem dos PERCENTIS DO CHECKPOINT, nao de numeros escritos a mao. A versao
anterior desta figura, em exploracoes/obsoletos/eos_mxr_2026-09-10/plot_mr.py,
tinha U_LO, U_HI = 0.617, 0.641 codificados, e esses limites nao sao a posterior:
o intervalo de 95 por cento em u e [0,6159; 0,6380]. A auditoria de 29/09/2026
apontou isso no achado 7, junto com o fato de que BBB2 entra no intervalo e SLy
tangencia sua borda, ao contrario do que o manuscrito afirmava.

Este script tambem imprime, para o texto, os u_max de cada curva e quais delas
alcancam a mediana e o intervalo, para que nenhuma afirmacao do artigo sobre
equacoes de estado seja escrita a mao outra vez.

Uso:  python3 scripts/figura_mr_eos.py [--saida DIR] [--janela 0.5]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eos_mr

RAIZ = Path(__file__).resolve().parents[1]
CKPT = ("exploracoes/convergencia_bacias_2026-09-25/B_i95/checkpoint.npz")
HAMBARYAN_MR = 0.087          # M/R de Hambaryan et al. 2011, z = 0.16
ESTILOS = {
    "SLy":               dict(color="#1d3557", lw=1.6, ls="-"),
    "APR4":              dict(color="#457b9d", lw=1.6, ls="-"),
    "MPA1":              dict(color="#2a9d8f", lw=1.6, ls="-"),
    "MIT bag (145 MeV)": dict(color="#bc4749", lw=1.6, ls="--"),
    "BBB2 (tabela)":     dict(color="0.45",    lw=1.4, ls=":"),
}
ROTULOS = {"MIT bag (145 MeV)": "MIT bag (quark star)", "BBB2 (tabela)": "BBB2"}


def posterior(caminho, janela):
    dados = np.load(caminho, allow_pickle=True)
    nomes = [str(v) for v in dados["names"]]
    amostras = dados["samples"]
    corte = int(amostras.shape[1] * (1.0 - janela))
    u = amostras[:, corte:, nomes.index("compactness")].ravel().astype(float)
    q = np.percentile(u, [2.5, 16, 50, 84, 97.5])
    return dict(u_2p5=float(q[0]), u_16=float(q[1]), u_50=float(q[2]),
                u_84=float(q[3]), u_97p5=float(q[4]),
                z_50=float(1.0 / np.sqrt(1.0 - q[2]) - 1.0),
                amostras=int(u.size), corte=corte,
                passos=int(amostras.shape[1]))


def linha_M(R, u):
    """M(R) ao longo de uma compacidade constante, u = 2 G M / R c^2."""
    return u * R / (2.0 * eos_mr.Rsun_km)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--saida", default=str(RAIZ / "exploracoes/figuras_mr"))
    p.add_argument("--janela", type=float, default=0.5,
                   help="fracao final da cadeia usada, 0,5 = segunda metade")
    a = p.parse_args()
    saida = Path(a.saida); saida.mkdir(parents=True, exist_ok=True)

    post = posterior(RAIZ / CKPT, a.janela)
    cs = eos_mr.curvas(raiz=RAIZ)

    # Dois paineis. O da esquerda e o plano massa-raio de sempre. O da direita
    # existe porque a afirmacao do artigo e sobre COMPACIDADE, e no plano M-R as
    # pontas das curvas moles encostam na borda de baixo da faixa sem que se
    # veja: e ali que se decide quais equacoes de estado alcancam o que medimos.
    fig, (ax, axu) = plt.subplots(
        1, 2, figsize=(9.6, 4.6), gridspec_kw=dict(width_ratios=[1.75, 1.0]))
    Rgrid = np.linspace(6.0, 16.0, 200)

    # causalidade, u > 0.708 (Lattimer & Prakash 2001)
    ax.fill_between(Rgrid, linha_M(Rgrid, 0.708), 3.2, color="0.88", zorder=0)
    ax.plot(Rgrid, linha_M(Rgrid, 0.708), color="0.6", lw=0.8)

    # a faixa: percentis reais, 95 por cento, com a mediana por dentro
    ax.fill_between(Rgrid, linha_M(Rgrid, post["u_2p5"]),
                    linha_M(Rgrid, post["u_97p5"]), color="#bc4749", alpha=0.22,
                    zorder=2, label=r"this work, $95\%%$ ($u=%.3f^{+%.3f}_{-%.3f}$)"
                    % (post["u_50"], post["u_97p5"] - post["u_50"],
                       post["u_50"] - post["u_2p5"]))
    ax.plot(Rgrid, linha_M(Rgrid, post["u_50"]), color="#bc4749", lw=1.4, zorder=3)
    ax.plot(Rgrid, linha_M(Rgrid, 2.0 * HAMBARYAN_MR), color="0.4", lw=1.1,
            ls="-.", zorder=2,
            label=r"Hambaryan et al. (2011), $M/R=0.087$")

    resumo = []
    for nome, (R, M) in cs.items():
        Re, Me = eos_mr.ramo_estavel(R, M)
        u = eos_mr.compactness(Re, Me)
        ax.plot(Re[Me >= (0.1 if "MIT" in nome else 1.0)],
                Me[Me >= (0.1 if "MIT" in nome else 1.0)],
                **ESTILOS[nome], zorder=4, label=ROTULOS.get(nome, nome))
        ax.plot(Re[-1], Me[-1], marker="|", color=ESTILOS[nome]["color"],
                ms=7, mew=1.6, zorder=5)   # a ponta: onde a compacidade para
        umax = float(u.max())
        alcanca_mediana = umax >= post["u_50"]
        entra_no_intervalo = umax >= post["u_2p5"]
        if alcanca_mediana:      # onde a curva cruza a mediana
            i = int(np.argmin(np.abs(u - post["u_50"])))
            ax.plot(Re[i], Me[i], "o", color=ESTILOS[nome]["color"], ms=6,
                    mec="white", mew=1.0, zorder=6)
            cruz = dict(R=float(Re[i]), M=float(Me[i]))
        else:
            cruz = None
        resumo.append(dict(eos=nome, u_max=umax, m_max=float(Me.max()),
                           alcanca_mediana=bool(alcanca_mediana),
                           entra_no_intervalo_95=bool(entra_no_intervalo),
                           margem_ate_u_2p5=umax - post["u_2p5"],
                           cruzamento_na_mediana=cruz))

    # painel da direita: a compacidade maxima de cada curva contra a posterior
    ordem = sorted(resumo, key=lambda d: d["u_max"])
    ys = np.arange(len(ordem))
    axu.axvspan(post["u_2p5"], post["u_97p5"], color="#bc4749", alpha=0.22,
                zorder=1, label=r"this work, $95\%$")
    axu.axvline(post["u_50"], color="#bc4749", lw=1.4, zorder=2)
    for y, r in zip(ys, ordem):
        cor = ESTILOS[r["eos"]]["color"]
        axu.plot([0.50, r["u_max"]], [y, y], color=cor, lw=2.4, solid_capstyle="butt",
                 zorder=3)
        axu.plot(r["u_max"], y, "o", color=cor, ms=6, mec="white", mew=1.0, zorder=4)
    axu.set_yticks(ys)
    axu.set_yticklabels([ROTULOS.get(r["eos"], r["eos"]).replace(" (quark star)", "")
                         for r in ordem], fontsize=9)
    axu.set_xlim(0.50, 0.685); axu.set_ylim(-0.7, len(ordem) - 0.3)
    axu.set_xlabel(r"maximum compactness $u=2GM/Rc^{2}$", fontsize=11)
    axu.tick_params(labelsize=9)
    axu.grid(axis="x", alpha=0.25, lw=0.6)
    axu.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    ax.set_xlim(7.0, 15.5); ax.set_ylim(0.4, 2.7)
    ax.set_xlabel("R (km)", fontsize=12)
    ax.set_ylabel(r"M ($M_\odot$)", fontsize=12)
    ax.grid(alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=10)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(saida / ("fig_mr_eos.%s" % ext), dpi=140)

    relatorio = dict(checkpoint=CKPT, janela=a.janela,
                     corte_em_passos=post["corte"], passos=post["passos"],
                     posterior={k: v for k, v in post.items()
                                if k.startswith(("u_", "z_"))},
                     equacoes_de_estado=resumo)
    (saida / "fig_mr_eos.json").write_text(json.dumps(relatorio, indent=2) + "\n")

    print("u 95%% = [%.6f, %.6f], mediana %.6f, z = %.4f"
          % (post["u_2p5"], post["u_97p5"], post["u_50"], post["z_50"]))
    for r in sorted(resumo, key=lambda d: -d["u_max"]):
        estado = ("alcanca a mediana" if r["alcanca_mediana"] else
                  "entra no intervalo de 95%" if r["entra_no_intervalo_95"] else
                  "fica fora do intervalo de 95%")
        print("  %-20s u_max = %.6f  %-28s margem ate u_2,5 = %+.6f"
              % (r["eos"], r["u_max"], estado, r["margem_ate_u_2p5"]))
    print("figura em", saida / "fig_mr_eos.pdf")


if __name__ == "__main__":
    main()
