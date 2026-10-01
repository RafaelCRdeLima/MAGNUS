#!/usr/bin/env python3
"""Diagnóstico da fatia angular theta_B = 90 graus, antes de mesclar na v5.

Duas perguntas. A primeira, de qualidade: onde esta a cauda de modelos que nao
conservaram fluxo, e ela cai na regiao que o ajuste da RBS 1223 usa? A segunda,
e a que justifica o trabalho: quanto o grampeamento em 60 graus estava custando,
ou seja, quanto a fisica em 90 difere da de 60, que o motor usava no lugar dela.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mesclar_tabelas as M  # noqa: E402

V4 = ROOT / "tabelas/magnus_campoBg_vacuo_v4.magnus"
SAIDA = ROOT / "exploracoes/theta90_diagnostico"
LGB_AJUSTE = 13.842          # lg B_p do ajuste fiducial da RBS 1223


def carrega_fatia():
    eixos = None
    blocos = []
    for nome in ("grade_theta90_A.magnus", "grade_theta90_B.magnus"):
        a, w = M.read_table(ROOT / "build" / nome)
        blocos.append((a, w))
    # une os dois blocos ao longo de lg g
    gs = sorted({round(float(g), 3) for a, _ in blocos for g in a[2]})
    a0 = blocos[0][0]
    W = np.full((len(a0[0]), len(a0[1]), len(gs), 1, len(a0[4]), len(a0[5])), np.nan)
    for a, w in blocos:
        for ig, g in enumerate(a[2]):
            W[:, :, gs.index(round(float(g), 3)), 0] = w[:, :, ig, 0]
    return [a0[0], a0[1], np.array(gs), np.array([90.0]), a0[4], a0[5]], W


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    eixos90, w90 = carrega_fatia()
    eixos4, w4 = M.read_table(V4)
    lgb, lgt, lgg = eixos90[0], eixos90[1], eixos90[2]
    print("fatia 90: %d B x %d T x %d g x 8 mu x %d E"
          % (len(lgb), len(lgt), len(lgg), len(eixos90[5])))

    # ---------- painel 1: onde esta a cauda ruim ----------
    dados = np.loadtxt(ROOT / "build/fluxo_theta90.txt")
    erro = np.full((len(lgb), len(lgt)), np.nan)
    for b, t, g, e in dados:                      # pior por celula (B, T)
        i, j = int(np.argmin(abs(lgb - b))), int(np.argmin(abs(lgt - t)))
        erro[i, j] = e if np.isnan(erro[i, j]) else max(erro[i, j], e)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    m = ax1.pcolormesh(lgt, lgb, np.log10(np.maximum(erro, 1e-4)),
                       cmap="magma", vmin=-3, vmax=0.5, shading="nearest")
    cb = fig.colorbar(m, ax=ax1); cb.set_label("lg (pior erro de fluxo por célula)")
    ruins = dados[dados[:, 3] > 0.30]
    ax1.plot(ruins[:, 1], ruins[:, 0], "o", mfc="none", mec="#39ff88", mew=1.8,
             ms=11, label="acima de 30 %")
    ax1.axhline(LGB_AJUSTE, color="#56e1d0", lw=1.6, ls="--",
                label="lg B do ajuste da RBS 1223")
    ax1.set_xlabel("lg T"); ax1.set_ylabel("lg B")
    ax1.set_title("Qualidade da fatia θ$_B$ = 90°", fontsize=11)
    ax1.legend(loc="lower left", fontsize=8, framealpha=.9)

    # ---------- painel 2: quanto 60 graus errava ----------
    # compara a fatia nova com o no de 60 graus da v4, nos eixos comuns
    ig4 = [int(np.argmin(abs(eixos4[2] - g))) for g in lgg]
    w60 = w4[:, :, ig4, 2]                        # theta_B = 60 na v4
    # Excluir as celulas grampeadas em |lg w| = 30 e ficar na banda do ajuste.
    # Sem isso a estatistica mistura a cauda de Wien, que o ajuste nao usa; e o
    # efeito e MAIOR sem ela, nao menor.
    E_ = 10 ** eixos90[5]
    banda = ((E_ >= 0.15) & (E_ <= 1.2))[None, None, None, None, :]
    bom = (np.isfinite(w90[:, :, :, 0:1]) & np.isfinite(w60[:, :, :, None])
           & (np.abs(w90[:, :, :, 0:1]) < 29.9) & (np.abs(w60[:, :, :, None]) < 29.9)
           & banda[:, :, :, 0, :][:, :, :, None, :])
    d = np.abs(w90[:, :, :, 0:1] - w60[:, :, :, None])[bom]
    d = d[np.isfinite(d)]
    ax2.hist(d, bins=np.linspace(0, 2.0, 80), color="#9e8cff", alpha=.85)
    ax2.axvline(np.median(d), color="#ffb86b", lw=2,
                label="mediana %.3f dex" % np.median(d))
    ax2.axvline(np.percentile(d, 90), color="#bc4749", lw=2, ls="--",
                label="p90 %.3f dex" % np.percentile(d, 90))
    ax2.set_xlabel("|lg w(90°) − lg w(60°)|   [dex]   ·   0,15–1,2 keV, sem grampeadas")
    ax2.set_ylabel("células")
    ax2.set_title("O que o grampeamento em 60° substituía  (%.0f %% acima de 0,1 dex)"
                  % (100 * np.mean(d > 0.1)), fontsize=11)
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(SAIDA / "theta90_qualidade.png", dpi=140)
    fig.savefig(SAIDA / "theta90_qualidade.pdf")

    print("diferença 90° contra 60°: mediana %.3f dex, p90 %.3f, p99 %.3f, max %.3f"
          % (np.median(d), np.percentile(d, 90), np.percentile(d, 99), d.max()))
    frac = 100 * np.mean(d > 0.1)
    print("células que diferem por mais de 0,1 dex (26 %%): %.1f %%" % frac)

    # ---------- painel 3: o espectro no no do ajuste ----------
    ib = int(np.argmin(abs(lgb - LGB_AJUSTE)))
    it = int(np.argmin(abs(lgt - 6.0)))
    ig = int(np.argmin(abs(lgg - 14.4)))
    ig4b = int(np.argmin(abs(eixos4[2] - lgg[ig])))
    ib4 = int(np.argmin(abs(eixos4[0] - lgb[ib])))
    it4 = int(np.argmin(abs(eixos4[1] - lgt[it])))
    IMU = 1          # mu = 0,102, a 84 graus da normal: onde a diferenca vive
    E = 10 ** eixos4[5]
    fig2, ax = plt.subplots(figsize=(7.4, 5.0))
    for k, (rot, cor) in enumerate(zip(("0°", "30°", "60°"),
                                       ("#1d3557", "#457b9d", "#2a9d8f"))):
        ax.plot(E, w4[ib4, it4, ig4b, k, IMU], color=cor, lw=1.6, label="θ$_B$ = " + rot)
    ax.plot(E, w90[ib, it, ig, 0, IMU], color="#bc4749", lw=2.2,
            label="θ$_B$ = 90°  (novo)")
    ax.set_xscale("log"); ax.set_xlabel("E (keV)"); ax.set_ylabel("lg w")
    ax.set_xlim(0.1, 2.0)
    ax.axvspan(0.15, 1.0, color="#56e1d0", alpha=.07, zorder=0)
    ax.set_title("lg B = %.2f, lg T = %.2f, lg g = %.1f, μ = 0,102 (84° da normal)"
                 % (lgb[ib], lgt[it], lgg[ig]), fontsize=11)
    ax.set_ylim(-2.5, 4)
    ax.grid(alpha=.25, lw=.6); ax.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(SAIDA / "theta90_espectro.png", dpi=140)
    fig2.savefig(SAIDA / "theta90_espectro.pdf")
    print("figuras em", SAIDA)


if __name__ == "__main__":
    main()
