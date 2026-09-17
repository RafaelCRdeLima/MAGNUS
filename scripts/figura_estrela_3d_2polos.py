#!/usr/bin/env python3
"""Render 3D da superficie da RBS 1223: esfera projetada para o observador,
colorida pelo mapa de DOIS POLOS DESIGUAIS (polo quente + polo frio sobre fundo
mais frio), com graticula, eixo de rotacao Omega, marcadores dos polos e uma
tira de fases. Estilo do render 3D (figura_estrela.py), conteudo de 2 polos
(ajuste_2polos_2026-09-07). Uso: python3 <script> [saida_sem_extensao]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec

plt.rcParams.update({"font.family": "serif", "font.size": 12,
                     "mathtext.fontset": "cm", "figure.dpi": 120})
CMAP = cm.inferno
KEV_TO_MK = 11.604518
# vista OBLIQUA ilustrativa (a geometria e degenerada): de quase-perfil, para
# mostrar os DOIS casquetes ao mesmo tempo. Nao e a inclinacao do ajuste (40 graus),
# que poe o polo quente no limbo; a curva de luz (Fig. 1) e que carrega a geometria.
VIEW_INCL = 90.0
BASE_KT = 0.065                   # fundo
# (colatitude theta, azimute phi, raio angular, kT) em graus/keV
POLE_COOL = dict(theta=62.95, phi=0.0,     radius=55.08, kT=0.076)
POLE_HOT  = dict(theta=146.43, phi=-137.49, radius=43.68, kT=0.109)


def unit(theta, phi):
    t, p = np.deg2rad(theta), np.deg2rad(phi)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def rot_z(v, a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([ca * v[0] - sa * v[1], sa * v[0] + ca * v[1], v[2]])


def camera(incl_deg):  # incl_deg = VIEW_INCL (ilustrativo)
    i = np.deg2rad(incl_deg)
    w = np.array([np.sin(i), 0.0, np.cos(i)])    # estrela -> observador
    u = np.array([-np.cos(i), 0.0, np.sin(i)])   # "cima" (proj. do eixo Omega)
    r = np.array([0.0, 1.0, 0.0])                # "direita"
    return w, u, r


def _bump(delta, R):
    """perfil radial suave: 1 no centro, 0 em delta=R (cosseno levantado, C^1)."""
    x = np.clip(delta / R, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def two_pole_T(P, c1, r1, k1, c2, r2, k2):
    """kT (keV) para direcoes P (...,3): fundo + dois casquetes com GRADIENTE.
    As contribuicoes somam em T^4 (fluxos somam); cada casquete decai suave do
    centro (kT_i) ate o fundo na borda (raio angular r_i)."""
    b4 = BASE_KT**4
    d1 = np.arccos(np.clip(P @ c1, -1, 1))
    d2 = np.arccos(np.clip(P @ c2, -1, 1))
    T4 = (b4 + (k1**4 - b4) * _bump(d1, np.deg2rad(r1))
             + (k2**4 - b4) * _bump(d2, np.deg2rad(r2)))
    return T4**0.25


def render_disk(ax, phase, npix, norm, c1_0, c2_0):
    w, u, r = camera(VIEW_INCL)
    a = 2 * np.pi * phase
    c1, c2 = rot_z(c1_0, a), rot_z(c2_0, a)
    lin = np.linspace(-1, 1, npix)
    xs, ys = np.meshgrid(lin, lin)
    rr2 = xs * xs + ys * ys
    inside = rr2 <= 1.0
    zs = np.zeros_like(xs); zs[inside] = np.sqrt(1.0 - rr2[inside])
    P = xs[..., None] * r + ys[..., None] * u + zs[..., None] * w
    T = two_pole_T(P, c1, POLE_COOL["radius"], POLE_COOL["kT"],
                   c2, POLE_HOT["radius"], POLE_HOT["kT"])
    rgba = CMAP(norm(T)); rgba[~inside] = [0, 0, 0, 0]
    ax.imshow(rgba, extent=[-1, 1, -1, 1], origin="lower", zorder=1,
              interpolation="bilinear")
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, lw=1.1, color="0.15", zorder=6))
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal"); ax.axis("off")
    return w, u, r, c1, c2


def project(v, w, u, r):
    return (v @ r), (v @ u), (v @ w > 1e-9)


def draw_graticule(ax, w, u, r, a, lw=0.45):
    t = np.linspace(0, np.pi, 180); ph = np.linspace(0, 2 * np.pi, 360)
    for phi in np.deg2rad(np.arange(0, 360, 30)):
        v = np.stack([np.sin(t) * np.cos(phi), np.sin(t) * np.sin(phi), np.cos(t)])
        v = np.stack([rot_z(c, a) for c in v.T]).T
        x, y, f = project(v.T, w, u, r)
        ax.plot(np.where(f, x, np.nan), np.where(f, y, np.nan),
                color="0.25", lw=lw, zorder=4, alpha=0.65)
    for th in np.deg2rad(np.arange(30, 180, 30)):
        v = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                      np.full_like(ph, np.cos(th))])
        v = np.stack([rot_z(c, a) for c in v.T]).T
        x, y, f = project(v.T, w, u, r)
        ax.plot(np.where(f, x, np.nan), np.where(f, y, np.nan),
                color="0.25", lw=lw, zorder=4, alpha=0.65)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fig_star")
    c1_0 = unit(POLE_COOL["theta"], POLE_COOL["phi"])
    c2_0 = unit(POLE_HOT["theta"], POLE_HOT["phi"])
    norm = colors.Normalize(vmin=0.055, vmax=POLE_HOT["kT"])
    # fase que traz o polo quente para a frente, em 3/4 de perfil
    mid_phi = 0.5 * (POLE_COOL["phi"] + POLE_HOT["phi"])  # -68.7 deg
    main_phase = -mid_phi / 360.0   # traz os dois casquetes para +-68 da frente

    fig = plt.figure(figsize=(7.2, 8.4))
    gs = GridSpec(2, 4, height_ratios=[3.0, 1.0], hspace=0.18, wspace=0.08,
                  left=0.02, right=0.85, top=0.97, bottom=0.05)
    axm = fig.add_subplot(gs[0, :])
    w, u, r, c1, c2 = render_disk(axm, main_phase, 720, norm, c1_0, c2_0)
    draw_graticule(axm, w, u, r, 2 * np.pi * main_phase)
    # eixo de rotacao Omega
    xn, yn, _ = project(np.array([0.0, 0.0, 1.0]), w, u, r)
    axm.annotate("", xy=(xn, 1.28), xytext=(xn, yn - 0.05),
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#111"), zorder=7)
    axm.text(xn + 0.05, 1.20, r"$\Omega$", fontsize=13, zorder=7)
    # marcadores dos dois polos (quando visiveis)
    for c, lbl, col in ((c1, "cooler cap\n%.0f eV" % (POLE_COOL["kT"]*1e3), "#2ea3ff"),
                        (c2, "hotter cap\n%.0f eV" % (POLE_HOT["kT"]*1e3), "#ffd166")):
        xp, yp, front = project(c, w, u, r)
        if front:
            axm.scatter([xp], [yp], s=80, marker="o", facecolor=col,
                        edgecolor="0.1", linewidth=1.2, zorder=8)
            axm.text(xp, yp - 0.12, lbl, fontsize=9.5, color="0.1", ha="center",
                     va="top", zorder=8,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    axm.text(-1.32, 1.26, "illustrative view\n(geometry degenerate)",
             fontsize=10, ha="left", va="top", color="0.35")

    # tira de fases
    for k, ph in enumerate([0.0, 0.25, 0.5, 0.75]):
        axk = fig.add_subplot(gs[1, k])
        wk, uk, rk, c1k, c2k = render_disk(axk, ph, 300, norm, c1_0, c2_0)
        draw_graticule(axk, wk, uk, rk, 2 * np.pi * ph, lw=0.3)
        for c, col in ((c1k, "#2ea3ff"), (c2k, "#ffd166")):
            xp, yp, front = project(c, wk, uk, rk)
            if front:
                axk.scatter([xp], [yp], s=18, marker="o", facecolor=col,
                            edgecolor="0.1", linewidth=0.6, zorder=8)
        axk.set_title("phase %.2f" % ph, fontsize=10, pad=2)

    # barra de calor keV/MK
    cax = fig.add_axes([0.875, 0.44, 0.03, 0.50])
    sm = cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.yaxis.set_ticks_position("left"); cb.ax.yaxis.set_label_position("left")
    cb.ax.tick_params(direction="in")
    cb.set_label(r"Effective temperature $kT$ (keV)", fontsize=12, labelpad=8)
    cax2 = cax.twinx()
    cax2.set_ylim(0.055 * KEV_TO_MK, POLE_HOT["kT"] * KEV_TO_MK)
    cax2.set_ylabel(r"$T$ (MK)", fontsize=12, labelpad=6)
    cax2.tick_params(direction="in")

    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"figura 3D de 2 polos salva em {out}.pdf / .png")


if __name__ == "__main__":
    main()
