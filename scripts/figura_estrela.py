#!/usr/bin/env python3
"""Renderiza a superfície da estrela do ajuste de espessura: mapa T(theta) com
os polos magnéticos quentes, geometria (Theta_B, phi_B, i) e barra de calor.

Convenções idênticas ao motor (write_spectral_grid / sample_full_sphere):
- referencial do corpo com z = eixo de rotação;
- eixo do dipolo em colatitude Theta_B e azimute phi_B;
- T^4(theta_m) = T_p^4 cos^2/(cos^2 + a sin^2) + T_min^4, cos theta_m = n.b_hat.

Uso: python3 scripts/figura_estrela.py <dir_do_resultado> [saida_sem_extensao]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 120,
})

CMAP = cm.inferno
KEV_TO_MK = 11.604518  # 1 keV -> 1.16e7 K = 11.6 MK


def load_geometry(result_dir: Path):
    d = json.loads((result_dir / "resultado.json").read_text())
    s = {row["name"]: row["median"] for row in d["summary"]}
    return dict(
        theta_b=s["magColat"], phi_b=s["magAzim"], incl=s["inclination"],
        a=s["peaking"], ktp=s["baseKT"], tmin_frac=0.3,
    )


def temperature_kev(P, m_hat, g):
    """kT(theta_m) em keV para direções P (…,3) e eixo magnético m_hat."""
    cos_tm = P @ m_hat
    c2 = cos_tm * cos_tm
    s2 = 1.0 - c2
    ktp4 = g["ktp"] ** 4
    ktmin4 = (g["tmin_frac"] * g["ktp"]) ** 4
    frac = np.divide(c2, c2 + g["a"] * s2, out=np.ones_like(c2),
                     where=(c2 + g["a"] * s2) > 0)
    return (ktp4 * frac + ktmin4) ** 0.25


def rot_z(v, alpha):
    ca, sa = np.cos(alpha), np.sin(alpha)
    x, y, z = v
    return np.array([ca * x - sa * y, sa * x + ca * y, z])


def camera(incl_deg):
    i = np.deg2rad(incl_deg)
    w = np.array([np.sin(i), 0.0, np.cos(i)])   # estrela -> observador
    u = np.array([-np.cos(i), 0.0, np.sin(i)])  # "cima" na imagem (proj. do eixo Omega)
    r = np.array([0.0, 1.0, 0.0])               # "direita" na imagem
    return w, u, r


def render_disk(ax, g, phase, npix, norm, m0):
    """Desenha o disco visível colorido por kT, com o campo girado à fase dada."""
    w, u, r = camera(g["incl"])
    m_hat = rot_z(m0, 2 * np.pi * phase)
    lin = np.linspace(-1, 1, npix)
    xs, ys = np.meshgrid(lin, lin)
    rr2 = xs * xs + ys * ys
    inside = rr2 <= 1.0
    zs = np.zeros_like(xs)
    zs[inside] = np.sqrt(1.0 - rr2[inside])
    # P = xs*r + ys*u + zs*w  (ponto na frente da esfera)
    P = (xs[..., None] * r + ys[..., None] * u + zs[..., None] * w)
    kt = temperature_kev(P, m_hat, g)
    rgba = CMAP(norm(kt))
    rgba[~inside] = [0, 0, 0, 0]
    ax.imshow(rgba, extent=[-1, 1, -1, 1], origin="lower", zorder=1,
              interpolation="bilinear")
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, lw=1.1, color="0.15", zorder=6))
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    return w, u, r, m_hat


def project(vec, w, u, r):
    front = vec @ w > 1e-9
    return (vec @ r), (vec @ u), front


def draw_graticule(ax, w, u, r, alpha, color="0.25", lw=0.45):
    t = np.linspace(0, np.pi, 180)
    ph = np.linspace(0, 2 * np.pi, 360)
    # meridianos (phi const)
    for phi in np.deg2rad(np.arange(0, 360, 30)):
        v = np.stack([np.sin(t) * np.cos(phi), np.sin(t) * np.sin(phi), np.cos(t)])
        v = np.stack([rot_z(c, alpha) for c in v.T]).T
        x, y, f = project(v.T, w, u, r)
        x = np.where(f, x, np.nan); y = np.where(f, y, np.nan)
        ax.plot(x, y, color=color, lw=lw, zorder=4, alpha=0.7)
    # paralelos (theta const)
    for th in np.deg2rad(np.arange(30, 180, 30)):
        v = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                      np.full_like(ph, np.cos(th))])
        v = np.stack([rot_z(c, alpha) for c in v.T]).T
        x, y, f = project(v.T, w, u, r)
        x = np.where(f, x, np.nan); y = np.where(f, y, np.nan)
        ax.plot(x, y, color=color, lw=lw, zorder=4, alpha=0.7)


def draw_axes_and_poles(ax, w, u, r, m_hat, g):
    # eixo de rotação Omega (polo norte do corpo, invariante sob giro em z)
    npole = np.array([0.0, 0.0, 1.0])
    xn, yn, _ = project(npole, w, u, r)
    ax.annotate("", xy=(xn, 1.28), xytext=(xn, yn - 0.05),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#111111"), zorder=7)
    ax.text(xn + 0.05, 1.20, r"$\Omega$", fontsize=13, ha="left", va="center",
            zorder=7)
    # eixo magnético (dipolo): linha tracejada entre os dois polos projetados
    for sign, tag in ((+1, "N"), (-1, "S")):
        pole = sign * m_hat
        xp, yp, front = project(pole, w, u, r)
        if front:
            ax.plot([0, xp * 1.18], [0, yp * 1.18], ls=(0, (5, 3)), lw=1.4,
                    color="#2ea3ff", zorder=7)
            ax.scatter([xp], [yp], s=70, marker="o", facecolor="#2ea3ff",
                       edgecolor="white", linewidth=1.1, zorder=8)
            ax.text(xp + 0.06 * sign, yp - 0.10, "magnetic\npole",
                    fontsize=10, color="#1c6fb0", ha="left" if sign > 0 else "right",
                    va="top", zorder=8)


def main():
    result_dir = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else result_dir / "fig_star"
    g = load_geometry(result_dir)
    tb, pb = np.deg2rad(g["theta_b"]), np.deg2rad(g["phi_b"])
    m0 = np.array([np.sin(tb) * np.cos(pb), np.sin(tb) * np.sin(pb), np.cos(tb)])

    ktmin = g["tmin_frac"] * g["ktp"]
    norm = colors.Normalize(vmin=ktmin, vmax=g["ktp"])

    # fase que traz um polo magnético para a frente, em 3/4 de perfil
    main_phase = (-g["phi_b"] / 360.0) + 0.11

    fig = plt.figure(figsize=(7.2, 8.2))
    gs = GridSpec(2, 4, height_ratios=[3.0, 1.0], hspace=0.16, wspace=0.08,
                  left=0.02, right=0.86, top=0.97, bottom=0.06)
    axm = fig.add_subplot(gs[0, :])
    w, u, r, m_hat = render_disk(axm, g, main_phase, 720, norm, m0)
    draw_graticule(axm, w, u, r, 2 * np.pi * main_phase)
    draw_axes_and_poles(axm, w, u, r, m_hat, g)
    axm.text(-1.32, 1.24,
             (r"$\Theta_B=%.0f^\circ$, $i=%.0f^\circ$" % (g["theta_b"], g["incl"])
              + "\n" + r"$k T_p=%.3f$ keV, $a=%.2f$" % (g["ktp"], g["a"])),
             fontsize=11, ha="left", va="top")

    # tira de fases mostrando os polos varrendo o disco
    for k, ph in enumerate([0.0, 0.25, 0.5, 0.75]):
        axk = fig.add_subplot(gs[1, k])
        wk, uk, rk, mk = render_disk(axk, g, ph, 300, norm, m0)
        draw_graticule(axk, wk, uk, rk, 2 * np.pi * ph, lw=0.3)
        for sign in (+1, -1):
            xp, yp, front = project(sign * mk, wk, uk, rk)
            if front:
                axk.scatter([xp], [yp], s=16, marker="o", facecolor="#2ea3ff",
                            edgecolor="white", linewidth=0.6, zorder=8)
        axk.set_title("phase %.2f" % ph, fontsize=10, pad=2)

    # barra de calor: keV à esquerda, MK à direita, sem sobreposição
    cax = fig.add_axes([0.885, 0.44, 0.03, 0.50])
    sm = cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.ax.tick_params(direction="in")
    cb.set_label(r"Effective temperature $kT$ (keV)", fontsize=12, labelpad=8)
    cax2 = cax.twinx()
    cax2.set_ylim(ktmin * KEV_TO_MK, g["ktp"] * KEV_TO_MK)
    cax2.yaxis.set_ticks_position("right")
    cax2.yaxis.set_label_position("right")
    cax2.set_ylabel(r"$T$ (MK)", fontsize=12, labelpad=6)
    cax2.tick_params(direction="in")

    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"figura da estrela salva em {out}.pdf / .png")


if __name__ == "__main__":
    main()
