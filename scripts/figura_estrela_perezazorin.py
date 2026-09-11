#!/usr/bin/env python3
"""Render 3D da superficie da RBS 1223 usando a lei de dois polos de Perez-Azorin
que o motor ajustou. Agora com polo 2 NAO-antipodal (parametro beta = poleTilt2):
   eixo1 = m_hat (colat, azim);  eixo2 = -m_hat girado por beta em torno de
           k = (m_y, -m_x, 0)  (plano magneto-rotacional), como no motor.
   T^4(P) = T_min^4 + lobe(P.eixo1; T_p1,a1) + lobe(P.eixo2; T_p2,a2),
   lobe(c;Tp,a) = Tp^4 c^2/(c^2+a(1-c^2)) para c>0, senao 0;  T_min=0.3 T_p1.
Le os parametros do resultado do ajuste (summary/median).
Uso: python3 <script> <dir_do_resultado> [saida_sem_extensao]
"""
from __future__ import annotations
import sys, json
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
VIEW_INCL = 70.0          # vista obliqua ilustrativa (geometria degenerada)
COL1, COL2 = "#ffd166", "#a9aef0"   # cores neutras dos dois polos


def load(result_dir):
    r = json.loads((Path(result_dir) / "resultado.json").read_text())
    s = {x["name"]: x["median"] for x in r["summary"]}
    g = dict(colat=s.get("magColat", 45.0), azim=s.get("magAzim", 0.0),
             tp1=s.get("baseKT", 0.09), a1=s.get("peaking", 0.5),
             tp2=s.get("baseKT2", s.get("baseKT", 0.09)),
             a2=s.get("peaking2", s.get("peaking", 0.5)),
             beta=s.get("poleTilt2", 0.0), tmin_frac=0.3)
    return g


def unit(theta, phi):
    t, p = np.deg2rad(theta), np.deg2rad(phi)
    return np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])


def rot_z(v, a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([ca*v[0]-sa*v[1], sa*v[0]+ca*v[1], v[2]])


def rodrigues(v, k, ang):
    c, s = np.cos(ang), np.sin(ang)
    return v*c + np.cross(k, v)*s + k*(k @ v)*(1-c)


def pole_axes(mhat, beta_deg):
    """(eixo1, eixo2) para o m_hat corrente (ja girado pela fase)."""
    k = np.array([mhat[1], -mhat[0], 0.0])
    nk = np.linalg.norm(k)
    k = k/nk if nk > 1e-9 else np.array([1.0, 0.0, 0.0])
    axis2 = rodrigues(-mhat, k, np.deg2rad(beta_deg))
    return mhat, axis2


def camera(incl_deg):
    i = np.deg2rad(incl_deg)
    return (np.array([np.sin(i), 0, np.cos(i)]),
            np.array([-np.cos(i), 0, np.sin(i)]),
            np.array([0.0, 1.0, 0.0]))


def _lobe(c, tp4, a):
    cc = c*c
    s2 = np.clip(1-cc, 0, 1)
    return np.where(c > 0, tp4*cc/(cc + a*s2), 0.0)


def perez_azorin_T(P, mhat, g):
    """kT (keV) pela lei de dois polos, com polo 2 possivelmente nao-antipodal."""
    e1, e2 = pole_axes(mhat, g["beta"])
    c1 = P @ e1
    c2 = P @ e2
    tmin4 = (g["tmin_frac"]*g["tp1"])**4
    t4 = tmin4 + _lobe(c1, g["tp1"]**4, g["a1"]) + _lobe(c2, g["tp2"]**4, g["a2"])
    return np.maximum(t4, 1e-12)**0.25


def render_disk(ax, phase, npix, norm, g, mhat0):
    w, u, r = camera(VIEW_INCL)
    mhat = rot_z(mhat0, 2*np.pi*phase)
    lin = np.linspace(-1, 1, npix)
    xs, ys = np.meshgrid(lin, lin)
    rr2 = xs*xs + ys*ys; inside = rr2 <= 1.0
    zs = np.zeros_like(xs); zs[inside] = np.sqrt(1-rr2[inside])
    P = xs[..., None]*r + ys[..., None]*u + zs[..., None]*w
    T = perez_azorin_T(P, mhat, g)
    rgba = CMAP(norm(T)); rgba[~inside] = [0, 0, 0, 0]
    ax.imshow(rgba, extent=[-1, 1, -1, 1], origin="lower", zorder=1, interpolation="bilinear")
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, lw=1.1, color="0.15", zorder=6))
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_aspect("equal"); ax.axis("off")
    return w, u, r, mhat


def project(v, w, u, r):
    return (v @ r), (v @ u), (v @ w > 1e-9)


def graticule(ax, w, u, r, a, lw=0.45):
    t = np.linspace(0, np.pi, 180); ph = np.linspace(0, 2*np.pi, 360)
    for phi in np.deg2rad(np.arange(0, 360, 30)):
        v = np.stack([np.sin(t)*np.cos(phi), np.sin(t)*np.sin(phi), np.cos(t)])
        v = np.stack([rot_z(c, a) for c in v.T]).T
        x, y, f = project(v.T, w, u, r)
        ax.plot(np.where(f, x, np.nan), np.where(f, y, np.nan), color="0.25", lw=lw, alpha=0.6, zorder=4)
    for th in np.deg2rad(np.arange(30, 180, 30)):
        v = np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.full_like(ph, np.cos(th))])
        v = np.stack([rot_z(c, a) for c in v.T]).T
        x, y, f = project(v.T, w, u, r)
        ax.plot(np.where(f, x, np.nan), np.where(f, y, np.nan), color="0.25", lw=lw, alpha=0.6, zorder=4)


def main():
    result_dir = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("fig_star")
    g = load(result_dir)
    mhat0 = unit(g["colat"], g["azim"])
    hot = max(g["tp1"], g["tp2"])
    norm = colors.Normalize(vmin=0.055, vmax=hot)
    main_phase = (90.0 - g["azim"]) / 360.0

    fig = plt.figure(figsize=(7.2, 8.4))
    gs = GridSpec(2, 4, height_ratios=[3.0, 1.0], hspace=0.18, wspace=0.08,
                  left=0.02, right=0.85, top=0.97, bottom=0.05)
    axm = fig.add_subplot(gs[0, :])
    w, u, r, mhat = render_disk(axm, main_phase, 720, norm, g, mhat0)
    graticule(axm, w, u, r, 2*np.pi*main_phase)
    xn, yn, _ = project(np.array([0, 0, 1.0]), w, u, r)
    axm.annotate("", xy=(xn, 1.28), xytext=(xn, yn-0.05),
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#111"), zorder=7)
    axm.text(xn+0.05, 1.20, r"$\Omega$", fontsize=13, zorder=7)
    # marca os dois polos magneticos com rotulos neutros (temperaturas ~iguais)
    e1, e2 = pole_axes(mhat, g["beta"])
    for axis, tp, col, name in [(e1, g["tp1"], COL1, "pole 1"), (e2, g["tp2"], COL2, "pole 2")]:
        xp, yp, front = project(axis, w, u, r)
        if front:
            axm.scatter([xp], [yp], s=80, marker="o", facecolor=col, edgecolor="0.1",
                        linewidth=1.2, zorder=8)
            axm.text(xp, yp-0.12, "%s\n%.0f eV" % (name, tp*1e3),
                     fontsize=9.5, color="0.1", ha="center", va="top", zorder=8,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    axm.text(-1.32, 1.26, "illustrative view\n(geometry degenerate)",
             fontsize=10, ha="left", va="top", color="0.35")

    for k, ph in enumerate([0.0, 0.25, 0.5, 0.75]):
        axk = fig.add_subplot(gs[1, k])
        wk, uk, rk, mk = render_disk(axk, ph, 300, norm, g, mhat0)
        graticule(axk, wk, uk, rk, 2*np.pi*ph, lw=0.3)
        e1k, e2k = pole_axes(mk, g["beta"])
        for axis, col in [(e1k, COL1), (e2k, COL2)]:
            xp, yp, front = project(axis, wk, uk, rk)
            if front:
                axk.scatter([xp], [yp], s=18, marker="o", facecolor=col,
                            edgecolor="0.1", linewidth=0.6, zorder=8)
        axk.set_title("phase %.2f" % ph, fontsize=10, pad=2)

    cax = fig.add_axes([0.875, 0.44, 0.03, 0.50])
    sm = cm.ScalarMappable(norm=norm, cmap=CMAP); cb = fig.colorbar(sm, cax=cax)
    cb.ax.yaxis.set_ticks_position("left"); cb.ax.yaxis.set_label_position("left")
    cb.ax.tick_params(direction="in")
    cb.set_label(r"Effective temperature $kT$ (keV)", fontsize=12, labelpad=8)
    cax2 = cax.twinx(); cax2.set_ylim(0.055*KEV_TO_MK, hot*KEV_TO_MK)
    cax2.set_ylabel(r"$T$ (MK)", fontsize=12, labelpad=6); cax2.tick_params(direction="in")

    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"figura Perez-Azorin (2 polos, beta={g['beta']:.1f} deg) salva: "
          f"T_p1={g['tp1']*1e3:.0f} eV (a1={g['a1']:.2f}), "
          f"T_p2={g['tp2']*1e3:.0f} eV (a2={g['a2']:.2f})")


if __name__ == "__main__":
    main()
