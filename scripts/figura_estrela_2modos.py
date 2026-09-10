#!/usr/bin/env python3
"""Renderiza os DOIS modos geometricos do ajuste z,B-fixos lado a lado, cada um
colorido por kT(theta_m), compartilhando a barra de calor. Reaproveita as funcoes
de figura_estrela.py.

Uso: python3 scripts/figura_estrela_2modos.py <saida_sem_extensao>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figura_estrela as fe  # noqa: E402

# Os dois modos (medianas por modo do ajuste ajuste_zBfix_2026-09-06)
MODES = [
    dict(label="Mode A (64%)", theta_b=36.0, phi_b=-50.0, incl=36.0, a=7.2,
         ktp=0.101, tmin_frac=0.3),
    dict(label="Mode B (36%)", theta_b=90.0, phi_b=36.0, incl=43.0, a=2.0,
         ktp=0.101, tmin_frac=0.3),
]


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fig_star")
    ktp = MODES[0]["ktp"]
    ktmin = MODES[0]["tmin_frac"] * ktp
    norm = colors.Normalize(vmin=ktmin, vmax=ktp)

    fig = plt.figure(figsize=(9.2, 5.4))
    for k, g in enumerate(MODES):
        ax = fig.add_axes([0.015 + 0.435 * k, 0.05, 0.40, 0.82])
        tb, pb = np.deg2rad(g["theta_b"]), np.deg2rad(g["phi_b"])
        m0 = np.array([np.sin(tb) * np.cos(pb), np.sin(tb) * np.sin(pb), np.cos(tb)])
        phase = (-g["phi_b"] / 360.0) + 0.11
        w, u, r, m_hat = fe.render_disk(ax, g, phase, 620, norm, m0)
        fe.draw_graticule(ax, w, u, r, 2 * np.pi * phase)
        fe.draw_axes_and_poles(ax, w, u, r, m_hat, g)
        ax.set_title(g["label"] +
                     r": $\Theta_B=%.0f^\circ$, $i=%.0f^\circ$" % (g["theta_b"], g["incl"]),
                     fontsize=12, pad=6)

    cax = fig.add_axes([0.905, 0.20, 0.022, 0.55])
    sm = cm.ScalarMappable(norm=norm, cmap=fe.CMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.ax.tick_params(direction="in")
    cb.set_label(r"Effective temperature $kT$ (keV)", fontsize=12, labelpad=8)
    cax2 = cax.twinx()
    cax2.set_ylim(ktmin * fe.KEV_TO_MK, ktp * fe.KEV_TO_MK)
    cax2.set_ylabel(r"$T$ (MK)", fontsize=12)
    cax2.tick_params(direction="in")

    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"figura dos 2 modos salva em {out}.pdf / .png")


if __name__ == "__main__":
    main()
