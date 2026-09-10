#!/usr/bin/env python3
"""Mapa Mollweide da temperatura de superficie de dois polos desiguais
(ilustrativo; a geometria detalhada e degenerada). Uso: python3 <script> <saida>"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

KEV_TO_MK = 11.604518
# medianas do ajuste de 2 polos (ajuste_2polos_2026-09-07); ilustrativo
POLE1 = dict(theta=62.95, phi=0.0,     radius=55.08, kT=0.076)  # polo frio
POLE2 = dict(theta=146.43, phi=-137.49, radius=43.68, kT=0.109)  # polo quente
BASE_KT = 0.066


def unit(theta, phi):
    t, p = np.deg2rad(theta), np.deg2rad(phi)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fig_star")
    c1, c2 = unit(POLE1["theta"], POLE1["phi"]), unit(POLE2["theta"], POLE2["phi"])
    cr1, cr2 = np.cos(np.deg2rad(POLE1["radius"])), np.cos(np.deg2rad(POLE2["radius"]))
    nlon, nlat = 360, 180
    lon = np.linspace(-np.pi, np.pi, nlon)
    lat = np.linspace(-np.pi / 2, np.pi / 2, nlat)
    LON, LAT = np.meshgrid(lon, lat)
    # direcao: lat = 90 - theta (colatitude), lon = phi
    nx = np.cos(LAT) * np.cos(LON)
    ny = np.cos(LAT) * np.sin(LON)
    nz = np.sin(LAT)
    d1 = nx * c1[0] + ny * c1[1] + nz * c1[2]
    d2 = nx * c2[0] + ny * c2[1] + nz * c2[2]
    T = np.full(LON.shape, BASE_KT)
    T = np.where(d1 >= cr1, np.maximum(T, POLE1["kT"]), T)
    T = np.where(d2 >= cr2, np.maximum(T, POLE2["kT"]), T)

    # plasma (roxo-azul -> magenta -> amarelo) e vmin abaixo do fundo, para o
    # fundo nao ficar preto e os tres niveis (fundo/frio/quente) se separarem.
    cmap = cm.plasma
    norm = colors.Normalize(vmin=BASE_KT - 0.006, vmax=POLE2["kT"])
    fig = plt.figure(figsize=(8.6, 4.6))
    # deixa folga a direita para a barra nao encostar na figura
    ax = fig.add_axes([0.02, 0.06, 0.74, 0.88], projection="mollweide")
    ax.pcolormesh(LON, LAT, T, cmap=cmap, norm=norm, shading="auto")
    # centros dos polos
    def moll_pt(theta, phi):
        la = np.pi / 2 - np.deg2rad(theta)
        lo = (np.deg2rad(phi) + np.pi) % (2 * np.pi) - np.pi
        return lo, la
    for pole, lbl in [(POLE1, "cooler, %.0f eV" % (POLE1["kT"] * 1e3)),
                      (POLE2, "hotter, %.0f eV" % (POLE2["kT"] * 1e3))]:
        lo, la = moll_pt(pole["theta"], pole["phi"])
        ax.plot(lo, la, "+", color="0.15", ms=13, mew=2.2)
        ax.annotate(lbl, xy=(lo, la), xytext=(0, -16), textcoords="offset points",
                    color="0.1", fontsize=10, ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    ax.grid(True, color="0.5", lw=0.4, alpha=0.6)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    cax = fig.add_axes([0.86, 0.20, 0.025, 0.58])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax); cb.set_label(r"$kT$ (keV)", fontsize=12)
    cb.ax.tick_params(direction="in")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print("mapa de 2 polos salvo em", out)


if __name__ == "__main__":
    main()
