#!/usr/bin/env python3
"""Compara a interpolacao em lg B da tabela GROSSA (4 nos, 0,3-0,5 dex) com a
tabela DENSA (0,05 dex) no campo ajustado da RBS 1223.

Mostra o artefato: com a feicao ciclotron dentro da tabela, interpolar lg w entre
nos distantes produz duas depressoes (nas energias dos nos vizinhos) em vez de
uma em E_cp(B). O fluxo medio-em-angulo e calculado como no motor (trapezio em mu).

Uso: python3 scripts/compara_tabelas_B.py [lgB ...]   (padrao: 13.62 13.72 13.42)
Saida: exploracoes/tabela_densa_2026-09-13/compara_tabelas_B.png/.pdf
"""
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read_table(path):
    f = open(path, "rb")
    assert f.read(8) == b"MAGNUSI2", path
    def axis():
        n = struct.unpack("<i", f.read(4))[0]
        return np.frombuffer(f.read(4 * n), dtype="<f4").astype(float)
    axes = [axis() for _ in range(6)]           # lgB, lgT, lgG, thetaB, mu, lgE
    shape = tuple(len(a) for a in axes)
    w = np.frombuffer(f.read(), dtype="<f4").reshape(shape).astype(float)
    return axes, w


def bracket(axis, value):
    """Mesma regra do motor: grampeia nas bordas, linear entre nos."""
    if value <= axis[0]:
        return 0, 0.0
    if value >= axis[-1]:
        return len(axis) - 2, 1.0
    i = int(np.searchsorted(axis, value, side="right") - 1)
    i = min(i, len(axis) - 2)
    return i, (value - axis[i]) / (axis[i + 1] - axis[i])


def flux_lgw(axes, w, lgb, lgt, lgg, thb):
    """lg do fluxo medio-em-angulo (razao p/ corpo negro) vs E, interpolado
    linearmente em lg w nos eixos B, T, g, theta_B (como o motor)."""
    lgB, lgT, lgG, thB, mu, lgE = axes
    wt = np.zeros(len(mu)); wt[0] = 0.5 * (mu[1] - mu[0]); wt[-1] = 0.5 * (mu[-1] - mu[-2])
    wt[1:-1] = 0.5 * (mu[2:] - mu[:-2]); den = (mu * wt).sum()
    def node(ib, it, ig, ith):
        return np.log10((10 ** w[ib, it, ig, ith] * (mu * wt)[:, None]).sum(0) / den)
    out = np.zeros(len(lgE))
    for (ax, val) in ((lgB, lgb), (lgT, lgt), (lgG, lgg), (thB, thb)):
        pass
    ib, fb = bracket(lgB, lgb); it, ft = bracket(lgT, lgt); ig, fg = bracket(lgG, lgg); ih, fh = bracket(thB, thb)
    for db, cb in ((0, 1 - fb), (1, fb)):
        for dt, ct in ((0, 1 - ft), (1, ft)):
            for dg, cg in ((0, 1 - fg), (1, fg)):
                for dh, ch in ((0, 1 - fh), (1, fh)):
                    c = cb * ct * cg * ch
                    if c:
                        out += c * node(ib + db, it + dt, ig + dg, ih + dh)
    return 10 ** lgE, out


def main():
    fields = [float(x) for x in sys.argv[1:]] or [13.62, 13.72, 13.42]
    coarse = read_table(ROOT / "tabelas" / "magnus_campoBg.magnus")
    dense_path = ROOT / "tabelas" / "magnus_campoBg_denso.magnus"
    if not dense_path.exists():
        dense_path = ROOT / "build" / "magnus_campoBg_denso.magnus"
    dense = read_table(dense_path)
    lgt, lgg, thb = 6.114, 14.2, 0.0
    fig, axs = plt.subplots(1, len(fields), figsize=(4.6 * len(fields), 3.9), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, lgb in zip(axs, fields):
        E, yc = flux_lgw(*coarse, lgb, lgt, lgg, thb)
        E2, yd = flux_lgw(*dense, lgb, lgt, lgg, thb)
        ecp = 0.63 * 10 ** lgb / 1e14
        sel = (E > 0.07) & (E < 1.5)
        ax.plot(E[sel], yc[sel], color="0.5", lw=1.6, ls="--", label="coarse grid (4 nodes), interpolated")
        ax.plot(E2[sel], yd[sel], color="#1f4e79", lw=2.0, label="dense grid (0.05 dex)")
        ax.axvline(ecp, color="#bc4749", lw=1.0, ls=":", label=r"$E_{c,p}(B)$")
        ib, fb = bracket(coarse[0][0], lgb)
        for k in (ib, ib + 1):
            ax.axvline(0.63 * 10 ** coarse[0][0][k] / 1e14, color="0.6", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_xlabel("emitted energy (keV)")
        ax.set_title(fr"$\lg B={lgb:.2f}$  ($E_{{c,p}}={ecp:.2f}$ keV)", fontsize=10)
        ax.grid(alpha=0.25)
        d = yd[sel] - yc[sel]
        print(f"lgB={lgb:.2f}: max |dlg w| grossa-densa em 0.07-1.5 keV = {np.abs(d).max():.3f} dex "
              f"(fator {10**np.abs(d).max():.2f}) em E={E[sel][np.argmax(np.abs(d))]:.3f} keV")
    axs[0].set_ylabel(r"$\lg\,\langle w\rangle_\mu$  (flux / blackbody)")
    axs[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(fr"Atmosphere table interpolation in $\lg B$ (lg T={lgt}, lg g={lgg}, $\alpha_B$={thb:.0f}°)", fontsize=10)
    fig.tight_layout()
    out = ROOT / "exploracoes" / "tabela_densa_2026-09-13"; out.mkdir(exist_ok=True)
    fig.savefig(out / "compara_tabelas_B.png", dpi=140); fig.savefig(out / "compara_tabelas_B.pdf")
    print("figura:", out / "compara_tabelas_B.png")


if __name__ == "__main__":
    main()
