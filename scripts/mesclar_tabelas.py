#!/usr/bin/env python3
"""Mescla tabelas MAGNUSI2 com os mesmos eixos de mu e E, fazendo a uniao dos
nos de lg B, lg T, lg g e theta_B. Cada fatia (B, T, g, theta_B) tem de existir
em exatamente um dos arquivos; faltas ou conflitos abortam.

A uniao em theta_B entrou em 29/09/2026: a grade parava em 60 graus e alpha_B
chega a 90 na superficie ajustada, com 5,2 por cento do fluxo saindo grampeado
na borda, que e o achado 15 da auditoria. Antes disso o eixo de theta_B tinha de
ser identico em todas as entradas, e uma fatia angular nova nao podia ser
acrescentada sem refazer a grade inteira.

A uniao em lg T entrou em 18/09/2026: a lei de temperatura de Perez-Azorin poe o
equador em 0,3 T_polo, e no melhor ajuste isso da 0,45 MK contra o piso de 0,5 MK
da grade, ou seja 11 por cento da area extrapolada.

Uso: python3 scripts/mesclar_tabelas.py <saida.magnus> <a.magnus> <b.magnus> ...
"""
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tabela_intensidade as formato  # noqa: E402


def read_table(path):
    f = open(path, "rb")
    assert f.read(8) == b"MAGNUSI2", path
    def axis():
        n = struct.unpack("<i", f.read(4))[0]
        return np.frombuffer(f.read(4 * n), dtype="<f4").astype(float)
    axes = [axis() for _ in range(6)]
    shape = tuple(len(a) for a in axes)
    w = np.frombuffer(f.read(), dtype="<f4").reshape(shape).astype(float)
    return axes, w


def main():
    out = Path(sys.argv[1]); inputs = [Path(p) for p in sys.argv[2:]]
    tables = [read_table(p) for p in inputs]
    ref = tables[0][0]
    for (axes, _), p in zip(tables, inputs):
        for k, name in ((4, "mu"), (5, "lg E")):
            if not np.allclose(axes[k], ref[k], atol=1e-5):
                raise SystemExit(f"{p}: eixo {name} difere do de {inputs[0]}")
    key = lambda x: round(float(x), 3)
    Bs = sorted({key(b) for axes, _ in tables for b in axes[0]})
    Ts = sorted({key(t) for axes, _ in tables for t in axes[1]})
    Gs = sorted({key(g) for axes, _ in tables for g in axes[2]})
    Th = sorted({key(a) for axes, _ in tables for a in axes[3]})
    nmu, nE = len(ref[4]), len(ref[5])
    W = np.full((len(Bs), len(Ts), len(Gs), len(Th), nmu, nE), np.nan)
    filled = np.zeros((len(Bs), len(Ts), len(Gs), len(Th)), dtype=int)
    for (axes, w), p in zip(tables, inputs):
        for ib, b in enumerate(axes[0]):
            for it, t in enumerate(axes[1]):
                for ig, g in enumerate(axes[2]):
                    for ia, a in enumerate(axes[3]):
                        i, j, k, l = (Bs.index(key(b)), Ts.index(key(t)),
                                      Gs.index(key(g)), Th.index(key(a)))
                        if filled[i, j, k, l]:
                            raise SystemExit(f"fatia (lgB={b:.2f}, lgT={t:.3f}, "
                                             f"lgg={g:.2f}, theta_B={a:.1f}) "
                                             f"duplicada em {p}")
                        W[i, j, k, l] = w[ib, it, ig, ia]; filled[i, j, k, l] += 1
    faltando = np.argwhere(filled == 0)
    if len(faltando):
        primeiras = [(Bs[i], Ts[j], Gs[k], Th[l]) for i, j, k, l in faltando[:8]]
        raise SystemExit(f"{len(faltando)} fatias ausentes "
                         f"(lgB, lgT, lgg, theta_B), as primeiras: {primeiras}")
    formato.write_with_field(out, Bs, Ts, Gs, Th, ref[4], ref[5], W)
    print(f"GRAVADA {out}: {len(Bs)} B ({Bs[0]}..{Bs[-1]}) x {len(Ts)} T "
          f"({Ts[0]}..{Ts[-1]}) x {len(Gs)} g ({Gs[0]}..{Gs[-1]}) x {len(Th)} "
          f"theta_B ({Th[0]}..{Th[-1]}) x {nmu} mu x {nE} E")


if __name__ == "__main__":
    main()
