#!/usr/bin/env python3
"""Mescla tabelas MAGNUSI2 com os mesmos eixos de T, theta_B, mu e E, fazendo a
uniao dos nos de lg B e lg g. Cada fatia (B, g) tem de existir em exatamente um
dos arquivos; faltas ou conflitos abortam.

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
        for k, name in ((1, "lg T"), (3, "theta_B"), (4, "mu"), (5, "lg E")):
            if not np.allclose(axes[k], ref[k], atol=1e-5):
                raise SystemExit(f"{p}: eixo {name} difere do de {inputs[0]}")
    key = lambda x: round(float(x), 3)
    Bs = sorted({key(b) for axes, _ in tables for b in axes[0]})
    Gs = sorted({key(g) for axes, _ in tables for g in axes[2]})
    nT, nth, nmu, nE = (len(ref[k]) for k in (1, 3, 4, 5))
    W = np.full((len(Bs), nT, len(Gs), nth, nmu, nE), np.nan)
    filled = np.zeros((len(Bs), len(Gs)), dtype=int)
    for (axes, w), p in zip(tables, inputs):
        for ib, b in enumerate(axes[0]):
            for ig, g in enumerate(axes[2]):
                i, j = Bs.index(key(b)), Gs.index(key(g))
                if filled[i, j]:
                    raise SystemExit(f"fatia (lgB={b:.2f}, lgg={g:.2f}) duplicada em {p}")
                W[i, :, j] = w[ib, :, ig]; filled[i, j] += 1
    missing = [(Bs[i], Gs[j]) for i in range(len(Bs)) for j in range(len(Gs)) if not filled[i, j]]
    if missing:
        raise SystemExit(f"fatias ausentes (lgB, lgg): {missing}")
    formato.write_with_field(out, Bs, ref[1], Gs, ref[3], ref[4], ref[5], W)
    print(f"GRAVADA {out}: {len(Bs)} B ({Bs[0]}..{Bs[-1]}) x {nT} T x {len(Gs)} g ({Gs[0]}..{Gs[-1]}) x {nth} theta_B x {nmu} mu x {nE} E")


if __name__ == "__main__":
    main()
