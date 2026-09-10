#!/usr/bin/env python3
"""Co-adiciona observações XMM (CSVs de eventos do XREDUX) numa grade
fase x energia. Cada obs e dobrada no SEU proprio periodo (do cabecalho) e
alinhada por correlacao cruzada do perfil de pulso; as grades sao somadas.
Saida no formato preparedDataNpz que o scripts/mcmc_fit.py le.

Uso: python3 scripts/coadicionar.py <saida.npz> <ev1.csv> <ev2.csv> ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PHASE_BINS = 32
ENERGY_BINS = 40
EMIN, EMAX = 0.15, 1.2
EDGES = np.linspace(EMIN, EMAX, ENERGY_BINS + 1)
ECEN = 0.5 * (EDGES[:-1] + EDGES[1:])


def read_header(path: Path) -> dict:
    meta = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if "=" in line:
                k, v = line[1:].split("=", 1)
                meta[k.strip()] = v.strip()
    return meta


def fold_obs(path: Path):
    meta = read_header(path)
    period = float(meta["period_s"])
    pref = float(meta.get("phase_reference_s", 0.0))
    exp = float(meta["exposure_s"])
    # pula o bloco de metadados (#) + a linha de cabecalho (TIME,PI,DETECTED_ENERGY_KEV)
    n_skip = 0
    with open(path) as fh:
        for line in fh:
            n_skip += 1
            if not line.startswith("#"):
                break  # esta linha e o cabecalho de colunas; tambem pula
    arr = np.loadtxt(path, delimiter=",", skiprows=n_skip, usecols=(0, 2))
    t = arr[:, 0]
    e = arr[:, 1]
    m = (e >= EMIN) & (e < EMAX)
    t, e = t[m], e[m]
    ph = ((t - pref) / period) % 1.0
    ip = np.minimum((ph * PHASE_BINS).astype(int), PHASE_BINS - 1)
    ie = np.minimum(((e - EMIN) / (EMAX - EMIN) * ENERGY_BINS).astype(int), ENERGY_BINS - 1)
    grid = np.zeros((PHASE_BINS, ENERGY_BINS), dtype=np.int64)
    np.add.at(grid, (ip, ie), 1)
    return grid, exp, meta, period


def read_background(events_path: Path):
    cand = list(events_path.parent.glob("*background*.csv"))
    if not cand:
        return np.zeros(ENERGY_BINS)
    d = np.loadtxt(cand[0], delimiter=",", comments="#")
    return np.interp(ECEN, d[:, 0], d[:, 1])


def xcorr_shift(prof, ref):
    """deslocamento circular inteiro que maximiza a correlacao de prof com ref."""
    best, bestc = 0, -np.inf
    for s in range(PHASE_BINS):
        c = float(np.dot(np.roll(prof, s), ref))
        if c > bestc:
            bestc, best = c, s
    return best


def main():
    out = Path(sys.argv[1])
    paths = [Path(p) for p in sys.argv[2:]]
    grids, exps, bgs, metas, periods = [], [], [], [], []
    for p in paths:
        g, exp, meta, per = fold_obs(p)
        grids.append(g); exps.append(exp); metas.append(meta); periods.append(per)
        bgs.append(read_background(p))
        print(f"  {meta.get('obsid','?'):11s} P={per:.7f}s exp={exp:8.0f}s eventos(0.15-1.2)={g.sum():6d}")
    # referencia = obs com mais contagens
    ref_i = int(np.argmax([g.sum() for g in grids]))
    ref_prof = grids[ref_i].sum(1).astype(float)
    print(f"  referencia de fase: {metas[ref_i].get('obsid')} (mais contagens)")
    observed = np.zeros((PHASE_BINS, ENERGY_BINS), dtype=np.int64)
    shifts = []
    for i, g in enumerate(grids):
        s = 0 if i == ref_i else xcorr_shift(g.sum(1).astype(float), ref_prof)
        shifts.append(s)
        observed += np.roll(g, s, axis=0)
        print(f"    {metas[i].get('obsid')}: shift={s} bins")
    exposure = float(np.sum(exps))
    # fundo combinado: taxa media ponderada pela exposicao
    background = np.average(np.array(bgs), axis=0, weights=np.array(exps))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             observed=observed.ravel(),
             exposure=exposure,
             background=background,
             shifts=np.array(shifts),
             exposures=np.array(exps),
             obsids=np.array([m.get("obsid", "?") for m in metas]),
             periods=np.array(periods))
    print(f"\nGRAVADO {out}")
    print(f"  {len(paths)} obs | contagens totais={observed.sum()} | exposicao={exposure:.0f}s ({exposure/1e3:.1f} ks)")


if __name__ == "__main__":
    main()
