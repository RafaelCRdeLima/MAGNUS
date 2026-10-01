#!/usr/bin/env python3
"""Resposta efetiva da coadicao: soma dos operadores ponderada pela exposicao.

A coadicao soma contagens de 13 observacoes, mas o ajuste dobrava o modelo por
UMA resposta, a de 2019, que contribui 6123 s de 131978 s. Achado 8 da auditoria
de 29/09 e 7 da de 01/10. Para uma fonte estacionaria o correto e

    R_ef(ch, E) = sum_i t_i A_i(E) P_i(ch|E) / sum_i t_i

e o formato do motor guarda area e redistribuicao em separado, entao a
decomposicao certa e

    A_ef(E)      = sum_i t_i A_i(E) / T
    P_ef(ch|E)   = sum_i t_i A_i(E) P_i(ch|E) / (T A_ef(E))

cujo produto devolve R_ef. Ponderar a redistribuicao so pela exposicao, sem a
area, daria a resposta errada onde as areas efetivas diferem entre epocas.

As 13 respostas vem do XREDUX, em Codes/XMM-NEWTON/products, que conduziu o SAS.
A leitura reaproveita o parser FITS do PULSARIS, que nao depende de astropy.

Uso:  python3 scripts/resposta_ponderada.py [--saida DIR]
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PULSARIS = Path(os.environ.get("MAGNUS_PULSARIS", Path.home() / "Codes/PULSARIS"))
sys.path.insert(0, str(PULSARIS / "scripts"))
import build_instrument_profiles as B  # noqa: E402

PRODUTOS = Path.home() / "Codes/XMM-NEWTON/products"
COMBINADO = ROOT / "exploracoes/dados_combinados_13obs_2026-09-12/combinado13.npz"
ID = "xmm_epn_imaging_combinado13"


def acha(obsid):
    for p in (PRODUTOS / obsid, *PRODUTOS.glob("*/" + obsid)):
        if (p / "src.arf").is_file() and (p / "src.rmf").is_file():
            return p
    raise SystemExit("sem resposta para %s" % obsid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(ROOT / "instrumento"))
    a = ap.parse_args()
    saida = Path(a.saida) / "instrument_data" / "profiles"
    saida.mkdir(parents=True, exist_ok=True)

    d = np.load(COMBINADO, allow_pickle=True)
    obsids = [str(x) for x in d["obsids"]]
    tempos = np.asarray(d["exposures"], float)
    T = float(tempos.sum())
    print("%d observacoes, %.1f s no total" % (len(obsids), T))

    energias = None
    area_acc = None
    # acumulador esparso: por indice de energia, canal -> soma de t*A*P
    linhas = None
    canais_bounds = {}

    for obsid, t in zip(obsids, tempos):
        p = acha(obsid)
        arf = B.read_arf(p / "src.arf")
        rmf = B.read_rmf_response(p / "src.rmf")
        E = np.array([e for e, _ in rmf])
        if energias is None:
            energias = E
            area_acc = np.zeros(len(E))
            linhas = [dict() for _ in E]
        elif not np.allclose(E, energias, rtol=0, atol=1e-6):
            raise SystemExit("%s tem grade de energia diferente" % obsid)

        A = np.array([B.interpolate(arf, e) for e in E])
        area_acc += t * A
        for k, (_, amostras) in enumerate(rmf):
            peso = t * A[k]
            if peso <= 0:
                continue
            alvo = linhas[k]
            for canal, e_min, e_max, prob in amostras:
                alvo[canal] = alvo.get(canal, 0.0) + peso * prob
                if canal not in canais_bounds:
                    canais_bounds[canal] = (e_min, e_max)
        print("  %s  t = %8.1f s  A(1 keV) = %7.1f cm2  %d canais"
              % (obsid, t, B.interpolate(arf, 1.0), len(canais_bounds)), flush=True)
        del rmf, arf

    area_ef = area_acc / T
    resposta = []
    for k, e in enumerate(energias):
        norma = T * area_ef[k]
        amostras = ([(c, *canais_bounds[c], v / norma) for c, v in sorted(linhas[k].items())]
                    if norma > 0 else [])
        resposta.append((float(e), amostras))

    area_pts = list(zip(energias.tolist(), area_ef.tolist()))
    B.write_sparse_response(saida / (ID + ".rmfbin"), area_pts, resposta)

    momentos = B.response_moments(resposta)
    passo = max(1, len(momentos) // 256)
    sel = momentos[::passo]
    if sel[-1] != momentos[-1]:
        sel.append(momentos[-1])
    with (saida / (ID + ".csv")).open("w", newline="") as f:
        f.write("# energy_keV,area_cm2,rmf_mean_keV,rmf_sigma_keV,rmf_efficiency\n")
        w = csv.writer(f)
        for e, mean, sigma, eff in sel:
            w.writerow((f"{e:.9g}", f"{B.interpolate(area_pts, e):.9g}",
                        f"{mean:.9g}", f"{sigma:.9g}", f"{eff:.9g}"))

    # manifesto proprio, com as entradas do PULSARIS mais a nova
    orig = json.loads((PULSARIS / "instrument_data/profiles/manifest.json").read_text())
    perfis = orig["profiles"]
    modelo = next(p for p in (perfis if isinstance(perfis, list) else perfis.values())
                  if p["id"] == "xmm_epn_imaging_0844140101")
    novo = dict(modelo)
    novo.update(id=ID, label="XMM-Newton / EPN coadd 13 obs",
                calibration="Soma dos operadores de resposta das 13 observacoes, "
                            "ponderada pela exposicao; ARF e RMF por observacao "
                            "gerados pelo SAS via XREDUX.",
                profile_file="instrument_data/profiles/%s.csv" % ID,
                response_file="instrument_data/profiles/%s.rmfbin" % ID,
                raw_files=[])
    if isinstance(perfis, list):
        orig["profiles"] = [p for p in perfis if p["id"] != ID] + [novo]
    else:
        orig["profiles"][ID] = novo
    (saida / "manifest.json").write_text(json.dumps(orig, indent=1) + "\n")

    # o resto do acervo tem de continuar resolvendo
    for sub in ("absorption",):
        alvo = Path(a.saida) / "instrument_data" / sub
        if not alvo.exists():
            alvo.symlink_to(PULSARIS / "instrument_data" / sub)
    for p in (PULSARIS / "instrument_data/profiles").glob("*"):
        if p.name != "manifest.json" and not (saida / p.name).exists():
            (saida / p.name).symlink_to(p)

    print()
    print("gravado %s" % (saida / (ID + ".rmfbin")))
    print("area efetiva a 1 keV: %.1f cm2" % B.interpolate(area_pts, 1.0))


if __name__ == "__main__":
    main()
