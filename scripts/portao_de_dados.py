#!/usr/bin/env python3
"""Portao de dados: congela os produtos atuais e audita as 13 entradas efetivas.

PEDIDO DO AUDITOR, em 07/10/2026, depois da auditoria do XREDUX. Ele separou as
duas coisas, e a separacao importa: o erro de P1 aparece num controle ANALITICO,
sem dado observado, logo e independente destes achados. Podemos ter os dois
problemas ao mesmo tempo.

O que este script faz, que sao os passos 1 e 2 do portao:

  1. CONGELA os produtos atuais, com sha256 e o vinculo de cada observacao a
     eventos, regiao, GTIs, exposicao, RMF, ARF e fundo;
  2. AUDITA as 13 entradas efetivas: decimacao, camera e exposicao, correcao
     baricentrica, origem temporal, selecao de energia, cobertura de fase e
     medidas de pile-up.

O ponto que o auditor marcou como imediato: `coadicionar.py` le a exposicao e a
referencia de fase DOS CSVS, de modo que metadado antigo incorreto se propaga
mesmo com a coadicao funcionando exatamente como escrita. Entao a auditoria
compara CADA metadado do CSV contra o cabecalho FITS de onde ele deveria ter
saido.

O QUE ELE NAO FAZ: nao roda reducao nenhuma, nao altera nada em
~/Codes/XMM-NEWTON, e nao certifica retrospectivamente produto algum. Os passos 3
e 4 do portao, reducao completa em diretorio limpo e reconstrucao das 13 com
procedencia validada, pertencem ao XREDUX e nao a este repositorio.

Uso: python3 scripts/portao_de_dados.py [--saida docs/portao_de_dados.json]
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
PULSARIS = Path(os.environ.get("MAGNUS_PULSARIS", Path.home() / "Codes/PULSARIS"))
sys.path.insert(0, str(PULSARIS / "scripts"))
import build_instrument_profiles as B  # noqa: E402

PRODUTOS = Path(os.environ.get("MAGNUS_PRODUTOS",
                               Path.home() / "Codes/XMM-NEWTON/products"))
NPZ = RAIZ / "exploracoes/dados_combinados_13obs_2026-09-12/combinado13.npz"
NPZ_V2 = RAIZ / "exploracoes/dados_combinados_13obs_v2_2026-10-06/combinado13_v2.npz"
#: Chaves do CSV que o coadicionar LE e que portanto propagam metadado antigo.
CHAVES_CONSUMIDAS = ("period_s", "phase_reference_s", "exposure_s")
FASES = 32


def sha256(p, limite=None):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        lido = 0
        while True:
            pedaco = fh.read(1 << 20)
            if not pedaco:
                break
            h.update(pedaco)
            lido += len(pedaco)
            if limite and lido >= limite:
                break
    return h.hexdigest()


def cabecalho_csv(caminho):
    meta, linhas = {}, 0
    with open(caminho, encoding="utf-8") as fh:
        for linha in fh:
            if linha.startswith("#"):
                if "=" in linha:
                    k, v = linha[1:].split("=", 1)
                    meta[k.strip()] = v.strip()
                continue
            linhas += 1
    return meta, linhas - 1   # menos a linha de nomes de coluna


def fits_campos(caminho, extensao=None):
    saida = {}
    for h, _ in B.read_hdus(caminho):
        nome = str(h.get("EXTNAME", "")).strip()
        if extensao and nome != extensao:
            continue
        for k in ("EXPOSURE", "ONTIME", "LIVETIME", "TSTART", "TSTOP", "TIMEREF",
                  "TIMESYS", "INSTRUME", "SUBMODE", "DATAMODE", "FILTER",
                  "NAXIS2", "BACKSCAL"):
            if h.get(k) is not None and k not in saida:
                saida[k] = h.get(k)
        if nome == "EVENTS":
            saida["eventos_NAXIS2"] = h.get("NAXIS2")
    return saida


def acha_obs(obsid):
    for cand in (PRODUTOS / obsid, *PRODUTOS.glob("*/" + obsid)):
        if cand.is_dir() and (cand / "session.json").is_file():
            return cand
    for cand in (PRODUTOS / obsid, *PRODUTOS.glob("*/" + obsid)):
        if cand.is_dir():
            return cand
    return None


def acha_eventos_csv(dirobs):
    p = sorted(dirobs.glob("pulsaris/*_events.csv"))
    return p[0] if p else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(RAIZ / "docs/portao_de_dados.json"))
    args = ap.parse_args()

    d = np.load(NPZ, allow_pickle=True)
    obsids = [str(x) for x in d["obsids"]]
    expos = np.asarray(d["exposures"], float)
    periodos = np.asarray(d["periods"], float)
    shifts = np.asarray(d["shifts"], int)
    observado = np.asarray(d["observed"], float).reshape(FASES, 40)

    print("=== as 13 entradas efetivas ===")
    print("%-12s %12s %12s %10s %9s %9s %s"
          % ("obsid", "exp CSV", "LIVETIME", "dif rel", "eventos", "NAXIS2",
             "decimado?"))
    registros, achados = [], []
    for obsid, exp_npz, per_npz, shift in zip(obsids, expos, periodos, shifts):
        dirobs = acha_obs(obsid)
        if dirobs is None:
            achados.append({"obsid": obsid, "achado": "diretorio nao encontrado"})
            continue
        csv = acha_eventos_csv(dirobs)
        if csv is None:
            achados.append({"obsid": obsid, "achado": "CSV de eventos ausente"})
            continue
        meta, linhas = cabecalho_csv(csv)
        fonte = dirobs / "epn_bary_source.fits"
        limpo = dirobs / "epn_clean_bary.fits"
        cf = fits_campos(fonte) if fonte.is_file() else {}
        cl = fits_campos(limpo) if limpo.is_file() else {}

        exp_csv = float(meta.get("exposure_s", "nan"))
        live = float(cf.get("LIVETIME", cl.get("LIVETIME", float("nan"))))
        dif_exp = (exp_csv / live - 1.0) if live == live and live else float("nan")
        declarados = int(meta.get("detected_events", -1))
        naxis = int(cf.get("eventos_NAXIS2", -1))
        # DECIMACAO: o achado XR-04 diz que decimar mantinha a exposicao integral
        # e omitia o metadado de amostragem. O sinal observavel e o numero de
        # linhas ficar ABAIXO do NAXIS2 da lista de eventos da fonte, sem que
        # nenhuma chave de amostragem apareca.
        chaves_amostragem = [k for k in meta
                             if any(t in k.lower() for t in
                                    ("thin", "decim", "sample", "fraction",
                                     "effective", "seed"))]
        decimado = (naxis > 0 and linhas > 0 and linhas < naxis - 1)
        print("%-12s %12.3f %12.3f %10.2e %9d %9d %s"
              % (obsid, exp_csv, live, dif_exp, linhas, naxis,
                 "SIM" if decimado else "nao"))
        reg = {
            "obsid": obsid,
            "diretorio": str(dirobs),
            "eventos_csv": {"caminho": str(csv), "sha256": sha256(csv),
                            "bytes": csv.stat().st_size, "linhas_de_dados": linhas},
            "metadados_consumidos_pelo_coadd": {
                k: meta.get(k) for k in CHAVES_CONSUMIDAS},
            "do_csv": {k: meta.get(k) for k in
                       ("instrument", "obsid", "detected_events",
                        "time_origin_mjd_s", "barycentric", "energy_min_keV",
                        "energy_max_keV", "region", "response_rmf",
                        "time_resolution_us", "dead_time_us", "folded_in_phase")},
            "do_fits_da_fonte": cf, "do_fits_limpo": cl,
            "conferencias": {
                "exposicao_csv_contra_LIVETIME": dif_exp,
                "eventos_declarados_contra_linhas": declarados - linhas,
                "linhas_contra_NAXIS2_da_fonte": linhas - naxis,
                "decimacao_aparente": bool(decimado),
                "chaves_de_amostragem_no_csv": chaves_amostragem,
                "baricentrico_no_csv": meta.get("barycentric"),
                "TIMEREF_no_fits": cf.get("TIMEREF", cl.get("TIMEREF")),
                "TIMESYS_no_fits": cl.get("TIMESYS"),
                "origem_do_csv_menos_TSTART_dos_eventos":
                    (float(meta["time_origin_mjd_s"]) - float(cf["TSTART"]))
                    if ("time_origin_mjd_s" in meta and "TSTART" in cf) else None,
                "exposicao_no_npz": float(exp_npz),
                "exposicao_npz_contra_csv": float(exp_npz) / exp_csv - 1.0
                    if exp_csv == exp_csv and exp_csv else None,
                "periodo_no_npz_contra_csv":
                    float(per_npz) / float(meta["period_s"]) - 1.0
                    if "period_s" in meta else None,
                "shift_de_fase_aplicado": int(shift),
            },
        }
        for extra, papel in (("src_spec.fits", "espectro da fonte"),
                             ("bkg_spec.fits", "espectro do fundo"),
                             ("src.rmf", "RMF"), ("src.arf", "ARF"),
                             ("epn_gti.fits", "GTI")):
            p = dirobs / extra
            if p.is_file():
                reg.setdefault("produtos", {})[extra] = {
                    "papel": papel, "sha256": sha256(p),
                    "bytes": p.stat().st_size,
                    "campos": fits_campos(p)}
        registros.append(reg)

    print()
    print("=== o que as conferencias acusam ===")
    def acusa(cond, texto, detalhe=None):
        if cond:
            achados.append({"achado": texto, "detalhe": detalhe})
            print("  ACHADO: %s" % texto)

    decimados = [r["obsid"] for r in registros
                 if r["conferencias"]["decimacao_aparente"]]
    acusa(bool(decimados), "decimacao aparente em %s" % decimados, decimados)
    exp_fora = [(r["obsid"], r["conferencias"]["exposicao_csv_contra_LIVETIME"])
                for r in registros
                if abs(r["conferencias"]["exposicao_csv_contra_LIVETIME"] or 0) > 1e-9]
    acusa(bool(exp_fora), "exposicao do CSV diverge do LIVETIME em %s" % exp_fora,
          exp_fora)
    sem_bary = [r["obsid"] for r in registros
                if r["conferencias"]["baricentrico_no_csv"] != "SOLARSYSTEM"
                or r["conferencias"]["TIMEREF_no_fits"] != "SOLARSYSTEM"]
    acusa(bool(sem_bary), "correcao baricentrica nao confirmada em %s" % sem_bary,
          sem_bary)
    origens = [(r["obsid"],
                r["conferencias"]["origem_do_csv_menos_TSTART_dos_eventos"])
               for r in registros]
    fora_origem = [(o, v) for o, v in origens if v is not None and abs(v) > 1e-6]
    acusa(bool(fora_origem),
          "a origem temporal do CSV nao e o TSTART dos eventos em %d de %d "
          "observacoes" % (len(fora_origem), len(origens)), fora_origem)
    bandas = sorted({(r["do_csv"]["energy_min_keV"], r["do_csv"]["energy_max_keV"])
                     for r in registros})
    acusa(len(bandas) > 1,
          "as 13 nao foram exportadas na MESMA banda: %s" % bandas, bandas)
    npz_fora = [(r["obsid"], r["conferencias"]["exposicao_npz_contra_csv"])
                for r in registros
                if abs(r["conferencias"]["exposicao_npz_contra_csv"] or 0) > 1e-12]
    acusa(bool(npz_fora), "exposicao do npz diverge do CSV em %s" % npz_fora,
          npz_fora)

    # pile-up: existe medida na sessao?
    pileup = {}
    for r in registros:
        s = Path(r["diretorio"]) / "session.json"
        if not s.is_file():
            pileup[r["obsid"]] = "sem session.json"
            continue
        texto = s.read_text(errors="replace")
        pileup[r["obsid"]] = ("mencao a pile-up" if "pile" in texto.lower()
                              else "SEM mencao a pile-up")
    sem_pile = [o for o, v in pileup.items() if v != "mencao a pile-up"]
    acusa(bool(sem_pile), "sem medida de pile-up registrada em %s" % sem_pile,
          sem_pile)

    # cobertura de fase: bins vazios no observado
    vazios = int((observado.sum(axis=1) == 0).sum())
    acusa(vazios > 0, "%d dos %d bins de fase estao vazios" % (vazios, FASES))
    print("  cobertura de fase: %d de %d bins com contagem; minimo %d, maximo %d"
          % (FASES - vazios, FASES, int(observado.sum(axis=1).min()),
             int(observado.sum(axis=1).max())))

    saida = {
        "o_que_e": "portao de dados: congelamento e auditoria das 13 entradas efetivas",
        "data": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pedido_por": "auditoria do XREDUX e recomendacao do auditor em 07/10/2026",
        "separacao_declarada": (
            "o erro de P1 aparece num controle ANALITICO, sem dado observado, "
            "logo e independente destes achados; podemos ter os dois problemas"),
        "o_que_este_script_NAO_faz": [
            "nao roda reducao nenhuma",
            "nao altera nada em ~/Codes/XMM-NEWTON",
            "nao certifica retrospectivamente produto algum",
            "os passos 3 e 4 do portao, reducao completa em diretorio limpo e "
            "reconstrucao das 13 com procedencia validada, pertencem ao XREDUX"],
        "congelamento": {
            "npz_v1": {"caminho": str(NPZ), "sha256": sha256(NPZ)},
            "npz_v2": ({"caminho": str(NPZ_V2), "sha256": sha256(NPZ_V2)}
                       if NPZ_V2.is_file() else None),
            "observacoes": registros},
        "achados": achados,
        "pile_up": pileup,
        "cobertura_de_fase": {"bins": FASES, "vazios": vazios,
                              "minimo": int(observado.sum(axis=1).min()),
                              "maximo": int(observado.sum(axis=1).max())},
    }
    Path(args.saida).write_text(json.dumps(saida, indent=2, ensure_ascii=False,
                                           default=str) + "\n")
    print()
    print("%d achados | gravado em %s" % (len(achados), args.saida))


if __name__ == "__main__":
    main()
