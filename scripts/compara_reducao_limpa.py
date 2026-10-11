#!/usr/bin/env python3
"""Compara a reducao limpa de 0402850301 com o produto antigo, por componente.

PEDIDO DO AUDITOR: "a comparacao final deve separar mudancas em eventos,
exposicao, fase, padroes e respostas". E: "registre commit do codigo montado,
identificacao da imagem e comando efetivo. Caso contrario, uma reducao
aparentemente nova poderia validar o codigo antigo."

A PROCEDENCIA DO QUE RODOU esta embutida abaixo, porque o conteiner NAO monta o
repositorio: `docker/run.sh` monta so external/ccf, o diretorio de trabalho e o
PULSARIS, de modo que pelo caminho normal ele rodaria o codigo embutido na
imagem, anterior as correcoes. A reducao foi feita com o codigo corrigido
montado por cima.

Uso: python3 scripts/compara_reducao_limpa.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from exposicao_por_fase import cabecalho_csv, tabela_numpy  # noqa: E402
import build_instrument_profiles as B  # noqa: E402

XMM = Path.home() / "Codes/XMM-NEWTON"
ANTIGO = XMM / "products/0402850301"
LIMPO = XMM / "products_limpo_0402850301/0402850301"
PROCEDENCIA = {
    "imagem": "xredux:22.1.0",
    "imagem_sha256": ("sha256:6eca925f5ae196392a330cdf425513a64695f18ab392e4a"
                      "173571602e80c1c59"),
    "codigo_montado_commit": "b27878b7e710424df71fa24c72110cb0520806c4",
    "codigo_montado_data": "2026-10-07 10:20:18 -0300",
    "codigo_montado_assunto": "Corrige os 13 achados da auditoria de codigo do XREDUX",
    "por_que_montado": (
        "docker/run.sh monta so external/ccf, o diretorio de trabalho e o "
        "PULSARIS; o conteiner NAO monta o repositorio, logo pelo caminho normal "
        "rodaria o codigo embutido na imagem, construida antes das correcoes"),
    "comando_efetivo": (
        "docker run --rm --init --user $(id -u):$(id -g) "
        "-v $ROOT/external/ccf:/ccf:ro -v $LIMPO:/work "
        "-v $HOME/Codes/PULSARIS:/pulsaris "
        "-v $ROOT/xredux:/opt/xredux/xredux:ro -v $ROOT/tools:/opt/xredux/tools:ro "
        "-e XREDUX_THREADS=$(nproc) xredux:22.1.0 reduce --obsid 0402850301 "
        "--target 'RX J1308.6+2127' --ra 197.20292 --dec 21.45222 "
        "--period 10.3131714585 --band 150 1200 --export"),
    "prova_de_que_o_codigo_novo_rodou": (
        "os produtos sairam com a nomenclatura corrigida epn_s003_*, que e a "
        "correcao XR-03; o produto antigo usa epn_*"),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for pedaco in iter(lambda: fh.read(1 << 20), b""):
            h.update(pedaco)
    return h.hexdigest()


def acha(base, *nomes):
    for n in nomes:
        p = base / n
        if p.is_file():
            return p
    return None


def campos(caminho, ext="EVENTS"):
    for h, _ in B.read_hdus(caminho):
        if str(h.get("EXTNAME", "")).strip() == ext:
            return {k: h.get(k) for k in
                    ("EXPOSURE", "ONTIME", "LIVETIME", "TSTART", "TSTOP",
                     "TIMEREF", "TIMESYS", "NAXIS2", "SUBMODE", "BACKSCAL")
                    if h.get(k) is not None}
    return {}


def main():
    t0 = time.time()
    relato = {"o_que_e": "comparacao da reducao limpa de 0402850301 com o produto antigo",
              "procedencia": PROCEDENCIA, "componentes": {}}
    print("=== procedencia do que rodou ===")
    for k in ("imagem", "imagem_sha256", "codigo_montado_commit"):
        print("  %-24s %s" % (k, PROCEDENCIA[k]))
    print("  %-24s %s" % ("prova", PROCEDENCIA["prova_de_que_o_codigo_novo_rodou"]))

    # --- 1. eventos --------------------------------------------------------
    fa = acha(ANTIGO, "epn_bary_source.fits")
    fl = acha(LIMPO, "epn_s003_bary_source.fits", "epn_bary_source.fits")
    ca, cl = campos(fa), campos(fl)
    pa = tabela_numpy(fa, "EVENTS", {"PI", "PATTERN", "TIME"})
    pl = tabela_numpy(fl, "EVENTS", {"PI", "PATTERN", "TIME"})
    ba = (pa["PI"] >= 150) & (pa["PI"] < 1200)
    bl = (pl["PI"] >= 150) & (pl["PI"] < 1200)
    print()
    print("=== 1. EVENTOS ===")
    print("  antigo %d na lista, %d na banda | limpo %d na lista, %d na banda"
          % (pa["PI"].size, int(ba.sum()), pl["PI"].size, int(bl.sum())))
    print("  diferenca na banda: %+d (%.4f%%)"
          % (int(bl.sum()) - int(ba.sum()),
             100.0 * (int(bl.sum()) / int(ba.sum()) - 1.0)))
    relato["componentes"]["eventos"] = {
        "antigo_lista": int(pa["PI"].size), "limpo_lista": int(pl["PI"].size),
        "antigo_banda": int(ba.sum()), "limpo_banda": int(bl.sum()),
        "diferenca_na_banda": int(bl.sum()) - int(ba.sum()),
        "diferenca_relativa": float(int(bl.sum()) / int(ba.sum()) - 1.0)}

    # --- 2. exposicao ------------------------------------------------------
    print()
    print("=== 2. EXPOSICAO ===")
    for k in ("LIVETIME", "ONTIME", "TSTART", "TSTOP"):
        va, vl = ca.get(k), cl.get(k)
        if va is None or vl is None:
            continue
        print("  %-10s antigo %18.6f | limpo %18.6f | dif %+.6f"
              % (k, va, vl, vl - va))
    relato["componentes"]["exposicao"] = {"antigo": ca, "limpo": cl}

    # --- 3. fase -----------------------------------------------------------
    csva = sorted((ANTIGO / "pulsaris").glob("*_events.csv"))
    csvl = sorted((LIMPO / "pulsaris").glob("*_events.csv"))
    ma = cabecalho_csv(csva[0]) if csva else {}
    ml = cabecalho_csv(csvl[0]) if csvl else {}
    print()
    print("=== 3. FASE ===")
    for k in ("period_s", "phase_reference_s", "time_origin_mjd_s",
              "exposure_s", "detected_events", "time_resolution_us",
              "energy_min_keV", "energy_max_keV"):
        print("  %-20s antigo %-22s limpo %-22s %s"
              % (k, ma.get(k), ml.get(k),
                 "IGUAL" if ma.get(k) == ml.get(k) else "DIFERE"))
    # a origem antiga desviava 200,08 s do TSTART; a corrigida deveria ser o TSTART
    for rotulo, meta, c in (("antigo", ma, ca), ("limpo", ml, cl)):
        if "time_origin_mjd_s" in meta and "TSTART" in c:
            print("  origem %s menos TSTART dos eventos: %+.6f s"
                  % (rotulo, float(meta["time_origin_mjd_s"]) - float(c["TSTART"])))
    relato["componentes"]["fase"] = {
        "antigo": {k: ma.get(k) for k in ma if not k.startswith("PULSARIS")},
        "limpo": {k: ml.get(k) for k in ml if not k.startswith("PULSARIS")},
        "origem_menos_tstart_antigo":
            float(ma["time_origin_mjd_s"]) - float(ca["TSTART"])
            if "time_origin_mjd_s" in ma and "TSTART" in ca else None,
        "origem_menos_tstart_limpo":
            float(ml["time_origin_mjd_s"]) - float(cl["TSTART"])
            if "time_origin_mjd_s" in ml and "TSTART" in cl else None}

    # --- 4. padroes --------------------------------------------------------
    print()
    print("=== 4. PADROES, na banda ===")
    ha = np.bincount(pa["PATTERN"][ba].astype(int), minlength=13)[:13]
    hl = np.bincount(pl["PATTERN"][bl].astype(int), minlength=13)[:13]
    print("  antigo:", ha.tolist())
    print("  limpo :", hl.tolist())
    print("  PATTERN maximo: antigo %d, limpo %d  (o corte do pipeline e <= 4)"
          % (int(pa["PATTERN"].max()), int(pl["PATTERN"].max())))
    relato["componentes"]["padroes"] = {
        "antigo": ha.tolist(), "limpo": hl.tolist(),
        "maximo_antigo": int(pa["PATTERN"].max()),
        "maximo_limpo": int(pl["PATTERN"].max()),
        "nota": ("o maximo 4 nas duas listas mostra que triplos e quadruplos "
                 "foram ELIMINADOS pelo corte de padrao, e nao estao ausentes "
                 "fisicamente: a minha leitura anterior estava errada")}

    # --- 5. respostas ------------------------------------------------------
    print()
    print("=== 5. RESPOSTAS ===")
    for nome_a, nome_l, papel in (("src.rmf", "epn_s003_src.rmf", "RMF"),
                                  ("src.arf", "epn_s003_src.arf", "ARF"),
                                  ("src_spec.fits", "epn_s003_src_spec.fits",
                                   "espectro da fonte"),
                                  ("bkg_spec.fits", "epn_s003_bkg_spec.fits",
                                   "espectro do fundo")):
        a, l = acha(ANTIGO, nome_a), acha(LIMPO, nome_l, nome_a)
        if a is None or l is None:
            print("  %-18s antigo %s limpo %s" % (papel, bool(a), bool(l)))
            continue
        iguais = sha256(a) == sha256(l)
        extra = ""
        if papel.startswith("espectro"):
            fa_, fl_ = campos(a, "SPECTRUM"), campos(l, "SPECTRUM")
            extra = ("BACKSCAL %s -> %s, EXPOSURE %s -> %s"
                     % (fa_.get("BACKSCAL"), fl_.get("BACKSCAL"),
                        fa_.get("EXPOSURE"), fl_.get("EXPOSURE")))
        print("  %-18s sha256 %s  %s" % (papel, "IGUAL" if iguais else "DIFERE",
                                         extra))
        relato["componentes"].setdefault("respostas", {})[papel] = {
            "antigo": str(a), "limpo": str(l),
            "sha256_antigo": sha256(a), "sha256_limpo": sha256(l),
            "identicos": iguais}
        if papel.startswith("espectro"):
            relato["componentes"]["respostas"][papel].update(
                {"campos_antigo": campos(a, "SPECTRUM"),
                 "campos_limpo": campos(l, "SPECTRUM")})

    relato["segundos"] = time.time() - t0
    destino = RAIZ / "docs/compara_reducao_limpa.json"
    destino.write_text(json.dumps(relato, indent=2, ensure_ascii=False,
                                  default=str) + "\n")
    print()
    print("gravado em %s" % destino)


if __name__ == "__main__":
    main()
