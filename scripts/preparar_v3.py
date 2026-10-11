#!/usr/bin/env python3
"""Produto preparado v3, da reconstrucao com procedencia validada.

Fonte: <XMM>/reconstrucao_5906a57, feita pela sessao
xredux-c1 com o commit 5906a57, codigo congelado por git archive e semente do
SAS fixa pelo ObsID. O indice da reconstrucao traz, por conjunto, os caminhos, o
submodo, a resolucao temporal, a exposicao, o periodo, a origem, a semente e o
sha256 do manifesto.

O QUE O v3 TRAZ, alem do que o v1 e o v2 tinham:
  - exposicao por fase POR OBSERVACAO e coadicionada, da tabela
    _phase_exposure.csv, na convencao declarada no cabecalho dela, com as duas
    receitas, por quadro e por GTI, para comparacao;
  - resolucao temporal POR OBSERVACAO, porque seis das 13 sao PrimeLargeWindow
    com 47700 us e o perfil combinado declarava 73400 para todas;
  - o fundo pelo estimador por AREA de sobreposicao canal-bin, normalizado pela
    cobertura, que e o que a verossimilhanca pede;
  - procedencia: sha256 do manifesto de cada conjunto, commit, impressao digital
    do codigo e id da imagem.

TRES DECISOES QUE NAO SAO MINHAS e que o v3 apenas registra:
  1. o limiar de surto de 0305900201 e 0402851001, que corta 52% e 18% da
     exposicao; medi que o tempo removido tem a MESMA taxa da fonte;
  2. o recentramento, porque as coordenadas do lote ficam 5,6" a 10,3" do
     centroide nas 13, perdendo 0,27% a 1,57% das contagens;
  3. o empilhamento, com 9 das 13 em excesso de duplos acima de 3 sigma.

Uso: python3 scripts/preparar_v3.py [--saida DIR]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import coadicionar as co  # noqa: E402

RECON = Path(os.environ.get(
    "MAGNUS_RECONSTRUCAO",
    Path.home() / "Codes/XMM-NEWTON/reconstrucao_5906a57"))
V1 = RAIZ / "exploracoes/dados_combinados_13obs_2026-09-12/combinado13.npz"
FASES, ENERGIAS = 32, 40
E_MIN, E_MAX = 0.15, 1.2
BORDAS = np.linspace(E_MIN, E_MAX, ENERGIAS + 1)
LARGURA = (E_MAX - E_MIN) / ENERGIAS
VERSAO = "3.0.0"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for pedaco in iter(lambda: fh.read(1 << 20), b""):
            h.update(pedaco)
    return h.hexdigest()


def cabecalho(p):
    m = {}
    with open(p, encoding="utf-8") as fh:
        for linha in fh:
            if not linha.startswith("#"):
                break
            if "=" in linha:
                k, v = linha[1:].split("=", 1)
                m[k.strip()] = v.strip()
    return m


def le_phase_exposure(p):
    """Le a tabela por fase, com o cabecalho e as quatro colunas."""
    meta, linhas = cabecalho(p), []
    nomes = None
    with open(p, encoding="utf-8") as fh:
        for linha in fh:
            if linha.startswith("#"):
                continue
            partes = [x.strip() for x in linha.strip().split(",")]
            if nomes is None:
                nomes = partes
                continue
            if partes and partes[0]:
                linhas.append([float(x) for x in partes])
    tab = {n: np.array([l[i] for l in linhas]) for i, n in enumerate(nomes)}
    return meta, tab


def fundo_por_area(dirobs, prefixo):
    """Fundo pelo estimador por area, dos espectros do conjunto.

    Reusa a receita do v2: rebin ponderado pela area de sobreposicao
    canal-bin, normalizado pela largura coberta. Aqui sobre os espectros
    RECONSTRUIDOS.
    """
    sys.path.insert(0, str(RAIZ / "scripts"))
    from exposicao_por_fase import tabela_numpy
    import build_instrument_profiles as B
    fonte = dirobs / ("%s_src_spec.fits" % prefixo)
    fundo = dirobs / ("%s_bkg_spec.fits" % prefixo)
    rmf = dirobs / ("%s_src.rmf" % prefixo)
    if not (fonte.is_file() and fundo.is_file() and rmf.is_file()):
        return None, {"erro": "espectros ou rmf ausentes", "prefixo": prefixo}
    cab_b, _ = B.extension(fundo, "SPECTRUM")
    cab_s, _ = B.extension(fonte, "SPECTRUM")
    col = tabela_numpy(fundo, "SPECTRUM", {"CHANNEL", "COUNTS"})
    escala = float(cab_s["BACKSCAL"]) / float(cab_b["BACKSCAL"])
    exposicao = float(cab_b["EXPOSURE"])
    eb = tabela_numpy(rmf, "EBOUNDS", {"CHANNEL", "E_MIN", "E_MAX"})
    lim = {int(c): (lo, hi) for c, lo, hi in
           zip(eb["CHANNEL"], eb["E_MIN"], eb["E_MAX"])}
    conta = dict(zip(col["CHANNEL"].astype(int), col["COUNTS"]))
    elo, ehi, taxa = [], [], []
    for c in sorted(lim):
        a, b = lim[c]
        if b <= a:
            continue
        meio = 0.5 * (a + b)
        if not (0.150 <= meio <= 1.200):
            continue
        elo.append(a); ehi.append(b)
        taxa.append(conta.get(c, 0.0) * escala / (exposicao * (b - a)))
    elo, ehi, taxa = np.array(elo), np.array(ehi), np.array(taxa)
    v = np.zeros(ENERGIAS)
    for k in range(ENERGIAS):
        a, b = BORDAS[k], BORDAS[k + 1]
        ov = np.clip(np.minimum(ehi, b) - np.maximum(elo, a), 0.0, None)
        if ov.sum() > 0:
            v[k] = float((taxa * ov).sum() / ov.sum())
    return v, {"backscal_fonte": float(cab_s["BACKSCAL"]),
               "backscal_fundo": float(cab_b["BACKSCAL"]),
               "escala": escala, "exposicao_do_espectro_s": exposicao,
               "canais_na_banda": int(elo.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(
        RAIZ / "exploracoes/dados_combinados_13obs_v3_2026-10-07"))
    args = ap.parse_args()
    t0 = time.time()
    indice = json.loads((RECON / "indice.json").read_text())
    ambiente = json.loads((RECON / "ambiente.json").read_text())
    conjuntos = indice["sets"]

    # --- a coadicao, pelo proprio coadicionar, sobre os CSVs novos ---------
    caminhos = [RECON / s["files"]["events"] for s in conjuntos]
    grids, exps, bgs, metas, periodos = [], [], [], [], []
    print("%-12s %-22s %8s %12s %10s" % ("obsid", "submodo", "res us",
                                         "exposicao", "eventos"))
    for s, p in zip(conjuntos, caminhos):
        g, exp, meta, per = co.fold_obs(p)
        grids.append(g); exps.append(exp); metas.append(meta); periodos.append(per)
        bgs.append(co.read_background(p))
        print("%-12s %-22s %8s %12.1f %10d"
              % (s["obsid"], s["submode"], s["time_resolution_us"], exp, g.sum()))
    ref = int(np.argmax([g.sum() for g in grids]))
    perfil_ref = grids[ref].sum(1).astype(float)
    observado = np.zeros((FASES, ENERGIAS), dtype=np.int64)
    shifts = []
    for i, g in enumerate(grids):
        sh = 0 if i == ref else co.xcorr_shift(g.sum(1).astype(float), perfil_ref)
        shifts.append(sh)
        observado += np.roll(g, sh, axis=0)
    exposicao = float(np.sum(exps))

    # --- exposicao por fase, as duas receitas, alinhada ---------------------
    por_quadro = np.zeros(FASES)
    por_gti = np.zeros(FASES)
    convencoes, faltando = [], []
    for s, sh in zip(conjuntos, shifts):
        pe = s["files"].get("phase_exposure")
        if not pe:
            faltando.append(s["obsid"]); continue
        meta, tab = le_phase_exposure(RECON / pe)
        if len(tab.get("exposure_s", [])) != FASES:
            faltando.append(s["obsid"]); continue
        por_quadro += np.roll(np.asarray(tab["exposure_s"], float), int(sh))
        if "exposure_gti_s" in tab:
            por_gti += np.roll(np.asarray(tab["exposure_gti_s"], float), int(sh))
        convencoes.append({
            "obsid": s["obsid"], "shift": int(sh),
            "phase_convention": meta.get("phase_convention"),
            "method": meta.get("method"), "recipe": meta.get("recipe"),
            "time_origin_s": meta.get("time_origin_s"),
            "period_s": meta.get("period_s"),
            "phase_reference_s": meta.get("phase_reference_s"),
            "frame_cycle_s": meta.get("frame_cycle_s"),
            "frame_live_vs_livetime_relative":
                meta.get("frame_live_vs_livetime_relative"),
            "gti_vs_ontime_relative": meta.get("gti_vs_ontime_relative")})

    # --- fundo pelo estimador por area, dos espectros reconstruidos --------
    fundo_area = np.zeros(ENERGIAS)
    fundo_info = []
    T = exposicao
    for s, exp in zip(conjuntos, exps):
        ev = RECON / s["files"]["events"]
        # Os CSVs levam o prefixo da FONTE, RXJ1308.6_2127_<obsid>_epn_sNNN,
        # mas os espectros e as respostas ficam um nivel acima com o prefixo
        # CURTO, epn_sNNN. Descobri isso olhando o diretorio, nao supondo.
        longo = ev.name.replace("_events.csv", "")
        curto = longo.split("_")[-2] + "_" + longo.split("_")[-1] \
            if longo.count("_") >= 2 else longo
        v, info = fundo_por_area(ev.parent.parent, curto)
        if v is None:
            v, info = fundo_por_area(ev.parent.parent, longo)
        info["obsid"] = s["obsid"]
        if v is None:
            fundo_info.append(info); continue
        fundo_area += (exp / T) * v
        fundo_info.append(info)

    destino = Path(args.saida)
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / "combinado13_v3.npz"
    np.savez(
        arquivo,
        observed=observado.ravel(),
        exposure=exposicao,
        background=(fundo_area if fundo_area.any()
                    else np.average(np.array(bgs), axis=0, weights=np.array(exps))),
        background_interp=np.average(np.array(bgs), axis=0,
                                     weights=np.array(exps)),
        shifts=np.array(shifts),
        exposures=np.array(exps),
        obsids=np.array([s["obsid"] for s in conjuntos]),
        periods=np.array(periodos),
        versao=np.array(VERSAO),
        # o que o v3 acrescenta
        exposicao_por_fase=por_quadro,
        exposicao_por_fase_gti=por_gti,
        resolucao_temporal_us=np.array(
            [float(s["time_resolution_us"]) for s in conjuntos]),
        submodos=np.array([s["submode"] for s in conjuntos]),
        sementes=np.array([s.get("sas_rand_seed", -1) for s in conjuntos]),
    )

    v1 = np.load(V1, allow_pickle=True)
    proc = {
        "versao": VERSAO, "data": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arquivo": str(arquivo), "sha256": sha256(arquivo),
        "fonte": {
            "reconstrucao": str(RECON), "commit": ambiente.get("commit"),
            "sha256_do_codigo": ambiente.get("source_sha256",
                                             ambiente.get("xredux_source_sha256")),
            "imagem": ambiente.get("image", ambiente.get("imagem")),
            "feita_por": "a sessao xredux-c1, com codigo congelado por git archive"},
        "conjuntos": [{"obsid": s["obsid"], "set_id": s["set_id"],
                       "manifest_sha256": s.get("manifest_sha256"),
                       "sas_rand_seed": s.get("sas_rand_seed"),
                       "submode": s["submode"],
                       "time_resolution_us": s["time_resolution_us"],
                       "exposure_s": s["exposure_s"],
                       "period_s": s["period_s"],
                       "time_origin_s": s["time_origin_s"],
                       "pileup": s.get("pileup")} for s in conjuntos],
        "contra_o_v1": {
            "contagens": {"v1": int(np.asarray(v1["observed"]).sum()),
                          "v3": int(observado.sum())},
            "exposicao": {"v1": float(v1["exposure"]), "v3": exposicao},
            "variacao_de_contagens": float(observado.sum()
                                           / np.asarray(v1["observed"]).sum() - 1),
            "variacao_de_exposicao": float(exposicao / float(v1["exposure"]) - 1)},
        "exposicao_por_fase": {
            "convencao_por_observacao": convencoes,
            "faltando": faltando,
            "amplitude_por_quadro": float((por_quadro.max() - por_quadro.min())
                                          / por_quadro.mean()) if por_quadro.any() else None,
            "amplitude_por_gti": float((por_gti.max() - por_gti.min())
                                       / por_gti.mean()) if por_gti.any() else None},
        "fundo": {"estimador": "rebin por area de sobreposicao canal-bin, "
                               "normalizado pela cobertura",
                  "por_observacao": fundo_info,
                  "soma_v3": float(fundo_area.sum()),
                  "soma_interp_v3": float(np.average(np.array(bgs), axis=0,
                                                     weights=np.array(exps)).sum())},
        "decisoes_que_nao_sao_minhas": [
            "o limiar de surto de 0305900201 e 0402851001, que corta 52 e 18 por "
            "cento da exposicao",
            "o recentramento: as coordenadas do lote ficam 5,6 a 10,3 arcsec do "
            "centroide nas 13",
            "o empilhamento: 9 das 13 com excesso de duplos acima de 3 sigma"],
    }
    (destino / "proveniencia.json").write_text(
        json.dumps(proc, indent=2, ensure_ascii=False) + "\n")
    print()
    print("v3: %d contagens em %.1f s (v1: %d em %.1f s)"
          % (observado.sum(), exposicao, np.asarray(v1["observed"]).sum(),
             float(v1["exposure"])))
    print("  contagens %.2f%%, exposicao %.2f%%"
          % (100 * proc["contra_o_v1"]["variacao_de_contagens"],
             100 * proc["contra_o_v1"]["variacao_de_exposicao"]))
    if por_quadro.any():
        print("  exposicao por fase, amplitude: por quadro %.4f%%, por GTI %.4f%%"
              % (100 * proc["exposicao_por_fase"]["amplitude_por_quadro"],
                 100 * (proc["exposicao_por_fase"]["amplitude_por_gti"] or 0)))
    print("  fundo somado: area %.6f, interpolacao %.6f"
          % (fundo_area.sum(), proc["fundo"]["soma_interp_v3"]))
    print("  resolucoes: %s" % sorted({float(s["time_resolution_us"])
                                       for s in conjuntos}))
    print("gravado %s" % arquivo)
    print("%.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
