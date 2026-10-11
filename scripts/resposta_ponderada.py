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


def acha(obsid, origem=None):
    """O diretorio com src.arf e src.rmf daquela observacao.

    Dois layouts. O antigo, em Codes/XMM-NEWTON/products, tem os arquivos com
    nome fixo `src.arf` e `src.rmf`. O da RECONSTRUCAO tem prefixo CURTO,
    `epn_sNNN_src.arf`, um nivel acima dos CSVs do pulsaris. Devolve
    (diretorio, prefixo), com prefixo vazio no layout antigo.
    """
    base = Path(origem) if origem else PRODUTOS
    for p in (base / obsid, *base.glob("*/" + obsid), *base.glob("*/*/" + obsid)):
        if (p / "src.arf").is_file() and (p / "src.rmf").is_file():
            return p, ""
        achados = sorted(p.glob("*_src.arf"))
        for arf in achados:
            prefixo = arf.name[:-len("_src.arf")]
            if (p / (prefixo + "_src.rmf")).is_file():
                return p, prefixo
    raise SystemExit("sem resposta para %s em %s" % (obsid, base))


def arquivos(dirobs, prefixo):
    pre = (prefixo + "_") if prefixo else ""
    return dirobs / (pre + "src.arf"), dirobs / (pre + "src.rmf")


def sha256(caminho):
    import hashlib
    h = hashlib.sha256()
    with open(caminho, "rb") as fh:
        for pedaco in iter(lambda: fh.read(1 << 20), b""):
            h.update(pedaco)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(ROOT / "instrumento"))
    ap.add_argument("--origem", default=None,
                    help="raiz das respostas; sem ela, Codes/XMM-NEWTON/products")
    ap.add_argument("--combinado", default=str(COMBINADO),
                    help="npz coadicionado de onde vem obsids e exposicoes")
    ap.add_argument("--id", dest="ident", default=ID)
    ap.add_argument("--procedencia", default=None,
                    help="rotulo da procedencia, por exemplo o commit da reducao")
    a = ap.parse_args()
    ident = a.ident
    COMB = Path(a.combinado)
    saida = Path(a.saida) / "instrument_data" / "profiles"
    saida.mkdir(parents=True, exist_ok=True)

    d = np.load(COMB, allow_pickle=True)
    obsids = [str(x) for x in d["obsids"]]
    tempos = np.asarray(d["exposures"], float)
    T = float(tempos.sum())
    print("%d observacoes, %.1f s no total" % (len(obsids), T))

    energias = None
    area_acc = None
    # acumulador esparso: por indice de energia, canal -> soma de t*A*P
    linhas = None
    canais_bounds = {}

    proveniencia = []
    for obsid, t in zip(obsids, tempos):
        p, prefixo = acha(obsid, a.origem)
        f_arf, f_rmf = arquivos(p, prefixo)
        proveniencia.append({"obsid": obsid, "exposicao": float(t),
                             "arf": str(f_arf), "rmf": str(f_rmf),
                             "sha256_arf": sha256(f_arf),
                             "sha256_rmf": sha256(f_rmf)})
        arf = B.read_arf(f_arf)
        rmf = B.read_rmf_response(f_rmf)
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
    B.write_sparse_response(saida / (ident + ".rmfbin"), area_pts, resposta)

    momentos = B.response_moments(resposta)
    passo = max(1, len(momentos) // 256)
    sel = momentos[::passo]
    if sel[-1] != momentos[-1]:
        sel.append(momentos[-1])
    with (saida / (ident + ".csv")).open("w", newline="") as f:
        f.write("# energy_keV,area_cm2,rmf_mean_keV,rmf_sigma_keV,rmf_efficiency\n")
        w = csv.writer(f)
        for e, mean, sigma, eff in sel:
            w.writerow((f"{e:.9g}", f"{B.interpolate(area_pts, e):.9g}",
                        f"{mean:.9g}", f"{sigma:.9g}", f"{eff:.9g}"))

    # manifesto proprio. SE JA EXISTIR UM LOCAL, ELE E A BASE: comecar do
    # PULSARIS apagaria as entradas por observacao registradas aqui, que o
    # PULSARIS nao tem. Isto quase me custou os 13 perfis da reconstrucao.
    local = saida / "manifest.json"
    fonte = local if local.is_file() else (
        PULSARIS / "instrument_data/profiles/manifest.json")
    orig = json.loads(fonte.read_text())
    perfis = orig["profiles"]
    modelo = next(p for p in (perfis if isinstance(perfis, list) else perfis.values())
                  if p["id"] == "xmm_epn_imaging_0844140101")
    novo = dict(modelo)
    novo.update(id=ident, label="XMM-Newton / EPN coadd 13 obs",
                calibration="Soma dos operadores de resposta das 13 observacoes, "
                            "ponderada pela exposicao; ARF e RMF por observacao "
                            "gerados pelo SAS via XREDUX.",
                profile_file=str(Path(a.saida).relative_to(ROOT)
                                 / "instrument_data/profiles/%s.csv" % ident)
                if Path(a.saida).is_relative_to(ROOT) else
                "instrument_data/profiles/%s.csv" % ident,
                response_file=str(Path(a.saida).relative_to(ROOT)
                                  / "instrument_data/profiles/%s.rmfbin" % ident)
                if Path(a.saida).is_relative_to(ROOT) else
                "instrument_data/profiles/%s.rmfbin" % ident,
                raw_files=[])
    # As resolucoes temporais que entraram, e a escolha DECLARADA. O revisor:
    # "a mistura de quadros deve ser representada pela combinacao dos operadores
    # de suavizacao de cada observacao. Usar o tempo medio ponderado nao e, em
    # geral, equivalente." Logo nao existe uma resolucao correta para uma
    # coadicao de duas; a ponderada entra como CONVENCAO da formulacao
    # coadicionada, e o custo dela foi medido: residuo de forma de 3,34e-02
    # entre os dois grupos de resolucao, em docs/reuso_por_observacao.json.
    resol = ([float(x) for x in d["resolucao_temporal_us"]]
             if "resolucao_temporal_us" in d.files else None)
    if resol:
        media = float(np.average(resol, weights=tempos))
        novo["time_resolution_us"] = media
        novo["resolucao_e_uma_CONVENCAO"] = {
            "valores_que_entraram": sorted(set(resol)),
            "ponderada_pela_exposicao": media,
            "por_que_isto_nao_e_equivalente": (
                "a mistura de quadros tem de ser a combinacao dos operadores de "
                "suavizacao de cada observacao, e o tempo medio ponderado nao e "
                "equivalente a isso"),
            "custo_medido": ("residuo de forma de 3,34e-02 entre os dois grupos "
                             "de resolucao, em docs/reuso_por_observacao.json")}
    novo["procedencia"] = {
        "rotulo": a.procedencia, "combinado": str(COMB),
        "origem_das_respostas": a.origem or str(PRODUTOS),
        "exposicao_total_s": T, "observacoes": proveniencia}
    if isinstance(perfis, list):
        orig["profiles"] = [p for p in perfis if p["id"] != ident] + [novo]
    else:
        orig["profiles"][ident] = novo
    if local.is_file():
        import shutil
        shutil.copy2(local, local.with_suffix(".json.antes_" + ident))
    local.write_text(json.dumps(orig, indent=1) + "\n")

    # o resto do acervo tem de continuar resolvendo
    for sub in ("absorption",):
        alvo = Path(a.saida) / "instrument_data" / sub
        if not alvo.exists():
            alvo.symlink_to(PULSARIS / "instrument_data" / sub)
    for p in (PULSARIS / "instrument_data/profiles").glob("*"):
        if p.name != "manifest.json" and not (saida / p.name).exists():
            (saida / p.name).symlink_to(p)

    print()
    print("gravado %s" % (saida / (ident + ".rmfbin")))
    print("area efetiva a 1 keV: %.1f cm2" % B.interpolate(area_pts, 1.0))


if __name__ == "__main__":
    main()
