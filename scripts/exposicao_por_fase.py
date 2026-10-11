#!/usr/bin/env python3
"""Exposição por fase a partir da exposição por QUADRO, contra `represented_time`.

PRIORIDADE DO AUDITOR, em 07/10/2026: "o achado mais importante agora é a
exposição por fase usada na verossimilhança não vir das GTIs. Ter eventos nos 32
bins não demonstra exposição uniforme; essa aproximação precisa ser medida e, se
relevante, corrigida."

O QUE A VEROSSIMILHANÇA USA HOJE. Em `FitProblem.score_document`:

    ciclos  = floor(exposicao / periodo)
    parcial = exposicao - ciclos * periodo
    t[p]    = ciclos * (periodo/32) + max(0, min(periodo/32, parcial - p*periodo/32))

isto é, trata a coadição como UMA observação contínua de `exposicao` segundos num
único `periodo`. São 13 observações, com períodos diferentes, lacunas de GTI e
deslocamentos de fase inteiros. A aproximação precisa ser medida.

O QUE ESTE SCRIPT USA. A extensão `EXPOSUnn` da lista de eventos traz TIME e
FRACEXP **por quadro**, o que é mais fino que a GTI: além das lacunas, ela carrega
a fração de exposição perdida por pixel ruim e por coluna. Validado em
0157360101: 366925 quadros de 73,372 ms dão 26921,9 s, que é o ONTIME; vezes a
média de FRACEXP, 0,9608, dá 25867,1 s; e o LIVETIME de 24220,78 s sai disso pelo
fator de tempo vivo do modo, 0,9364. A normalização final força a soma a dar
exatamente a exposição que a coadição usou, de modo que a comparação é só de
FORMA em fase.

A convenção temporal é a MESMA da coadição: fase = ((t − origem)/periodo) mod 1,
com a origem e o período do cabeçalho do CSV, e o mesmo deslocamento inteiro
guardado no npz.

Uso: python3 scripts/exposicao_por_fase.py [--saida docs/exposicao_por_fase.json]
"""
import argparse
import json
import math
import os
import re
import struct
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
FASES, ENERGIAS = 32, 40

_TAM = {"B": 1, "I": 2, "J": 4, "K": 8, "E": 4, "D": 8, "A": 1}
_NP = {"B": "u1", "I": ">i2", "J": ">i4", "K": ">i8", "E": ">f4", "D": ">f8"}


def tabela_numpy(caminho, extensao, colunas):
    """Le colunas de um BINTABLE direto em numpy, sem dict por linha.

    O parser do PULSARIS devolve um dict por linha, o que para a EXPOSUnn, com
    366925 quadros por observacao e treze observacoes, seriam 4,8 milhoes de
    dicts. Aqui o buffer vira um dtype estruturado de uma vez.
    """
    cab, dados = B.extension(caminho, extensao)
    nlin = int(cab["NAXIS2"])
    campos, deslocamento = [], 0
    for i in range(1, int(cab["TFIELDS"]) + 1):
        nome = str(cab["TTYPE%d" % i]).strip()
        forma = str(cab["TFORM%d" % i]).strip()
        m = re.match(r"(\d*)([A-Z])", forma)
        rep = int(m.group(1) or 1)
        tipo = m.group(2)
        if tipo not in _TAM:
            raise ValueError("TFORM %s nao suportado" % forma)
        largura = rep * _TAM[tipo]
        campos.append((nome, tipo, rep, deslocamento, largura))
        deslocamento += largura
    passo = int(cab["NAXIS1"])
    bruto = np.frombuffer(bytes(dados[:nlin * passo]), dtype=np.uint8)
    bruto = bruto.reshape(nlin, passo)
    saida = {}
    for nome, tipo, rep, off, largura in campos:
        if nome not in colunas:
            continue
        pedaco = bruto[:, off:off + largura].copy()
        saida[nome] = np.frombuffer(pedaco.tobytes(),
                                    dtype=np.dtype(_NP[tipo])).astype(float)
        if rep != 1:
            saida[nome] = saida[nome].reshape(nlin, rep)
    return saida


def cabecalho_csv(caminho):
    meta = {}
    with open(caminho, encoding="utf-8") as fh:
        for linha in fh:
            if not linha.startswith("#"):
                break
            if "=" in linha:
                k, v = linha[1:].split("=", 1)
                meta[k.strip()] = v.strip()
    return meta


def acha_obs(obsid):
    for cand in (PRODUTOS / obsid, *PRODUTOS.glob("*/" + obsid)):
        if cand.is_dir() and (cand / "epn_bary_source.fits").is_file():
            return cand
    return None


def ccd_da_fonte(caminho):
    """CCD onde a fonte esta, pela moda do CCDNR dos eventos da fonte."""
    col = tabela_numpy(caminho, "EVENTS", {"CCDNR"})
    v = col["CCDNR"].astype(int)
    return int(np.bincount(v).argmax())


def exposicao_por_fase(dirobs, meta, exposicao_alvo):
    """Exposicao por bin de fase, da EXPOSUnn, normalizada a exposicao_alvo."""
    fonte = dirobs / "epn_bary_source.fits"
    ccd = ccd_da_fonte(fonte)
    ext = "EXPOSU%02d" % ccd
    col = tabela_numpy(fonte, ext, {"TIME", "FRACEXP"})
    t, f = col["TIME"], col["FRACEXP"]
    ordem = np.argsort(t)
    t, f = t[ordem], f[ordem]
    passos = np.diff(t)
    quadro = float(np.median(passos))
    origem = float(meta["time_origin_mjd_s"])
    periodo = float(meta["period_s"])
    fase = ((t - origem) / periodo) % 1.0
    ib = np.minimum((fase * FASES).astype(int), FASES - 1)
    cru = np.bincount(ib, weights=f * quadro, minlength=FASES)
    soma = float(cru.sum())
    fator = exposicao_alvo / soma if soma > 0 else float("nan")
    return (cru * fator, {"ccd": ccd, "extensao": ext, "quadros": int(t.size),
                          "quadro_s": quadro,
                          "ontime_implicado_s": float(t.size * quadro),
                          "fracexp_media": float(f.mean()),
                          "soma_fracexp_x_quadro_s": soma,
                          "fator_de_tempo_vivo": float(fator),
                          "lacunas_acima_de_2_quadros":
                              int((passos > 2 * quadro).sum())})


def representado(exposicao, periodo):
    """O `represented_time` do MAGNUS, reproduzido do codigo."""
    ciclos = math.floor(exposicao / periodo)
    parcial = exposicao - ciclos * periodo
    dur = periodo / FASES
    return np.array([ciclos * dur + max(0.0, min(dur, parcial - ip * dur))
                     for ip in range(FASES)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(RAIZ / "docs/exposicao_por_fase.json"))
    args = ap.parse_args()
    t0 = time.time()

    d = np.load(NPZ, allow_pickle=True)
    obsids = [str(x) for x in d["obsids"]]
    expos = np.asarray(d["exposures"], float)
    periodos = np.asarray(d["periods"], float)
    shifts = np.asarray(d["shifts"], int)
    observado = np.asarray(d["observed"], float).reshape(FASES, ENERGIAS)
    T = float(d["exposure"])

    print("=== exposicao por fase, de EXPOSUnn, por observacao ===")
    print("%-12s %4s %8s %10s %10s %8s %12s %12s"
          % ("obsid", "ccd", "quadros", "quadro ms", "FRACEXP", "lacunas",
             "tempo vivo", "amplitude %"))
    total = np.zeros(FASES)
    por_obs = []
    for obsid, exp, per, shift in zip(obsids, expos, periodos, shifts):
        dirobs = acha_obs(obsid)
        csv = sorted(dirobs.glob("pulsaris/*_events.csv"))[0]
        meta = cabecalho_csv(csv)
        tp, info = exposicao_por_fase(dirobs, meta, float(exp))
        # o MESMO deslocamento inteiro que a coadicao aplicou
        tp_alinhado = np.roll(tp, int(shift))
        total += tp_alinhado
        amplitude = 100.0 * (tp.max() - tp.min()) / tp.mean()
        print("%-12s %4d %8d %10.4f %10.4f %8d %12.3f %12.3f"
              % (obsid, info["ccd"], info["quadros"], info["quadro_s"] * 1e3,
                 info["fracexp_media"], info["lacunas_acima_de_2_quadros"],
                 float(tp.sum()), amplitude))
        por_obs.append({"obsid": obsid, "shift": int(shift),
                        "exposicao_alvo_s": float(exp), "periodo_s": float(per),
                        "por_fase_s": tp.tolist(),
                        "por_fase_alinhada_s": tp_alinhado.tolist(),
                        "amplitude_relativa_por_cento": amplitude,
                        **info})

    rep = representado(T, float(periodos[0]))
    print()
    print("=== a coadicao: das GTIs contra o represented_time do MAGNUS ===")
    print("soma das GTIs %.3f s | soma do represented_time %.3f s | exposicao %.3f s"
          % (total.sum(), rep.sum(), T))
    print("%5s %16s %16s %12s %12s %14s"
          % ("fase", "das GTIs", "represented", "dif s", "dif rel", "contagens"))
    dif = total - rep
    for ip in range(FASES):
        print("%5d %16.4f %16.4f %12.4f %12.3e %14d"
              % (ip, total[ip], rep[ip], dif[ip], dif[ip] / rep[ip],
                 int(observado[ip].sum())))
    rel = dif / rep
    print()
    print("diferenca relativa: mediana %.3e, pior bin %.3e (bin %d), amplitude %.3e"
          % (float(np.median(np.abs(rel))), float(np.abs(rel).max()),
             int(np.argmax(np.abs(rel))), float(rel.max() - rel.min())))
    print("o represented_time e uniforme? amplitude %.3e; e o das GTIs? %.3e"
          % ((rep.max() - rep.min()) / rep.mean(),
             (total.max() - total.min()) / total.mean()))

    # consequencia por fase E energia: mu = (S + B) * largura * t[p], logo a
    # mudanca relativa de t[p] e a mudanca relativa de TODAS as energias daquela
    # fase. O efeito em contagens previstas sai do proprio dado.
    print()
    print("=== consequencia em contagens previstas ===")
    ct_por_fase = observado.sum(axis=1)
    desloc_ct = ct_por_fase * rel
    print("mudanca implicada nas contagens previstas, por fase:")
    print("  mediana %.2f ct, pior %.2f ct no bin %d, soma em modulo %.1f ct"
          % (float(np.median(np.abs(desloc_ct))), float(np.abs(desloc_ct).max()),
             int(np.argmax(np.abs(desloc_ct))), float(np.abs(desloc_ct).sum())))
    print("  contra o Poisson do dado por fase: mediana %.3f sigma, pior %.3f sigma"
          % (float(np.median(np.abs(desloc_ct) / np.sqrt(ct_por_fase))),
             float((np.abs(desloc_ct) / np.sqrt(ct_por_fase)).max())))

    saida = {
        "o_que_e": "exposicao por fase da EXPOSUnn contra o represented_time do MAGNUS",
        "prioridade_do_auditor": (
            "a exposicao por fase usada na verossimilhanca nao vem das GTIs; ter "
            "eventos nos 32 bins nao demonstra exposicao uniforme"),
        "o_que_o_magnus_usa": (
            "t[p] = ciclos*(P/32) + max(0, min(P/32, parcial - p*P/32)), isto e "
            "UMA observacao continua num unico periodo"),
        "fonte_usada": (
            "extensao EXPOSUnn da lista de eventos, TIME e FRACEXP por QUADRO, "
            "mais fino que a GTI: carrega lacunas e perda por pixel ruim"),
        "convencao": ("fase = ((t - origem)/periodo) mod 1 com a origem e o "
                      "periodo do CSV, e o mesmo deslocamento inteiro do npz"),
        "normalizacao": ("a soma por observacao e forcada a dar a exposicao que a "
                         "coadicao usou, logo a comparacao e so de FORMA"),
        "por_observacao": por_obs,
        "coadicao": {
            "das_gtis_s": total.tolist(),
            "represented_time_s": rep.tolist(),
            "diferenca_s": dif.tolist(),
            "diferenca_relativa": rel.tolist(),
            "mediana_do_modulo": float(np.median(np.abs(rel))),
            "pior_bin": int(np.argmax(np.abs(rel))),
            "pior_relativo": float(np.abs(rel).max()),
            "amplitude_do_represented_time": float((rep.max() - rep.min()) / rep.mean()),
            "amplitude_das_gtis": float((total.max() - total.min()) / total.mean())},
        "consequencia": {
            "contagens_por_fase": ct_por_fase.tolist(),
            "deslocamento_implicado_ct": desloc_ct.tolist(),
            "em_sigmas_de_poisson":
                (np.abs(desloc_ct) / np.sqrt(ct_por_fase)).tolist(),
            "nota": ("mu = (S + B) * largura * t[p], logo a mudanca relativa de "
                     "t[p] vale para TODAS as energias daquela fase")},
        "o_que_isto_NAO_diz": [
            "nao mede o efeito nos parametros nem nos intervalos de z, que exige "
            "reajuste",
            "a normalizacao por observacao e forcada, logo um erro de exposicao "
            "TOTAL nao seria visto aqui, so a forma em fase",
            "o fator de tempo vivo do modo entra como razao global e nao foi "
            "verificado contra a calibracao"],
        "segundos": time.time() - t0,
    }
    Path(args.saida).write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n")
    print()
    print("%.1f s | gravado em %s" % (saida["segundos"], args.saida))


if __name__ == "__main__":
    main()
