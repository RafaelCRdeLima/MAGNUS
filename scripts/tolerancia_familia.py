#!/usr/bin/env python3
"""Erro da familia fechada contra a soma exata, nos 40 bins, com esforco crescente.

Executa o desenho PRE-REGISTRADO em
docs/incidente_resolucao_2026-10-04/pre_registro_tolerancia_familia.md, commit
f9e125f, anterior a esta medida.

POR QUE EXISTE. Na comparacao anterior, num unico esforco de Monte Carlo, um dos
cinco bins relatados teve |GL - MC| acima do proprio erro de MC, e eu nao o
expliquei. A causa possivel e dupla, e um esforco so nao as separa: o RUIDO do
Monte Carlo cai como n^(-1/2), e o ERRO DA FAMILIA fechada nao cai com n. Tres
esforcos, com fator 16 entre o menor e o maior, separam os dois.

TOLERANCIAS, fixadas no pre-registro e repetidas aqui:
  (1) razao |GL-MC| entre n=200000 e n=3200000 dentro de [2, 8], isto e a um
      fator 2 do valor 4 que o decaimento puro prediz para fator 16 em n;
  (2) patamar, estimado por |GL-MC| em n=3200000 menos 2 erros de MC, com piso
      em zero, abaixo de 1e-4 em lnL por bin;
  (3) soma dos patamares dos 40 bins abaixo de 4e-3 em lnL.

Memoria: o sorteio vai em blocos de 200000, isto e cerca de 125 MB por bloco,
liberados antes do seguinte. Nunca os 3,2 milhoes de uma vez, que seriam 2 GB.

Uso: python3 scripts/tolerancia_familia.py [--saida docs/tolerancia_familia.json]
"""
import argparse
import gc
import json
import math
import resource
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
V2 = RAIZ / "exploracoes/dados_combinados_13obs_v2_2026-10-06/combinado13_v2.npz"
E_MIN, E_MAX, N_E, N_P = 0.15, 1.2, 40, 32
LARGURA = (E_MAX - E_MIN) / N_E
A0 = 0.01            # convencao fiducial escolhida pelo revisor
BETA = 0.0
ESFORCOS = ((200000, 20261006), (800000, 777), (3200000, 4242))
BLOCO = 200000
TOL_RAZAO = (2.0, 8.0)
TOL_PATAMAR = 1.0e-4
TOL_SOMA = 4.0e-3


def cumulantes(a, forma):
    return (float((a * forma).sum()), float((a ** 2 * forma).sum()),
            float(2.0 * (a ** 3 * forma).sum()))


def gama3(k1, k2, k3):
    if not (k2 > 0.0):
        return 1.0, 0.0, k1
    if k3 <= 0.0:
        return k1 * k1 / k2, k2 / k1, 0.0
    escala = k3 / (2.0 * k2)
    forma = 4.0 * k2 ** 3 / (k3 * k3)
    return forma, escala, k1 - forma * escala


def densidade(b, forma, escala, desloc):
    if not (escala > 0.0):
        return np.zeros_like(np.asarray(b, float))
    x = (np.asarray(b, float) - desloc) / escala
    saida = np.zeros_like(x)
    bom = x > 0.0
    saida[bom] = np.exp((forma - 1.0) * np.log(x[bom]) - x[bom]
                        - math.lgamma(forma) - math.log(escala))
    return saida


def ll(b, S_p, n_p, passo_p):
    b = np.atleast_1d(np.asarray(b, float))
    mu = np.maximum((S_p[None, :] + b[:, None]) * passo_p[None, :], 1e-300)
    return (n_p[None, :] * np.log(mu) - mu).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(RAIZ / "docs/tolerancia_familia.json"))
    args = ap.parse_args()

    d = np.load(V2, allow_pickle=True)
    pesos = np.asarray(d["fundo_pesos"], float)
    contagens = np.asarray(d["fundo_contagens"], float)
    larguras = np.asarray(d["fundo_largura_do_canal"], float)
    desl = np.asarray(d["fundo_deslocamento"], int)
    observado = np.asarray(d["observed"], float).reshape(N_P, N_E)
    fundo = np.asarray(d["background"], float)
    T = float(np.asarray(d["exposures"], float).sum())
    passo_p = LARGURA * np.full(N_P, T / N_P)

    print("a0 = %g por keV, beta = %g, esforcos %s"
          % (A0, BETA, [n for n, _ in ESFORCOS]))
    print("%4s %18s %11s %11s %11s %11s %9s %11s"
          % ("bin", "GL", "|GL-MC| n1", "|GL-MC| n2", "|GL-MC| n3",
             "erro MC n3", "razao", "patamar"))
    linhas = []
    for ie in range(N_E):
        fatia = slice(desl[ie], desl[ie + 1])
        a = pesos[fatia]
        forma_j = contagens[fatia] + A0 * larguras[fatia]
        k1, k2, k3 = cumulantes(a, forma_j)
        n_p = observado[:, ie]
        S_p = np.maximum(n_p / passo_p - fundo[ie], 0.0)
        # GL, deterministico
        sd = math.sqrt(k2)
        lo, hi = max(0.0, k1 - 8.0 * sd), k1 + 12.0 * sd
        x, w = np.polynomial.legendre.leggauss(64)
        b = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        peso = 0.5 * (hi - lo) * w
        f, e, s = gama3(k1, k2, k3)
        pi = densidade(b, f, e, s)
        referencia = float(ll(k1, S_p, n_p, passo_p)[0])
        num = float((peso * pi * np.exp(ll(b, S_p, n_p, passo_p) - referencia)).sum())
        den = float((peso * pi).sum())
        gl = referencia + math.log(num / den)
        # MC em blocos, sobre a soma EXATA
        difs, erros = [], []
        for n_total, semente in ESFORCOS:
            rng = np.random.default_rng(semente)
            soma, soma2, visto = 0.0, 0.0, 0
            while visto < n_total:
                m = min(BLOCO, n_total - visto)
                g = rng.gamma(shape=np.repeat(forma_j[None, :], m, axis=0),
                              scale=1.0)
                bb = (g @ a) / (1.0 + BETA)
                del g
                v = np.exp(ll(bb, S_p, n_p, passo_p) - referencia)
                soma += float(v.sum())
                soma2 += float((v ** 2).sum())
                visto += m
                del bb, v
                gc.collect()
            media = soma / n_total
            var = max(0.0, soma2 / n_total - media ** 2)
            erro_rel = math.sqrt(var / n_total) / media if media > 0 else float("inf")
            mc = referencia + math.log(media)
            difs.append(abs(gl - mc))
            erros.append(erro_rel)
        razao = difs[0] / difs[2] if difs[2] > 0 else float("inf")
        patamar = max(0.0, difs[2] - 2.0 * erros[2])
        print("%4d %18.8f %11.3e %11.3e %11.3e %11.3e %9.2f %11.3e"
              % (ie, gl, difs[0], difs[1], difs[2], erros[2], razao, patamar),
              flush=True)
        linhas.append({"bin": ie, "gauss_legendre": gl,
                       "dif_por_esforco": difs, "erro_de_mc": erros,
                       "razao_n1_sobre_n3": razao, "patamar": patamar,
                       "k1": k1, "contagens_brutas": float(contagens[fatia].sum()),
                       "termos_com_zero": int((contagens[fatia] == 0).sum())})

    fora_razao = [l["bin"] for l in linhas
                  if not (TOL_RAZAO[0] <= l["razao_n1_sobre_n3"] <= TOL_RAZAO[1])]
    fora_patamar = [l["bin"] for l in linhas if l["patamar"] > TOL_PATAMAR]
    soma = sum(l["patamar"] for l in linhas)
    print()
    print("=== as tres tolerancias do pre-registro ===")
    print("(1) razao em [2, 8]: %d bins fora -> %s"
          % (len(fora_razao), fora_razao if fora_razao else "nenhum"))
    print("(2) patamar < %.0e por bin: %d bins fora -> %s"
          % (TOL_PATAMAR, len(fora_patamar),
             fora_patamar if fora_patamar else "nenhum"))
    print("(3) soma dos patamares < %.0e: %.3e" % (TOL_SOMA, soma))
    aprova = not fora_razao and not fora_patamar and soma < TOL_SOMA
    print()
    print("VEREDITO: %s" % ("PASSA nas tres" if aprova else "NAO PASSA"))
    print("pico de memoria: %.0f MB"
          % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0))
    saida = {
        "pre_registro": "docs/incidente_resolucao_2026-10-04/pre_registro_tolerancia_familia.md, commit f9e125f",
        "a0": A0, "beta": BETA, "esforcos": [n for n, _ in ESFORCOS],
        "sementes": [s for _, s in ESFORCOS],
        "o_que_mede": ("|GL - MC| mistura ruido de Monte Carlo, que cai como "
                       "n^(-1/2), e erro da familia fechada, que nao cai; tres "
                       "esforcos separam os dois"),
        "tolerancias": {"razao": list(TOL_RAZAO), "patamar_por_bin": TOL_PATAMAR,
                        "soma_dos_patamares": TOL_SOMA},
        "por_bin": linhas,
        "bins_fora_da_razao": fora_razao,
        "bins_fora_do_patamar": fora_patamar,
        "soma_dos_patamares": soma,
        "veredito": "PASSA" if aprova else "NAO PASSA",
        "pico_de_memoria_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "o_que_nao_cobre": ("mede a familia nesta configuracao, com a0 = 0,01 por "
                            "keV, no banco de teste em que S[p] e casado ao dado; "
                            "nao se estende a outros a0, outros vetores de "
                            "parametros, nem outras resolucoes"),
    }
    Path(args.saida).write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n")
    print("gravado em %s" % args.saida)


if __name__ == "__main__":
    main()
