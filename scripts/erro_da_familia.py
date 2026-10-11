#!/usr/bin/env python3
"""Erro da familia fechada, medido SEM Monte Carlo, nos 40 bins.

POR QUE ESTE ARQUIVO EXISTE, e e uma correcao do meu proprio pre-registro. O
desenho pre-registrado em `pre_registro_tolerancia_familia.md` separava ruido de
viés por ESFORCO CRESCENTE de Monte Carlo, com a tolerancia (1) exigindo que a
razao de |GL - MC| entre dois esforcos ficasse em [2, 8]. Essa tolerancia e MAL
POSTA no regime em que o erro da familia esta ABAIXO do ruido: ali |GL - MC| e
quase puro ruido, a razao de dois ruidos independentes tem distribuicao larga em
torno de 1, e valores como 0,25 ou 18 aparecem sem que nada esteja errado com a
familia. Medido: a tolerancia (1) reprova muitos bins exatamente por isso.

A ROTA CERTA e determinística: a soma de Gammas tem funcao caracteristica
fechada,

    phi(t) = produto_j (1 - i a_j t)^(-k_j),

e a densidade exata sai por inversao de Fourier, sem amostragem. A verificacao de
que a inversao esta certa nao e opinião: a densidade recuperada tem de integrar 1
e reproduzir os cumulantes fechados k1, k2 e k3. MEDIDO: norma 1,0000000000, k1 a
1e-13, k2 a 4e-9 e k3 a 9e-6.

Com isso o erro da familia e

    delta_e = lnL com Gama deslocada de tres cumulantes  -  lnL com a densidade exata

sem ruido, nos 40 bins, e as tolerancias (2) e (3) do pre-registro, que estavam
BEM postas, se aplicam direto: patamar por bin abaixo de 1e-4 e soma abaixo de
4e-3.

Uso: python3 scripts/erro_da_familia.py [--saida docs/erro_da_familia.json]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
V2 = RAIZ / "exploracoes/dados_combinados_13obs_v2_2026-10-06/combinado13_v2.npz"
E_MIN, E_MAX, N_E, N_P = 0.15, 1.2, 40, 32
LARGURA = (E_MAX - E_MIN) / N_E
A0, BETA = 0.01, 0.0
M_FFT, B_FFT = 1 << 16, 0.25
TOL_PATAMAR, TOL_SOMA = 1.0e-4, 4.0e-3


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


def densidade_gama(b, forma, escala, desloc):
    if not (escala > 0.0):
        return np.zeros_like(np.asarray(b, float))
    x = (np.asarray(b, float) - desloc) / escala
    saida = np.zeros_like(x)
    bom = x > 0.0
    saida[bom] = np.exp((forma - 1.0) * np.log(x[bom]) - x[bom]
                        - math.lgamma(forma) - math.log(escala))
    return saida


def densidade_exata(a, forma, M=M_FFT, B=B_FFT):
    """Densidade da soma ponderada de Gammas, por inversao da caracteristica.

    phi(t) = produto (1 - i a_j t)^(-k_j), somada em logaritmo para nao estourar.
    A grade em b vai de 0 a B com passo h = B/M; o passo em t e 2 pi / B, o que
    faz o produto t_n b_m virar 2 pi n m / M e a soma virar uma FFT.
    """
    h = B / M
    dt = 2.0 * math.pi / B
    t = dt * np.arange(M)
    logphi = -(forma[None, :] * np.log1p(-1j * a[None, :] * t[:, None])).sum(axis=1)
    phi = np.exp(logphi)
    f = (dt / math.pi) * (np.real(np.fft.fft(phi))
                          - 0.5 * np.real(phi[0]))
    return h * np.arange(M), np.maximum(f, 0.0), h


def ll(b, S_p, n_p, passo_p):
    b = np.atleast_1d(np.asarray(b, float))
    mu = np.maximum((S_p[None, :] + b[:, None]) * passo_p[None, :], 1e-300)
    return (n_p[None, :] * np.log(mu) - mu).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(RAIZ / "docs/erro_da_familia.json"))
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

    print("a0 = %g por keV, beta = %g, FFT com M = %d e B = %g"
          % (A0, BETA, M_FFT, B_FFT))
    print("%4s %18s %18s %12s %10s %10s %10s"
          % ("bin", "GL (familia)", "exata (FFT)", "delta", "norma", "k1 rel",
             "k3 rel"))
    linhas = []
    for ie in range(N_E):
        fatia = slice(desl[ie], desl[ie + 1])
        a = pesos[fatia]
        forma_j = contagens[fatia] + A0 * larguras[fatia]
        k1, k2, k3 = cumulantes(a, forma_j)
        n_p = observado[:, ie]
        S_p = np.maximum(n_p / passo_p - fundo[ie], 0.0)
        referencia = float(ll(k1, S_p, n_p, passo_p)[0])

        # a familia fechada, por Gauss-Legendre
        sd = math.sqrt(k2)
        lo, hi = max(0.0, k1 - 8.0 * sd), k1 + 12.0 * sd
        x, w = np.polynomial.legendre.leggauss(64)
        b = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        peso = 0.5 * (hi - lo) * w
        f3, e3, s3 = gama3(k1, k2, k3)
        pi = densidade_gama(b, f3, e3, s3)
        gl = referencia + math.log(
            float((peso * pi * np.exp(ll(b, S_p, n_p, passo_p) - referencia)).sum())
            / float((peso * pi).sum()))

        # a densidade EXATA, e as verificacoes que a autorizam
        be, fe, h = densidade_exata(a, forma_j)
        norma = float(fe.sum() * h)
        m1 = float((be * fe).sum() * h) / norma
        m2 = float(((be - m1) ** 2 * fe).sum() * h) / norma
        m3 = float(((be - m1) ** 3 * fe).sum() * h) / norma
        exata = referencia + math.log(
            float((fe * np.exp(ll(be, S_p, n_p, passo_p) - referencia)).sum() * h)
            / norma)
        delta = gl - exata
        print("%4d %18.8f %18.8f %12.3e %10.7f %10.1e %10.1e"
              % (ie, gl, exata, delta, norma, abs(m1 / k1 - 1.0),
                 abs(m3 / k3 - 1.0) if k3 > 0 else 0.0))
        linhas.append({"bin": ie, "gauss_legendre_familia": gl, "exata": exata,
                       "delta": delta, "modulo_delta": abs(delta),
                       "verificacao_da_fft": {
                           "norma": norma, "k1_relativo": abs(m1 / k1 - 1.0),
                           "k2_relativo": abs(m2 / k2 - 1.0),
                           "k3_relativo": abs(m3 / k3 - 1.0) if k3 > 0 else 0.0},
                       "k1": k1, "assimetria": k3 / k2 ** 1.5,
                       "contagens_brutas": float(contagens[fatia].sum()),
                       "termos_com_zero": int((contagens[fatia] == 0).sum())})

    fora = [l["bin"] for l in linhas if l["modulo_delta"] > TOL_PATAMAR]
    soma = sum(l["modulo_delta"] for l in linhas)
    pior = max(linhas, key=lambda l: l["modulo_delta"])
    print()
    print("=== as tolerancias BEM POSTAS do pre-registro ===")
    print("(2) |delta| < %.0e por bin: %d de %d fora -> %s"
          % (TOL_PATAMAR, len(fora), N_E, fora if fora else "nenhum"))
    print("(3) soma dos |delta| < %.0e: %.3e" % (TOL_SOMA, soma))
    print("pior bin: %d, com |delta| = %.3e, assimetria %.4f, %d canais vazios"
          % (pior["bin"], pior["modulo_delta"], pior["assimetria"],
             pior["termos_com_zero"]))
    print("verificacao da FFT no pior caso: norma %.10f, k1 a %.1e, k3 a %.1e"
          % (min(l["verificacao_da_fft"]["norma"] for l in linhas),
             max(l["verificacao_da_fft"]["k1_relativo"] for l in linhas),
             max(l["verificacao_da_fft"]["k3_relativo"] for l in linhas)))
    aprova = not fora and soma < TOL_SOMA
    print()
    print("VEREDITO: %s" % ("PASSA nas duas tolerancias bem postas"
                            if aprova else "NAO PASSA"))
    saida = {
        "o_que_mede": "erro da familia fechada contra a densidade EXATA, sem Monte Carlo",
        "por_que_nao_e_por_monte_carlo": (
            "a tolerancia (1) do meu pre-registro, a razao de |GL - MC| entre "
            "dois esforcos, e MAL POSTA quando o erro da familia esta abaixo do "
            "ruido: ali a razao e de dois ruidos independentes, com distribuicao "
            "larga em torno de 1"),
        "rota": "inversao da funcao caracteristica phi(t) = produto (1 - i a_j t)^(-k_j), por FFT",
        "verificacao_que_autoriza_a_rota": (
            "a densidade recuperada integra 1 e reproduz os cumulantes fechados"),
        "a0": A0, "beta": BETA, "fft": {"M": M_FFT, "B": B_FFT},
        "tolerancias_aplicadas": {"por_bin": TOL_PATAMAR, "soma": TOL_SOMA},
        "por_bin": linhas,
        "bins_fora": fora, "soma_dos_modulos": soma,
        "pior_bin": pior["bin"], "pior_modulo": pior["modulo_delta"],
        "veredito": "PASSA" if aprova else "NAO PASSA",
        "o_que_nao_cobre": (
            "mede a familia nesta configuracao, com a0 = 0,01 por keV, no banco "
            "de teste em que S[p] e casado ao dado; nao se estende a outros a0, "
            "outros vetores de parametros, nem outras resolucoes"),
    }
    Path(args.saida).write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n")
    print("gravado em %s" % args.saida)


if __name__ == "__main__":
    main()
