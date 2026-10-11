#!/usr/bin/env python3
"""Protótipo: marginalizar o fundo com UMA integral por canal de energia.

O QUE O REVISOR CORRIGIU, em 06/10/2026, e que define este arquivo. A
diagonalidade da covariância separa as integrais ENTRE canais, sob as hipóteses
usadas. Ela não separa nada DENTRO de um canal: o mesmo fundo `B[e]` entra nos 32
bins de fase, logo tem de ser integrado UMA VEZ sobre a verossimilhança conjunta
dessas fases. Integrar cada célula fase-energia por conta própria criaria 32
fundos independentes artificiais, cada um com a mesma priori, e estreitaria a
incerteza por um fator espúrio de raiz de 32.

A integral correta, por canal de energia e:

    L_e(theta) = integral de 0 a infinito  pi_e(b)  produto sobre p de
                 Poisson( n[p][e] | (S[p][e] + b) * Delta * t[p] )  db

e a verossimilhança é o produto dos L_e, porque os canais são independentes.

A DISTRIBUIÇÃO pi_e NÃO É GAMMA, e isto também foi correção do revisor: conhecer
média e variância não autoriza escolher uma família. Derivando das contagens
brutas e dos pesos,

    b_e = soma_j a_j lambda_j ,   lambda_j | N_j ~ Gamma(N_j + alpha, 1)

com j percorrendo os pares observação-canal que se sobrepõem ao bin e a_j os
pesos do estimador por área. É uma soma ponderada de Gammas independentes com
escalas DIFERENTES, que não é Gamma. Os cumulantes saem fechados,

    k1 = soma a_j (N_j + alpha)        k3 = 2 soma a_j^3 (N_j + alpha)
    k2 = soma a_j^2 (N_j + alpha)      k4 = 6 soma a_j^4 (N_j + alpha)

e o script mede quais aproximações fechadas reproduzem a referência, que é uma
amostra de Monte Carlo da soma exata, não outra aproximação.

TRÊS VALIDAÇÕES INDEPENDENTES da quadratura, como o portão exige: Gauss-Legendre
com nós fixos, soma de Riemann muito fina e Monte Carlo sobre sorteios da própria
pi_e. As três têm de concordar.

LIMITE DECLARADO: isto é protótipo de um TRATAMENTO, não a adoção dele. Entrar na
produção depende da decisão do revisor sobre a definição do alvo, e de o
estimador de fundo ser corrigido antes, com versão nova do dado preparado.

Uso: python3 scripts/marginaliza_fundo.py [--vetores] [--saida ARQ]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
PULSARIS = Path(os.environ.get("MAGNUS_PULSARIS", Path.home() / "Codes/PULSARIS"))
sys.path.insert(0, str(PULSARIS / "scripts"))
import build_instrument_profiles as B  # noqa: E402

PRODUTOS = Path(os.environ.get("MAGNUS_PRODUTOS",
                               Path.home() / "Codes/XMM-NEWTON/products"))
COMBINADO = RAIZ / "exploracoes/dados_combinados_13obs_2026-09-12/combinado13.npz"
E_MIN, E_MAX, N_E, N_P = 0.15, 1.2, 40, 32
BORDAS = np.linspace(E_MIN, E_MAX, N_E + 1)
ECEN = 0.5 * (BORDAS[:-1] + BORDAS[1:])
LARGURA = (E_MAX - E_MIN) / N_E
#: Forma da priori por canal. alpha = 1 é priori plana em lambda, cuja posterior
#: tem média e variância N+1, a mesma convenção de scripts/incerteza_fundo.py.
#: alpha = 1/2 é Jeffreys. A escolha injeta alpha pseudo-contagens POR CANAL, e o
#: relatório mede quanto isso move o resultado, porque são muitos canais.
ALPHA = 1.0


def pesos_por_bin(alpha=ALPHA):
    """Pesos a_j e contagens N_j de cada par observação-canal, por bin.

    Usa o estimador por ÁREA normalizado pela cobertura, que é o que corresponde
    ao que a verossimilhança pede (ver scripts/estimador_do_fundo.py). Os pesos
    já incluem a escala de BACKSCAL, a exposição do espectro e o peso da
    coadição.
    """
    d = np.load(COMBINADO, allow_pickle=True)
    obsids = [str(x) for x in d["obsids"]]
    expos = np.asarray(d["exposures"], float)
    T = float(expos.sum())
    pesos = [[] for _ in range(N_E)]
    contagens = [[] for _ in range(N_E)]
    rotulos = [[] for _ in range(N_E)]
    for obsid, exp in zip(obsids, expos):
        dirobs = None
        for cand in (PRODUTOS / obsid, *PRODUTOS.glob("*/" + obsid)):
            if (cand / "bkg_spec.fits").is_file():
                dirobs = cand
                break
        cab_b, dados_b = B.extension(dirobs / "bkg_spec.fits", "SPECTRUM")
        conta = {int(r["CHANNEL"]): float(r["COUNTS"])
                 for r in B.table_rows(cab_b, dados_b)}
        cab_s, _ = B.extension(dirobs / "src_spec.fits", "SPECTRUM")
        escala = float(cab_s["BACKSCAL"]) / float(cab_b["BACKSCAL"])
        exposicao = float(cab_b["EXPOSURE"])
        cabe, dadose = B.extension(dirobs / "src.rmf", "EBOUNDS")
        eb = list(B.table_rows(cabe, dadose))
        cl = np.array([int(r["CHANNEL"]) for r in eb])
        elo = np.array([float(r["E_MIN"]) for r in eb])
        ehi = np.array([float(r["E_MAX"]) for r in eb])
        ecen = 0.5 * (elo + ehi)
        sel = (ecen >= 0.150) & (ecen <= 1.200) & (ehi > elo)
        cl, elo, ehi = cl[sel], elo[sel], ehi[sel]
        N = np.array([conta.get(int(c), 0.0) for c in cl])
        for k in range(N_E):
            a, b = BORDAS[k], BORDAS[k + 1]
            ov = np.clip(np.minimum(ehi, b) - np.maximum(elo, a), 0.0, None)
            coberto = ov.sum()
            if coberto <= 0:
                continue
            quais = ov > 0
            # b_e^o = escala * soma_k (N_k * ov_k / w_k) / (exposicao * coberto)
            # e a coadicao pesa por exp/T.
            a_j = (exp / T) * escala * ov[quais] / (
                (ehi[quais] - elo[quais]) * exposicao * coberto)
            pesos[k].extend(a_j.tolist())
            contagens[k].extend(N[quais].tolist())
            rotulos[k].extend([(obsid, int(c)) for c in cl[quais]])
    return ([np.array(x) for x in pesos], [np.array(x) for x in contagens],
            rotulos, T)


def cumulantes(a, N, alpha=ALPHA):
    """Cumulantes exatos de soma a_j * Gamma(N_j + alpha, 1)."""
    f = N + alpha
    k1 = float((a * f).sum())
    k2 = float((a ** 2 * f).sum())
    k3 = float(2.0 * (a ** 3 * f).sum())
    k4 = float(6.0 * (a ** 4 * f).sum())
    return k1, k2, k3, k4


def amostra_exata(a, N, n, rng, alpha=ALPHA):
    """Sorteia da distribuição EXATA somando as Gammas. Referência do teste."""
    f = N + alpha
    # Gamma(forma, escala=1) por termo, escalado por a_j
    g = rng.gamma(shape=np.repeat(f[None, :], n, axis=0), scale=1.0)
    return g @ a


def gama_de_dois_cumulantes(k1, k2):
    forma = k1 ** 2 / k2
    escala = k2 / k1
    return forma, escala, 0.0


def gama_de_tres_cumulantes(k1, k2, k3):
    """Gamma DESLOCADA casada em três cumulantes, o ajuste clássico para soma
    de Gammas: escala = k3/(2 k2), forma = 4 k2^3/k3^2, deslocamento = k1 -
    forma*escala. O deslocamento pode sair negativo, e aí a família não serve."""
    if k3 <= 0:
        return gama_de_dois_cumulantes(k1, k2)
    escala = k3 / (2.0 * k2)
    forma = 4.0 * k2 ** 3 / k3 ** 2
    return forma, escala, k1 - forma * escala


def densidade_gama(b, forma, escala, desloc):
    x = (np.asarray(b, float) - desloc) / escala
    out = np.zeros_like(x)
    bom = x > 0
    out[bom] = np.exp((forma - 1.0) * np.log(x[bom]) - x[bom]
                      - math.lgamma(forma) - math.log(escala))
    return out


def log_verossimilhanca_do_canal(b, S_p, n_p, passo_p):
    """log do produto de Poisson sobre as fases, para um b escalar ou vetor.

    mu[p] = (S[p] + b) * passo_p[p], com passo_p = Delta * t[p]. Fatorial de n
    omitido: constante em b e em theta.
    """
    b = np.atleast_1d(np.asarray(b, float))
    mu = (S_p[None, :] + b[:, None]) * passo_p[None, :]
    mu = np.maximum(mu, 1e-300)
    return (n_p[None, :] * np.log(mu) - mu).sum(axis=1)


def marginal_do_canal(S_p, n_p, passo_p, k1, k2, k3, nos=64, familia="3cum",
                      ref_log=None):
    """integral de pi(b) L(b) db por Gauss-Legendre num intervalo generoso.

    O integrando é suave e concentrado; o intervalo vai de max(0, k1-8 sigma) a
    k1 + 12 sigma, com cauda à direita mais longa porque a soma de Gammas é
    assimétrica à direita.
    """
    sigma = math.sqrt(k2)
    lo = max(0.0, k1 - 8.0 * sigma)
    hi = k1 + 12.0 * sigma
    x, w = np.polynomial.legendre.leggauss(nos)
    b = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
    peso = 0.5 * (hi - lo) * w
    if familia == "3cum":
        forma, escala, desloc = gama_de_tres_cumulantes(k1, k2, k3)
    else:
        forma, escala, desloc = gama_de_dois_cumulantes(k1, k2)
    pi = densidade_gama(b, forma, escala, desloc)
    ll = log_verossimilhanca_do_canal(b, S_p, n_p, passo_p)
    pico = ref_log if ref_log is not None else float(ll.max())
    integral = float((peso * pi * np.exp(ll - pico)).sum())
    norma = float((peso * pi).sum())
    return pico + math.log(max(integral, 1e-300)), norma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(RAIZ / "docs/marginaliza_fundo.json"))
    ap.add_argument("--amostras", type=int, default=200000)
    args = ap.parse_args()
    rng = np.random.default_rng(20261006)

    pesos, contagens, rotulos, T = pesos_por_bin()
    d = np.load(COMBINADO, allow_pickle=True)
    observado = np.asarray(d["observed"], float).reshape(N_P, N_E)
    guardado = np.asarray(d["background"], float)

    print("=== pi_e, a distribuicao do fundo por canal de energia ===")
    print("derivada das contagens brutas e dos pesos: soma ponderada de Gammas "
          "independentes com escalas diferentes, que NAO e Gamma")
    print("%4s %7s %9s %12s %12s %10s %10s %10s"
          % ("bin", "termos", "contagens", "media", "desvio", "assim.", "exc.curt",
             "pseudo%"))
    resumo = []
    for k in (0, 10, 20, 30, 39):
        a, N = pesos[k], contagens[k]
        k1, k2, k3, k4 = cumulantes(a, N)
        assimetria = k3 / k2 ** 1.5
        curtose = k4 / k2 ** 2
        pseudo = 100.0 * ALPHA * float(a.sum()) / k1
        print("%4d %7d %9.0f %12.6f %12.6f %10.4f %10.4f %10.2f"
              % (k, a.size, N.sum(), k1, math.sqrt(k2), assimetria, curtose,
                 pseudo))
        resumo.append({"bin": k, "termos": int(a.size),
                       "contagens_brutas": float(N.sum()),
                       "media": k1, "desvio": math.sqrt(k2),
                       "assimetria": assimetria, "excesso_de_curtose": curtose,
                       "peso_das_pseudo_contagens_por_cento": pseudo})

    print()
    print("=== quais familias fechadas reproduzem pi_e ===")
    print("referencia: %d sorteios da SOMA EXATA de Gammas, nao outra aproximacao"
          % args.amostras)
    print("%4s %12s %12s %12s %12s"
          % ("bin", "KS 2 cumul", "KS 3 cumul", "KS lognormal", "KS gaussiana"))
    familias = []
    for k in (0, 10, 20, 30, 39):
        a, N = pesos[k], contagens[k]
        k1, k2, k3, _ = cumulantes(a, N)
        amostra = np.sort(amostra_exata(a, N, args.amostras, rng))
        emp = np.arange(1, amostra.size + 1) / amostra.size

        def ks_de(cdf):
            return float(np.abs(cdf(amostra) - emp).max())

        from math import erf, sqrt as _sqrt
        import scipy.special as sp  # noqa: E402  só para a CDF da Gamma
        f2, e2, d2 = gama_de_dois_cumulantes(k1, k2)
        f3, e3, d3 = gama_de_tres_cumulantes(k1, k2, k3)
        ks2 = ks_de(lambda x: sp.gammainc(f2, np.maximum(x - d2, 0) / e2))
        ks3 = ks_de(lambda x: sp.gammainc(f3, np.maximum(x - d3, 0) / e3))
        s2 = math.log1p(k2 / k1 ** 2)
        mu = math.log(k1) - 0.5 * s2
        ks_ln = ks_de(lambda x: 0.5 * (1.0 + sp.erf(
            (np.log(np.maximum(x, 1e-300)) - mu) / math.sqrt(2.0 * s2))))
        ks_g = ks_de(lambda x: 0.5 * (1.0 + sp.erf(
            (x - k1) / math.sqrt(2.0 * k2))))
        print("%4d %12.5f %12.5f %12.5f %12.5f" % (k, ks2, ks3, ks_ln, ks_g))
        familias.append({"bin": k, "ks_dois_cumulantes": ks2,
                         "ks_tres_cumulantes": ks3, "ks_lognormal": ks_ln,
                         "ks_gaussiana": ks_g,
                         "deslocamento_da_3cum": d3,
                         "deslocamento_negativo": bool(d3 < 0)})

    # ---- sensibilidade a ESCOLHA da priori, que nao e detalhe ----
    print()
    print("=== quanto a priori por canal injeta, e a alternativa por bin ===")
    print("alpha pseudo-contagens POR CANAL, e sao 65 a 91 canais por bin: no "
          "bin 39 isso e 28 por cento da media. Colocar a priori no RATE DO BIN,")
    print("uma por observacao, injeta 13 pseudo-contagens em vez de 91.")
    print("%4s %10s %10s %10s %10s %10s"
          % ("bin", "a=1 canal", "a=.5 canal", "a=1 bin", "a=.5 bin", "sem a"))
    priori = []
    for k in (0, 10, 20, 30, 39):
        a, N = pesos[k], contagens[k]
        linha = {}
        for nome, al in (("canal_alpha_1", 1.0), ("canal_alpha_meio", 0.5),
                         ("sem_alpha", 0.0)):
            linha[nome] = cumulantes(a, N, al)[0]
        # Priori por BIN: agrupa os canais de cada observacao num unico termo,
        # com peso igual ao peso medio ponderado pela contagem e forma igual a
        # contagem somada daquela observacao no bin. A hipotese e a mesma que o
        # estimador ja faz, densidade uniforme dentro do canal, de modo que o
        # que carrega informacao e o TOTAL do bin.
        porobs = {}
        for aj, nj, (obsid, _canal) in zip(a, N, rotulos[k]):
            soma_a, soma_n, soma_an = porobs.get(obsid, (0.0, 0.0, 0.0))
            porobs[obsid] = (soma_a + aj, soma_n + nj, soma_an + aj * nj)
        a_bin, n_bin = [], []
        for obsid, (soma_a, soma_n, soma_an) in porobs.items():
            # peso efetivo: o que reproduz a media quando multiplicado por soma_n
            a_bin.append(soma_an / soma_n if soma_n > 0 else soma_a / max(1, 1))
            n_bin.append(soma_n)
        a_bin, n_bin = np.array(a_bin), np.array(n_bin)
        for nome, al in (("bin_alpha_1", 1.0), ("bin_alpha_meio", 0.5)):
            linha[nome] = cumulantes(a_bin, n_bin, al)[0]
        print("%4d %10.6f %10.6f %10.6f %10.6f %10.6f"
              % (k, linha["canal_alpha_1"], linha["canal_alpha_meio"],
                 linha["bin_alpha_1"], linha["bin_alpha_meio"],
                 linha["sem_alpha"]))
        linha["bin"] = k
        linha["termos_por_canal"] = int(a.size)
        linha["termos_por_bin"] = int(a_bin.size)
        linha["vies_relativo_canal_alpha_1"] = (
            linha["canal_alpha_1"] / linha["sem_alpha"] - 1.0
            if linha["sem_alpha"] > 0 else None)
        linha["vies_relativo_bin_alpha_1"] = (
            linha["bin_alpha_1"] / linha["sem_alpha"] - 1.0
            if linha["sem_alpha"] > 0 else None)
        priori.append(linha)

    # ---- a integral, por tres rotas independentes ----
    print()
    print("=== a integral por canal, tres rotas independentes ===")
    print("banco de teste: S[p] casado ao DADO, S = n/(Delta t) - B, e t[p] = "
          "exposicao/32. Valida a maquinaria de integracao, nao um vetor fisico.")
    expos_total = float(np.asarray(d["exposures"], float).sum())
    t_p = np.full(N_P, expos_total / N_P)
    passo_p = LARGURA * t_p
    print("%4s %16s %16s %16s %12s %12s"
          % ("bin", "Gauss-Legendre", "Riemann fino", "MC exato", "GL-Riemann",
             "GL-MC"))
    rotas = []
    for k in (0, 10, 20, 30, 39):
        a, N = pesos[k], contagens[k]
        k1, k2, k3, _ = cumulantes(a, N)
        n_p = observado[:, k]
        S_p = np.maximum(n_p / passo_p - guardado[k], 0.0)
        pico = float(log_verossimilhanca_do_canal(k1, S_p, n_p, passo_p)[0])
        gl, norma = marginal_do_canal(S_p, n_p, passo_p, k1, k2, k3, nos=64,
                                      ref_log=pico)
        gl = gl - math.log(norma)
        # Riemann muito fino sobre a MESMA pi, para isolar o erro de quadratura
        sigma = math.sqrt(k2)
        lo, hi = max(0.0, k1 - 8 * sigma), k1 + 12 * sigma
        bb = np.linspace(lo, hi, 200001)
        f3, e3, d3 = gama_de_tres_cumulantes(k1, k2, k3)
        pi = densidade_gama(bb, f3, e3, d3)
        ll = log_verossimilhanca_do_canal(bb, S_p, n_p, passo_p)
        peso = (hi - lo) / (bb.size - 1)
        ri = pico + math.log(float((pi * np.exp(ll - pico)).sum() * peso)) \
            - math.log(float(pi.sum() * peso))
        # Monte Carlo sobre a distribuicao EXATA: mede quadratura E familia juntas
        draws = amostra_exata(a, N, args.amostras, rng)
        llm = log_verossimilhanca_do_canal(draws, S_p, n_p, passo_p)
        mc = pico + math.log(float(np.exp(llm - pico).mean()))
        print("%4d %16.8f %16.8f %16.8f %12.2e %12.2e"
              % (k, gl, ri, mc, abs(gl - ri), abs(gl - mc)))
        rotas.append({"bin": k, "gauss_legendre": gl, "riemann_fino": ri,
                      "monte_carlo_exato": mc,
                      "erro_de_quadratura": abs(gl - ri),
                      "erro_total_incluindo_familia": abs(gl - mc),
                      "sem_marginalizar": float(
                          log_verossimilhanca_do_canal(guardado[k], S_p, n_p,
                                                       passo_p)[0]),
                      "desvio_de_monte_carlo": float(
                          np.exp(llm - pico).std(ddof=1)
                          / math.sqrt(llm.size) / np.exp(llm - pico).mean())})
    total_marg = sum(r["gauss_legendre"] for r in rotas)
    total_fixo = sum(r["sem_marginalizar"] for r in rotas)
    print("  nos cinco canais medidos: marginalizado %.6f contra fixo %.6f, "
          "diferenca %.6f" % (total_marg, total_fixo, total_marg - total_fixo))
    print("  o erro de quadratura e %.1e e o erro total, que inclui a familia "
          "fechada, e %.1e" % (max(r["erro_de_quadratura"] for r in rotas),
                               max(r["erro_total_incluindo_familia"] for r in rotas)))

    saida = {
        "o_que_este_arquivo_faz": (
            "protótipo da marginalização do fundo com UMA integral por canal de "
            "energia sobre a verossimilhança conjunta das 32 fases"),
        "correcao_do_revisor": (
            "a diagonalidade separa as integrais ENTRE canais; DENTRO de um "
            "canal o mesmo B[e] entra nos 32 bins de fase e tem de ser "
            "integrado uma vez só. Integrar célula por célula criaria 32 fundos "
            "independentes artificiais."),
        "distribuicao": {
            "forma": "b_e = soma_j a_j lambda_j, com lambda_j | N_j ~ Gamma(N_j + alpha, 1)",
            "nao_e_gamma": "soma ponderada de Gammas com escalas diferentes",
            "alpha": ALPHA,
            "cumulantes_fechados": "k1=soma a f, k2=soma a^2 f, k3=2 soma a^3 f, k4=6 soma a^4 f",
            "por_bin": resumo},
        "familias_testadas": familias,
        "piso_de_ruido_do_KS": 1.36 / math.sqrt(args.amostras),
        "escolha_da_priori": {
            "problema": (
                "alpha injeta pseudo-contagens POR CANAL e sao 65 a 91 canais "
                "por bin; no bin 39 isso e 28 por cento da media"),
            "alternativa": (
                "colocar a priori no rate do BIN, uma por observacao, injeta 13 "
                "pseudo-contagens, e usa a mesma hipotese que o estimador ja "
                "faz, densidade uniforme dentro do canal"),
            "por_bin": priori,
            "decisao": "PENDENTE do revisor; o script mede, nao escolhe"},
        "tres_rotas": {
            "banco_de_teste": (
                "S[p] casado ao dado, S = n/(Delta t) - B, e t[p] = exposicao/32. "
                "Valida a maquinaria, nao um vetor fisico."),
            "por_bin": rotas,
            "leitura": (
                "Gauss-Legendre com 64 nos contra Riemann de 200001 pontos mede "
                "o erro de QUADRATURA; contra Monte Carlo sobre a soma exata de "
                "Gammas mede quadratura MAIS erro da familia fechada")},
        "estimador_usado": "por área normalizado pela cobertura, de scripts/estimador_do_fundo.py",
    }
    Path(args.saida).write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n")
    print()
    print("gravado em %s" % args.saida)


if __name__ == "__main__":
    main()
