#!/usr/bin/env python3
"""Propaga a incerteza das contagens brutas do fundo pela interpolacao e pela
coadicao, com a matriz de covariancia entre os 40 canais de energia.

POR QUE A MATRIZ, e nao 40 barras independentes: o vetor de fundo que o ajuste
usa nao e uma medida por canal. Ele e

    B_k = soma_o (t_o / T) soma_i W_ki^o c_i^o N_i^o

onde N_i^o sao as CONTAGENS BRUTAS do espectro de fundo da observacao o no canal
i, c_i^o = escala_o / (exposicao_o * largura_i) e a receita OGIP, e W^o e a
interpolacao linear de np.interp dos centros de canal para os 40 centros de
energia. Duas operacoes misturam canais: a interpolacao, porque cada B_k vem de
uma combinacao de canais, e a coadicao, porque as 13 observacoes entram todas em
todos os 40 canais. Logo os 40 numeros sao correlacionados, e reportar desvios
por canal sem a matriz descreveria uma incerteza que nao existe.

AS CONTAGENS BRUTAS vem dos produtos preservados, nao de inversao da tabela: de
`bkg_spec.fits` de cada observacao, coluna COUNTS, com BACKSCAL e EXPOSURE dos
cabecalhos, o BACKSCAL da fonte de `src_spec.fits` e as bordas de canal da
extensao EBOUNDS de `src.rmf`. O script CONFERE que a receita aplicada a essas
contagens reproduz o CSV que o co-add leu, o que fecha a cadeia dos dados ate o
produto do SAS.

O QUE ELE NAO FAZ, e isto importa: calcular a covariancia NAO e marginalizar o
fundo. O alvo estatistico de hoje trata B como constante conhecida. Entrar com a
incerteza exige decidir COMO ela entra, e a secao final do relatorio enumera as
opcoes com seus custos, sem escolher por conta propria.

Uso: python3 scripts/incerteza_fundo.py [--saida docs/incerteza_fundo.json]
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
PULSARIS = Path(os.environ.get("MAGNUS_PULSARIS", Path.home() / "Codes/PULSARIS"))
sys.path.insert(0, str(PULSARIS / "scripts"))
import build_instrument_profiles as B  # noqa: E402  parser FITS sem astropy

PRODUTOS = Path(os.environ.get("MAGNUS_PRODUTOS",
                               Path.home() / "Codes/XMM-NEWTON/products"))
COMBINADO = RAIZ / "exploracoes/dados_combinados_13obs_2026-09-12/combinado13.npz"
E_MIN, E_MAX, N_E = 0.15, 1.2, 40
BORDAS = np.linspace(E_MIN, E_MAX, N_E + 1)
ECEN = 0.5 * (BORDAS[:-1] + BORDAS[1:])
LARGURA = (E_MAX - E_MIN) / N_E
#: Banda do filtro aplicado pelo exportador do XREDUX, em keV.
BANDA = (0.150, 1.200)


def acha_obs(obsid):
    for p in (PRODUTOS / obsid, *PRODUTOS.glob("*/" + obsid)):
        if (p / "bkg_spec.fits").is_file():
            return p
    raise SystemExit("sem bkg_spec.fits para %s" % obsid)


def acha_csv(diretorio):
    p = sorted(diretorio.glob("pulsaris/*_background.csv"))
    return p[0] if p else None


def le_canais(caminho):
    cab, dados = B.extension(caminho, "SPECTRUM")
    linhas = list(B.table_rows(cab, dados))
    canal = np.array([r["CHANNEL"] for r in linhas], dtype=int)
    contagem = np.array([r["COUNTS"] for r in linhas], dtype=float)
    return cab, canal, contagem


def bordas_de_canal(rmf):
    cab, dados = B.extension(rmf, "EBOUNDS")
    linhas = list(B.table_rows(cab, dados))
    return {int(r["CHANNEL"]): (float(r["E_MIN"]), float(r["E_MAX"]))
            for r in linhas}


def matriz_interp(origem, destino):
    """Matriz W com destino = W @ valores(origem), identica a np.interp.

    Cada linha tem no maximo dois nao-nulos, os canais que cercam o centro de
    energia pedido. Fora da faixa np.interp SEGURA o valor da ponta, o que a
    linha reproduz com um unico nao-nulo igual a 1; `coadicionar.read_background`
    hoje levanta erro nesse caso, e aqui so registramos se aconteceria.
    """
    W = np.zeros((len(destino), len(origem)))
    extrapola = 0
    for k, x in enumerate(destino):
        if x <= origem[0]:
            W[k, 0] = 1.0
            extrapola += int(x < origem[0])
            continue
        if x >= origem[-1]:
            W[k, -1] = 1.0
            extrapola += int(x > origem[-1])
            continue
        j = int(np.searchsorted(origem, x)) - 1
        passo = origem[j + 1] - origem[j]
        f = (x - origem[j]) / passo
        W[k, j] = 1.0 - f
        W[k, j + 1] = f
    return W, extrapola


def uma_observacao(obsid, exposicao_usada):
    """Contagens brutas, receita conferida e a matriz A com b = A @ N.

    OS NOS DA INTERPOLACAO SAO OS DO CSV, nao os canais que eu escolheria: o CSV
    e o arquivo que `coadicionar.read_background` leu, e `np.interp` usa as
    energias dele como nos. Casar cada no ao canal do FITS, em vez de refiltrar
    o FITS por conta propria, e o que garante que a matriz seja a da cadeia real.
    Isto importa porque as 13 exportacoes nao sao uniformes: doze CSVs vem
    cortados em 0,150 a 1,200 keV e o de 2019 vem com a banda inteira ate 12 keV,
    com 2372 linhas. Para os 40 centros, todos interiores nas duas versoes, os
    pesos sao os mesmos; a diferenca apareceu aqui como numero de linhas.
    """
    dir_obs = acha_obs(obsid)
    cab_b, canal, contagem = le_canais(dir_obs / "bkg_spec.fits")
    cab_s, _ = B.extension(dir_obs / "src_spec.fits", "SPECTRUM")
    bscal_fundo = float(cab_b["BACKSCAL"])
    bscal_fonte = float(cab_s["BACKSCAL"])
    exposicao = float(cab_b["EXPOSURE"])
    escala = bscal_fonte / bscal_fundo
    bordas = bordas_de_canal(dir_obs / "src.rmf")

    csv = acha_csv(dir_obs)
    if csv is None:
        raise SystemExit("sem CSV de fundo em %s: a cadeia nao pode ser fechada"
                         % dir_obs)
    tabela = np.loadtxt(csv, delimiter=",", comments="#")
    nos = tabela[:, 0]
    taxa_csv = tabela[:, 1]

    # Canal de cada no, pelo centro de energia da EBOUNDS.
    centro_de = {}
    for c in canal.tolist():
        lim = bordas.get(int(c))
        if lim is None or lim[1] <= lim[0]:
            continue
        centro_de[int(c)] = (0.5 * (lim[0] + lim[1]), lim[1] - lim[0])
    canais = np.array(sorted(centro_de))
    centros = np.array([centro_de[c][0] for c in canais])
    larguras = np.array([centro_de[c][1] for c in canais])
    conta = dict(zip(canal.tolist(), contagem.tolist()))

    idx = np.searchsorted(centros, nos)
    idx = np.clip(idx, 1, centros.size - 1)
    esq = np.abs(nos - centros[idx - 1]) < np.abs(nos - centros[idx])
    idx = np.where(esq, idx - 1, idx)
    desencontro = float(np.abs(nos - centros[idx]).max())

    bruto = np.array([conta[int(canais[i])] for i in idx], dtype=float)
    coef = escala / (exposicao * larguras[idx])
    taxa = coef * bruto
    rel = np.abs(taxa - taxa_csv) / np.maximum(np.abs(taxa_csv), 1e-300)

    W, extrapola = matriz_interp(nos, ECEN)
    A = W * coef                     # b_k = sum_i A_ki N_i

    # Estimador alternativo, para medir o preco da interpolacao: REBIN POR SOMA,
    # que usa TODOS os canais caidos dentro do bin de energia,
    #     taxa_k = escala * (soma de N_i no bin) / (exposicao * soma das larguras)
    # com a mesma receita OGIP e a mesma unidade. Ele NAO e adotado: trocar o
    # estimador mudaria o vetor de dados e invalidaria os ajustes feitos.
    A_reb = np.zeros((N_E, nos.size))
    indice = np.clip(((nos - E_MIN) / LARGURA).astype(int), 0, N_E - 1)
    dentro = (nos >= E_MIN) & (nos < E_MAX)
    for k in range(N_E):
        sel = dentro & (indice == k)
        if not sel.any():
            continue
        A_reb[k, sel] = escala / (exposicao * larguras[idx][sel].sum())

    na_banda = dentro
    registro = {
        "obsid": obsid,
        "diretorio": str(dir_obs),
        "csv_lido_pelo_coadd": str(csv),
        "sha256_do_csv": hashlib.sha256(csv.read_bytes()).hexdigest()[:16],
        "linhas_do_csv": int(nos.size),
        "faixa_do_csv_keV": [float(nos.min()), float(nos.max())],
        "backscal_fonte": bscal_fonte, "backscal_fundo": bscal_fundo,
        "escala": escala,
        "exposicao_do_espectro_s": exposicao,
        "exposicao_no_npz_s": float(exposicao_usada),
        "maior_desencontro_de_energia_no_casamento_keV": desencontro,
        "maior_diferenca_relativa_da_taxa": float(rel.max()),
        "contagens_brutas_nos_nos": float(bruto.sum()),
        "contagens_brutas_na_grade_de_energia": float(bruto[na_banda].sum()),
        "nos_com_zero_contagens": int((bruto == 0).sum()),
        "nos_na_grade_de_energia": int(na_banda.sum()),
        "nos_efetivamente_usados_pela_interpolacao":
            int((np.abs(W).sum(axis=0) > 0).sum()),
        "extrapolaria": int(extrapola),
    }
    return A, A_reb, bruto, taxa, nos, registro


def cov_de_convencao(A, bruto, variancia):
    """Cov(b) = A diag(var) A^T para uma observacao."""
    return (A * variancia) @ A.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(RAIZ / "docs/incerteza_fundo.json"))
    a = ap.parse_args()

    d = np.load(COMBINADO, allow_pickle=True)
    obsids = [str(x) for x in d["obsids"]]
    expos = np.asarray(d["exposures"], dtype=float)
    guardado = np.asarray(d["background"], dtype=float)
    observado = np.asarray(d["observed"], dtype=float).reshape(32, N_E)
    T = float(expos.sum())

    print("%-12s %6s %7s %9s %6s %6s %7s %11s" %
          ("obsid", "linhas", "na grade", "contagens", "zeros", "usados",
           "extrap", "confere CSV"))
    As, Arebs, brutos, registros = [], [], [], []
    for obsid, exp in zip(obsids, expos):
        A, A_reb, bruto, taxa, centro, reg = uma_observacao(obsid, exp)
        As.append(A); Arebs.append(A_reb); brutos.append(bruto); registros.append(reg)
        print("%-12s %6d %7d %9.0f %6d %6d %7d %11.1e" %
              (obsid, reg["linhas_do_csv"], reg["nos_na_grade_de_energia"],
               reg["contagens_brutas_na_grade_de_energia"],
               reg["nos_com_zero_contagens"],
               reg["nos_efetivamente_usados_pela_interpolacao"],
               reg["extrapolaria"], reg["maior_diferenca_relativa_da_taxa"]))

    pesos = expos / T
    # O vetor reconstruido, para confirmar que a cadeia inteira bate.
    recon = sum(w * (A @ n) for w, A, n in zip(pesos, As, brutos))
    rel_recon = np.abs(recon - guardado) / np.maximum(np.abs(guardado), 1e-300)

    # Duas convencoes para a variancia de Poisson das contagens brutas. A
    # diferenca so aparece nos canais vazios, onde var = N daria zero, que e
    # subestimativa: com priori plana a posterior de lambda dado N tem media e
    # variancia N+1. Fiducial: N+1, por nao atribuir certeza a canal vazio.
    convencoes = {}
    for nome, f in (("N", lambda n: n), ("N+1", lambda n: n + 1.0)):
        cov = sum((w ** 2) * cov_de_convencao(A, n, f(n))
                  for w, A, n in zip(pesos, As, brutos))
        convencoes[nome] = cov
    cov = convencoes["N+1"]
    sigma = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sigma, sigma)
    fora = corr - np.diag(np.diag(corr))

    # Contagens de fundo previstas na coadicao inteira, por canal de energia, e a
    # incerteza propagada nessas contagens, contra o ruido de Poisson do dado.
    fundo_ct = guardado * LARGURA * T
    sigma_ct = sigma * LARGURA * T
    dado_ct = observado.sum(axis=0)
    poisson_dado = np.sqrt(np.maximum(dado_ct, 1.0))

    print()
    print("=== cadeia completa: contagens brutas -> vetor do NPZ ===")
    print("maior diferenca relativa por canal: %.3e | mediana %.3e"
          % (rel_recon.max(), float(np.median(rel_recon))))
    print("concordancia numerica, nao identidade binaria: os CSVs guardam"
          " %.9g por linha, e a reconstrucao parte das contagens inteiras")
    print()
    print("=== incerteza propagada, convencao var(N) = N+1 ===")
    print("%4s %8s %12s %10s %9s %12s %10s" %
          ("bin", "E keV", "fundo ct", "sigma ct", "sigma %", "dado ct",
           "sigma/sqrtN"))
    for k in range(N_E):
        print("%4d %8.4f %12.1f %10.2f %9.2f %12.0f %10.4f" %
              (k, ECEN[k], fundo_ct[k], sigma_ct[k],
               100.0 * sigma_ct[k] / max(fundo_ct[k], 1e-300),
               dado_ct[k], sigma_ct[k] / poisson_dado[k]))

    print()
    print("fundo total previsto %.1f ct | sigma do total %.1f ct (%.2f%%)"
          % (fundo_ct.sum(),
             math.sqrt(float(np.ones(N_E) @ cov @ np.ones(N_E))) * LARGURA * T,
             100.0 * math.sqrt(float(np.ones(N_E) @ cov @ np.ones(N_E)))
             * LARGURA * T / fundo_ct.sum()))
    print("dado total %.0f ct | fundo e %.2f%% do dado"
          % (dado_ct.sum(), 100.0 * fundo_ct.sum() / dado_ct.sum()))
    # A diagonalidade tem que ser DEMONSTRADA, nao inferida de "correlacao 0,0000":
    # dois bins so se correlacionam se compartilharem algum no, porque cada
    # observacao e independente e cada bin usa dois nos. Contamos os pares que
    # compartilham.
    partilha, vizinhos = 0, []
    for A in As:
        suporte = [set(np.nonzero(A[k])[0].tolist()) for k in range(N_E)]
        for k in range(N_E):
            for j in range(k + 1, N_E):
                if suporte[k] & suporte[j]:
                    partilha += 1
        for k in range(N_E - 1):
            if suporte[k] and suporte[k + 1]:
                vizinhos.append(min(suporte[k + 1]) - max(suporte[k]))
    print()
    print("=== por que a matriz sai diagonal ===")
    print("pares de bins que compartilham algum no: %d, somados as 13 "
          "observacoes" % partilha)
    print("distancia em nos entre bins consecutivos: minima %d, mediana %.0f"
          % (min(vizinhos), float(np.median(vizinhos))))
    print("a grade de energia tem passo %.5f keV e o canal tem %.5f keV, "
          "razao %.1f" % (LARGURA, 0.00533, LARGURA / 0.00533))
    print("os suportes so passariam a se tocar com mais de %d bins na banda"
          % int((E_MAX - E_MIN) / (2 * 0.00533)))
    print()
    print("=== estrutura da matriz ===")
    print("maior correlacao fora da diagonal: %.4f | mediana do |valor| %.2e"
          % (fora.max(), float(np.median(np.abs(fora)))))
    print("pares com |correlacao| > 0,1: %d de %d"
          % (int((np.abs(fora) > 0.1).sum() // 2), N_E * (N_E - 1) // 2))
    autov = np.linalg.eigvalsh(cov)[::-1]
    frac = autov / autov.sum()
    print("fracao da variancia nos 1, 3, 5 e 10 maiores autovalores: "
          "%.3f %.3f %.3f %.3f"
          % (frac[0], frac[:3].sum(), frac[:5].sum(), frac[:10].sum()))
    coerente = float(np.ones(N_E) @ cov @ np.ones(N_E)) / float(np.trace(cov) * N_E)
    print("parte COERENTE (um unico fator de escala) contra o total: %.4f" % coerente)

    print()
    print("=== quantos canais a interpolacao usa ===")
    for reg in registros:
        print("  %-12s %3d nos na grade de energia, %3d usados" %
              (reg["obsid"], reg["nos_na_grade_de_energia"],
               reg["nos_efetivamente_usados_pela_interpolacao"]))
    # Quanto a interpolacao custa em variancia contra um rebin que SOMA os canais
    # dentro de cada bin de energia.
    cov_reb = sum((w ** 2) * cov_de_convencao(A, n, n + 1.0)
                  for w, A, n in zip(pesos, Arebs, brutos))
    sigma_reb = np.sqrt(np.diag(cov_reb))
    media_reb = sum(w * (A @ n) for w, A, n in zip(pesos, Arebs, brutos))
    razao = sigma / np.maximum(sigma_reb, 1e-300)
    print("desvio da interpolacao contra o do rebin por soma, por bin: "
          "mediana %.2f, minimo %.2f, maximo %.2f"
          % (float(np.median(razao)), float(razao.min()), float(razao.max())))
    vies = (recon - media_reb) / np.maximum(media_reb, 1e-300)
    print("os dois estimadores diferem no VALOR por %.2f%% na mediana e %.2f%% "
          "no pior bin" % (100.0 * float(np.median(np.abs(vies))),
                           100.0 * float(np.abs(vies).max())))
    print("isto mede o preco do estimador escolhido, e NAO autoriza trocar o "
          "vetor de dados: a troca invalidaria todos os ajustes feitos")

    saida = {
        "receita": "rate = N * (BACKSCAL_fonte/BACKSCAL_fundo) / (exposicao * largura_do_canal)",
        "operacoes_que_misturam_canais": [
            "interpolacao linear de np.interp dos centros de canal para os 40 centros de energia",
            "media das 13 observacoes ponderada pela exposicao"],
        "convencao_de_variancia": "var(N) = N+1, posterior de Poisson com priori plana; "
                                  "var(N) = N tambem calculada, difere so nos canais vazios",
        "observacoes": registros,
        "reconstrucao_das_contagens_brutas": {
            "maior_diferenca_relativa": float(rel_recon.max()),
            "mediana_da_diferenca_relativa": float(np.median(rel_recon)),
            "nota": "concordancia numerica, nao identidade binaria"},
        "sigma_do_fundo_ct_por_s_por_keV": sigma.tolist(),
        "sigma_relativo_por_bin": (sigma / np.maximum(guardado, 1e-300)).tolist(),
        "covariancia_ct_por_s_por_keV_ao_quadrado": cov.tolist(),
        "covariancia_com_var_igual_a_N": convencoes["N"].tolist(),
        "diagonalidade": {
            "pares_de_bins_que_compartilham_no": int(partilha),
            "distancia_minima_em_nos_entre_bins_consecutivos": int(min(vizinhos)),
            "passo_da_grade_keV": LARGURA,
            "largura_tipica_do_canal_keV": 0.00533,
            "bins_na_banda_a_partir_dos_quais_os_suportes_se_tocam":
                int((E_MAX - E_MIN) / (2 * 0.00533)),
            "consequencia": "com 40 bins os dois nos de cada bin sao disjuntos "
                            "dos dos outros bins, e como as observacoes sao "
                            "independentes a covariancia propagada e DIAGONAL; "
                            "a correlacao que eu havia anunciado existe no "
                            "mecanismo e nao nesta grade"},
        "maior_correlacao_fora_da_diagonal": float(fora.max()),
        "mediana_do_modulo_fora_da_diagonal": float(np.median(np.abs(fora))),
        "pares_com_correlacao_acima_de_0_1": int((np.abs(fora) > 0.1).sum() // 2),
        "autovalores": autov.tolist(),
        "fracao_da_variancia_no_maior_autovalor": float(frac[0]),
        "parte_coerente_normalizada": coerente,
        "contagens": {
            "fundo_previsto_por_bin": fundo_ct.tolist(),
            "sigma_do_fundo_por_bin": sigma_ct.tolist(),
            "dado_por_bin": dado_ct.tolist(),
            "sigma_do_fundo_sobre_poisson_do_dado":
                (sigma_ct / poisson_dado).tolist(),
            "fundo_total": float(fundo_ct.sum()),
            "sigma_do_fundo_total": float(
                math.sqrt(float(np.ones(N_E) @ cov @ np.ones(N_E))) * LARGURA * T),
            "dado_total": float(dado_ct.sum())},
        "estimador_alternativo_rebin_por_soma": {
            "descricao": "soma as contagens de todos os canais caidos no bin, "
                         "em vez de interpolar entre os dois canais vizinhos "
                         "do centro",
            "razao_dos_desvios_interp_sobre_rebin": razao.tolist(),
            "mediana_da_razao": float(np.median(razao)),
            "diferenca_relativa_no_valor": vies.tolist(),
            "mediana_do_modulo_da_diferenca": float(np.median(np.abs(vies))),
            "nota": "medida do preco do estimador; trocar o estimador mudaria "
                    "o vetor de dados e invalidaria os ajustes feitos"},
        "limites_declarados": [
            "a covariancia cobre so a estatistica de contagem do espectro de "
            "fundo; nao cobre erro sistematico de BACKSCAL, variacao espacial "
            "do fundo entre a regiao da fonte e a do fundo, nem o fundo de "
            "particulas dependente do tempo",
            "a incerteza do BACKSCAL entra como fator comum por observacao e "
            "nao foi propagada: os produtos nao guardam uma estimativa dela",
            "calcular esta matriz NAO e marginalizar o fundo; o alvo atual "
            "trata B como constante conhecida"],
    }
    Path(a.saida).write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n")
    print()
    print("gravado em %s" % a.saida)


if __name__ == "__main__":
    main()
