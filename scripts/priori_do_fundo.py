#!/usr/bin/env python3
"""A priori do fundo latente: coerência sob reagrupamento em energia.

O QUE O REVISOR EXIGIU, em 06/10/2026, e que este arquivo responde:

    Não escolheria alpha=1 por canal nem por bin com base no tamanho do
    deslocamento. A priori precisa corresponder a uma grandeza física e ser
    definida numa representação explícita. Distribuições atribuídas
    independentemente a cada canal ou bin geralmente não preservam o mesmo prior
    ao reagrupar a energia; portanto, as duas opções não são apenas
    implementações equivalentes.

Isto não é observação de estilo: é um defeito verificável, e o script o mede.

O FUNDO LATENTE, definido explicitamente. Para a observação o e o canal j, o
espectro da REGIÃO DE FUNDO conta `N_oj` fótons, com

    N_oj ~ Poisson( Lambda_o(I_j) )

onde `I_j` é o intervalo de energia do canal e `Lambda_o` é a MEDIDA de contagem
esperada na região de fundo daquela observação. A grandeza que o alvo usa é a
densidade de taxa escalada para a região da fonte,

    b(E) = soma_o (tau_o / T) * s_o * dLambda_o/dE / tau_o
    com s_o = BACKSCAL_fonte / BACKSCAL_fundo  e  tau_o a exposição,

e o que entra na verossimilhança é a MÉDIA de b sobre o bin de energia, não o
valor dela no centro (ver scripts/estimador_do_fundo.py). Exposição e escala de
área ficam portanto explícitas, e a incógnita é uma MEDIDA, não um vetor de
números soltos.

O DEFEITO. Atribuir Gamma(alpha, 1) independente a cada canal é uma afirmação
sobre a GRADE, não sobre o fundo: a priori implicada para o total de um intervalo
é Gamma(n_canais * alpha, 1), que depende de em quantos canais o intervalo foi
dividido. Reagrupar a energia muda a priori sem que nada físico tenha mudado. O
script mede o tamanho disso em três reagrupamentos do MESMO intervalo.

A FAMÍLIA COERENTE. Se a forma é proporcional à LARGURA,

    alpha_j = a0 * w_j ,   a0 em pseudo-contagens por keV,

então somar canais adjacentes soma as formas e dá `a0 * largura do grupo`: a
afirmação a priori sobre um intervalo depende só do intervalo. É o processo Gama,
uma medida aleatória completamente aditiva, e a posterior segue na mesma família,
com forma `soma de N + a0 * largura`. Com isso a discrepância entre "por canal" e
"por bin" que eu havia medido desaparece POR CONSTRUÇÃO, e não por escolha.

O QUE CONTINUA PENDENTE, e o script não decide: o valor de `a0`. Ele é uma
afirmação física sobre quantas contagens de fundo por keV se admite antes de
olhar o dado, e nenhuma medida deste arquivo o justifica. As variantes entram
como SENSIBILIDADE declarada, como o revisor pediu.

Uso: python3 scripts/priori_do_fundo.py [--saida docs/priori_do_fundo.json]
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
#: Agrupamentos do MESMO intervalo, para medir a dependência de representação.
#: 30 canais nativos agrupados de 1, 2, 5, 10 e 30 em 30: todos exatos, porque
#: 30 é divisível por todos eles e o intervalo é união de canais inteiros.
AGRUPAMENTOS = (1, 2, 3, 5, 6, 10, 15, 30)
N_CANAIS = 30


def canais_da_observacao(obsid):
    for cand in (PRODUTOS / obsid, *PRODUTOS.glob("*/" + obsid)):
        if (cand / "bkg_spec.fits").is_file():
            dirobs = cand
            break
    else:
        raise SystemExit("sem bkg_spec.fits para %s" % obsid)
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
    cen = 0.5 * (elo + ehi)
    sel = (cen >= 0.150) & (cen <= 1.200) & (ehi > elo)
    cl, elo, ehi = cl[sel], elo[sel], ehi[sel]
    N = np.array([conta.get(int(c), 0.0) for c in cl])
    return elo, ehi, N, escala, exposicao


def posterior_do_intervalo(elo, ehi, N, inicio, n_canais, grupo, a0, alpha_fixo):
    """Forma e escala da posterior do TOTAL de contagens no intervalo.

    Agrupa `grupo` canais por termo, trata cada grupo como uma observação de
    Poisson do total daquele grupo, e soma as Gammas. As escalas são todas 1, em
    unidade de contagem, logo a soma é Gamma exata com formas somadas: é
    exatamente aí que a coerência, ou a falta dela, aparece.

    `alpha_fixo` é a priori por GRUPO, independente da largura, que é a opção que
    não reagrupa. `a0` é a priori proporcional à largura, em pseudo-contagens por
    keV.
    """
    fim = inicio + n_canais
    bordas_lo = elo[inicio:fim]
    bordas_hi = ehi[inicio:fim]
    contagens = N[inicio:fim]
    forma = 0.0
    for k in range(0, n_canais, grupo):
        largura = float(bordas_hi[k + grupo - 1] - bordas_lo[k])
        soma = float(contagens[k:k + grupo].sum())
        forma += soma + (alpha_fixo if alpha_fixo is not None else a0 * largura)
    return forma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(RAIZ / "docs/priori_do_fundo.json"))
    args = ap.parse_args()

    d = np.load(COMBINADO, allow_pickle=True)
    obsids = [str(x) for x in d["obsids"]]
    obsid = obsids[0]
    elo, ehi, N, escala, exposicao = canais_da_observacao(obsid)
    # Intervalo de 30 canais no meio da banda, união de canais inteiros.
    inicio = int(np.searchsorted(0.5 * (elo + ehi), 0.60))
    a, b = float(elo[inicio]), float(ehi[inicio + N_CANAIS - 1])
    largura = b - a
    contagens = float(N[inicio:inicio + N_CANAIS].sum())
    print("observacao %s, intervalo de %d canais: %.6f a %.6f keV, largura "
          "%.6f keV, %d contagens brutas"
          % (obsid, N_CANAIS, a, b, largura, int(contagens)))
    print("largura media do canal: %.6f keV" % (largura / N_CANAIS))
    print()

    print("=== o defeito: priori por GRUPO, independente da largura ===")
    print("a mesma afirmacao a priori, aplicada a grades diferentes do MESMO")
    print("intervalo fisico, da posteriores diferentes")
    print("%8s %8s %14s %14s %14s %12s"
          % ("agrupa", "grupos", "forma a=1", "media a=1", "desvio a=1", "vies %"))
    por_grupo = {}
    sem_priori = posterior_do_intervalo(elo, ehi, N, inicio, N_CANAIS, 1, 0.0, 0.0)
    for grupo in AGRUPAMENTOS:
        forma = posterior_do_intervalo(elo, ehi, N, inicio, N_CANAIS, grupo,
                                       0.0, 1.0)
        vies = 100.0 * (forma / sem_priori - 1.0)
        print("%8d %8d %14.2f %14.2f %14.3f %12.2f"
              % (grupo, N_CANAIS // grupo, forma, forma, math.sqrt(forma), vies))
        por_grupo[str(grupo)] = {"grupos": N_CANAIS // grupo, "forma": forma,
                                 "media": forma, "desvio": math.sqrt(forma),
                                 "vies_relativo_por_cento": vies}
    amplitude = (max(v["forma"] for v in por_grupo.values())
                 - min(v["forma"] for v in por_grupo.values()))
    print("  amplitude da forma entre agrupamentos: %.2f contagens, ou %.2f%% da "
          "contagem bruta" % (amplitude, 100.0 * amplitude / contagens))
    print("  a priori nao e uma afirmacao sobre o fundo: e uma afirmacao sobre a "
          "grade")
    print()

    print("=== a familia coerente: forma proporcional a LARGURA ===")
    print("alpha_j = a0 * w_j, somar canais soma as formas e da a0 * largura do "
          "grupo")
    print("%8s %8s %14s %14s %12s"
          % ("agrupa", "grupos", "forma a0=100", "desvio", "vies %"))
    a0 = 100.0           # pseudo-contagens por keV, VARIANTE e nao escolha
    coerente = {}
    for grupo in AGRUPAMENTOS:
        forma = posterior_do_intervalo(elo, ehi, N, inicio, N_CANAIS, grupo,
                                       a0, None)
        vies = 100.0 * (forma / sem_priori - 1.0)
        print("%8d %8d %14.6f %14.3f %12.4f"
              % (grupo, N_CANAIS // grupo, forma, math.sqrt(forma), vies))
        coerente[str(grupo)] = {"grupos": N_CANAIS // grupo, "forma": forma,
                                "vies_relativo_por_cento": vies}
    formas = [v["forma"] for v in coerente.values()]
    print("  amplitude da forma entre agrupamentos: %.3e contagens"
          % (max(formas) - min(formas)))
    print("  IDENTICA em todos os agrupamentos, a menos de arredondamento: a "
          "afirmacao depende so do intervalo")
    print()

    print("=== sensibilidade a a0, que NAO esta justificado ===")
    print("a0 em pseudo-contagens por keV; o intervalo tem %.6f keV e %d "
          "contagens" % (largura, int(contagens)))
    print("%12s %14s %14s %12s" % ("a0", "pseudo-cont.", "forma", "vies %"))
    sens = {}
    for valor in (0.0, 1.0, 10.0, 100.0, 1000.0):
        forma = posterior_do_intervalo(elo, ehi, N, inicio, N_CANAIS, 1,
                                       valor, None)
        pseudo = valor * largura
        vies = 100.0 * (forma / sem_priori - 1.0) if sem_priori > 0 else float('nan')
        print("%12.1f %14.4f %14.4f %12.4f" % (valor, pseudo, forma, vies))
        sens[str(valor)] = {"pseudo_contagens_no_intervalo": pseudo,
                            "forma": forma, "vies_relativo_por_cento": vies}
    print("  a0 = 0 nao injeta nada e e proprio enquanto o intervalo tiver "
          "contagem; a0 = 1000 injeta %.1f contagens contra %d medidas"
          % (1000.0 * largura, int(contagens)))
    print()

    print("=== e a mesma conta nos 40 bins do alvo, com as 13 observacoes ===")
    print("quanto a priori injeta no bin, por canal contra proporcional a largura")
    E_MIN, E_MAX, N_E = 0.15, 1.2, 40
    BORDAS = np.linspace(E_MIN, E_MAX, N_E + 1)
    LARG = (E_MAX - E_MIN) / N_E
    expos = np.asarray(d["exposures"], float)
    T = float(expos.sum())
    print("%4s %10s %12s %12s %12s"
          % ("bin", "canais", "pseudo a=1", "pseudo a0=100", "contagens"))
    nos_bins = []
    for k in (0, 10, 20, 30, 39):
        n_can, pseudo_fixo, pseudo_coer, brutas = 0, 0.0, 0.0, 0.0
        for obsid_i, exp in zip(obsids, expos):
            e_lo, e_hi, N_i, esc, tau = canais_da_observacao(obsid_i)
            ov = np.clip(np.minimum(e_hi, BORDAS[k + 1])
                         - np.maximum(e_lo, BORDAS[k]), 0.0, None)
            quais = ov > 0
            n_can += int(quais.sum())
            pseudo_fixo += float(quais.sum()) * 1.0
            pseudo_coer += a0 * float(ov[quais].sum())
            brutas += float(N_i[quais].sum())
        print("%4d %10d %12.2f %12.4f %12.0f"
              % (k, n_can, pseudo_fixo, pseudo_coer, brutas))
        nos_bins.append({"bin": k, "canais": n_can,
                         "pseudo_contagens_alpha_1_por_canal": pseudo_fixo,
                         "pseudo_contagens_a0_100_por_keV": pseudo_coer,
                         "contagens_brutas": brutas,
                         "razao_entre_as_duas": pseudo_fixo / max(pseudo_coer, 1e-300)})
    print("  a priori por canal injeta %0.1f vezes mais que a proporcional a "
          "largura com a0 = 100, e a razao DEPENDE do numero de canais"
          % (nos_bins[0]["razao_entre_as_duas"]))

    saida = {
        "exigencia_do_revisor": (
            "a priori precisa corresponder a uma grandeza fisica e ser definida "
            "numa representacao explicita; distribuicoes independentes por canal "
            "ou por bin geralmente nao preservam o mesmo prior ao reagrupar a "
            "energia"),
        "fundo_latente": {
            "modelo": "N_oj ~ Poisson(Lambda_o(I_j)), com Lambda_o a medida de "
                      "contagem esperada na regiao de fundo da observacao o",
            "exposicao_explicita": "tau_o, do cabecalho EXPOSURE de bkg_spec.fits",
            "escala_de_area_explicita": "s_o = BACKSCAL_fonte / BACKSCAL_fundo",
            "grandeza_do_alvo": "media da densidade de taxa escalada sobre o bin "
                                "de energia, nao o valor no centro",
            "incognita": "uma MEDIDA em energia, nao um vetor de numeros soltos"},
        "intervalo_de_teste": {
            "observacao": obsid, "canais": N_CANAIS,
            "de_keV": a, "ate_keV": b, "largura_keV": largura,
            "contagens_brutas": contagens,
            "por_que_este": "uniao de canais inteiros e divisivel por todos os "
                            "agrupamentos testados, logo os tres sao exatos"},
        "defeito_medido": {
            "o_que": "priori por GRUPO, independente da largura, aplicada a "
                     "agrupamentos diferentes do MESMO intervalo",
            "por_agrupamento": por_grupo,
            "amplitude_da_forma": amplitude,
            "amplitude_relativa_a_contagem_por_cento":
                100.0 * amplitude / contagens,
            "leitura": "a priori nao e afirmacao sobre o fundo: e afirmacao "
                       "sobre a grade"},
        "familia_coerente": {
            "forma": "alpha_j = a0 * w_j, com a0 em pseudo-contagens por keV",
            "nome": "processo Gama: medida aleatoria aditiva, cuja posterior "
                    "segue na mesma familia com forma soma de N + a0 * largura",
            "por_agrupamento": coerente,
            "amplitude_da_forma": max(formas) - min(formas),
            "consequencia": "a discrepancia entre por canal e por bin que eu "
                            "havia medido desaparece POR CONSTRUCAO"},
        "a0_PENDENTE_DE_JUSTIFICATIVA": {
            "o_que_e": "quantas contagens de fundo por keV se admite antes de "
                       "olhar o dado",
            "nada_aqui_o_justifica": True,
            "entra_como": "SENSIBILIDADE declarada",
            "sensibilidade": sens,
            "limite_a0_zero": "nao injeta nada e e proprio enquanto o intervalo "
                              "tiver contagem"},
        "nos_40_bins_do_alvo": nos_bins,
    }
    Path(args.saida).write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n")
    print()
    print("gravado em %s" % args.saida)


if __name__ == "__main__":
    main()
