#!/usr/bin/env python3
"""Qual estimador de fundo corresponde ao que a verossimilhanca pede?

A PERGUNTA, levantada pelo revisor em 06/10/2026 depois de medirmos 79,6 por
cento de diferenca entre a interpolacao e o rebin por soma no pior bin: a
diferenca nao prova que o rebin esteja certo. Ela exige descobrir qual dos dois
corresponde as contagens INTEGRADAS nos bins da verossimilhanca.

O QUE A VEROSSIMILHANCA PEDE. Em `FitProblem.score_document` o modelo e

    mu[p][e] = (S[p][e] + B[e]) * energy_width * tempo_representado[p]

e `energy_width` e a largura do bin de energia, 0,02625 keV. Multiplicar B[e] por
essa largura tem de dar a INTEGRAL da densidade de taxa sobre o bin:

    B[e] * Delta = integral de b(E) dE sobre o bin   =>   B[e] = media de b no bin

Logo o alvo define B[e] como a MEDIA da densidade no bin, nao o valor dela no
centro. Os dois coincidem para b linear e divergem na curvatura, por Euler-
Maclaurin, com erro de ordem Delta^2 b''/24.

O TESTE, com espectros sinteticos de integral CONHECIDA, na grade de canais real:
  constante     os dois exatos
  linear        os dois exatos (centro = media)
  exponencial   mede a ordem do erro de cada um
  curvatura     potencia, para separar ordem 2 de ordem maior
  linha estreita o caso que produz os 79,6 por cento
Em cada caso a referencia e a integral analitica sobre o bin, dividida pela
largura, e nao a opiniao de nenhum dos dois estimadores.

CONSERVACAO: soma de B[e]*Delta sobre os bins contra a integral analitica na
banda. Um estimador que nao conserva contagem total nao pode entrar num alvo
de Poisson.

UNIDADES: o CSV traz ct/s/keV ja escalado pela razao de BACKSCAL. As duas
estimativas ficam na mesma unidade; o teste confere isso numericamente, com um
espectro constante de valor conhecido.

Uso: python3 scripts/estimador_do_fundo.py [--saida docs/estimador_do_fundo.json]
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
E_MIN, E_MAX, N_E = 0.15, 1.2, 40
BORDAS = np.linspace(E_MIN, E_MAX, N_E + 1)
ECEN = 0.5 * (BORDAS[:-1] + BORDAS[1:])
LARGURA = (E_MAX - E_MIN) / N_E
#: Observacao de onde sai a grade de canais real, a mais longa das 13.
OBS = "0157360101"


def grade_de_canais():
    """Centros e larguras dos canais na banda, da EBOUNDS da RMF real."""
    cab, dados = B.extension(PRODUTOS / OBS / "src.rmf", "EBOUNDS")
    linhas = list(B.table_rows(cab, dados))
    lo = np.array([float(r["E_MIN"]) for r in linhas])
    hi = np.array([float(r["E_MAX"]) for r in linhas])
    centro = 0.5 * (lo + hi)
    dentro = (centro >= 0.150) & (centro <= 1.200) & (hi > lo)
    return centro[dentro], lo[dentro], hi[dentro]


#: Espectros sinteticos: densidade b(E) em ct/s/keV e a primitiva, para a
#: integral exata sobre qualquer intervalo. Nada de quadratura na referencia.
def espectros():
    def linha_gauss(E, centro=0.65, sigma=0.012, altura=1.0, base=0.02):
        return base + altura * np.exp(-0.5 * ((E - centro) / sigma) ** 2)

    def integral_linha(a, b, centro=0.65, sigma=0.012, altura=1.0, base=0.02):
        raiz2 = math.sqrt(2.0)
        erf = np.vectorize(math.erf)
        return (base * (b - a) + altura * sigma * math.sqrt(math.pi / 2.0)
                * (erf((b - centro) / (sigma * raiz2))
                   - erf((a - centro) / (sigma * raiz2))))

    return [
        ("constante", lambda E: np.full_like(np.asarray(E, float), 0.05),
         lambda a, b: 0.05 * (b - a)),
        ("linear", lambda E: 0.10 - 0.05 * E,
         lambda a, b: 0.10 * (b - a) - 0.025 * (b ** 2 - a ** 2)),
        ("exponencial lenta", lambda E: 0.12 * np.exp(-E / 0.8),
         lambda a, b: 0.12 * 0.8 * (np.exp(-a / 0.8) - np.exp(-b / 0.8))),
        ("exponencial rapida", lambda E: 0.12 * np.exp(-E / 0.15),
         lambda a, b: 0.12 * 0.15 * (np.exp(-a / 0.15) - np.exp(-b / 0.15))),
        ("potencia -2", lambda E: 0.01 * np.asarray(E, float) ** -2.0,
         lambda a, b: 0.01 * (1.0 / a - 1.0 / b)),
        ("linha estreita", linha_gauss, integral_linha),
    ]


def estimativas(centro, lo, hi, densidade):
    """Os TRES estimadores, a partir das contagens de canal SEM ruido.

    As contagens entram como `densidade integrada no canal x exposicao`, em vez
    de `densidade no centro x largura`, porque e isso que o detector mede: cada
    canal conta fotons, e a contagem e a integral. Usar o valor central aqui
    embutiria no dado o erro que o teste quer medir no estimador.

      interpolacao  np.interp nos centros dos bins, o que o co-add faz hoje
      rebin inteiro soma dos canais cujo CENTRO cai no bin, o alternativo que eu
                    havia proposto
      rebin por area soma dos canais ponderada pela FRACAO de cada canal que cai
                    dentro do bin, que e o unico dos tres que estima a media no
                    intervalo que o alvo nomeia

    O terceiro entrou depois de medir: a grade de canal, 0,00484 a 0,00534 keV,
    nao divide o bin de 0,02625 keV, de modo que os canais de um bin NAO cobrem
    o bin. O rebin inteiro portanto faz a media sobre um intervalo DESLOCADO, e
    erra 6e-4 ate num espectro linear, onde curvatura nao explica nada.
    """
    # taxa por canal, em ct/s/keV, como o CSV a traz: contagem / (exposicao *
    # largura). Com contagem = integral da densidade * exposicao, isto e a MEDIA
    # da densidade no canal.
    taxa_canal = np.array([densidade[1](a, b) / (b - a) for a, b in zip(lo, hi)])
    interp = np.interp(ECEN, centro, taxa_canal)

    rebin = np.zeros(N_E)
    indice = np.clip(((centro - E_MIN) / LARGURA).astype(int), 0, N_E - 1)
    dentro = (centro >= E_MIN) & (centro < E_MAX)
    for k in range(N_E):
        sel = dentro & (indice == k)
        if sel.any():
            peso = hi[sel] - lo[sel]
            rebin[k] = float((taxa_canal[sel] * peso).sum() / peso.sum())

    area = np.zeros(N_E)
    coberto = np.zeros(N_E)
    for k in range(N_E):
        a, b = BORDAS[k], BORDAS[k + 1]
        sobrepoe = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None)
        if sobrepoe.sum() > 0:
            # contagem = taxa * largura do canal; a fracao do canal dentro do bin
            # leva a fracao da contagem, sob a hipotese de densidade uniforme
            # DENTRO do canal, que e a unica informacao que o canal carrega
            area[k] = float((taxa_canal * sobrepoe).sum() / (b - a))
            coberto[k] = float(sobrepoe.sum() / (b - a))
    # Quarta variante: normalizar pela largura COBERTA em vez da largura do bin.
    # A grade de canal nao cobre o bin 0 inteiro, porque o primeiro canal comeca
    # em 0,150088 keV e o bin comeca em 0,15, o que deixa 0,34 por cento de
    # fresta. Dividir pelo coberto da a media sobre o que foi medido, sob a
    # hipotese de que a densidade na fresta e a mesma; dividir pela largura do
    # bin trata a fresta como taxa zero.
    area_coberta = np.where(coberto > 0, area / np.maximum(coberto, 1e-300), 0.0)
    return interp, rebin, area, area_coberta, coberto, taxa_canal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(RAIZ / "docs/estimador_do_fundo.json"))
    args = ap.parse_args()

    centro, lo, hi = grade_de_canais()
    print("grade de canais real de %s: %d canais na banda, largura %.5f a %.5f keV"
          % (OBS, centro.size, (hi - lo).min(), (hi - lo).max()))
    print("grade de energia da verossimilhanca: %d bins de %.5f keV"
          % (N_E, LARGURA))
    print()
    print("REFERENCIA: media da densidade no bin, da integral ANALITICA, que e o "
          "que o alvo pede, porque B[e]*Delta tem de ser a integral no bin")
    print()
    print("erro relativo contra a media exata no bin: MEDIANA e depois PIOR BIN")
    print("%-20s %10s %10s %10s %10s | %7s %7s %7s %7s"
          % ("espectro", "interp", "inteiro", "area", "area norm",
             "interp", "inteiro", "area", "a.norm"))

    registro = {}
    for nome, dens, prim in espectros():
        densidade = (dens, prim)
        (interp, rebin, area, area_coberta, coberto,
         taxa_canal) = estimativas(centro, lo, hi, densidade)
        exato = np.array([prim(BORDAS[k], BORDAS[k + 1]) / LARGURA
                          for k in range(N_E)])
        rel = lambda v: np.abs(v - exato) / np.maximum(np.abs(exato), 1e-300)
        quatro = (("interpolacao", interp), ("rebin_inteiro", rebin),
                  ("rebin_por_area", area), ("area_normalizada", area_coberta))
        erros = {k: rel(v) for k, v in quatro}
        total_exato = float(prim(E_MIN, E_MAX))
        cons = {k: float(v.sum() * LARGURA / total_exato) for k, v in quatro}
        print("%-20s %10.2e %10.2e %10.2e %10.2e | %7.4f %7.4f %7.4f %7.4f"
              % ((nome,) + tuple(float(np.median(erros[k])) for k, _ in quatro)
                 + tuple(float(erros[k].max()) for k, _ in quatro)))
        registro[nome] = {
            "erro_relativo_mediano": {k: float(np.median(erros[k])) for k, _ in quatro},
            "erro_relativo_maximo": {k: float(erros[k].max()) for k, _ in quatro},
            "bin_do_pior_erro": {k: int(np.argmax(erros[k])) for k, _ in quatro},
            "conservacao_da_integral": cons,
            "integral_analitica_na_banda": total_exato,
        }

    print()
    print("=== unidades, com o espectro constante de 0,05 ct/s/keV ===")
    (interp, rebin, area, area_norm, coberto,
     taxa) = estimativas(centro, lo, hi, (espectros()[0][1], espectros()[0][2]))
    print("  taxa de canal reconstruida: %.10f (esperado 0,05)" % taxa.mean())
    print("  interp %.10f | inteiro %.10f | area %.10f | area norm %.10f"
          % (interp.mean(), rebin.mean(), area.mean(), area_norm.mean()))
    print()
    print("=== cobertura da grade de canal sobre os bins de energia ===")
    print("  minima %.6f no bin %d, mediana %.6f, bins com fresta: %s"
          % (coberto.min(), int(np.argmin(coberto)), float(np.median(coberto)),
             [int(k) for k in np.where(coberto < 0.9999)[0]]))
    print("  o primeiro canal da banda comeca em %.6f keV e o bin 0 em %.6f: a "
          "fresta e %.6f keV, ou %.4f do bin"
          % (lo.min(), E_MIN, lo.min() - E_MIN,
             (lo.min() - E_MIN) / LARGURA))
    print("  e e exatamente o erro do estimador por area no espectro constante, "
          "o que identifica a causa sem ambiguidade")
    print("  contagens no bin = B * %.5f keV * exposicao, mesma unidade nos dois"
          % LARGURA)

    print()
    print("=== ordem do erro: a grade de energia pela metade e pelo dobro ===")
    ordens = {}
    for nome, dens, prim in espectros():
        linha = []
        for fator in (0.5, 1.0, 2.0):
            n = max(2, int(round(N_E / fator)))
            bordas = np.linspace(E_MIN, E_MAX, n + 1)
            larg = (E_MAX - E_MIN) / n
            cen = 0.5 * (bordas[:-1] + bordas[1:])
            taxa_canal = np.array([prim(x, y) / (y - x) for x, y in zip(lo, hi)])
            ip = np.interp(cen, centro, taxa_canal)
            ex = np.array([prim(bordas[k], bordas[k + 1]) / larg for k in range(n)])
            linha.append(float(np.median(np.abs(ip - ex) / np.maximum(np.abs(ex), 1e-300))))
        razao = linha[2] / linha[1] if linha[1] > 0 else float('nan')
        ordens[nome] = {"erro_mediano_por_passo": linha,
                        "razao_dobrando_o_passo": razao,
                        "ordem_aparente": (math.log2(razao) if razao > 0
                                           and math.isfinite(razao) else None)}
        print("  %-20s passo/2 %.3e  passo %.3e  2*passo %.3e  ordem aparente %s"
              % (nome, linha[0], linha[1], linha[2],
                 ("%.2f" % math.log2(razao)) if razao > 0 and math.isfinite(razao)
                 else "-"))

    # ---- o que isto significa no DADO REAL, que e o numero que decide ----
    print()
    print("=== os quatro estimadores nas 13 observacoes reais ===")
    d = np.load(RAIZ / "exploracoes/dados_combinados_13obs_2026-09-12/combinado13.npz",
                allow_pickle=True)
    obsids = [str(x) for x in d["obsids"]]
    expos = np.asarray(d["exposures"], float)
    guardado = np.asarray(d["background"], float)
    observado = np.asarray(d["observed"], float).reshape(32, N_E)
    T = float(expos.sum())
    acumula = {k: np.zeros(N_E) for k in
               ("interpolacao", "rebin_inteiro", "rebin_por_area",
                "area_normalizada")}
    for obsid, exp in zip(obsids, expos):
        dirobs = None
        for cand in (PRODUTOS / obsid, *PRODUTOS.glob("*/" + obsid)):
            if (cand / "bkg_spec.fits").is_file():
                dirobs = cand
                break
        cab_b, dados_b = B.extension(dirobs / "bkg_spec.fits", "SPECTRUM")
        linhas = list(B.table_rows(cab_b, dados_b))
        conta = {int(r["CHANNEL"]): float(r["COUNTS"]) for r in linhas}
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
        cl, elo, ehi, ecen = cl[sel], elo[sel], ehi[sel], ecen[sel]
        n = np.array([conta.get(int(c), 0.0) for c in cl])
        taxa = n * escala / (exposicao * (ehi - elo))
        # os quatro, com a MESMA taxa de canal de entrada
        ip = np.interp(ECEN, ecen, taxa)
        ri = np.zeros(N_E); ar = np.zeros(N_E); cob = np.zeros(N_E)
        idx = np.clip(((ecen - E_MIN) / LARGURA).astype(int), 0, N_E - 1)
        dentro = (ecen >= E_MIN) & (ecen < E_MAX)
        for k in range(N_E):
            m = dentro & (idx == k)
            if m.any():
                w = ehi[m] - elo[m]
                ri[k] = float((taxa[m] * w).sum() / w.sum())
            a, b = BORDAS[k], BORDAS[k + 1]
            ov = np.clip(np.minimum(ehi, b) - np.maximum(elo, a), 0.0, None)
            if ov.sum() > 0:
                ar[k] = float((taxa * ov).sum() / (b - a))
                cob[k] = float(ov.sum() / (b - a))
        an = np.where(cob > 0, ar / np.maximum(cob, 1e-300), 0.0)
        for chave, vetor in (("interpolacao", ip), ("rebin_inteiro", ri),
                             ("rebin_por_area", ar), ("area_normalizada", an)):
            acumula[chave] += (exp / T) * vetor

    atual = acumula["interpolacao"]
    conferido = float(np.abs(atual - guardado).max() /
                      max(float(np.abs(guardado).max()), 1e-300))
    print("  a interpolacao reproduz o vetor do npz em %.2e relativo" % conferido)
    melhor = acumula["area_normalizada"]
    dif = (melhor - atual) / np.maximum(np.abs(atual), 1e-300)
    ct_atual = atual * LARGURA * T
    ct_melhor = melhor * LARGURA * T
    poisson = np.sqrt(np.maximum(observado.sum(axis=0), 1.0))
    sigma_bin = np.abs(ct_melhor - ct_atual) / poisson
    print("  entrada atual contra o estimador por area normalizada:")
    print("    diferenca relativa: mediana %.4f, pior bin %.4f (bin %d, %.4f keV)"
          % (float(np.median(np.abs(dif))), float(np.abs(dif).max()),
             int(np.argmax(np.abs(dif))), ECEN[int(np.argmax(np.abs(dif)))]))
    print("    em contagens, contra o Poisson do dado: mediana %.4f sigma, "
          "pior %.4f sigma" % (float(np.median(sigma_bin)), float(sigma_bin.max())))
    print("    fundo total: %.1f ct atual contra %.1f ct corrigido (%.2f%%)"
          % (ct_atual.sum(), ct_melhor.sum(),
             100.0 * (ct_melhor.sum() - ct_atual.sum()) / ct_atual.sum()))
    gap = np.abs(acumula["rebin_inteiro"] - atual) / np.maximum(np.abs(atual), 1e-300)
    print("  para comparar, a diferenca que eu havia reportado, interpolacao "
          "contra rebin inteiro: mediana %.4f, pior %.4f"
          % (float(np.median(gap)), float(gap.max())))
    print("  ela media a distancia entre DOIS estimadores defeituosos, e nao "
          "limita o erro de nenhum dos dois")
    real = {
        "a_interpolacao_reproduz_o_npz": conferido,
        "vetores": {k: v.tolist() for k, v in acumula.items()},
        "entrada_atual_contra_area_normalizada": {
            "diferenca_relativa_mediana": float(np.median(np.abs(dif))),
            "diferenca_relativa_pior_bin": float(np.abs(dif).max()),
            "bin_do_pior": int(np.argmax(np.abs(dif))),
            "em_sigmas_de_poisson_do_dado_mediana": float(np.median(sigma_bin)),
            "em_sigmas_de_poisson_do_dado_pior": float(sigma_bin.max()),
            "fundo_total_atual_ct": float(ct_atual.sum()),
            "fundo_total_corrigido_ct": float(ct_melhor.sum())},
        "a_diferenca_que_eu_havia_reportado": {
            "o_que_era": "interpolacao contra rebin inteiro",
            "mediana": float(np.median(gap)), "pior": float(gap.max()),
            "leitura": ("media a distancia entre DOIS estimadores defeituosos e "
                        "nao limita o erro de nenhum deles"),
            "sobre_os_79_6_por_cento": (
                "aquele numero era (interpolacao - rebin)/rebin no pior bin; "
                "dividido pela interpolacao o mesmo desacordo da 44,3 por cento. "
                "A diferenca e so o denominador, e nenhuma das duas versoes "
                "limita erro algum")},
    }

    veredito = (
        "NENHUM DOS DOIS estimadores anteriores e o que o alvo pede, e o rebin "
        "por soma que eu havia proposto e o PIOR dos dois em espectro suave: ele "
        "erra 1,4e-3 no pior bin ate num espectro LINEAR, onde curvatura nao "
        "explica nada, porque a grade de canal nao divide o bin e os canais "
        "agrupados pelo centro cobrem um intervalo DESLOCADO. A interpolacao no "
        "centro erra em segunda ordem no passo, ordem aparente 2,14 medida, e "
        "31,7 por cento numa linha estreita. O estimador correto e o rebin "
        "ponderado pela AREA de sobreposicao canal-bin, normalizado pela largura "
        "coberta: exato em espectro constante, 1e-4 no pior bin dos suaves, 5,3 "
        "por cento na linha estreita, que e a discretizacao irredutivel do canal, "
        "e conserva a integral na banda em 1e-5. Logo os 79,6 por cento que eu "
        "havia reportado mediam a distancia entre dois estimadores defeituosos e "
        "nao limitavam o erro de nenhum deles.")
    print()
    print("VEREDITO: %s" % veredito)
    saida = {
        "pergunta": "qual estimador corresponde as contagens integradas nos bins da verossimilhanca",
        "o_que_o_alvo_pede": (
            "mu[p][e] = (S + B[e]) * energy_width * tempo, logo B[e]*Delta tem de "
            "ser a integral da densidade no bin, logo B[e] e a MEDIA no bin"),
        "referencia": "integral ANALITICA sobre o bin, nao quadratura",
        "grade_de_canais": {"observacao": OBS, "canais_na_banda": int(centro.size),
                            "largura_min": float((hi - lo).min()),
                            "largura_max": float((hi - lo).max())},
        "espectros": registro,
        "ordem_do_erro_da_interpolacao": ordens,
        "dado_real": real,
        "veredito": veredito,
        "o_que_isto_NAO_diz": [
            "nao diz que o rebin por area seja exato: ele supoe densidade "
            "uniforme DENTRO do canal, que e toda a informacao que o canal "
            "carrega, e erra 5,3 por cento numa linha de 12 eV contra canais de "
            "5 eV",
            "nao autoriza mexer nas CONTAGENS OBSERVADAS, que nao mudam: o fundo "
            "e uma ENTRADA do alvo, e corrigi-la exige versao nova do dado "
            "preparado e revalidacao",
            "nao mede o efeito nos parametros; mede o estimador contra a "
            "definicao que o alvo usa"],
    }
    Path(args.saida).write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n")
    print("gravado em %s" % args.saida)


if __name__ == "__main__":
    main()
