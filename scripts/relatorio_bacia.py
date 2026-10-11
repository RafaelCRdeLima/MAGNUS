#!/usr/bin/env python3
"""Relatorio de uma bacia a partir do checkpoint, com erro de Monte Carlo.

Substitui a inspecao manual que eu vinha fazendo e atende tres exigencias do
plano de recuperacao que a ferramenta anterior nao cobria:

- r-hat classico entre caminhantes do MESMO enxame nao e valido: os caminhantes
  de um stretch move sao dependentes por construcao. Ele e reportado, porque e
  util como alarme de caminhante encalhado, mas vem rotulado como alarme e nao
  como diagnostico de convergencia. O substituto e split-rank-R-hat com
  ensembles independentes, que exige outra corrida.
- ESS pela sequencia positiva inicial de Geyer sobre a autocorrelacao media dos
  caminhantes, SEM o teto de 100 passos que `autocorrelation_ess` usa. Com
  cadeias de 1e5 passos aquele teto subestima tau sempre que tau passa de ~30.
- dispersao dos quantis sob reamostragem DE CAMINHANTES. Quando um caminhante
  encalha, a variabilidade entre caminhantes domina, e um bootstrap sobre
  amostras individuais a esconde. ESTE NUMERO NAO E ERRO DE MONTE CARLO
  CERTIFICADO, e eu o chamei disso antes: os caminhantes de um stretch move sao
  dependentes, entao a reamostragem nao produz replicas independentes, e um
  modo que o enxame nunca visitou fica invisivel para ela, com dispersao
  pequena. E diagnostico DESCRITIVO de sensibilidade a composicao do enxame,
  valido como alarme e nao como barra de erro, ate ser validado contra
  replicacoes independentes. A dependencia dentro do enxame esta explicitada na
  documentacao do emcee, em
  https://emcee.readthedocs.io/en/stable/tutorials/autocorr/

Uso: python3 scripts/relatorio_bacia.py <pasta com checkpoint.npz> [--janela 0.5]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

GM_SUN_C2_KM = 1.4766250385
#: Limiares de a1 em que a camada termica deixa de ser resolvida. A lei
#: lobe(c,a,T^4) = T^4 c^2/(c^2 + a s^2) cai a meia altura em c ~ sqrt(a) no
#: equador magnetico, e a celula da malha cobre Dc ~ pi/bandas, dividida por
#: `refino` nas celulas cortadas. Logo a camada e subamostrada quando
#: a1 < (pi/(bandas*refino))^2. Sem isto, dizer que a1 "pressiona o zero" e
#: impressao: o que importa e a fracao de amostras DENTRO da regiao em que a
#: quadratura erra, e nao a distancia ao limite do prior.
RESOLUCOES_DE_REFERENCIA = ((26, 1), (60, 1), (120, 1), (26, 8), (120, 8), (400, 8))
ATRASO_MAXIMO = 100.0
#: Passos por tau abaixo dos quais a propria estimativa de tau e pouco
#: confiavel. Regra de bolso usual para a sequencia positiva inicial. Isto
#: sinaliza POUCA CONFIABILIDADE e nao transforma o tau num limite inferior.
PASSOS_POR_TAU_MINIMO = 50.0
#: fracao da posteriori encostada num limite que torna o limite ativo
FRACAO_LIMITE = 0.10
MARGEM_LIMITE = 0.10


def rhat_entre_caminhantes(x):
    cadeias, passos = x.shape
    dentro = x.var(axis=1, ddof=1).mean()
    entre = passos * x.mean(axis=1).var(ddof=1)
    if dentro <= 0.0:
        return float("nan")
    return float(np.sqrt(((passos - 1) / passos * dentro + entre / passos) / dentro))


def autocorrelacao(x):
    """ACF media dos caminhantes, por FFT, normalizada em lag 0."""
    n = x.shape[1]
    m = 1
    while m < 2 * n:
        m *= 2
    centrado = x - x.mean(axis=1, keepdims=True)
    f = np.fft.rfft(centrado, n=m, axis=1)
    acf = np.fft.irfft(f * np.conj(f), n=m, axis=1)[:, :n].real
    norma = acf[:, :1].copy()
    norma[norma == 0.0] = 1.0
    return (acf / norma).mean(axis=0)


def tau_geyer(x):
    """Tempo de autocorrelacao pela sequencia positiva inicial de Geyer."""
    rho = autocorrelacao(x)
    soma = 0.0
    for m in range(1, len(rho) - 1, 2):
        par = rho[m] + rho[m + 1]
        if par <= 0.0:
            break
        soma += par
    return 1.0 + 2.0 * soma


#: teto de amostras por caminhante no bootstrap. O CENTRO sai dos dados
#: completos; so a DISPERSAO usa a subamostra, porque 400 sorteios sobre
#: (96, 40000) custam 12 min por parametro e a informacao independente de uma
#: cadeia com tau na casa das centenas cabe em 2000 pontos.
SUBAMOSTRA_BOOTSTRAP = 2000


def quantis_com_dispersao(valores_por_caminhante, qs=(16, 50, 84), sorteios=400,
                          semente=20261004):
    """Quantis nos dados completos; dispersao sob reamostragem DE CAMINHANTES.

    A dispersao devolvida mede sensibilidade a composicao do enxame. Nao e erro
    de Monte Carlo: ver o cabecalho do modulo.
    """
    rng = np.random.default_rng(semente)
    nc, np_ = valores_por_caminhante.shape
    centro = {q: float(np.percentile(valores_por_caminhante, q)) for q in qs}
    passo = max(1, np_ // SUBAMOSTRA_BOOTSTRAP)
    reduzido = valores_por_caminhante[:, ::passo]
    amostra = {q: [] for q in qs}
    for _ in range(sorteios):
        bloco = reduzido[rng.integers(0, nc, nc)]
        for q in qs:
            amostra[q].append(np.percentile(bloco, q))
    disp = {q: float(np.std(amostra[q], ddof=1)) for q in qs}
    return centro, disp, {"sorteios": sorteios, "passo_subamostra": passo,
                          "por_caminhante": reduzido.shape[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta")
    ap.add_argument("--janela", type=float, default=0.5,
                    help="fracao final da cadeia usada, 0.5 = segunda metade")
    ap.add_argument("--atraso", type=float, default=ATRASO_MAXIMO)
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()

    pasta = Path(a.pasta)
    dados = np.load(pasta / "checkpoint.npz", allow_pickle=True)
    nomes = [str(v) for v in dados["names"]]
    amostras = dados["samples"]
    logps = dados["logps"].astype(np.float64)
    nc, passos, _ = amostras.shape
    corte = int(passos * (1.0 - a.janela))

    janela = logps[:, corte:]
    atraso = np.median(janela) - janela.mean(axis=1)
    perdidos = np.where(atraso > a.atraso)[0]
    bons = np.setdiff1d(np.arange(nc), perdidos)

    print("%s: %d caminhantes x %d passos, janela %d a %d"
          % (pasta.name, nc, passos, corte, passos))
    print("lnL na janela: mediana %.2f, melhor %.2f" % (np.median(janela), janela.max()))
    if len(perdidos):
        print("caminhantes atrasados mais de %.0f em lnL: %s" % (a.atraso, list(perdidos)))
        for w in perdidos:
            print("   walker %3d  atraso %8.1f" % (w, atraso[w]))
    else:
        print("nenhum caminhante atrasado mais de %.0f em lnL" % a.atraso)

    rel = {"pasta": str(pasta), "caminhantes": nc, "passos": passos,
           "janela": [corte, passos], "criterio_atraso": a.atraso,
           "atrasados": [{"walker": int(w), "atraso_lnL": float(atraso[w])}
                         for w in perdidos],
           "aviso_rhat": ("r-hat classico entre caminhantes do mesmo enxame NAO e "
                          "diagnostico valido de convergencia, porque os caminhantes "
                          "sao dependentes. Vale como alarme de caminhante encalhado."),
           "parametros": {}}

    print()
    print("%-18s %10s %10s %10s %9s %9s %12s"
          % ("parametro", "mediana", "p16", "p84", "r-hat", "tau", "ESS"))
    for i, nome in enumerate(nomes):
        fatia = amostras[:, corte:, i].astype(np.float64)
        if fatia.std() == 0.0:
            rel["parametros"][nome] = {"congelado": True,
                                       "valor": float(fatia.flat[0])}
            print("%-18s %10.6g %10s %10s %9s %9s %12s"
                  % (nome, fatia.flat[0], "congelado", "", "", "", ""))
            continue
        centro, disp, boot = quantis_com_dispersao(fatia)
        tau = tau_geyer(fatia)
        ess = nc * (passos - corte) / tau if tau > 0 else float("nan")
        r = rhat_entre_caminhantes(fatia)
        rel["parametros"][nome] = {
            "mediana": centro[50], "p16": centro[16], "p84": centro[84],
            "dispersaoEnxame_mediana": disp[50],
            "dispersaoEnxame_p16": disp[16], "dispersaoEnxame_p84": disp[84],
            "rhat_alarme": r, "tau": float(tau), "ESS": float(ess),
            "passosPorTau": float((passos - corte) / tau) if tau > 0 else None,
            "largura_do_intervalo": centro[84] - centro[16],
            "dispersao_sobre_largura": (disp[50] / (centro[84] - centro[16])
                                        if centro[84] > centro[16] else float("nan")),
            "media_por_caminhante_min": float(fatia.mean(axis=1).min()),
            "media_por_caminhante_max": float(fatia.mean(axis=1).max()),
            "rhat_sem_atrasados": (rhat_entre_caminhantes(fatia[bons])
                                   if len(perdidos) and len(bons) > 2 else None),
            "bootstrap": boot,
        }
        print("%-18s %10.6g %10.6g %10.6g %9.3f %9.1f %12.0f"
              % (nome, centro[50], centro[16], centro[84], r, tau, ess))

    # derivados: z, massa, M/R
    iu, ir = nomes.index("compactness"), nomes.index("radius")
    u = amostras[:, corte:, iu].astype(np.float64)
    raio = amostras[:, corte:, ir].astype(np.float64)
    deriv = {"z": 1.0 / np.sqrt(1.0 - u) - 1.0,
             "massa": u * raio / (2.0 * GM_SUN_C2_KM),
             "MsobreR_Msol_por_km": (u * raio / (2.0 * GM_SUN_C2_KM)) / raio,
             "raio": raio}
    print()
    print("%-22s %10s %10s %10s %12s %10s"
          % ("derivado", "mediana", "p16", "p84", "dispersao", "disp/largura"))
    rel["derivados"] = {}
    for nome, arr in deriv.items():
        centro, disp, boot = quantis_com_dispersao(arr)
        larg = centro[84] - centro[16]
        rel["derivados"][nome] = {"mediana": centro[50], "p16": centro[16],
                                  "p84": centro[84], "dispersaoEnxame_mediana": disp[50],
                                  "dispersao_sobre_largura": disp[50] / larg if larg > 0 else float("nan"),
                                  "bootstrap": boot}
        print("%-22s %10.5f %10.5f %10.5f %12.5f %10.3f"
              % (nome, centro[50], centro[16], centro[84], disp[50],
                 disp[50] / larg if larg > 0 else float("nan")))

    massa = deriv["massa"]
    rel["massa_perto_do_teto"] = {
        "maximo": float(massa.max()), "p99": float(np.percentile(massa, 99)),
        "fracao_acima": {("%.3f" % lim): float((massa > lim).mean())
                         for lim in (2.45, 2.48, 2.49, 2.495, 2.499)},
        "nota": ("calculada AMOSTRA POR AMOSTRA. O produto dos percentis marginais de u e R "
                 "nao da o percentil da massa: medido nesta corrida, p84(u)*p84(R)/(2GM) "
                 "dava 2,4970 contra 2,4813 do p84 das massas."),
    }
    print()
    print("massa perto do teto, amostra por amostra: maximo %.5f, p99 %.5f"
          % (massa.max(), np.percentile(massa, 99)))
    for lim in (2.45, 2.48, 2.49, 2.495, 2.499):
        print("   fracao com M > %.3f: %.3f %%" % (lim, 100 * float((massa > lim).mean())))

    print()
    z = rel["derivados"]["z"]
    razao = z["dispersao_sobre_largura"]
    print("dispersao do enxame na mediana de z: %.5f, largura do intervalo %.5f,"
          % (z["dispersaoEnxame_mediana"], z["p84"] - z["p16"]))
    print("razao %.3f. Isto NAO e o criterio do plano, que pede erro de Monte"
          % razao)
    print("Carlo abaixo de 10 % da largura CONFIRMADO ENTRE REPLICACOES: com um")
    print("enxame so, uma razao pequena tambem e o que se ve quando o enxame")
    print("deixou de visitar um modo. O criterio fica pendente de replicas.")
    rel["dispersao_z"] = {
        "razao": razao,
        "nao_e_o_criterio_do_plano": (
            "o criterio pede erro de Monte Carlo dos quantis de z abaixo de 10 % da largura, "
            "confirmado entre replicacoes independentes. Esta razao mede sensibilidade a "
            "composicao de UM enxame de caminhantes dependentes, e fica pequena tanto quando "
            "a amostragem e boa quanto quando um modo nunca foi visitado."),
    }

    # a1 contra as resolucoes: fracao de amostras na regiao mal resolvida
    if "peaking" in nomes:
        a1 = amostras[:, corte:, nomes.index("peaking")].astype(np.float64)
        rel["a1_contra_a_resolucao"] = {
            "regra": ("camada mal resolvida quando a1 < (pi/(bandas*refino))^2, porque a "
                      "meia-largura da camada vai como sqrt(a1) e a celula cobre pi/bandas"),
            "minimo_na_janela": float(a1.min()),
            "p1": float(np.percentile(a1, 1)), "p5": float(np.percentile(a1, 5)),
            "por_configuracao": {
                "%db/ref%d" % (b, r): {
                    "a1_critico": (math.pi / b / r) ** 2,
                    "fracao_abaixo": float((a1 < (math.pi / b / r) ** 2).mean()),
                    "camadas_por_celula_no_minimo":
                        float(math.sqrt(a1.min()) / (math.pi / b / r)),
                } for b, r in RESOLUCOES_DE_REFERENCIA},
        }
        print()
        print("a1 contra a resolucao: minimo na janela %.6g" % a1.min())
        print("%-20s %12s %14s %14s"
              % ("configuracao", "a1 critico", "fracao abaixo", "celulas no min"))
        for b, r in RESOLUCOES_DE_REFERENCIA:
            e = rel["a1_contra_a_resolucao"]["por_configuracao"]["%db/ref%d" % (b, r)]
            print("%-20s %12.3e %13.4f %% %14.2f"
                  % ("%d bandas, refino %d" % (b, r), e["a1_critico"],
                     100 * e["fracao_abaixo"], e["camadas_por_celula_no_minimo"]))

    por_tau = [(d["passosPorTau"], n) for n, d in rel["parametros"].items()
               if d.get("passosPorTau") is not None]
    pior_tau = min(por_tau) if por_tau else (float("nan"), "nenhum")
    print()
    print("RESSALVAS deste relatorio:")
    print("1. O ESS acima e ESTIMATIVA POTENCIALMENTE OTIMISTA, nao limite")
    print("   superior certificado. A sequencia positiva inicial de Geyer")
    print("   subestimou tau numa realizacao de AR(1) conferida, 184,6 contra")
    print("   199 teoricos, e isso NAO estabelece vies unilateral garantido.")
    print("   Menos de %.0f passos por tau marca pouca confiabilidade e nao"
          % PASSOS_POR_TAU_MINIMO)
    print("   transforma o tau num limite inferior. Pior caso aqui: %.1f passos"
          % pior_tau[0])
    print("   por tau em %s%s." % (pior_tau[1],
          "" if pior_tau[0] >= PASSOS_POR_TAU_MINIMO else ", ABAIXO do limiar"))
    print("2. A coluna de dispersao mede sensibilidade a composicao do enxame,")
    print("   nao erro de Monte Carlo. Os caminhantes de um stretch move sao")
    print("   dependentes, entao a reamostragem nao gera replicas independentes,")
    print("   e um modo que o enxame nunca visitou fica invisivel para ela. E")
    print("   diagnostico descritivo ate ser validado contra replicacoes.")
    sem = [(d["rhat_sem_atrasados"], n) for n, d in rel["parametros"].items()
           if d.get("rhat_sem_atrasados") is not None]
    if sem:
        pior_com = max((d["rhat_alarme"], n) for n, d in rel["parametros"].items()
                       if d.get("rhat_alarme") is not None)
        pior_sem = max(sem)
        print("   Excluindo os atrasados, o pior r-hat cai de %.3f em %s para "
              "%.3f em %s." % (pior_com[0], pior_com[1], pior_sem[0], pior_sem[1]))
    print("3. O r-hat e alarme de caminhante encalhado, nao diagnostico de")
    print("   convergencia: o substituto e split-rank-R-hat com ensembles")
    print("   independentes, que exige outra corrida.")
    print("Tres coisas que nenhum numero daqui separa: precisao do estimador,")
    print("mistura do enxame e exploracao dos modos.")
    rel["passosPorTau_pior"] = {"valor": pior_tau[0], "parametro": pior_tau[1],
                                "limiar": PASSOS_POR_TAU_MINIMO,
                                "acima_do_limiar": bool(pior_tau[0] >= PASSOS_POR_TAU_MINIMO)}
    rel["ressalvas"] = [
        "o ESS e estimativa potencialmente otimista, nao limite superior certificado; a "
        "subestimacao de tau conferida numa realizacao de AR(1), 184,6 contra 199, nao "
        "estabelece vies unilateral garantido",
        "a dispersao sob reamostragem de caminhantes mede sensibilidade a composicao do enxame, "
        "nao erro de Monte Carlo: os caminhantes sao dependentes e um modo nao visitado fica "
        "invisivel, com dispersao pequena",
        "o r-hat entre caminhantes do mesmo enxame e alarme de caminhante encalhado, nao "
        "diagnostico de convergencia",
        "nenhum numero deste relatorio separa precisao do estimador, mistura do enxame e "
        "exploracao dos modos",
    ]

    destino = Path(a.saida) if a.saida else pasta / "relatorio_bacia.json"
    destino.write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print("gravado em", destino)


if __name__ == "__main__":
    main()
