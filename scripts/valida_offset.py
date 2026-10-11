#!/usr/bin/env python3
"""As cinco validações do offset de fase OBSERVACIONAL.

O revisor, 07/10/2026: "Para offsets nao nulos, valide sinal, periodicidade,
deslocamentos inteiros e precisao sub-bin contra uma referencia mais fina. Somar
a `arrival_phase` define a translacao, mas nao certifica sua integracao
numerica."

O QUE ESTE TESTE VALIDA E O QUE NAO VALIDA. Ele valida o offset contra A FUNCAO
HOJE IMPLEMENTADA, que amostra a fase no CENTRO de cada bin. Ele NAO valida a
resposta temporal fisica: integracao sobre bins de fase e resolucao instrumental
sao operacoes distintas, e a resolucao e zerada aqui justamente para comparar a
mesma funcao dos dois lados. A omissao da resposta temporal sub-bin e item
separado, em docs/suavizacao_inerte.json.

AS CINCO:

  V1 offset ZERO reproduz a previsao e o ln L do binario de REFERENCIA, que foi
     preservado em referencia/. O criterio nao e o hash do binario, que muda por
     existir a flag; e a previsao e o ln L.
  V2 SINAL: o deslocamento medido tem o sinal e a escala da derivacao,
     `m_offset[p] = m_0[p + offset.N]`, logo offset positivo move o padrao para
     bins MENORES.
  V3 PERIODICIDADE: offset e offset + 1 ciclo dao a mesma previsao.
  V4 DESLOCAMENTO INTEIRO: offset = k/N tem de dar EXATAMENTE roll(grade_0, -k),
     porque `arrival_phase(p)` com offset k/N e a mesma fase que `arrival_phase`
     do bin p+k sem offset, e nada se move dentro da malha. Isto e o teste que
     distingue este offset do `magAzim`, que errava 3,0e-04 por celula porque
     movia a configuracao magnetica DENTRO da malha fixa.
  V5 SUB-BIN contra referencia mais fina, por serie de Fourier de uma corrida
     com muitos bins de fase, com a convergencia da propria referencia medida.

Uso: python3 scripts/valida_offset.py [--ate V4] [--bandas 120]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import mcmc_fit as m  # noqa: E402
import por_observacao as PO  # noqa: E402

REFERENCIA_BIN = RAIZ / "referencia/magnus_engine_antes_do_offset"
OBS = "0163560101"
FASES = 32


def roda(obs, offsets, bandas, refino, binario=None, bins_de_fase=None,
         resolucao_zero=True, precisao=17):
    """As grades e os ln L naquele conjunto de offsets, uma chamada cada."""
    base = PO.monta_pedido(obs, bandas, refino)
    # PRECISAO DA SAIDA. O padrao do motor, 10 digitos, poe um piso de 1e-9
    # relativo na comparacao, e uma tolerancia de 1e-12 nunca passaria por ele.
    # Os testes pedem 17. O V1 e a EXCECAO: o binario de REFERENCIA nao tem a
    # flag, logo os dois lados correm em 10, que e o unico ponto comum.
    if precisao is not None:
        base["outputPrecision"] = int(precisao)
    if resolucao_zero:
        # A resolucao e zerada para comparar a MESMA funcao dos dois lados. A
        # 32 bins ela ja e inerte (medido, diferenca 0,000e+00), mas numa grade
        # fina ela LIGA, e ai os dois lados deixariam de ser a mesma funcao.
        base["responseProfile"]["time_resolution_us"] = 0.0
    if bins_de_fase is not None:
        # SEM o dado preparado: ele tem forma 32 x 40 e outro numero de bins de
        # fase seria incompativel. A referencia fina so e usada para comparar
        # GRADES, nao ln L, entao o dado nao entra.
        base.pop("preparedDataNpz", None)
        base["phaseBins"] = int(bins_de_fase)
    meta, ev = m.read_events(RAIZ / "dados/eventos.csv")
    prob = m.FitProblem(base, ev, meta, None)
    comando = list(prob.worker_command())
    if binario is not None:
        comando[0] = str(binario)
    nome, caminho, rotulo = PO.VETORES[0]
    v = PO.le_vetor(prob, caminho, rotulo)
    saida = {}
    for d in offsets:
        cmd = list(comando)
        if d != 0.0:
            if binario is not None:
                raise SystemExit("o binario de referencia nao tem a flag; "
                                 "offset nao nulo nele seria silenciosamente "
                                 "ignorado, e isso falsificaria o V1")
            cmd += ["--observation-phase-offset", repr(float(d))]
        trab = m.EngineWorker(cmd)
        t0 = time.time()
        doc = trab.evaluate(prob.worker_line(v))
        dt = time.time() - t0
        trab.close()
        if doc.get("status") != "ok":
            raise SystemExit("motor falhou em offset %r: %s" % (d, doc))
        g = np.asarray(doc["detected_count_rate"], float)
        try:
            lnl = float(prob.score_document(doc))
        except Exception:                        # noqa: BLE001
            lnl = float("nan")                   # grade fina, sem dado comparavel
        saida[d] = {"grade": g, "ln_verossimilhanca": lnl, "segundos": dt}
        print("    offset %+12.9f  soma %.12e  lnL %16.6f  (%.1f s)"
              % (d, g.sum(), saida[d]["ln_verossimilhanca"], dt), flush=True)
    return saida, prob


def desloca_medido(g0, g):
    """O deslocamento em bins pelo primeiro harmonico do perfil somado."""
    a = np.fft.rfft(g0.sum(1))
    b = np.fft.rfft(g.sum(1))
    ang = np.angle(b[1] / a[1])
    return -ang / (2 * np.pi) * FASES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandas", type=int, default=120)
    ap.add_argument("--refino", type=int, default=8)
    ap.add_argument("--ate", default="V5", choices=["V1", "V2", "V3", "V4", "V5"])
    ap.add_argument("--finos", type=int, nargs=2, default=[320, 640])
    ap.add_argument("--saida", default=str(RAIZ / "docs/valida_offset.json"))
    a = ap.parse_args()
    obss = {o["obsid"]: o for o in PO.prepara()}
    obs = obss[OBS]
    rel = {"o_que_e": "as cinco validacoes do offset de fase observacional",
           "o_que_valida": ("o offset contra a FUNCAO HOJE IMPLEMENTADA, que "
                            "amostra a fase no centro do bin"),
           "o_que_NAO_valida": ("a resposta temporal fisica. Integracao sobre "
                                "bins de fase e resolucao instrumental sao "
                                "operacoes distintas, e a resolucao e zerada "
                                "aqui para comparar a mesma funcao dos dois "
                                "lados"),
           "observacao": OBS, "vetor": PO.VETORES[0][0],
           "bandas": a.bandas, "refino": a.refino}
    ate = int(a.ate[1])

    print("=== V1: offset zero reproduz o binario de REFERENCIA ===")
    novo, prob = roda(obs, [0.0], a.bandas, a.refino, precisao=None)
    velho, _ = roda(obs, [0.0], a.bandas, a.refino, binario=REFERENCIA_BIN,
                    precisao=None)
    g_novo, g_velho = novo[0.0]["grade"], velho[0.0]["grade"]
    dmax = float(np.abs(g_novo - g_velho).max())
    dlnl = abs(novo[0.0]["ln_verossimilhanca"] - velho[0.0]["ln_verossimilhanca"])
    identico = bool((g_novo == g_velho).all()) and dlnl == 0.0
    print("  diferenca maxima na grade: %.3e" % dmax)
    print("  diferenca no ln L:         %.3e" % dlnl)
    print("  %s" % ("PASSA: previsao e ln L identicos bit a bit"
                    if identico else "FALHA: offset zero mudou a previsao"))
    rel["V1"] = {"diferenca_maxima_da_grade": dmax, "diferenca_do_lnL": dlnl,
                 "identico": identico,
                 "binario_de_referencia": str(REFERENCIA_BIN),
                 "criterio": ("previsao e ln L, nao o hash do binario, que muda "
                              "por existir a flag"),
                 "precisao_da_saida": ("10, o padrao, nos DOIS lados: o binario "
                                       "de referencia nao tem --output-precision, "
                                       "logo 10 e o unico ponto comum. Os outros "
                                       "testes usam 17")}
    if not identico:
        Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
        raise SystemExit("V1 falhou: nao sigo")
    if ate < 2:
        Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
        return

    print()
    print("=== V2: SINAL e escala ===")
    print("  derivacao: m_offset[p] = m_0[p + offset.N], logo offset positivo "
          "move o padrao para bins MENORES, e o deslocamento medido e -offset.N")
    sinais, _ = roda(obs, [2.0 / FASES, -2.0 / FASES], a.bandas, a.refino)
    v2 = []
    for d, esperado in ((2.0 / FASES, -2.0), (-2.0 / FASES, 2.0)):
        medido = float(desloca_medido(g_novo, sinais[d]["grade"]))
        v2.append({"offset_ciclos": d, "bins_esperados": esperado,
                   "bins_medidos": medido, "razao": medido / esperado})
        print("  offset %+9.6f ciclos: esperado %+5.2f bins, medido %+8.4f, "
              "razao %+.6f" % (d, esperado, medido, medido / esperado))
    ok2 = all(abs(x["razao"] - 1.0) < 1e-3 for x in v2)
    print("  %s" % ("PASSA: sinal e escala conferem" if ok2 else "FALHA"))
    rel["V2"] = {"tabela": v2, "passa": ok2}

    print()
    print("=== V3: PERIODICIDADE, offset e offset + 1 ciclo ===")
    per, _ = roda(obs, [1.0, 0.25, 1.25], a.bandas, a.refino)
    v3 = []
    for a1, b1 in ((1.0, 0.0), (1.25, 0.25)):
        ga = per[a1]["grade"]
        gb = g_novo if b1 == 0.0 else per[b1]["grade"]
        d = float(np.abs(ga - gb).max())
        rel_d = d / max(float(np.abs(gb).max()), 1e-300)
        v3.append({"par": [a1, b1], "diferenca_maxima": d,
                   "diferenca_relativa": rel_d})
        print("  offset %.2f contra %.2f: diferenca maxima %.3e (relativa %.3e)"
              % (a1, b1, d, rel_d))
    ok3 = all(x["diferenca_relativa"] < 1e-12 for x in v3)
    print("  %s" % ("PASSA: periodica em 1 ciclo" if ok3 else "FALHA"))
    rel["V3"] = {"tabela": v3, "passa": ok3,
                 "tolerancia": "1e-12 relativo, que e nivel de ponto flutuante"}

    print()
    print("=== V4: DESLOCAMENTO INTEIRO contra roll da grade sem offset ===")
    print("  arrival_phase(p) com offset k/N e a fase do bin p+k sem offset, e "
          "nada se move dentro da malha: a igualdade tem de ser de ponto "
          "flutuante, nao de quadratura")
    intei, _ = roda(obs, [1.0 / FASES, 8.0 / FASES], a.bandas, a.refino)
    v4 = []
    for k in (1, 8):
        g = intei[k / FASES]["grade"]
        esperado = np.roll(g_novo, -k, axis=0)
        d = float(np.abs(g - esperado).max())
        rd = float(np.abs((g - esperado) / np.maximum(np.abs(esperado), 1e-300)).max())
        v4.append({"k": k, "diferenca_maxima": d, "diferenca_relativa_max": rd})
        print("  k = %d: diferenca maxima %.3e, relativa %.3e" % (k, d, rd))
    ok4 = all(x["diferenca_relativa_max"] < 1e-12 for x in v4)
    print("  %s" % ("PASSA: exato em deslocamento inteiro" if ok4 else "FALHA"))
    print("  contraste com magAzim, medido antes: 3,0e-04 por celula")
    rel["V4"] = {"tabela": v4, "passa": ok4,
                 "contraste_com_magAzim": 3.003e-04,
                 "o_que_o_contraste_demonstra": (
                   "a unica coisa que difere entre os dois operadores e se a "
                   "configuracao se move DENTRO da malha fixa. O offset nao "
                   "move, e e exato; o azimute move, e erra 3,0e-04. Isso "
                   "transforma a atribuicao a orientacao da malha de "
                   "compatibilidade em ordem de grandeza para demonstracao")}
    if ate < 5:
        Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
        print()
        print("gravado em %s" % a.saida)
        return

    print()
    print("=== V5: SUB-BIN contra referencia mais fina, EXATAMENTE ===")
    print("  O plano antigo era interpolar por Fourier uma grade fina, o que")
    print("  custaria ~18 min e misturaria erro de interpolacao ao do offset.")
    print("  A aritmetica de paridade da uma comparacao EXATA: a fase")
    print("    (p+0,5)/N + k/(NM) = [pM + 0,5M + k]/(NM)")
    print("  e o centro do bin j = pM + 0,5(M-1) + k da grade de NM fases, e isso")
    print("  e inteiro quando M e IMPAR. Logo rodar N fases com offset k/(NM) e")
    print("  rodar NM fases com offset zero avaliam A MESMA FASE por duas rotas.")
    v5 = []
    for M in (3, 5):
        Nf = FASES * M
        print("  M = %d, grade fina de %d fases" % (M, Nf))
        fino, _ = roda(obs, [0.0], a.bandas, a.refino, bins_de_fase=Nf)
        gf = fino[0.0]["grade"]
        for k in range(1, M):
            d = k / Nf
            gross, _ = roda(obs, [d], a.bandas, a.refino)
            g = gross[d]["grade"]
            j = np.array([p * M + (M - 1) // 2 + k for p in range(FASES)])
            esperado = gf[j % Nf, :]
            dif = float(np.abs(g - esperado).max())
            rd = float(np.abs((g - esperado)
                              / np.maximum(np.abs(esperado), 1e-300)).max())
            v5.append({"M": M, "k": k, "offset_ciclos": d,
                       "offset_em_bins": k / M,
                       "bins_finos_usados": j.tolist(),
                       "diferenca_maxima": dif, "diferenca_relativa_max": rd})
            print("    k = %d (offset %.4f bins): diferenca maxima %.3e, "
                  "relativa %.3e" % (k, k / M, dif, rd))
    ok5 = all(x["diferenca_relativa_max"] < 1e-12 for x in v5)
    print("  %s" % ("PASSA: o offset fracionario avalia a fase certa, no nivel "
                    "de ponto flutuante" if ok5 else "FALHA"))
    rel["V5"] = {
      "tabela": v5, "passa": ok5, "tolerancia": "1e-12 relativo",
      "como_foi_feito": (
        "comparacao EXATA, nao interpolada: N fases com offset k/(NM) contra NM "
        "fases com offset zero, lendo o bin j = pM + 0,5(M-1) + k, que e a mesma "
        "fase fisica. So funciona com M IMPAR, por causa da convencao de centro "
        "de bin, e isso foi verificado com fracoes para M de 1 a 8"),
      "o_plano_que_eu_abandonei": (
        "interpolar por Fourier uma grade de 640 fases, a ~18 min, misturando "
        "erro de interpolacao ao erro do offset. A rota exata custa ~2,5 min e "
        "nao tem esse erro"),
      "o_que_isto_valida": (
        "que o offset fracionario faz o motor avaliar a fase pretendida. NAO "
        "valida a resposta temporal fisica, que esta zerada nos dois lados de "
        "proposito, nem a propagacao da incerteza do alinhamento")}
    Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print()
    print("gravado em %s" % a.saida)


if __name__ == "__main__":
    main()
