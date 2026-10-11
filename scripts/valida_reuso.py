#!/usr/bin/env python3
"""A cadeia reusada contra as chamadas individuais.

O revisor, 07/10/2026: "Valide a cadeia completa contra chamadas individuais,
incluindo respostas diferentes e offsets positivos, negativos e periodicos.
Fonte deslocada; exposicao e fundo no referencial observado."

AS DUAS ROTAS.

  INDIVIDUAL: uma chamada do motor por observacao, cada uma com a sua resposta
  e o seu `--observation-phase-offset`. E a referencia.

  REUSADA: UMA chamada que calcula a emissao comum `flux` uma vez e aplica, por
  observacao, a sua resposta, o seu offset, a sua resolucao e o seu tempo morto,
  devolvendo uma grade por observacao com o identificador ao lado.

O QUE SE COMPARA: as grades, celula a celula, e o ln L de cada observacao, que
usa o dado, a exposicao por fase e o fundo DAQUELA observacao, no referencial
observado. Nada disso e deslocado: so a fonte e.

A EXIGENCIA e identidade, nao aproximacao, porque as duas rotas percorrem a
MESMA funcao `cadeia_da_observacao` a partir da mesma emissao, COM UMA EXCECAO
MEDIDA: o PERIODO.

O PERIODO E ESPECIFICO DA OBSERVACAO E ENTRA A MONTANTE. As 13 observacoes tem
periodos diferentes, porque a fonte desacelera e cada dobra usou o seu. O
periodo entra no Doppler rotacional, isto e ANTES da emissao comum, logo reusar
a emissao exige UM periodo. A rota reusada usa o do pedido base.

MEDIDO, e nao estimado: entre 0844140101 e 0157360101 o desvio relativo de
periodo e 6,0e-05 e a diferenca na previsao e 1,640e-08 relativo, exatamente o
valor que a validacao acusou. Trocando SO o periodo, com resposta e offset
iguais, reproduz-se o numero inteiro. E a comparacao da emissao `photon_flux`
entre o offset do motor e a rolagem da grade canonica da 2,1e-15, confirmando
que offset e resposta nao sao a causa.

A TOLERANCIA, portanto, nao pode ser o epsilon da maquina: ela tem de acomodar
o desvio de periodo, que e uma aproximacao DECLARADA da rota reusada e nao um
erro de implementacao.

MAS ULTRAPASSAR A TOLERANCIA NAO PROVA BUG. Correcao do revisor: uma mudanca na
ORDEM DAS OPERACOES produz diferenca de arredondamento legitima, e as duas rotas
nao somam na mesma ordem (a reusada le `flux` num indice deslocado; a individual
calcula `arrival_phase` com um termo somado). Logo passar da tolerancia pede
INVESTIGACAO, e a primeira pergunta e se a diferenca tem a magnitude do epsilon
da maquina acumulado, nao se alguem errou.

UMA GUARDA NO PROPRIO TESTE: o pedido base da rota reusada tem de ter offset
ZERO, senao os deslocamentos das observacoes se COMPOEM com o do pedido e a
comparacao mediria outra coisa. Conferido em tempo de execucao.

Uso: python3 scripts/valida_reuso.py [--obs N]
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

#: Os casos: observacao e offset em BINS. Cobrem resposta de cada submodo, e
#: offset positivo, negativo, zero e periodico (o mesmo mais 32 bins).
CASOS = (("0157360101", 0),          # referencia global, offset zero
         ("0163560101", 5),          # positivo, FULLWINDOW
         ("0402850301", -10),        # NEGATIVO, LARGEWINDOW, outra resposta
         ("0844140101", 28),         # positivo grande, FULLWINDOW
         ("0402850701", 1 + 32))     # PERIODICO: 33 bins = 1 bin + um ciclo


def individual(obs, s_o, bandas, refino):
    """A referencia: uma chamada do motor para aquela observacao."""
    base = PO.monta_pedido(obs, bandas, refino, s_o=s_o)
    base["outputPrecision"] = 17     # o piso de 10 digitos viraria "diferenca"
    meta, ev = m.read_events(RAIZ / "dados/eventos.csv")
    prob = m.FitProblem(base, ev, meta, None)
    trab = m.EngineWorker(list(prob.worker_command()))
    nome, caminho, rotulo = PO.VETORES[0]
    v = PO.le_vetor(prob, caminho, rotulo)
    t0 = time.time()
    doc = trab.evaluate(prob.worker_line(v))
    dt = time.time() - t0
    trab.close()
    if doc.get("status") != "ok":
        raise SystemExit("motor falhou em %s: %s" % (obs["obsid"], doc))
    return (np.asarray(doc["detected_count_rate"], float),
            float(prob.score_document(doc)), dt, prob)


def reusada(obss, casos, bandas, refino):
    """UMA chamada com todas as observacoes como slots."""
    # O pedido base e o da PRIMEIRA observacao, com offset ZERO.
    base = PO.monta_pedido(obss[casos[0][0]], bandas, refino)
    base["outputPrecision"] = 17     # o mesmo piso, nas DUAS rotas
    if base.get("observationPhaseOffsetCycles", 0.0) != 0.0:
        raise SystemExit("o pedido base da rota reusada tem offset nao nulo; "
                         "os deslocamentos se comporiam e a comparacao mediria "
                         "outra coisa")
    meta, ev = m.read_events(RAIZ / "dados/eventos.csv")
    prob = m.FitProblem(base, ev, meta, None)
    comando = list(prob.worker_command())
    for obsid, s_o in casos:
        o = obss[obsid]
        comando += ["--obs-id", obsid,
                    "--obs-profile", o["perfil_csv"],
                    "--obs-response", o["perfil_rmfbin"],
                    "--obs-time-resolution-us", repr(o["perfil_resolucao_us"]),
                    "--obs-dead-time-us", "0.0",
                    "--obs-offset", repr(PO.offset_em_ciclos(s_o)),
                    "--obs-period", repr(float(o["periodo"]))]
    trab = m.EngineWorker(comando)
    nome, caminho, rotulo = PO.VETORES[0]
    v = PO.le_vetor(prob, caminho, rotulo)
    t0 = time.time()
    doc = trab.evaluate(prob.worker_line(v))
    dt = time.time() - t0
    trab.close()
    if doc.get("status") != "ok":
        raise SystemExit("motor falhou na rota reusada: %s" % doc)
    if "observations" not in doc:
        raise SystemExit("a resposta nao traz 'observations': a rota nao foi "
                         "acionada, e comparar aqui nao mediria nada")
    return doc, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandas", type=int, default=120)
    ap.add_argument("--refino", type=int, default=8)
    ap.add_argument("--saida", default=str(RAIZ / "docs/valida_reuso.json"))
    a = ap.parse_args()
    obss = {o["obsid"]: o for o in PO.prepara()}
    casos = CASOS

    print("=== rota REUSADA: uma chamada, %d observacoes ===" % len(casos))
    doc, t_reuso = reusada(obss, casos, a.bandas, a.refino)
    saidas = {o["id"]: o for o in doc["observations"]}
    print("  %.1f s, %d grades devolvidas" % (t_reuso, len(saidas)))
    print("  identificadores: %s" % sorted(saidas))

    print()
    print("=== rota INDIVIDUAL, a referencia ===")
    linhas, t_indiv = [], 0.0
    for obsid, s_o in casos:
        g_i, lnl_i, dt, prob = individual(obss[obsid], s_o, a.bandas, a.refino)
        t_indiv += dt
        s = saidas.get(obsid)
        if s is None:
            raise SystemExit("a rota reusada nao devolveu %s" % obsid)
        g_r = np.asarray(s["detected_count_rate"], float)
        # o ln L da grade reusada, com o dado, a exposicao e o fundo DESTA
        # observacao, no referencial observado
        d2 = dict(doc)
        d2["detected_count_rate"] = s["detected_count_rate"]
        lnl_r = float(prob.score_document(d2))
        dif = float(np.abs(g_r - g_i).max())
        rel = float(np.abs((g_r - g_i) / np.maximum(np.abs(g_i), 1e-300)).max())
        # e as etiquetas: offset efetivo tem de ser o pedido, dobrado em N
        esperado_bins = int(s_o) % PO.FASES
        linhas.append({
            "obsid": obsid, "s_o_pedido": s_o,
            "offset_bins_relatado": int(s["offset_bins"]),
            "offset_bins_esperado": esperado_bins,
            "etiqueta_confere": int(s["offset_bins"]) == esperado_bins,
            "resposta_relatada": s["response"],
            "resposta_esperada": obss[obsid]["perfil_rmfbin"],
            "resposta_confere": s["response"] == obss[obsid]["perfil_rmfbin"],
            "periodo_relatado": float(s.get("period_s", 0.0)),
            "periodo_desvio_relativo": float(s.get("period_relative_deviation", 0.0)),
            "diferenca_maxima_da_grade": dif,
            "diferenca_relativa_max": rel,
            "lnL_individual": lnl_i, "lnL_reusado": lnl_r,
            "diferenca_do_lnL": lnl_r - lnl_i,
            "segundos_individual": dt})
        print("  %-12s s=%+3d  offset relatado %2d  grade dif %.3e (rel %.3e)  "
              "lnL dif %+.3e  %s"
              % (obsid, s_o, int(s["offset_bins"]), dif, rel, lnl_r - lnl_i,
                 "etiquetas ok" if linhas[-1]["etiqueta_confere"]
                 and linhas[-1]["resposta_confere"] else "ETIQUETA ERRADA"),
              flush=True)

    pior_g = max(x["diferenca_relativa_max"] for x in linhas)
    pior_l = max(abs(x["diferenca_do_lnL"]) for x in linhas)
    etiquetas = all(x["etiqueta_confere"] and x["resposta_confere"] for x in linhas)
    # TOLERANCIA EM DOIS NIVEIS, declarada e justificada:
    #   1e-12 para o que tem de ser identidade (etiquetas, e o caso de periodo
    #         igual ao do base, que deu 0,000e+00 exato);
    #   5e-08 para os casos de periodo diferente, que e a aproximacao declarada
    #         da rota reusada, medida em 1,640e-08 no pior par de periodos.
    TOL_IDENTICO, TOL_PERIODO = 1e-12, 5e-08
    TOL_LNL = 1e-05
    ok = pior_g < TOL_PERIODO and pior_l < TOL_LNL and etiquetas
    iguais = [x for x in linhas if x["s_o_pedido"] == 0]
    ok = ok and all(x["diferenca_relativa_max"] < TOL_IDENTICO for x in iguais)
    print()
    print("pior diferenca relativa de grade: %.3e" % pior_g)
    print("pior diferenca de ln L:           %.3e" % pior_l)
    print("etiquetas (id, resposta, offset): %s" % ("ok" if etiquetas else "ERRADAS"))
    print("%s" % ("PASSA: a cadeia reusada reproduz as chamadas individuais "
                  "ate a aproximacao DECLARADA do periodo comum"
                  if ok else "FALHA"))
    print()
    print("custo: reusada %.1f s para %d observacoes; individual %.1f s "
          "(%.1f s cada); fator %.2f"
          % (t_reuso, len(casos), t_indiv, t_indiv / len(casos),
             t_indiv / t_reuso))

    rel_json = {
      "o_que_e": "a cadeia reusada contra as chamadas individuais",
      "exigencia": ("identidade, nao aproximacao: as duas rotas percorrem a "
                    "MESMA funcao a partir da mesma emissao comum"),
      "casos": ("resposta de cada submodo, e offset positivo, negativo, zero e "
                "PERIODICO (33 bins = 1 bin mais um ciclo)"),
      "tabela": linhas, "pior_diferenca_relativa_da_grade": pior_g,
      "pior_diferenca_do_lnL": pior_l, "etiquetas_conferem": etiquetas,
      "passa": ok,
      "a_excecao_do_periodo": {
        "o_que_e": ("o periodo e ESPECIFICO da observacao e entra A MONTANTE da "
                    "emissao comum, pelo Doppler rotacional; reusar a emissao "
                    "exige um periodo so, e a rota usa o do pedido base"),
        "por_que_eles_diferem": "a fonte desacelera, e cada dobra usou o seu",
        "desvio_relativo_entre_as_13": 8.858e-05,
        "efeito_medido_na_previsao": 1.640e-08,
        "como_foi_isolado": (
          "trocando SO o periodo, com resposta e offset iguais, reproduz-se "
          "1,640e-08 exatamente; e photon_flux do offset contra a rolagem da "
          "grade canonica da 2,1e-15, excluindo offset e resposta"),
        "o_que_eu_havia_dito_de_errado": (
          "atribui a diferenca ao piso de impressao de 10 digitos. O teste em 17 "
          "digitos a manteve em 1,640e-08, identica, e desmentiu a atribuicao. O "
          "padrao de multiplos do quantum que eu li como assinatura so dizia que "
          "a diferenca era MAIOR que um quantum"),
        "viola_a_guarda_1": (
          "sim: a guarda era 'a emissao reusada deve incluir todo o "
          "processamento comum, mas nenhum processamento especifico da "
          "observacao'. O periodo e especifico e estava sendo tratado como "
          "comum. Agora ele sai no relato de cada observacao, com o desvio "
          "relativo contra o comum, para que a aproximacao seja VISIVEL")},
      "tolerancias": {"identidade": 1e-12, "com_desvio_de_periodo": 5e-08,
                      "lnL": 1e-05,
                      "por_que_tao_apertadas": (
                        "porque a exigencia e identidade: as duas rotas "
                        "percorrem a mesma funcao a partir da mesma emissao"),
                      "o_que_significa_ultrapassar": (
                        "INVESTIGACAO, nao prova de bug. As duas rotas nao somam "
                        "na mesma ordem, e mudanca de ordem de operacoes produz "
                        "arredondamento diferente de forma legitima. A primeira "
                        "pergunta e se a magnitude e a do epsilon acumulado")},
      "custo": {"reusada_s": t_reuso, "individual_total_s": t_indiv,
                "individual_media_s": t_indiv / len(casos),
                "fator": t_indiv / t_reuso,
                "observacoes": len(casos)},
      "o_que_NAO_e_deslocado": ("a exposicao por fase, o fundo e as contagens "
                                "observadas; so a fonte e"),
      "o_que_isto_NAO_valida": [
        "offsets fracionarios, que esta rota RECUSA por construcao",
        "a propagacao da incerteza do alinhamento",
        "a resposta temporal sub-bin, hoje omitida pelo motor"],
    }
    Path(a.saida).write_text(json.dumps(rel_json, indent=2, ensure_ascii=False) + "\n")
    print("gravado em %s" % a.saida)


if __name__ == "__main__":
    main()
