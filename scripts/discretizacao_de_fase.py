#!/usr/bin/env python3
"""O erro da REGRA DO PONTO MÉDIO na integração de fase.

O motor amostra a fase no CENTRO de cada bin observado. O que entra na
verossimilhança é a previsão INTEGRADA sobre o bin. A diferença entre as duas é
o erro de discretização de fase, e este script o mede.

O REVISOR CORRIGIU UM TESTE QUE EU IA FAZER E QUE NÃO MEDIRIA ISSO. Eu ia
comparar o mesmo offset contra grades de 96 e 288 fases e chamar o resultado de
"estabilidade da referência". Mas as duas avaliam EXATAMENTE A MESMA FASE, logo a
concordância valida o ENDEREÇAMENTO e não a integral. Para a integral, compara-se
a MÉDIA das M amostras finas dentro de cada bin observado, em M = 3, 9 e 15.

MÉDIA, NÃO SOMA. A saída do motor é densidade de TAXA, em ct/s/keV, então o
representante do bin é a média das amostras finas, e só depois se multiplica pela
exposição. Somar daria M vezes o valor.

A LIMITAÇÃO, DECLARADA. Com exposição variável DENTRO do bin, a referência
rigorosa é a média ponderada pela exposição. A tabela de exposição por fase da
reconstrução tem 32 bins, não sub-bin, então a média UNIFORME é o que se pode
calcular aqui, e ela testa apenas a discretização do PERFIL. Medir o caso
rigoroso exigiria exposição por fase numa grade fina, o que é um pedido à
redução e não algo que eu possa derivar.

O que se reporta: diferenças por bin, `Q_num` e Δln L nos mesmos dados.

Uso: python3 scripts/discretizacao_de_fase.py [--Ms 3 9 15]
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

OBS = "0163560101"
FASES = PO.FASES
#: Qual vetor. O efeito ABSOLUTO nao decide: o que decide e quanto sobra nas
#: DIFERENCAS entre vetores, e para isso a medida tem de ser repetida em cada um.
VETOR = PO.VETORES[0][0]


def grade(obs, bins, bandas, refino, com_dado):
    """A densidade de taxa com `bins` fases, resolucao ZERADA.

    A resolucao e zerada para que o teste meça SÓ a discretização de fase. A
    resposta temporal instrumental é outro teste, e a 32 bins ela é inerte de
    todo modo (docs/suavizacao_inerte.json).
    """
    base = PO.monta_pedido(obs, bandas, refino)
    base["outputPrecision"] = 17     # o piso de 10 digitos viraria "diferenca"
    base["responseProfile"]["time_resolution_us"] = 0.0
    if not com_dado:
        base.pop("preparedDataNpz", None)
    base["phaseBins"] = int(bins)
    meta, ev = m.read_events(RAIZ / "dados/eventos.csv")
    prob = m.FitProblem(base, ev, meta, None)
    trab = m.EngineWorker(list(prob.worker_command()))
    nome, caminho, rotulo = [x for x in PO.VETORES if x[0] == VETOR][0]
    v = PO.le_vetor(prob, caminho, rotulo)
    t0 = time.time()
    doc = trab.evaluate(prob.worker_line(v))
    dt = time.time() - t0
    trab.close()
    if doc.get("status") != "ok":
        raise SystemExit("motor falhou em %d fases: %s" % (bins, doc))
    return np.asarray(doc["detected_count_rate"], float), prob, doc, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ms", type=int, nargs="+", default=[3, 9, 15])
    ap.add_argument("--bandas", type=int, default=120)
    ap.add_argument("--refino", type=int, default=8)
    ap.add_argument("--vetor", default=PO.VETORES[0][0])
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()
    global VETOR
    VETOR = a.vetor
    if a.saida is None:
        a.saida = str(RAIZ / ("docs/discretizacao_de_fase_%s.json"
                              % a.vetor.replace(" ", "_")))
    obss = {o["obsid"]: o for o in PO.prepara()}
    obs = obss[OBS]

    print("=== a amostra PONTUAL, 32 fases, que e o que o motor faz hoje ===")
    g_ponto, prob, doc_ponto, dt = grade(obs, FASES, a.bandas, a.refino, True)
    lnl_ponto = float(prob.score_document(doc_ponto))
    print("  %.1f s, ln L = %.6f" % (dt, lnl_ponto))
    T = np.asarray(prob.tempo_representado(), float)
    print("  exposicao por fase: %.1f a %.1f s, amplitude %.4f por cento"
          % (T.min(), T.max(), 100 * (T.max() - T.min()) / T.mean()))

    rel = {"o_que_e": "o erro da regra do ponto medio na integracao de fase",
           "observacao": OBS, "vetor": a.vetor,
           "bandas": a.bandas, "refino": a.refino,
           "resolucao_zerada": ("sim, para medir SO a discretizacao de fase; a "
                                "resposta temporal instrumental e outro teste"),
           "media_e_nao_soma": ("a saida e densidade de TAXA, logo o "
                                "representante do bin e a MEDIA das amostras "
                                "finas, e a exposicao multiplica depois"),
           "a_limitacao": ("com exposicao variavel DENTRO do bin a referencia "
                           "rigorosa e a media PONDERADA pela exposicao; a "
                           "tabela da reconstrucao tem 32 bins e nao sub-bin, "
                           "logo a media uniforme aqui testa apenas a "
                           "discretizacao do PERFIL"),
           "ln_L_pontual": lnl_ponto, "niveis": []}

    anterior = None
    for M in a.Ms:
        n = FASES * M
        print()
        print("=== M = %d, grade de %d fases ===" % (M, n))
        g_fino, _, _, dtf = grade(obs, n, a.bandas, a.refino, False)
        # a MEDIA das M amostras finas de cada bin observado
        g_media = g_fino.reshape(FASES, M, -1).mean(axis=1)
        d = g_media - g_ponto
        r_cel = d / np.maximum(np.abs(g_ponto), 1e-300)
        # por fase, na taxa somada em energia
        pf = (g_media.sum(1) / np.maximum(g_ponto.sum(1), 1e-300)) - 1.0
        # Q_num e D ln L, na forma do projeto, por celula
        doc2 = dict(doc_ponto)
        doc2["detected_count_rate"] = g_media.tolist()
        lnl_media = float(prob.score_document(doc2))
        mu_p = np.asarray(prob.score_document(doc_ponto, return_model=True)[1],
                          float).reshape(FASES, -1)
        mu_m = np.asarray(prob.score_document(doc2, return_model=True)[1],
                          float).reshape(FASES, -1)
        delta = mu_m - mu_p
        qnum = float((delta ** 2 / np.maximum(mu_p, 1e-300)).sum())
        nivel = {"M": M, "fases": n, "segundos": dtf,
                 "diferenca_relativa_mediana": float(np.median(np.abs(r_cel))),
                 "diferenca_relativa_max": float(np.abs(r_cel).max()),
                 "pior_celula": [int(x) for x in
                                 np.unravel_index(int(np.argmax(np.abs(r_cel))),
                                                  r_cel.shape)],
                 "por_fase_relativa_max": float(np.abs(pf).max()),
                 "por_fase_relativa": pf.tolist(),
                 "Q_num": qnum,
                 "ln_L_com_a_media": lnl_media,
                 "D_lnL": lnl_media - lnl_ponto}
        if anterior is not None:
            # ESTABILIDADE DA INTEGRAL: quanto a propria media muda de M para M
            dd = g_media - anterior[1]
            nivel["mudanca_da_media_contra_M_anterior"] = {
              "M_anterior": anterior[0],
              "relativa_mediana": float(np.median(np.abs(
                  dd / np.maximum(np.abs(anterior[1]), 1e-300)))),
              "relativa_max": float(np.abs(
                  dd / np.maximum(np.abs(anterior[1]), 1e-300)).max()),
              "D_lnL_entre_os_dois": lnl_media - anterior[2]}
            print("  estabilidade contra M = %d: mediana %.3e, max %.3e, "
                  "D lnL %.6f"
                  % (anterior[0],
                     nivel["mudanca_da_media_contra_M_anterior"]["relativa_mediana"],
                     nivel["mudanca_da_media_contra_M_anterior"]["relativa_max"],
                     nivel["mudanca_da_media_contra_M_anterior"]["D_lnL_entre_os_dois"]))
        anterior = (M, g_media, lnl_media)
        rel["niveis"].append(nivel)
        print("  media contra ponto medio: mediana %.3e, max %.3e (fase %d, en %d)"
              % (nivel["diferenca_relativa_mediana"],
                 nivel["diferenca_relativa_max"], *nivel["pior_celula"]))
        print("  por fase, max %.3e" % nivel["por_fase_relativa_max"])
        print("  Q_num = %.6f   D ln L = %+.6f" % (qnum, nivel["D_lnL"]))

    ult = rel["niveis"][-1]
    print()
    print("=== leitura ===")
    print("o ponto medio erra %.3e na mediana e %.3e no pior bin, com D ln L de "
          "%+.6f e Q_num de %.6f no nivel mais fino"
          % (ult["diferenca_relativa_mediana"], ult["diferenca_relativa_max"],
             ult["D_lnL"], ult["Q_num"]))
    if len(rel["niveis"]) > 1:
        e = ult["mudanca_da_media_contra_M_anterior"]
        print("a propria integral esta estavel a %.3e na mediana entre os dois "
              "niveis mais finos, com D ln L de %+.6f entre eles"
              % (e["relativa_mediana"], e["D_lnL_entre_os_dois"]))
        print("logo o erro do ponto medio esta %s do ruido de refinamento"
              % ("ACIMA" if abs(ult["D_lnL"]) > 10 * abs(e["D_lnL_entre_os_dois"])
                 else "na ordem"))
    rel["o_que_decide"] = (
      "se o ponto medio couber no orcamento de erro, ele FICA; nao e obrigatorio "
      "substituir toda a integracao de fase. O orcamento e relativo a conclusao "
      "pretendida, e nao e meu para fixar")
    Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print("gravado em %s" % a.saida)


if __name__ == "__main__":
    main()
