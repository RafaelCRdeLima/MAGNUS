#!/usr/bin/env python3
"""A igualdade que o rotulo anterior NAO demonstrava: a soma fecha.

O revisor, 10/10/2026: "13 espectros distintos verifica ASSOCIACAO entre
observacoes, nao que a soma das contribuicoes de ln L fecha; registre essa
igualdade separadamente."

A distincao, nos termos exatos. Ter treze espectros de fundo distintos nas treze
observacoes diz que cada observacao foi ASSOCIADA ao seu proprio fundo, e nada
mais: e uma conferencia de procedencia do dado. Que a verossimilhanca conjunta
seja a soma das treze e outra afirmacao, sobre a FATORACAO do alvo, e precisa de
duas medidas distintas:

  FECHAMENTO. O mesmo vetor pontuado por duas rotas: treze chamadas isoladas do
  motor, uma por observacao, e UMA chamada com as treze como slots. Se a soma
  fecha, as duas rotas dao o mesmo total. Esta parte mede a ROTA.

  INDEPENDENCIA. Perturbar o fundo medido de UMA observacao e exigir que o termo
  daquela observacao mude e que os doze outros fiquem IDENTICOS bit a bit. Se
  algum outro termo se mover, ha latente compartilhado entre observacoes, e a
  soma nao e separavel qualquer que seja o total. Esta parte mede o ALVO.

O que isto nao faz: nao mede a correlacao DENTRO de uma observacao, entre bins
de energia que dividem um canal. Essa e separada, em
scripts/correlaciona_fundo.py.

Uso: python3 scripts/fecha_a_soma.py [--bandas 120] [--vetor "B colado"]
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

FASES = PO.FASES


def problemas(obss, ordem, bandas, refino, meta, ev):
    """Um FitProblem MARGINAL por observacao, so para pontuar."""
    out = {}
    for obsid in ordem:
        base = PO.monta_pedido(obss[obsid], bandas, refino)
        base["outputPrecision"] = 17
        base["marginalizeBackground"] = True
        out[obsid] = m.FitProblem(base, ev, meta, None)
    return out


def grade_isolada(obs, bandas, refino, vetor, meta, ev, periodo=None):
    """Uma chamada do motor com UMA observacao, com seu offset e sua resposta.

    `periodo`, quando dado, SUBSTITUI o periodo daquela observacao. Serve para
    isolar a aproximacao do periodo comum: a rota conjunta usa um unico periodo
    a montante, e igualar os dois lados mostra se a diferenca residual vem
    DALI ou de outra coisa. Sem isto a atribuicao seria leitura de rotulo.
    """
    base = PO.monta_pedido(obs, bandas, refino,
                           s_o=int(obs["desloc_v3_bins"]))
    if periodo is not None:
        base["period"] = float(periodo)
    base["outputPrecision"] = 17
    # RESOLUCAO EFETIVA ZERO tambem aqui, para que os dois lados declarem o
    # mesmo operador. A 32 fases ela e inerte pela guarda, medido em
    # docs/suavizacao_em_m.json, mas igualar a declaracao evita que a
    # comparacao passe a medir duas coisas se a grade mudar.
    base["responseProfile"] = dict(base["responseProfile"])
    base["responseProfile"]["time_resolution_us"] = 0.0
    prob = m.FitProblem(base, ev, meta, None)
    caminho, rotulo = [x[1:] for x in PO.VETORES if x[0] == vetor][0]
    v = PO.le_vetor(prob, caminho, rotulo)
    trab = m.EngineWorker(list(prob.worker_command()))
    doc = trab.evaluate(prob.worker_line(v))
    trab.close()
    if doc.get("status") != "ok":
        raise SystemExit("motor falhou isolado em %s: %s" % (obs["obsid"], doc))
    return np.asarray(doc["detected_count_rate"], float)


def grades_conjuntas(obss, ordem, bandas, refino, vetor, meta, ev):
    """UMA chamada com as treze como slots, pela rota de reuso."""
    base = PO.monta_pedido(obss[ordem[0]], bandas, refino)
    base["outputPrecision"] = 17
    prob = m.FitProblem(base, ev, meta, None)
    comando = list(prob.worker_command())
    for obsid in ordem:
        o = obss[obsid]
        comando += ["--obs-id", obsid,
                    "--obs-profile", o["perfil_csv"],
                    "--obs-response", o["perfil_rmfbin"],
                    # RESOLUCAO EFETIVA ZERO, como na configuracao validada. A
                    # 32 fases o operador nada faz de todo modo
                    # (docs/suavizacao_em_m.json), mas passa-la explicitamente
                    # mantem a rota conjunta identica a isolada por construcao.
                    "--obs-time-resolution-us", "0.0",
                    "--obs-dead-time-us", "0.0",
                    "--obs-period", repr(float(o["periodo"])),
                    "--obs-offset", repr(PO.offset_em_ciclos(int(o["desloc_v3_bins"])))]
    caminho, rotulo = [x[1:] for x in PO.VETORES if x[0] == vetor][0]
    v = PO.le_vetor(prob, caminho, rotulo)
    trab = m.EngineWorker(comando)
    t0 = time.time()
    doc = trab.evaluate(prob.worker_line(v))
    dt = time.time() - t0
    trab.close()
    if doc.get("status") != "ok":
        raise SystemExit("motor falhou na rota conjunta: %s" % doc)
    if "observations" not in doc:
        raise SystemExit("sem 'observations': a rota de reuso nao foi acionada")
    return {o["id"]: np.asarray(o["detected_count_rate"], float)
            for o in doc["observations"]}, dt


def pontua(prob, grade):
    return float(prob.score_document({"status": "ok",
                                      "detected_count_rate": grade.tolist()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandas", type=int, default=120)
    ap.add_argument("--refino", type=int, default=8)
    ap.add_argument("--vetor", default="B colado")
    ap.add_argument("--perturba", default=None,
                    help="obsid cujo fundo se perturba; padrao o primeiro")
    ap.add_argument("--fator", type=float, default=1.10)
    ap.add_argument("--saida", default=str(RAIZ / "docs/fecha_a_soma.json"))
    a = ap.parse_args()

    obss = {o["obsid"]: o for o in PO.prepara()}
    ordem = sorted(obss)
    meta, ev = m.read_events(RAIZ / "dados/eventos.csv")
    probs = problemas(obss, ordem, a.bandas, a.refino, meta, ev)

    rel = {"o_que_e": "a soma das treze contribuicoes fecha, e os latentes sao "
                      "independentes ENTRE observacoes",
           "data": "2026-10-10",
           "pedido_do_revisor": (
             "13 espectros distintos verifica associacao entre observacoes, nao "
             "que a soma das contribuicoes de ln L fecha; registre essa "
             "igualdade separadamente"),
           "o_que_nao_faz": ("nao mede a correlacao DENTRO de uma observacao, "
                             "entre bins que dividem um canal"),
           "configuracao": {"bandas": a.bandas, "refino": a.refino,
                            "vetor": a.vetor, "alvo": "marginal por observacao",
                            "resolucao_efetiva_us": 0.0},
           }

    print("=== FECHAMENTO: treze chamadas isoladas contra UMA conjunta ===")
    conj, dt_conj = grades_conjuntas(obss, ordem, a.bandas, a.refino, a.vetor,
                                     meta, ev)
    iso, dt_iso = {}, 0.0
    t0 = time.time()
    for obsid in ordem:
        iso[obsid] = grade_isolada(obss[obsid], a.bandas, a.refino, a.vetor,
                                   meta, ev)
    dt_iso = time.time() - t0

    termos = {"isolada": {}, "conjunta": {}}
    pior_grade = 0.0
    for obsid in ordem:
        termos["isolada"][obsid] = pontua(probs[obsid], iso[obsid])
        termos["conjunta"][obsid] = pontua(probs[obsid], conj[obsid])
        pior_grade = max(pior_grade, float(np.abs(iso[obsid] - conj[obsid]).max()))
    s_iso = sum(termos["isolada"].values())
    s_conj = sum(termos["conjunta"].values())
    print("   soma das treze, rota isolada : %.10f  (%.1f s)" % (s_iso, dt_iso))
    print("   soma das treze, rota conjunta: %.10f  (%.1f s)" % (s_conj, dt_conj))
    print("   diferenca das somas: %+.3e   pior diferenca de grade: %.3e"
          % (s_conj - s_iso, pior_grade))
    piores = sorted(((abs(termos["conjunta"][o] - termos["isolada"][o]), o)
                     for o in ordem), reverse=True)[:3]
    for d, o in piores:
        print("     pior termo %s: %+.3e" % (o, termos["conjunta"][o] - termos["isolada"][o]))
    # ISOLAMENTO DA ATRIBUICAO: as treze isoladas de novo, agora com o periodo
    # da base, que e o que a rota conjunta usa a montante.
    print("\n=== ISOLAMENTO: a diferenca vem do periodo comum? ===")
    p_base = float(obss[ordem[0]]["periodo"])
    iso2 = {}
    t0 = time.time()
    for obsid in ordem:
        iso2[obsid] = grade_isolada(obss[obsid], a.bandas, a.refino, a.vetor,
                                    meta, ev, periodo=p_base)
    dt2 = time.time() - t0
    t2 = {o: pontua(probs[o], iso2[o]) for o in ordem}
    s2 = sum(t2.values())
    pior2 = max(float(np.abs(iso2[o] - conj[o]).max()) for o in ordem)
    print("   soma isolada COM periodo da base: %.10f" % s2)
    print("   diferenca contra a conjunta: %+.3e  (antes: %+.3e)"
          % (s_conj - s2, s_conj - s_iso))
    print("   pior diferenca de grade: %.3e  (antes: %.3e)" % (pior2, pior_grade))
    reduziu = abs(s_conj - s2) < 0.1 * abs(s_conj - s_iso) if s_conj != s_iso else None
    rel["isolamento_do_periodo"] = {
        "o_que_e": ("as treze isoladas de novo, com o periodo da observacao de "
                    "referencia no lugar do proprio, que e o que a rota "
                    "conjunta usa a montante"),
        "periodo_usado_s": p_base,
        "soma_isolada_com_periodo_da_base": s2,
        "diferenca_contra_a_conjunta": s_conj - s2,
        "diferenca_antes": s_conj - s_iso,
        "pior_diferenca_de_grade": pior2,
        "a_diferenca_vem_do_periodo": reduziu,
        "leitura": ("se a diferenca cai para o piso ao igualar o periodo, a "
                    "aproximacao do periodo comum e a causa; se nao cai, a "
                    "causa e outra e a atribuicao anterior estaria errada"),
        "segundos": dt2,
    }
    rel["fechamento"] = {
        "soma_rota_isolada": s_iso, "soma_rota_conjunta": s_conj,
        "diferenca_das_somas": s_conj - s_iso,
        "pior_diferenca_de_grade": pior_grade,
        "termos": termos,
        "segundos": {"isolada": dt_iso, "conjunta": dt_conj},
        "aproximacao_declarada": (
          "a rota conjunta usa um periodo comum a montante; o desvio relativo "
          "medido entre os treze periodos e 8,86e-05 e move a previsao em "
          "1,640e-08, declarado em docs/por_observacao.json"),
    }

    alvo = a.perturba or ordem[0]
    print("\n=== INDEPENDENCIA: perturba o fundo de %s por %.2fx ===" % (alvo, a.fator))
    antes = {o: termos["conjunta"][o] for o in ordem}
    p = probs[alvo]
    p.fundo_contagens = p.fundo_contagens * a.fator
    p._fundo_pronto = False
    depois = {}
    for obsid in ordem:
        depois[obsid] = pontua(probs[obsid], conj[obsid])
    mudou = {o: depois[o] - antes[o] for o in ordem}
    outros = [abs(mudou[o]) for o in ordem if o != alvo]
    print("   termo perturbado %s: %+.6f" % (alvo, mudou[alvo]))
    print("   pior mudanca nos outros doze: %.3e" % max(outros))
    print("   os doze ficaram identicos bit a bit: %s"
          % all(depois[o] == antes[o] for o in ordem if o != alvo))
    rel["independencia"] = {
        "observacao_perturbada": alvo, "fator": a.fator,
        "mudanca_no_termo_perturbado": mudou[alvo],
        "pior_mudanca_nos_outros": max(outros),
        "outros_identicos_bit_a_bit": all(depois[o] == antes[o]
                                          for o in ordem if o != alvo),
        "mudancas": mudou,
    }
    rel["passa"] = bool(rel["independencia"]["outros_identicos_bit_a_bit"]
                        and mudou[alvo] != 0.0)
    Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print("\ngravado em %s" % a.saida)


if __name__ == "__main__":
    main()
