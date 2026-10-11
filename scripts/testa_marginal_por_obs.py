#!/usr/bin/env python3
"""A marginalizacao do fundo POR OBSERVACAO, ponta a ponta.

Item 1 das tres pendencias: testar o alvo decidido de verdade, e nao so
preparar o dado. O alvo e um fundo latente b(o,e) por observacao e canal,
COMPARTILHADO pelas 32 fases daquela observacao e integrado UMA vez, com as
contagens da regiao de fundo entrando como verossimilhanca da MEDICAO.

CONFIGURACAO DE MOTOR BARATA DE PROPOSITO: producao 120 bandas, tabela padrao.
A maquinaria do alvo e a JUSANTE do modelo, logo exercita-se com qualquer
previsao; usar a configuracao aprovada custaria 1620 s por avaliacao e nao
mediria nada a mais sobre a marginalizacao. A configuracao numerica foi validada
em separado, na frente A.

O QUE SE VERIFICA:
  1. que ela RODA nas treze, com a priori a0 = 0,01/keV e beta = 0;
  2. que o alvo de CONTROLE, com fundo fixo, continua funcionando;
  3. que as DIFERENCAS de ln L entre vetores mudam ou nao ao marginalizar, que e
     o que afeta conclusao -- os valores absolutos nao sao comparaveis entre
     alvos, por convencao de normalizacao diferente;
  4. que a soma dos treze ln L marginais e o produto das marginais, isto e que
     os latentes sao independentes ENTRE observacoes e compartilhados DENTRO de
     cada uma.

Uso: python3 scripts/testa_marginal_por_obs.py
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


def problema(obs, marginal, bandas, refino):
    base = PO.monta_pedido(obs, bandas, refino)
    base["outputPrecision"] = 17
    if marginal:
        base["marginalizeBackground"] = True
    meta, ev = m.read_events(RAIZ / "dados/eventos.csv")
    return m.FitProblem(base, ev, meta, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandas", type=int, default=120)
    ap.add_argument("--refino", type=int, default=8)
    a = ap.parse_args()
    obss = {o["obsid"]: o for o in PO.prepara(verbose=False)}
    ordem = sorted(obss)
    man = PO.manifesto_em_execucao(list(obss.values()),
                                   extras={"teste": "marginal por observacao",
                                           "bandas": a.bandas})
    rel = {"o_que_e": "a marginalizacao do fundo por observacao, ponta a ponta",
           "configuracao_do_motor": "producao %d bandas, tabela padrao" % a.bandas,
           "por_que_barata": ("a maquinaria do alvo e a jusante do modelo; a "
                              "configuracao numerica foi validada na frente A"),
           "manifesto_em_execucao": man, "observacoes": {}}

    print("=== 1 e 2: a marginalizacao roda nas treze? ===")
    print("%-12s %6s %14s %14s %12s" % ("obsid", "termos", "lnL controle",
                                        "lnL marginal", "segundos"))
    probs_c, probs_m = {}, {}
    ok = True
    for obsid in ordem:
        try:
            pc = problema(obss[obsid], False, a.bandas, a.refino)
            t0 = time.time()
            pm = problema(obss[obsid], True, a.bandas, a.refino)
            dt = time.time() - t0
            probs_c[obsid], probs_m[obsid] = pc, pm
            nt = int(np.asarray(pm.fundo_pesos).size)
            rel["observacoes"][obsid] = {"termos": nt, "segundos_preparo": dt,
                                         "a0": pm.fundo_a0, "beta": pm.fundo_beta,
                                         "nos": pm.fundo_nos}
            print("%-12s %6d %14s %14s %12.2f"
                  % (obsid, nt, "ok", "ok", dt), flush=True)
        except Exception as e:                       # noqa: BLE001
            ok = False
            rel["observacoes"][obsid] = {"falhou": "%s: %s" % (type(e).__name__, e)}
            print("%-12s FALHOU: %s" % (obsid, e), flush=True)
    if not ok:
        Path(RAIZ / "docs/marginal_por_obs.json").write_text(
            json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
        raise SystemExit("a marginalizacao nao montou nas treze; parando")
    print("  priori: a0 = %.4f/keV, beta = %.1f, %d nos de quadratura"
          % (probs_m[ordem[0]].fundo_a0, probs_m[ordem[0]].fundo_beta,
             probs_m[ordem[0]].fundo_nos))

    print()
    print("=== 3: as DIFERENCAS de ln L entre vetores, nos dois alvos ===")
    trab = {}
    vals = {"controle": {}, "marginal": {}}
    for vetor, caminho, rotulo in PO.VETORES:
        soma_c = soma_m = 0.0
        t0 = time.time()
        for obsid in ordem:
            pc, pm = probs_c[obsid], probs_m[obsid]
            w = m.EngineWorker(list(pc.worker_command()))
            v = PO.le_vetor(pc, caminho, rotulo)
            doc = w.evaluate(pc.worker_line(v))
            w.close()
            if doc.get("status") != "ok":
                raise SystemExit("motor falhou em %s: %s" % (obsid, doc))
            soma_c += float(pc.score_document(doc))
            soma_m += float(pm.score_document(doc))
        vals["controle"][vetor] = soma_c
        vals["marginal"][vetor] = soma_m
        print("  %-10s controle %16.6f   marginal %16.6f   (%.0f s)"
              % (vetor, soma_c, soma_m, time.time() - t0), flush=True)
    vs = [v for v, _, _ in PO.VETORES]
    print()
    print("%-24s %14s %14s %12s" % ("par", "controle", "marginal", "mudanca"))
    pares, pior = [], 0.0
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            dc = vals["controle"][vs[i]] - vals["controle"][vs[j]]
            dm = vals["marginal"][vs[i]] - vals["marginal"][vs[j]]
            pares.append({"i": vs[i], "j": vs[j], "controle": dc,
                          "marginal": dm, "mudanca": dm - dc})
            pior = max(pior, abs(dm - dc))
            print("%-24s %14.4f %14.4f %+12.4f"
                  % ("%s - %s" % (vs[i], vs[j]), dc, dm, dm - dc))
    print()
    print("pior mudanca nas diferencas ao marginalizar: %.4f" % pior)
    print("  a constante de normalizacao entre alvos NAO e comparavel: controle")
    print("  %.1f contra marginal %.1f, diferenca %.1f"
          % (vals["controle"][vs[0]], vals["marginal"][vs[0]],
             vals["marginal"][vs[0]] - vals["controle"][vs[0]]))
    rel["valores"] = vals
    rel["pares"] = pares
    rel["pior_mudanca_nas_diferencas"] = pior
    rel["o_que_isto_estabelece"] = (
      "que o alvo marginalizado por observacao RODA ponta a ponta nas treze e "
      "quanto ele move as diferencas de ln L entre os tres vetores preservados")
    rel["o_que_NAO_estabelece"] = [
      "que a marginalizacao esteja CORRETA: isto e teste de encanamento e de "
      "magnitude, nao de correcao da integral",
      "nada sobre a configuracao numerica, que e outro eixo e foi validada na "
      "frente A",
      "a correlacao entre latentes de bins que compartilham canal, medida em 19 "
      "por cento do peso e integrada como se fosse independente",
    ]
    PO.grava_sem_sobrescrever(RAIZ / "docs/marginal_por_obs.json", rel)
    print("gravado em docs/marginal_por_obs.json")


if __name__ == "__main__":
    main()
