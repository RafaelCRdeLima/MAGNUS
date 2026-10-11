#!/usr/bin/env python3
"""A comparacao COM e SEM a 0402851001, nos novos melhores pontos.

Decisao do revisor em 10/10/2026: manter as treze no alvo provisorio, sem
interromper as reotimizacoes, e fazer uma comparacao limitada com e sem a
0402851001, primeiro nos novos melhores pontos e depois em pilotos se o efeito
justificar. O fechamento da inclusao fiducial vem da adequacao do fundo, dos
residuos e da estabilidade, NAO da bacia favorecida.

POR QUE ESSA OBSERVACAO. Ela tem nivel de particulas cerca de vinte vezes o das
outras doze, inclusive contra as cinco de MESMA area e MESMO submodo, e isso nao
e flare isolado: 77 a 80 por cento da exposicao dela passa de qualquer limiar
entre 0,2 e 1,0 ct/s (docs/diagnostico_flare.json). O latente b(o,e) trata a
incerteza do fundo estimado, mas nao garante que o fundo da regiao externa
represente o fundo sob a fonte, nem modela variacao temporal residual dentro da
observacao.

DUAS COISAS SEPARADAS, e confundi-las foi o erro da primeira versao deste
script. Nos MESMOS parametros e com latentes independentes por observacao, a
contribuicao das doze comuns tem de ser IDENTICA com ou sem a decima terceira no
alvo: isso e TESTE DE IMPLEMENTACAO, e se nao der zero ha defeito. Eu havia
reportado essa diferenca como se fosse informacao cientifica.

A sensibilidade cientifica aparece em outros tres lugares, que este script mede:
  quanto a 0402851001 acrescenta aos CONTRASTES entre pontos;
  quanto os RESIDUOS dela indicam inadequacao do modelo de fundo, comparados
    com os das outras doze;
  e quanto a exclusao desloca os OTIMOS, que exige reotimizar no alvo de doze e
    nao se obtem avaliando pontos, por isso fica como job separado.

O QUE ISTO NAO DECIDE: a inclusao final. A comparacao pontual decide se vale
ampliar o teste.

Uso: python3 scripts/compara_sem_1001.py --saidas a.json b.json [--threads 16]
"""
import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import mcmc_fit as m  # noqa: E402
import alvo_por_observacao as APO  # noqa: E402
import revalida_ponto as RP  # noqa: E402

FORA = "0402851001"


def avalia(ponto, cfg, threads, excluir, com_residuos=False):
    alvo = APO.AlvoPorObservacao(refino=8, marginal=True, threads=threads,
                                 excluir=excluir, **cfg)
    if list(alvo.names) != list(ponto["nomes"]):
        raise SystemExit("ordem dos nomes nao bate com o problema")
    trab = m.EngineWorker(list(alvo.worker_command()))
    t0 = time.time()
    termos = alvo.termos(trab, ponto["vetor"])
    doc = trab.evaluate(alvo.worker_line(ponto["vetor"])) if com_residuos else None
    dt = time.time() - t0
    trab.close()
    res = residuos(alvo, doc) if com_residuos else None
    return termos, dt, res


def residuos(alvo, doc):
    """Residuos de Poisson por observacao, no fundo ESPERADO a posteriori.

    O alvo marginaliza o fundo, logo nao ha "o" modelo: usa-se o valor central
    de pi_e, que e o primeiro cumulante k1 do latente daquele bin, que e o mesmo
    que `score_document(..., return_model=True)` emprega para os graficos. O
    objetivo aqui nao e pontuar, e ver se a ESTRUTURA dos residuos de uma
    observacao denuncia fundo inadequado: excesso sistematico num bloco de
    canais, por exemplo, que um escalar por canal nao conseguiria absorver.
    """
    import numpy as np
    saida = {}
    grades = {o["id"]: o["detected_count_rate"] for o in doc["observations"]}
    for obsid in alvo.ordem:
        prob = alvo.probs[obsid]
        ln, esperado = prob.score_document(
            {"status": "ok", "detected_count_rate": grades[obsid]},
            return_model=True)
        mu = np.asarray(esperado, float)
        n = np.asarray(prob.observed, float)
        bons = mu > 0
        # Residuo de Pearson para Poisson, com o cuidado de nao dividir por zero
        r = np.zeros_like(mu)
        r[bons] = (n[bons] - mu[bons]) / np.sqrt(mu[bons])
        E = prob.energy_bins
        rr = r.reshape(-1, E)
        por_energia = rr.sum(axis=0) / np.sqrt(max(rr.shape[0], 1))
        saida[obsid] = {
            "chi2_por_grau": float((r[bons] ** 2).sum() / max(bons.sum(), 1)),
            "residuo_medio": float(r[bons].mean()),
            "pior_bin_de_energia": int(np.argmax(np.abs(por_energia))),
            "pior_residuo_por_energia": float(por_energia[np.argmax(np.abs(por_energia))]),
            "fracao_de_bins_acima_de_3_sigma": float((np.abs(r[bons]) > 3).mean()),
        }
    return saida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saidas", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--configuracao", default="grosso",
                    choices=["grosso", "fino"])
    ap.add_argument("--saida", default=str(RAIZ / "docs/compara_sem_1001.json"))
    a = ap.parse_args()
    cfg = RP.GROSSO if a.configuracao == "grosso" else RP.FINO

    pontos = [RP.le_ponto(s) for s in a.saidas]
    nomes = [p["rotulo"] for p in pontos]
    if len(set(nomes)) != len(nomes):
        raise SystemExit(
            "identificadores repetidos entre as saidas: %s. Dois pontos com o "
            "mesmo nome se sobrescrevem e os contrastes saem errados." % nomes)
    rel = {"o_que_e": "o efeito da 0402851001 nos novos melhores pontos",
           "data": time.strftime("%Y-%m-%d"),
           "decisao_que_originou": (
             "revisor, 10/10/2026: manter as treze no alvo provisorio e fazer "
             "uma comparacao limitada com e sem a 0402851001"),
           "configuracao": a.configuracao, "o_que_NAO_decide": "a inclusao",
           "pontos": [], "contrastes": {}}
    doze = {}
    rel["teste_de_implementacao"] = {
        "o_que_exige": ("nos mesmos parametros e com latentes independentes por "
                        "observacao, cada termo das doze comuns tem de ser "
                        "IDENTICO com ou sem a decima terceira no alvo"),
        "por_observacao": [], "passa": True}
    for p in pontos:
        print("=== %s ===" % p["rotulo"], flush=True)
        t13, dt13, res13 = avalia(p, cfg, a.threads, (), com_residuos=True)
        t12, dt12, _ = avalia(p, cfg, a.threads, (FORA,))
        comuns = [o for o in t13 if o != FORA]
        # TESTE DE IMPLEMENTACAO, termo a termo e nao so na soma: uma soma igual
        # poderia esconder dois termos trocados.
        difs = {o: t12[o] - t13[o] for o in comuns}
        pior = max(difs, key=lambda o: abs(difs[o]))
        identicos = all(t12[o] == t13[o] for o in comuns)
        rel["teste_de_implementacao"]["por_observacao"].append(
            {"rotulo": p["rotulo"], "todos_identicos_bit_a_bit": identicos,
             "pior_termo": pior, "pior_diferenca": float(difs[pior])})
        if not identicos:
            rel["teste_de_implementacao"]["passa"] = False
        s13 = sum(t13[o] for o in comuns)
        s12 = sum(t12[o] for o in comuns)
        reg = {"rotulo": p["rotulo"], "arquivo": p["arquivo"],
               "soma_das_treze": float(sum(t13.values())),
               "soma_das_doze_no_alvo_de_13": float(s13),
               "soma_das_doze_no_alvo_de_12": float(s12),
               "termo_da_1001": float(t13[FORA]),
               "fracao_da_soma": float(t13[FORA] / sum(t13.values())),
               "residuos": res13,
               "segundos": {"treze": dt13, "doze": dt12}}
        rel["pontos"].append(reg)
        doze[p["rotulo"]] = float(s12)
        print("  TESTE DE IMPLEMENTACAO: as doze comuns identicas bit a bit: %s"
              "  (pior %s: %+.3e)" % (identicos, pior, difs[pior]), flush=True)
        print("  termo da %s: %.4f (%.2f%% da soma)"
              % (FORA, reg["termo_da_1001"], 100 * reg["fracao_da_soma"]))
        if res13:
            c = {o: res13[o]["chi2_por_grau"] for o in res13}
            doze_c = [v for o, v in c.items() if o != FORA]
            print("  residuos, chi2 por grau: %s = %.3f, as outras doze de "
                  "%.3f a %.3f" % (FORA, c.get(FORA, float("nan")),
                                   min(doze_c), max(doze_c)), flush=True)
    if len(doze) > 1:
        rots = list(doze)
        treze = {r["rotulo"]: r["soma_das_treze"] for r in rel["pontos"]}
        print("\n=== contrastes, com e sem ===")
        for i in range(len(rots)):
            for j in range(i + 1, len(rots)):
                a_, b_ = rots[i], rots[j]
                c13 = treze[a_] - treze[b_]
                c12 = doze[a_] - doze[b_]
                rel["contrastes"]["%s - %s" % (a_, b_)] = {
                    "com_as_treze": c13, "sem_a_1001": c12,
                    "mudanca": c12 - c13}
                print("  %-22s com 13 %+10.4f   sem a 1001 %+10.4f   muda %+.4f"
                      % ("%s - %s" % (a_, b_), c13, c12, c12 - c13), flush=True)
    Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print("\ngravado em %s" % a.saida)


if __name__ == "__main__":
    main()
