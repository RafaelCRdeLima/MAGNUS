#!/usr/bin/env python3
"""Paralelizar o laco de fases: a previsao continua identica, e quanto rende.

O revisor, 10/10/2026: "Priorizar um benchmark de paralelizacao das fases do
motor, verificando identidade das previsoes e desempenho com diferentes
combinacoes de threads e workers."

DUAS PERGUNTAS, e a primeira manda. Se a previsao mudar, o ganho de velocidade
nao interessa, porque toda a validacao numerica foi feita sobre o caminho
serial. A identidade e exigida BIT A BIT, e nao dentro de tolerancia: o laco foi
reescrito para que ela seja possivel, com os somadores que atravessavam fases
levados a buffers por fase e reduzidos depois NA ORDEM DAS FASES.

TRES COMPARACOES, porque ha tres mudancas distintas empilhadas:
  1. o binario VALIDADO contra o novo SEM OpenMP: isola a reestruturacao dos
     acumuladores. Se falhar aqui, o problema e a reescrita, nao o paralelismo;
  2. o novo sem OpenMP contra o novo COM OpenMP e --threads 1: isola o efeito de
     compilar com -fopenmp, que muda o binario mesmo sem thread nenhuma;
  3. --threads 1 contra 2, 4, 8, ...: isola o paralelismo propriamente.

O DESEMPENHO e medido depois, e em duas direcoes, porque o cluster pode gastar
seus nucleos de dois modos: T threads dentro de uma avaliacao, ou W avaliacoes
simultaneas. O que decide a producao e o throughput do par (T, W), nao o speedup
de T isolado.

Uso: python3 scripts/paraleliza_fases.py [--bandas 480] [--tabela 115200]
                                         [--threads 1 2 4 8] [--nucleos 12]
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))

VALIDADO = os.environ.get("MAGNUS_ENGINE_VALIDADO", "")
SERIAL = str(RAIZ / "build/magnus_engine")
OMP = str(RAIZ / "build/magnus_engine_omp")


def uma_avaliacao(binario, threads, bandas, tabela, malha, M, obsid=None):
    """Roda UMA avaliacao do alvo por observacao com o binario pedido.

    Em subprocesso, porque trocar `mcmc_fit.ENGINE` depois do import nao e
    confiavel: o modulo e lido por outros e a troca por variavel de ambiente e a
    porta declarada.
    """
    # `PO.prepara(regravar=False)` e OBRIGATORIO aqui, e a razao e um dano que
    # aconteceu: na varredura de pares varios processos constroem o alvo ao
    # mesmo tempo, cada um chamando prepara(), que REGRAVA os treze npz. Em
    # 10/10/2026 isso corrompeu tres deles, restaurados depois da copia do
    # cluster. A escrita passou a ser atomica em por_observacao.py, o que fecha
    # a janela de leitura parcial, mas escrever treze arquivos em paralelo para
    # nada continua sendo erro: aqui ninguem precisa regravar.
    codigo = """
import os, sys, time, json
sys.path.insert(0, %r)
import mcmc_fit as m, por_observacao as PO, alvo_por_observacao as APO
_prep = PO.prepara
PO.prepara = lambda *a, **k: _prep(*a, **{**k, "regravar": False})
alvo = APO.AlvoPorObservacao(bandas=%d, refino=8, media_de_fase=%d,
                             marginal=True, malha=%r, tabela=%r)
nome, caminho, rot = PO.VETORES[0]
v = PO.le_vetor(alvo, caminho, rot)
comando = list(alvo.worker_command())
if %d > 1 or %s:
    comando += ["--threads", "%d"]
t = m.EngineWorker(comando)
t0 = time.time()
termos = alvo.termos(t, v)
dt = time.time() - t0
doc = t.evaluate(alvo.worker_line(v))
t.close()
grades = {o["id"]: o["detected_count_rate"] for o in doc["observations"]}
print("@@" + json.dumps({"segundos": dt, "termos": termos,
                         "soma": sum(termos.values()), "grades": grades}))
""" % (str(RAIZ / "scripts"), bandas, M, malha, tabela,
       threads, "True" if threads >= 1 and binario == OMP else "False", threads)
    env = dict(os.environ)
    env["MAGNUS_ENGINE"] = binario
    env.setdefault("OMP_NUM_THREADS", str(threads))
    r = subprocess.run([sys.executable, "-I", "-c", codigo], capture_output=True,
                       text=True, env=env, cwd=str(RAIZ))
    linha = [l for l in r.stdout.splitlines() if l.startswith("@@")]
    if not linha:
        raise RuntimeError("avaliacao falhou:\n%s\n%s"
                           % (r.stdout[-800:], r.stderr[-800:]))
    return json.loads(linha[0][2:])


def compara(a, b):
    """Diferenca bit a bit entre duas avaliacoes, grade por grade."""
    pior_grade = 0.0
    identicas = True
    for obsid in a["grades"]:
        g1 = np.asarray(a["grades"][obsid], float)
        g2 = np.asarray(b["grades"][obsid], float)
        if not np.array_equal(g1, g2):
            identicas = False
        pior_grade = max(pior_grade, float(np.abs(g1 - g2).max()))
    return {"grades_identicas_bit_a_bit": identicas,
            "pior_diferenca_de_grade": pior_grade,
            "diferenca_da_soma_de_ln": b["soma"] - a["soma"],
            "somas_identicas": a["soma"] == b["soma"]}


#: LIMITE DE LEITURA DESTA MEDIDA, declarado porque mordeu: cada ponto da
#: varredura chama `uma_avaliacao`, que faz DUAS avaliacoes do alvo, uma
#: cronometrada em `termos` e outra para extrair as grades. Logo o campo
#: `avaliacoes_por_hora` subestima o throughput absoluto por um fator de cerca
#: de dois, uniforme entre os pares. O RANKING dos pares vale; para DIMENSIONAR
#: use os tempos da secao de identidade, que sao de uma avaliacao cada.
LIMITE_DA_VARREDURA = (
    "cada ponto faz duas avaliacoes do alvo, logo avaliacoes_por_hora "
    "subestima o throughput absoluto por cerca de 2x, de forma uniforme. O "
    "ranking dos pares vale; para dimensionar, usar os tempos da secao de "
    "identidade, que sao de uma avaliacao cada")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandas", type=int, default=480)
    ap.add_argument("--tabela", type=int, default=115200)
    ap.add_argument("--malha", default="psi0", choices=["psi0", "producao"])
    ap.add_argument("--media-de-fase", type=int, default=1, dest="M")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--nucleos", type=int, default=os.cpu_count() or 4,
                    help="teto para o produto threads x workers no benchmark")
    ap.add_argument("--pares", type=str, default="",
                    help="pares T:W separados por virgula; vazio = automatico")
    ap.add_argument("--so-identidade", action="store_true")
    ap.add_argument("--saida", default=str(RAIZ / "docs/paraleliza_fases.json"))
    a = ap.parse_args()

    cfg = dict(bandas=a.bandas, tabela=a.tabela,
               malha=(None if a.malha == "producao" else "psi0"), M=a.M)
    rel = {"o_que_e": "paralelizacao do laco de fases: identidade e desempenho",
           "data": "2026-10-10",
           "pedido_do_revisor": (
             "benchmark de paralelizacao das fases do motor, verificando "
             "identidade das previsoes e desempenho com diferentes combinacoes "
             "de threads e workers"),
           "criterio": ("identidade BIT A BIT da previsao; se falhar, o ganho de "
                        "velocidade nao interessa, porque a validacao numerica "
                        "toda foi feita sobre o caminho serial"),
           "configuracao": {"bandas": a.bandas, "tabela": a.tabela,
                            "malha": a.malha, "media_de_fase": a.M},
           "binarios": {}, "identidade": [], "desempenho": [],
           "limite_da_varredura_de_pares": LIMITE_DA_VARREDURA}

    for rotulo, caminho in (("validado", VALIDADO), ("serial", SERIAL), ("omp", OMP)):
        if caminho and Path(caminho).exists():
            import hashlib
            rel["binarios"][rotulo] = {
                "caminho": caminho,
                "sha256": hashlib.sha256(Path(caminho).read_bytes()).hexdigest()[:16]}

    print("=== IDENTIDADE ===", flush=True)
    base = None
    if VALIDADO and Path(VALIDADO).exists():
        print("  1. binario validado ...", flush=True)
        val = uma_avaliacao(VALIDADO, 1, **cfg)
        print("     %.1f s, soma %.10f" % (val["segundos"], val["soma"]), flush=True)
        base = val
    print("  2. novo, sem OpenMP ...", flush=True)
    ser = uma_avaliacao(SERIAL, 1, **cfg)
    print("     %.1f s, soma %.10f" % (ser["segundos"], ser["soma"]), flush=True)
    if base is not None:
        c = compara(base, ser)
        c["par"] = "validado contra novo sem OpenMP"
        c["o_que_isola"] = "a reestruturacao dos acumuladores por fase"
        rel["identidade"].append(c)
        print("     -> %s  pior |dif| %.3e  d(soma) %+.3e"
              % ("IDENTICAS" if c["grades_identicas_bit_a_bit"] else "DIFEREM",
                 c["pior_diferenca_de_grade"], c["diferenca_da_soma_de_ln"]), flush=True)
    anterior = ser
    for t in a.threads:
        print("  3. OpenMP com %d thread(s) ..." % t, flush=True)
        r = uma_avaliacao(OMP, t, **cfg)
        c = compara(ser, r)
        c["par"] = "novo sem OpenMP contra OpenMP com %d thread(s)" % t
        c["o_que_isola"] = ("compilar com -fopenmp" if t == 1
                            else "o paralelismo com %d threads" % t)
        c["threads"] = t
        c["segundos"] = r["segundos"]
        rel["identidade"].append(c)
        print("     %.1f s  -> %s  pior |dif| %.3e  d(soma) %+.3e"
              % (r["segundos"],
                 "IDENTICAS" if c["grades_identicas_bit_a_bit"] else "DIFEREM",
                 c["pior_diferenca_de_grade"], c["diferenca_da_soma_de_ln"]),
              flush=True)
        anterior = r
    rel["identidade_passa"] = all(c["grades_identicas_bit_a_bit"]
                                  for c in rel["identidade"])
    print("  VEREDITO DA IDENTIDADE: %s"
          % ("todas identicas bit a bit" if rel["identidade_passa"]
             else "HOUVE DIFERENCA, ver o relatorio"), flush=True)

    if not a.so_identidade:
        print("\n=== DESEMPENHO, pares (threads, workers) ===", flush=True)
        if a.pares:
            pares = [tuple(int(x) for x in p.split(":")) for p in a.pares.split(",")]
        else:
            pares = [(t, w) for t in (1, 2, 4, 8) for w in (1, 2, 4, 8)
                     if t * w <= a.nucleos]
        print("  teto de nucleos: %d; pares: %s" % (a.nucleos, pares), flush=True)
        for T, W in pares:
            import concurrent.futures
            t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=W) as ex:
                fut = [ex.submit(uma_avaliacao, OMP, T, **cfg) for _ in range(W)]
                res = [f.result() for f in fut]
            dt = time.time() - t0
            somas = {r["soma"] for r in res}
            p = {"threads": T, "workers": W, "segundos_da_rodada": dt,
                 "avaliacoes": W, "avaliacoes_por_hora": W * 3600.0 / dt,
                 "mesma_soma_em_todos": len(somas) == 1,
                 "nucleos_usados": T * W}
            rel["desempenho"].append(p)
            print("  T=%-3d W=%-3d  rodada %7.1f s   %8.1f aval/h   iguais=%s"
                  % (T, W, dt, p["avaliacoes_por_hora"], len(somas) == 1), flush=True)
        if rel["desempenho"]:
            ref = [p for p in rel["desempenho"] if p["threads"] == 1 and p["workers"] == 1]
            if ref:
                base_h = ref[0]["avaliacoes_por_hora"]
                for p in rel["desempenho"]:
                    p["ganho_contra_1x1"] = p["avaliacoes_por_hora"] / base_h
            melhor = max(rel["desempenho"], key=lambda p: p["avaliacoes_por_hora"])
            rel["melhor_par"] = {"threads": melhor["threads"], "workers": melhor["workers"],
                                 "avaliacoes_por_hora": melhor["avaliacoes_por_hora"]}
            print("\n  melhor par: T=%d W=%d, %.1f avaliacoes/h"
                  % (melhor["threads"], melhor["workers"],
                     melhor["avaliacoes_por_hora"]))
    Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print("\ngravado em %s" % a.saida)


if __name__ == "__main__":
    main()
