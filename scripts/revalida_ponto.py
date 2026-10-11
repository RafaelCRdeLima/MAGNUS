#!/usr/bin/env python3
"""Revalida um ponto novo contra a REFERENCIA FINA, psi_0 960 com M = 15.

A configuracao aprovada o foi "para reotimizacao e pilotos, COM REVALIDACAO DOS
NOVOS PONTOS" (docs/frenteA_fechada.json). A frente A mediu a discrepancia entre
grosso e fino nas SEMENTES ANTIGAS: -0,01815, -0,01424 e -0,00391 nos tres
contrastes. Um ponto novo, achado pela reotimizacao, esta em outro lugar da
superficie, e nada garante que a discrepancia ali seja a mesma. Este script
mede, ponto por ponto.

O QUE ELE COMPARA, e por que mais de uma coisa:
  o ln L de cada observacao e a soma, nas duas configuracoes;
  a grade de taxa detectada, celula por celula, porque cancelamento na soma
    pode esconder erro distribuido;
  e, quando mais de um ponto e dado, os CONTRASTES entre eles, que e o que move
    conclusao. A discrepancia de um valor absoluto nao e comparavel entre
    configuracoes; a de um contraste e.

O QUE ELE NAO FAZ: nao certifica o erro da propria referencia fina contra a
integral continua. Isso segue aberto, como a frente A declarou.

Uso: python3 scripts/revalida_ponto.py --saidas a.json b.json [--threads 16]
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
import alvo_por_observacao as APO  # noqa: E402

#: As duas configuracoes, pelos nomes que a frente A usou.
GROSSO = dict(bandas=480, tabela=115200, malha="psi0", media_de_fase=5)
FINO = dict(bandas=960, tabela=115200, malha="psi0", media_de_fase=15)


def le_ponto(caminho):
    """O vetor otimo de uma saida de reotimizacao, com os nomes."""
    d = json.loads(Path(caminho).read_text())
    if "otimo" not in d:
        raise SystemExit("%s nao tem 'otimo': nao e saida de reotimizacao"
                         % caminho)
    return {"arquivo": str(caminho), "nomes": d["nomes"],
            "vetor": list(d["otimo"]["vetor"]),
            "vetor_de_partida": list(d["partida"]["vetor"]),
            "ln_posterior_registrado": d["otimo"]["lnPosterior"],
            "ln_posterior_da_partida": d["partida"]["lnPosterior"],
            "parcial": d.get("parcial", False),
            "ganho": d.get("ganho"),
            "rotulo": _identificador(caminho, d)}


def _identificador(caminho, d):
    """Um nome que DISTINGUE os pontos, e nao o rotulo interno da semente.

    O rotulo de `origem` vem do arquivo de semente e so distingue "colado" de
    "aberto": o lancador passa "aberto" tanto para B aberto quanto para A,
    porque a condicao dele separa "B colado" do resto. Com isso dois pontos
    chegaram aqui com o mesmo nome e um SOBRESCREVEU o outro no dicionario de
    medidas, de modo que o unico contraste reportado nos jobs 47546 e 47547 era
    B colado menos A rotulado como "colado - aberto". O nome passa a vir do
    arquivo de saida, que carrega a semente, e duplicata e recusada.
    """
    nome = Path(caminho).stem
    for prefixo in ("reot_", "reotimizacao_"):
        if nome.startswith(prefixo):
            nome = nome[len(prefixo):]
    return nome.replace("_", " ") or d.get("origem", {}).get("rotulo", "?")


def avalia(ponto, cfg, threads):
    alvo = APO.AlvoPorObservacao(refino=8, marginal=True, threads=threads, **cfg)
    if list(alvo.names) != list(ponto["nomes"]):
        raise SystemExit("ordem dos nomes nao bate com o problema: um vetor "
                         "fora de ordem devolve ln L plausivel e errado")
    trab = m.EngineWorker(list(alvo.worker_command()))
    t0 = time.time()
    termos = alvo.termos(trab, ponto["vetor"])
    doc = trab.evaluate(alvo.worker_line(ponto["vetor"]))
    dt = time.time() - t0
    trab.close()
    # AS GRADES VEM COM 32*M FASES, e comparar grosso com fino direto quebra:
    # M = 5 da 160 linhas e M = 15 da 480. Foi o que derrubou o job 47537 depois
    # de ja ter medido as somas. A comparacao certa e na grade OBSERVADA, de 32
    # fases, que e a media das M amostras finas, exatamente o que o adaptador
    # faz para pontuar.
    fases = PO.FASES
    grades = {}
    for o in doc["observations"]:
        g = np.asarray(o["detected_count_rate"], float)
        if g.shape[0] % fases:
            raise SystemExit("grade com %d fases nao e multiplo de %d"
                             % (g.shape[0], fases))
        grades[o["id"]] = g.reshape(fases, g.shape[0] // fases, -1).mean(axis=1)
    return {"termos": termos, "soma": float(sum(termos.values())),
            "segundos": dt, "grades": grades}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saidas", nargs="+", required=True,
                    help="saidas de reotimizacao, uma por semente")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--ganho", action="store_true",
                    help="avalia TAMBEM o vetor de partida, nas duas "
                         "configuracoes, e reporta o GANHO em cada uma. E este "
                         "o teste de se o ganho da reotimizacao e real ou esta "
                         "explorando erro numerico da configuracao grossa: um "
                         "ganho que encolhe na referencia fina era erro")
    ap.add_argument("--saida", default=str(RAIZ / "docs/revalida_ponto.json"))
    a = ap.parse_args()

    pontos = [le_ponto(s) for s in a.saidas]
    nomes = [p["rotulo"] for p in pontos]
    if len(set(nomes)) != len(nomes):
        raise SystemExit(
            "identificadores repetidos entre as saidas: %s. Dois pontos com o "
            "mesmo nome se sobrescrevem e os contrastes saem errados." % nomes)
    rel = {"o_que_e": "revalidacao dos pontos novos contra a referencia fina",
           "data": time.strftime("%Y-%m-%d"),
           "por_que": ("a configuracao aprovada o foi para reotimizacao e "
                       "pilotos COM revalidacao dos novos pontos: a "
                       "discrepancia medida na frente A vale nas sementes "
                       "antigas, e um ponto novo esta em outro lugar da "
                       "superficie"),
           "configuracoes": {"grosso": GROSSO, "fino": FINO},
           "o_que_NAO_certifica": ("o erro da propria referencia fina contra a "
                                   "integral continua, que segue aberto"),
           "pontos": [], "contrastes": []}
    medidas = {}
    for p in pontos:
        print("=== %s (%s)%s ===" % (p["rotulo"], Path(p["arquivo"]).name,
                                     "  PARCIAL" if p["parcial"] else ""),
              flush=True)
        g = avalia(p, GROSSO, a.threads)
        print("  grosso: soma %.6f  (%.1f s)" % (g["soma"], g["segundos"]), flush=True)
        f = avalia(p, FINO, a.threads)
        print("  fino  : soma %.6f  (%.1f s)" % (f["soma"], f["segundos"]), flush=True)
        pior_cel, pior_obs = 0.0, None
        for obsid in g["grades"]:
            d = np.abs(g["grades"][obsid] - f["grades"][obsid])
            rel_d = d / np.maximum(np.abs(f["grades"][obsid]), 1e-300)
            if float(rel_d.max()) > pior_cel:
                pior_cel, pior_obs = float(rel_d.max()), obsid
        por_obs = {o: g["termos"][o] - f["termos"][o] for o in g["termos"]}
        pior_termo = max(por_obs, key=lambda o: abs(por_obs[o]))
        reg = {"rotulo": p["rotulo"], "arquivo": p["arquivo"],
               "parcial": p["parcial"], "ganho_da_reotimizacao": p["ganho"],
               "ln_posterior_registrado": p["ln_posterior_registrado"],
               "soma_grosso": g["soma"], "soma_fino": f["soma"],
               "discrepancia_absoluta": g["soma"] - f["soma"],
               "pior_termo": {"obsid": pior_termo, "diferenca": por_obs[pior_termo]},
               "pior_celula_relativa": {"obsid": pior_obs, "valor": pior_cel},
               "segundos": {"grosso": g["segundos"], "fino": f["segundos"]}}
        rel["pontos"].append(reg)
        medidas[p["rotulo"]] = (g["soma"], f["soma"])
        if a.ganho:
            pp = dict(p, vetor=p["vetor_de_partida"])
            gp = avalia(pp, GROSSO, a.threads)
            fp = avalia(pp, FINO, a.threads)
            reg["ganho_grosso"] = g["soma"] - gp["soma"]
            reg["ganho_fino"] = f["soma"] - fp["soma"]
            reg["ganho_encolhe_na_fina"] = reg["ganho_grosso"] - reg["ganho_fino"]
            reg["partida"] = {"soma_grosso": gp["soma"], "soma_fino": fp["soma"]}
            print("  GANHO: grosso %+.4f   fino %+.4f   diferenca %+.4f"
                  % (reg["ganho_grosso"], reg["ganho_fino"],
                     reg["ganho_encolhe_na_fina"]), flush=True)
            print("    leitura: um ganho que sobrevive na referencia fina nao "
                  "esta explorando erro numerico da configuracao grossa",
                  flush=True)
        print("  discrepancia da soma: %+.5f   pior celula relativa %.3e (%s)"
              % (reg["discrepancia_absoluta"], pior_cel, pior_obs), flush=True)
    if len(medidas) > 1:
        print("\n=== contrastes, que e o que move conclusao ===")
        rots = list(medidas)
        for i in range(len(rots)):
            for j in range(i + 1, len(rots)):
                a_, b_ = rots[i], rots[j]
                cg = medidas[a_][0] - medidas[b_][0]
                cf = medidas[a_][1] - medidas[b_][1]
                rel["contrastes"].append(
                    {"contraste": "%s - %s" % (a_, b_), "grosso": cg,
                     "fino": cf, "discrepancia": cg - cf})
                print("  %-24s grosso %+10.6f  fino %+10.6f  discrepancia %+.5f"
                      % ("%s - %s" % (a_, b_), cg, cf, cg - cf), flush=True)
        piores = max(abs(c["discrepancia"]) for c in rel["contrastes"])
        rel["pior_discrepancia_de_contraste"] = piores
        rel["tolerancia_de_producao"] = 0.1
        rel["frente_A_nas_sementes_antigas"] = 0.01815
        rel["cabe_na_tolerancia"] = bool(piores <= 0.1)
        print("\n  pior discrepancia de contraste: %.5f  (frente A nas antigas: "
              "0,01815; tolerancia 0,1)" % piores)
    Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False,
                                        default=float) + "\n")
    print("\ngravado em %s" % a.saida)


if __name__ == "__main__":
    main()
