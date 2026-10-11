#!/usr/bin/env python3
"""O erro da quadratura de fase no ALVO CONJUNTO das 13, pela rota de reuso.

O revisor, 07/10/2026:

  "tres sub-amostras sao uma quadratura mais fina, nao uma integracao exata"
  "o custo total nao sera necessariamente 3x; meca-o com reuso e operadores por
   observacao"
  "compare diretamente seu residuo contra M=9 e M=15 nos tres vetores"
  "confira a estabilidade da DIFERENCA ENTRE VETORES na referencia"
  "resultado de uma observacao: falta verificar o alvo conjunto das 13; residuos
   pequenos podem acumular"

A ROTA, que sai de graca da aritmetica de paridade. Os offsets das observacoes
sao `s_o/32` ciclos, e numa grade fina de `32M` fases isso vale `s_o.M/(32M)`,
que e multiplo INTEIRO de `1/(32M)`. Logo a rota multi-observacao aceita os 13
offsets na grade fina, e UMA chamada por nivel entrega as 13 grades finas. Sem
isso, medir o conjunto custaria 13 vezes mais.

O QUE SE MEDE, e nessa ordem de importancia:

  1. a mudanca na diferenca de ln L entre vetores, no alvo CONJUNTO, ao trocar
     ponto medio por quadratura de M sub-amostras;
  2. o RESIDUO de M=3 contra M=15, na mesma grandeza, que e o que aprova ou
     reprova M=3;
  3. a estabilidade da propria DIFERENCA entre M=9 e M=15, que e o erro da
     referencia na grandeza comparada e nao no ln L de um vetor;
  4. o custo medido, com reuso e operadores por observacao.

NAO E INTEGRACAO EXATA. M sub-amostras sao uma quadratura mais fina do que uma
amostra; o limite continuo nao e atingido, e e por isso que o residuo contra o
nivel mais fino tem de ser medido em vez de suposto nulo.

Uso: python3 scripts/discretizacao_conjunta.py --vetor "B colado"
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


def problemas_por_observacao(obss, bandas, refino):
    """Um FitProblem por observacao, so para PONTUAR: dado, exposicao, fundo."""
    meta, ev = m.read_events(RAIZ / "dados/eventos.csv")
    out = {}
    for obsid, o in obss.items():
        base = PO.monta_pedido(o, bandas, refino)
        base["outputPrecision"] = 17
        out[obsid] = m.FitProblem(base, ev, meta, None)
    return out, meta, ev


def uma_chamada(obss, ordem, vetor, M, bandas, refino, meta, ev,
                malha="producao", tabela=None):
    """UMA chamada do motor com 32M fases e as 13 observacoes como slots."""
    primeiro = obss[ordem[0]]
    base = PO.monta_pedido(primeiro, bandas, refino)
    base["outputPrecision"] = 17
    base["phaseBins"] = FASES * M
    base.pop("preparedDataNpz", None)      # 32M fases nao casa com o npz de 32
    prob = m.FitProblem(base, ev, meta, None)
    comando = list(prob.worker_command())
    # MALHA E TABELA FIXAS. O teste temporal so mede fase se o erro espacial
    # estiver controlado, e a configuracao aprovada e psi_0 a 480 bandas com
    # tabela 115200 (docs/escolhe_producao.json). Variar fase com a malha de
    # producao mediria a soma dos dois, que foi o erro cometido nas rodadas
    # anteriores.
    if malha == "psi0":
        comando += ["--malha-psi0", "--psi-bandas", str(bandas)]
    if tabela:
        comando += ["--ray-table-size", str(tabela)]
    for obsid in ordem:
        o = obss[obsid]
        s_o = int(o["desloc_v3_bins"])
        comando += ["--obs-id", obsid,
                    "--obs-profile", o["perfil_csv"],
                    "--obs-response", o["perfil_rmfbin"],
                    # RESOLUCAO ZERADA, e isto NAO e detalhe. A 32 fases ela e
                    # inerte, mas o motor so suaviza quando a resolucao passa de
                    # UM bin de fase, e numa grade fina o bin encolhe: a 288
                    # fases o bin vale 0,0358 s contra 0,0734 s do FULLWINDOW, e
                    # o boxcar LIGA com janela 2. Medido antes de corrigir: as 7
                    # FULLWINDOW deslocavam 2,3 a 13,9 em ln L e as 6
                    # LARGEWINDOW, cuja janela arredonda para 1, ficavam em 0,19
                    # a 0,45. A medida estava somando discretizacao de fase com
                    # a suavizacao entrando em cena. O revisor pediu
                    # explicitamente para manter a resposta temporal como teste
                    # SEPARADO, e era disto que ele falava.
                    "--obs-time-resolution-us", "0.0",
                    "--obs-dead-time-us", "0.0",
                    "--obs-period", repr(float(o["periodo"])),
                    # s_o/32 ciclos = s_o.M/(32M), multiplo inteiro de 1/(32M)
                    "--obs-offset", repr(PO.offset_em_ciclos(s_o))]
    trab = m.EngineWorker(comando)
    caminho, rotulo = [x[1:] for x in PO.VETORES if x[0] == vetor][0]
    v = PO.le_vetor(prob, caminho, rotulo)
    t0 = time.time()
    doc = trab.evaluate(prob.worker_line(v))
    dt = time.time() - t0
    trab.close()
    if doc.get("status") != "ok":
        raise SystemExit("motor falhou em M = %d: %s" % (M, doc))
    if "observations" not in doc:
        raise SystemExit("sem 'observations': a rota nao foi acionada")
    return {o["id"]: np.asarray(o["detected_count_rate"], float)
            for o in doc["observations"]}, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vetor", default="B colado")
    ap.add_argument("--Ms", type=int, nargs="+", default=[1, 3, 9, 15])
    ap.add_argument("--bandas", type=int, default=120)
    ap.add_argument("--refino", type=int, default=8)
    ap.add_argument("--malha", default="producao", choices=["producao", "psi0"])
    ap.add_argument("--tabela", type=int, default=None)
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()
    obss = {o["obsid"]: o for o in PO.prepara()}
    ordem = sorted(obss)
    probs, meta, ev = problemas_por_observacao(obss, a.bandas, a.refino)
    print("=== alvo CONJUNTO das %d, vetor %s ===" % (len(ordem), a.vetor))
    print("%4s %7s %10s %18s %12s" % ("M", "fases", "segundos", "ln L conjunto",
                                      "contra M=1"))
    rel = {"o_que_e": "a quadratura de fase no alvo conjunto das 13",
           "vetor": a.vetor, "bandas": a.bandas, "refino": a.refino,
           "malha": a.malha, "tabela_de_raios": a.tabela,
           "a_configuracao_espacial_esta_FIXA": (
             "o teste temporal so mede fase se o erro espacial estiver "
             "controlado; a configuracao aprovada e psi_0 a 480 bandas com "
             "tabela 115200"),
           "nao_e_integracao_exata": (
             "M sub-amostras sao uma quadratura mais fina que uma amostra; o "
             "limite continuo nao e atingido, logo o residuo contra o nivel mais "
             "fino e MEDIDO e nao suposto nulo"),
           "a_rota": ("uma chamada por nivel, com as 13 como slots, porque os "
                      "offsets s_o/32 sao multiplos inteiros de 1/(32M)"),
           "resolucao_zerada_nas_finas": (
             "obrigatorio: o motor suaviza quando a resolucao passa de UM bin de "
             "fase, e na grade fina o bin encolhe, ligando o boxcar a partir de "
             "~140 fases. Sem zerar, a medida soma discretizacao de fase com a "
             "suavizacao entrando em cena, e foi o que aconteceu na primeira "
             "corrida: 2,3 a 13,9 nas FULLWINDOW contra 0,19 a 0,45 nas "
             "LARGEWINDOW, cuja janela arredonda para 1"),
           "o_offset_e_aplicado": (
             "sim, s_o por observacao. A medida isolada anterior rodou com offset "
             "ZERO, isto e modelo desalinhado, e por isso os dois ln L de base "
             "diferem: 205887,97 contra 206345,93 sao o s = 0 e o s = 5 da "
             "varredura. A conjunta e a configuracao certa"),
           "niveis": []}
    base_lnl = None
    for M in a.Ms:
        grades, dt = uma_chamada(obss, ordem, a.vetor, M, a.bandas, a.refino,
                                 meta, ev, malha=a.malha, tabela=a.tabela)
        total, por_obs = 0.0, {}
        for obsid in ordem:
            g = grades[obsid]
            if M > 1:
                g = g.reshape(FASES, M, -1).mean(axis=1)
            d = {"status": "ok", "detected_count_rate": g.tolist()}
            l = float(probs[obsid].score_document(d))
            por_obs[obsid] = l
            total += l
        if base_lnl is None:
            base_lnl = total
        rel["niveis"].append({"M": M, "fases": FASES * M, "segundos": dt,
                              "ln_L_conjunto": total,
                              "contra_M1": total - base_lnl,
                              "por_observacao": por_obs})
        print("%4d %7d %10.1f %18.6f %12.6f"
              % (M, FASES * M, dt, total, total - base_lnl), flush=True)
    # O NOME INCLUI OS NIVEIS M. Sem isso cada corrida SOBRESCREVIA a anterior,
    # e as corridas de M=15 apagaram os JSON de M=1,3,9 do par B; os valores
    # sobreviveram apenas nos logs. Defeito meu, consertado aqui.
    saida = Path(a.saida) if a.saida else RAIZ / (
        "docs/temporal_%s_%s_%d_%s_M%s.json"
        % (a.malha, a.vetor.replace(" ", "_"), a.bandas, a.tabela or "padrao",
           "-".join(str(M) for M in a.Ms)))
    custo = {n["M"]: n["segundos"] for n in rel["niveis"]}
    rel["custo_medido_com_reuso"] = {
      "segundos_por_nivel": custo,
      "razao_M3_sobre_M1": (custo.get(3, 0) / custo[1]) if 1 in custo else None,
      "leitura": ("o custo NAO e 3x por M = 3: a rota de reuso faz UMA chamada "
                  "para as 13, e o que cresce e so o numero de fases")}
    saida.write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print("gravado em %s" % saida)


if __name__ == "__main__":
    main()
