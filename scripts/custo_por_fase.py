#!/usr/bin/env python3
"""Quanto do custo de uma chamada e FIXO e quanto e POR FASE.

Isto decide o desenho do reuso, e nao era obvio. Duas situacoes:

  OFFSETS INTEIROS: as 13 observacoes precisam do MESMO conjunto de 32 fases,
  apenas rerrotulado, porque offset k/N leva a fase do bin p na fase do bin p+k.
  Uma avaliacao de 32 fases serve as treze exatamente, e a economia e quase
  tudo: 13 chamadas viram 1 mais 13 respostas.

  OFFSETS FRACIONARIOS: cada observacao precisa de {(p+0,5)/N + delta_o}, que
  sao 13 conjuntos distintos, 416 fases. O trabalho POR FASE nao e reusavel, e
  sobra economizar apenas o custo FIXO: malha da superficie, tabela de raios,
  tabela de atmosfera, carga da resposta.

Logo o teto da economia no caso fracionario e a fracao FIXA do custo, que se mede
variando o numero de bins de fase e ajustando

    custo(n) = fixo + por_fase . n

Uso: python3 scripts/custo_por_fase.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import mcmc_fit as m  # noqa: E402
import por_observacao as PO  # noqa: E402

BINS = (16, 32, 64, 128)


def main():
    obss = {o["obsid"]: o for o in PO.prepara()}
    obs = obss["0163560101"]
    meta, ev = m.read_events(RAIZ / "dados/eventos.csv")
    nome, caminho, rotulo = PO.VETORES[0]
    print("=== custo por numero de bins de fase ===")
    print("%6s %12s %12s %14s" % ("bins", "chamada s", "s por fase", "soma"))
    linhas = []
    for n in BINS:
        base = PO.monta_pedido(obs, 120, 8)
        # SEM o dado preparado: ele tem forma 32 x 40 e qualquer outro numero de
        # bins de fase seria incompativel. Aqui so se mede CUSTO, e o custo nao
        # depende do dado; sem o npz o adaptador bina os eventos e a forma
        # acompanha o pedido.
        base.pop("preparedDataNpz", None)
        base["phaseBins"] = n
        prob = m.FitProblem(base, ev, meta, None)
        v = PO.le_vetor(prob, caminho, rotulo)
        comando = list(prob.worker_command())
        # o tempo de SUBIR o worker, separado do de avaliar
        t0 = time.time()
        trab = m.EngineWorker(comando)
        doc = trab.evaluate(prob.worker_line(v))
        t_total = time.time() - t0
        t1 = time.time()
        doc2 = trab.evaluate(prob.worker_line(v))
        t_segunda = time.time() - t1
        trab.close()
        if doc.get("status") != "ok" or doc2.get("status") != "ok":
            raise SystemExit("motor falhou em %d bins" % n)
        g = np.asarray(doc["detected_count_rate"], float)
        linhas.append({"bins": n, "primeira_avaliacao_s": t_total,
                       "segunda_avaliacao_s": t_segunda,
                       "soma": float(g.sum())})
        print("%6d %12.2f %12.4f %14.6e"
              % (n, t_segunda, t_segunda / n, g.sum()), flush=True)

    ns = np.array([l["bins"] for l in linhas], float)
    ts = np.array([l["segunda_avaliacao_s"] for l in linhas], float)
    A = np.vstack([np.ones_like(ns), ns]).T
    (fixo, por_fase), resid, *_ = np.linalg.lstsq(A, ts, rcond=None)
    ajuste = fixo + por_fase * ns
    print()
    print("ajuste custo(n) = fixo + por_fase . n")
    print("  fixo     = %+8.3f s" % fixo)
    print("  por_fase = %+8.4f s" % por_fase)
    print("  residuo maximo: %.3f s (%.1f por cento)"
          % (np.abs(ts - ajuste).max(),
             100 * np.abs((ts - ajuste) / ts).max()))
    print()
    f32 = fixo + por_fase * 32
    print("em 32 fases, custo previsto %.1f s, dos quais %.1f s fixos "
          "(%.1f por cento)" % (f32, fixo, 100 * fixo / f32))
    print()
    print("TETO DA ECONOMIA DO REUSO:")
    print("  offsets INTEIROS:     13 x %.1f s -> %.1f s mais 13 respostas, "
          "fator ~13" % (f32, f32))
    uniao = fixo + por_fase * 32 * 13
    print("  offsets FRACIONARIOS: 13 x %.1f s = %.1f s -> %.1f s pela uniao "
          "das 416 fases, fator %.2f" % (f32, 13 * f32, uniao, 13 * f32 / uniao))
    rel = {
      "o_que_e": "a reparticao do custo entre fixo e por fase, que decide o reuso",
      "tabela": linhas,
      "ajuste": {"fixo_s": float(fixo), "por_fase_s": float(por_fase),
                 "residuo_relativo_max": float(np.abs((ts - ajuste) / ts).max())},
      "em_32_fases": {"custo_s": float(f32), "fixo_s": float(fixo),
                      "fracao_fixa": float(fixo / f32)},
      "teto_do_reuso": {
        "offsets_inteiros": {
          "hoje_s": float(13 * f32), "com_reuso_s": float(f32),
          "fator": float(13.0),
          "por_que": ("offset k/N leva a fase do bin p na fase do bin p+k, logo "
                      "as 13 precisam do MESMO conjunto de 32 fases e uma "
                      "avaliacao serve as treze exatamente")},
        "offsets_fracionarios": {
          "hoje_s": float(13 * f32), "com_reuso_s": float(uniao),
          "fator": float(13 * f32 / uniao),
          "por_que": ("13 conjuntos distintos de 32 fases, 416 no total; o "
                      "trabalho por fase nao e reusavel e so o custo fixo e")}},
      "o_que_isto_NAO_diz": (
        "o custo da aplicacao das 13 respostas, que entra por cima; medido antes "
        "em outra configuracao como 0,03 s de 6,38, mas nao nesta"),
    }
    Path(RAIZ / "docs/custo_por_fase.json").write_text(
        json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print("gravado em docs/custo_por_fase.json")


if __name__ == "__main__":
    main()
