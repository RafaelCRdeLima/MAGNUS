#!/usr/bin/env python3
"""Quanto a previsão muda de observação para observação, e o que dá para reusar.

DUAS PERGUNTAS, UMA MEDIÇÃO. O revisor pediu duas coisas:

  "13 chamadas não precisam repetir 13 vezes a física mais cara. Investigue
   calcular o espectro emitido por fase uma vez e aplicar separadamente
   respostas, suavizações e exposições."

  "A condição de equivalência da coadição está correta se os pesos forem
   conhecidos e independentes dos parâmetros, com o mesmo tratamento de fundo."

As duas se respondem com a MESMA medição, porque o motor devolve
`detected_count_rate[fase][energia]`, que é uma DENSIDADE DE TAXA, em
contagens/s/keV, SEM a exposição. Logo, com o mesmo vetor de parâmetros, se as
13 observações tivessem a mesma resposta, a mesma resolução e o mesmo período,
as 13 grades seriam IDÊNTICAS. Tudo o que as separa é:

  - a resposta, 13 hashes distintos em 13;
  - a resolução temporal, 73400 ou 47700 us, dois valores;
  - o período, 13 valores distintos, espalhados por 8,86e-05 relativos.

Então medir as 13 grades no MESMO vetor responde:

  (a) REUSO: se as grades diferirem pouco, uma chamada do motor mais os
      operadores por observação bastam, e a física caríssima (a atmosfera, 89%
      do custo) é calculada uma vez;
  (b) COADIÇÃO: a condição é `mu_o = w_o m`, isto é mesma FORMA com um escalar
      por observação. Isso é testável: ajusta-se o melhor escalar `w_o` por
      mínimos quadrados e mede-se o RESÍDUO. Resíduo nulo é a condição
      satisfeita; resíduo grande é a condição violada, e por quanto.

O QUE ESTA MEDIÇÃO NÃO É. Um vetor, um conjunto de parâmetros. O revisor já
avisou: "as avaliações dos três vetores serão um diagnóstico inicial, não uma
validação da equivalência em toda a posterior". Aqui é um diagnóstico num ponto.

Uso: python3 scripts/reuso_por_observacao.py [--bandas 120] [--vetor 0]
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


def grade_do_motor(obs, vetor_nome, bandas, refino):
    """A densidade de taxa [fase][energia] daquela observacao, no mesmo vetor."""
    prob, trab = PO.problema_de(obs, bandas, refino)
    nome, caminho, rotulo = [v for v in PO.VETORES if v[0] == vetor_nome][0]
    v = PO.le_vetor(prob, caminho, rotulo)
    t0 = time.time()
    doc = trab.evaluate(prob.worker_line(v))
    dt = time.time() - t0
    trab.close()
    if doc.get("status") != "ok":
        raise SystemExit("motor falhou em %s: %s" % (obs["obsid"], doc))
    return np.asarray(doc["detected_count_rate"], float), dt, prob, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandas", type=int, default=120)
    ap.add_argument("--refino", type=int, default=8)
    ap.add_argument("--vetor", default="B colado")
    ap.add_argument("--saida", default=str(RAIZ / "docs/reuso_por_observacao.json"))
    a = ap.parse_args()

    obss = {o["obsid"]: o for o in PO.prepara()}
    ordem = sorted(obss)
    print("=== 13 grades de densidade de taxa, MESMO vetor (%s), %d bandas ==="
          % (a.vetor, a.bandas))
    print("%-12s %8s %9s %12s %12s" % ("obsid", "res us", "periodo", "soma",
                                       "segundos"))
    grades, tempos = {}, {}
    for obsid in ordem:
        g, dt, prob, v = grade_do_motor(obss[obsid], a.vetor, a.bandas, a.refino)
        grades[obsid] = g
        tempos[obsid] = dt
        print("%-12s %8.0f %9.6f %12.6e %12.2f"
              % (obsid, obss[obsid]["resolucao_us"], obss[obsid]["periodo"],
                 g.sum(), dt), flush=True)

    ref = PO.REFERENCIA
    gr = grades[ref]
    print()
    print("=== diferenca contra a referencia %s ===" % ref)
    print("%-12s %12s %12s %12s %12s" % ("obsid", "escala w_o", "dif rel max",
                                         "residuo max", "residuo rms"))
    comp = []
    for obsid in ordem:
        g = grades[obsid]
        bruto = np.max(np.abs(g - gr) / np.maximum(gr, 1e-300))
        # o melhor escalar w_o por minimos quadrados, que e a condicao da coadicao
        w = float(np.sum(g * gr) / np.sum(gr * gr))
        res = (g - w * gr) / np.maximum(w * gr, 1e-300)
        comp.append({"obsid": obsid, "escala": w,
                     "diferenca_relativa_max": float(bruto),
                     "residuo_relativo_max": float(np.max(np.abs(res))),
                     "residuo_relativo_rms": float(np.sqrt(np.mean(res ** 2))),
                     "resolucao_us": obss[obsid]["resolucao_us"],
                     "periodo": obss[obsid]["periodo"]})
        print("%-12s %12.6f %12.3e %12.3e %12.3e"
              % (obsid, w, bruto, np.max(np.abs(res)),
                 np.sqrt(np.mean(res ** 2))))

    # quanto disso e a RESOLUCAO e quanto e a RESPOSTA: os dois grupos de
    # resolucao, comparados dentro e entre
    g73 = [o for o in ordem if obss[o]["resolucao_us"] == 73400.0]
    g47 = [o for o in ordem if obss[o]["resolucao_us"] == 47700.0]

    def pior_dentro(grupo):
        pior = 0.0
        for i, a1 in enumerate(grupo):
            for b1 in grupo[i + 1:]:
                x, y = grades[a1], grades[b1]
                w = float(np.sum(x * y) / np.sum(y * y))
                pior = max(pior, float(np.max(np.abs((x - w * y)
                                                     / np.maximum(w * y, 1e-300)))))
        return pior

    dentro73, dentro47 = pior_dentro(g73), pior_dentro(g47)
    entre = 0.0
    for a1 in g73:
        for b1 in g47:
            x, y = grades[a1], grades[b1]
            w = float(np.sum(x * y) / np.sum(y * y))
            entre = max(entre, float(np.max(np.abs((x - w * y)
                                                   / np.maximum(w * y, 1e-300)))))
    print()
    print("=== separando resposta de resolucao, em residuo apos a escala ===")
    print("pior DENTRO do grupo 73400 us (%d obs): %.3e" % (len(g73), dentro73))
    print("pior DENTRO do grupo 47700 us (%d obs): %.3e" % (len(g47), dentro47))
    print("pior ENTRE os dois grupos:              %.3e" % entre)
    print("leitura: dentro de um grupo so a RESPOSTA e o PERIODO diferem; entre "
          "grupos entra a RESOLUCAO tambem")

    # o custo medido, e o que o reuso economizaria
    total = sum(tempos.values())
    print()
    print("=== custo medido ===")
    print("13 chamadas: %.1f s, media %.2f s, minimo %.2f, maximo %.2f"
          % (total, total / 13, min(tempos.values()), max(tempos.values())))
    print("uma chamada: %.2f s" % tempos[ref])
    print("teto do reuso: 1 chamada de %.2f s mais 13 operadores, contra %.1f s "
          "hoje. NAO uso os 89 por cento da atmosfera medidos antes: aquilo foi "
          "na esfera isotermica a 6,38 s por chamada, nao neste vetor a %.1f s"
          % (total / 13, total, total / 13))

    saida = {
      "o_que_e": ("quanto a densidade de taxa prevista muda entre as 13 "
                  "observacoes no MESMO vetor, e o que isso diz sobre reuso e "
                  "sobre a equivalencia da coadicao"),
      "por_que_a_medicao_responde_as_duas": (
        "detected_count_rate e densidade de TAXA, sem exposicao; no mesmo vetor "
        "as 13 grades so diferem por resposta, resolucao e periodo"),
      "vetor": a.vetor, "bandas": a.bandas, "refino": a.refino,
      "referencia": ref,
      "o_que_difere_entre_as_13": {
        "respostas_distintas": 13, "resolucoes_distintas": [47700.0, 73400.0],
        "periodos_distintos": 13,
        "espalhamento_relativo_dos_periodos": 8.858e-05},
      "comparacao": comp,
      "condicao_da_coadicao": {
        "enunciado": "mu_o = w_o m, mesma forma com um escalar por observacao",
        "residuo_relativo_max": max(c["residuo_relativo_max"] for c in comp),
        "residuo_relativo_rms_mediano": float(np.median(
            [c["residuo_relativo_rms"] for c in comp]))},
      "separando_resposta_de_resolucao": {
        "pior_dentro_de_73400": dentro73, "pior_dentro_de_47700": dentro47,
        "pior_entre_grupos": entre,
        "leitura": ("dentro de um grupo so resposta e periodo diferem; entre "
                    "grupos entra a resolucao")},
      "custo": {"treze_chamadas_s": total, "media_s": total / 13,
                "uma_chamada_s": tempos[ref],
                "fracao_da_atmosfera": 0.89,
                "teto_do_reuso": ("13 x %.2f s -> %.2f s mais 13 operadores"
                                  % (total / 13, total / 13))},
      "o_que_isto_NAO_e": (
        "um vetor, um ponto da posterior. O revisor ja declarou que avaliacoes "
        "de poucos vetores sao diagnostico inicial e nao validacao em toda a "
        "posterior"),
    }
    Path(a.saida).write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n")
    print("gravado em %s" % a.saida)


if __name__ == "__main__":
    main()
