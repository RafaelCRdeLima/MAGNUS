#!/usr/bin/env python3
"""Revalida um otimo reotimizado: por malha, por imagem e por bin.

Exigencia do revisor em 04/10/2026: a variacao de 0,25 medida entre 120, 240 e
400 bandas e uma DIFERENCA OBSERVADA nos tres vetores antigos, nao um limite do
erro da quadratura. Se um otimo novo cair em outra regiao do espaco de
parametros, aqueles testes nao o cobrem e a variacao deles nao se transfere.

Este script responde tres coisas para cada otimo:

1. quanto o lnL dele se move entre 120, 240 e 400 bandas, a refino 8;
2. quanto dessa variacao vem da SEGUNDA IMAGEM, isto e, dos raios que rasam o
   limbo, que e o contribuinte dominante identificado em df43ed7;
3. onde a variacao vive, por fase e por energia, com a contribuicao de cada bin
   para a diferenca de lnL de Poisson.

E informa a distancia do otimo a cada um dos tres vetores antigos, em unidades
de largura do prior, para que a pergunta "os testes anteriores cobrem este
ponto?" tenha resposta numerica em vez de impressao.

Uso: python3 scripts/valida_otimo.py <resultado da reotimizacao>.json [...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mcmc_fit as m  # noqa: E402
import reotimiza as r  # noqa: E402

PEDIDO = ROOT / "exploracoes/reotimizacao_2026-10-04/pedido_uniao.json"
ANTIGOS = (("B colado", "docs/incidente_resolucao_2026-10-04/modosB.json", "colado"),
           ("B aberto", "docs/incidente_resolucao_2026-10-04/modosB.json", "aberto"),
           ("A", "docs/incidente_resolucao_2026-10-04/modoA.json", "aberto"))


def avalia(vetor, bandas, imagens, eventos, meta, com_modelo=False):
    base = r.monta_pedido(PEDIDO, bandas, 8, False)
    base["maxImages"] = imagens
    prob = m.FitProblem(base, eventos, meta, None)
    v = list(vetor)
    for nome, valor in prob.frozen.items():
        v[prob.names.index(nome)] = float(valor)
    w = m.EngineWorker(prob.worker_command())
    t = time.time()
    if com_modelo:
        L, esperado = prob.evaluate(w, v, return_model=True)
    else:
        L, esperado = prob.evaluate(w, v), None
    w.close()
    return L, esperado, prob, time.time() - t


def distancia_em_larguras(vetor, prob):
    """Distancia aos vetores antigos, por coordenada, em larguras de prior."""
    fora = {}
    for rotulo, arq, lab in ANTIGOS:
        dados = json.loads((ROOT / arq).read_text())
        r.valida_ordem(dados["nomes"], prob.names)
        antigo = dados[lab]["vetor"]
        por_coord = {}
        for i, nome in enumerate(prob.names):
            if nome in prob.frozen:
                continue
            lo, hi = prob.bounds[i]
            if hi > lo:
                por_coord[nome] = abs(vetor[i] - antigo[i]) / (hi - lo)
        pior = max(por_coord, key=por_coord.get)
        fora[rotulo] = {
            "maior_desvio": por_coord[pior], "em": pior,
            "euclidiana": float(np.sqrt(sum(v * v for v in por_coord.values()))),
            "por_coordenada": por_coord,
        }
    return fora


def por_bin(prob, e120, e400):
    nf, ne = prob.phase_bins, prob.energy_bins
    a = np.asarray(e120, float).reshape(nf, ne)
    c = np.asarray(e400, float).reshape(nf, ne)
    obs = np.asarray(prob.observed, float).reshape(nf, ne)
    with np.errstate(divide="ignore", invalid="ignore"):
        dln = np.where(a > 0, obs * np.log(np.maximum(c, 1e-300) /
                                           np.maximum(a, 1e-300)) - (c - a), 0.0)
    fase = dln.sum(axis=1)
    energia = dln.sum(axis=0)
    ordem = np.argsort(-np.abs(fase))
    total = np.abs(fase).sum()
    acumulado, quantas = 0.0, 0
    for i in ordem:
        acumulado += abs(fase[i])
        quantas += 1
        if total > 0 and acumulado >= 0.70 * total:
            break
    return {
        "soma_por_bin": float(dln.sum()),
        "fases_para_70_por_cento": quantas,
        "fases_dominantes": {int(i): float(fase[i]) for i in ordem[:4]},
        "energias_dominantes": {
            "%.3f keV" % (prob.energy_min + (j + 0.5) * prob.energy_width):
            float(energia[j]) for j in np.argsort(-np.abs(energia))[:4]},
        "pior_bin_relativo_por_cento": float(
            100.0 * np.max(np.abs((c - a) / np.maximum(a, 1e-300)))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("otimos", nargs="+", help="json de saida de reotimiza.py")
    ap.add_argument("--eventos", default=str(ROOT / "dados/eventos.csv"))
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()
    meta, eventos = m.read_events(Path(a.eventos))

    relatorio = {"data": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "aviso": ("a variacao entre malhas medida aqui e diferenca OBSERVADA entre "
                           "tres resolucoes, nao limite demonstrado do erro da quadratura"),
                 "otimos": {}}
    for caminho in a.otimos:
        d = json.loads(Path(caminho).read_text())
        vetor = d["otimo"]["vetor"]
        rotulo = "%s/%s" % (Path(d["origem"]["vetor"]).stem, d["origem"]["rotulo"])
        print("=== %s (de %s) ===" % (rotulo, caminho), flush=True)

        malha = {}
        modelos = {}
        for bandas in (120, 240, 400):
            L, esperado, prob, dt = avalia(vetor, bandas, 2, eventos, meta,
                                           com_modelo=bandas in (120, 400))
            malha["%db/2img" % bandas] = L
            if esperado is not None:
                modelos[bandas] = esperado
            print("  %3db 2 imagens  %.4f  (%.0f s)" % (bandas, L, dt), flush=True)
        uma = {}
        for bandas in (120, 400):
            L, _, _, dt = avalia(vetor, bandas, 1, eventos, meta)
            uma["%db/1img" % bandas] = L
            print("  %3db 1 imagem   %.4f  (%.0f s)" % (bandas, L, dt), flush=True)

        vals = [malha["%db/2img" % b] for b in (120, 240, 400)]
        dep2 = malha["400b/2img"] - malha["120b/2img"]
        dep1 = uma["400b/1img"] - uma["120b/1img"]
        seg120 = malha["120b/2img"] - uma["120b/1img"]
        seg400 = malha["400b/2img"] - uma["400b/1img"]

        entrada = {
            "vetor": vetor,
            "lnPosterior_por_malha": malha,
            "lnPosterior_uma_imagem": uma,
            "variacao_120_240_400": float(max(vals) - min(vals)),
            "dependencia_de_malha": {"duas_imagens": float(dep2),
                                     "uma_imagem": float(dep1)},
            "segunda_imagem": {"em_120b": float(seg120), "em_400b": float(seg400),
                               "movimento_com_a_malha": float(seg400 - seg120)},
            "por_bin_120_para_400": por_bin(prob, modelos[120], modelos[400]),
            "distancia_aos_vetores_antigos": distancia_em_larguras(vetor, prob),
        }
        relatorio["otimos"][rotulo] = entrada

        print("  variacao entre as tres malhas: %.4f" % entrada["variacao_120_240_400"])
        print("  dependencia de malha: %+.4f com duas imagens, %+.4f com uma"
              % (dep2, dep1))
        print("  segunda imagem move %+.4f com a malha" % (seg400 - seg120))
        pb = entrada["por_bin_120_para_400"]
        print("  %d fases carregam 70 %% da variacao; pior bin %.4f %%"
              % (pb["fases_para_70_por_cento"], pb["pior_bin_relativo_por_cento"]))
        for rot, dist in entrada["distancia_aos_vetores_antigos"].items():
            print("  distancia a %-9s: euclidiana %.3f larguras, maior desvio "
                  "%.3f em %s" % (rot, dist["euclidiana"], dist["maior_desvio"],
                                  dist["em"]))
        longe = [rot for rot, dist in entrada["distancia_aos_vetores_antigos"].items()
                 if dist["maior_desvio"] > 0.10]
        if len(longe) == len(ANTIGOS):
            print("  ATENCAO: mais de 0,10 largura de TODOS os vetores antigos em "
                  "alguma coordenada. Os testes de convergencia feitos neles nao "
                  "cobrem este ponto.")
            entrada["coberto_pelos_testes_antigos"] = False
        else:
            entrada["coberto_pelos_testes_antigos"] = True
        print(flush=True)

    destino = Path(a.saida) if a.saida else ROOT / "exploracoes/reotimizacao_2026-10-04/validacao_otimos.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False) + "\n")
    print("gravado em", destino)


if __name__ == "__main__":
    main()
