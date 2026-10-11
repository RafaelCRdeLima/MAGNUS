#!/usr/bin/env python3
"""O erro do integrador DE PRODUÇÃO no modelo REAL, contra autoconvergência.

A PERGUNTA CERTA. Eu vinha perguntando se a candidata de P1 é boa. O que bloqueia
produção é outra coisa: **o erro do integrador de produção, no modelo real, cabe
no orçamento da conclusão pretendida?** Essa pergunta se responde com o
executável CONGELADO, sem implementar candidata nenhuma, e é o que este script
faz.

O MÉTODO. Para cada caso, avalia-se a mesma configuração com 120, 240, 480 e 960
bandas de superfície. Com erro `~ C h^p`, três níveis dão a ordem `p` e a
extrapolação de Richardson dá a referência. O que decide é

    D ln L = ln L(120 bandas, refino 8) - ln L(referência extrapolada)

porque é isso que a posterior vê. A meta conservadora do revisor é 0,1 em
diferenças de ln L.

O QUE ESTE TESTE COBRE, e que a esfera isotérmica não cobria: mapa térmico não
uniforme com `peaking`, dois polos inclinados, campo não uniforme pela lei
dipolar GR, e `max_images` 1 contra 2, que liga e desliga a segunda imagem.

A SALVAGUARDA CONTRA MIM MESMO. O erro da própria referência é estimado
comparando duas extrapolações, de (120,240,480) e de (240,480,960). Se as duas
discordarem mais que o efeito a medir, o caso sai **INCONCLUSIVO**, não
arredondado para "passou". E onde o erro a 120 bandas for MENOR que a 960 o caso
é marcado como **suspeito de cancelamento**, porque já medi um caso em que o erro
pequeno era cancelamento entre contribuições de sinais opostos.

Uso: python3 scripts/convergencia_producao.py --fatia I N
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import mcmc_fit as m  # noqa: E402
import por_observacao as PO  # noqa: E402

BANDAS = (120, 240, 480, 960)
OBS = "0163560101"
TETO_RSS_MB = 400.0
#: Os casos, do pre-registro. Cada um e (rotulo, vetor, mudancas no pedido).
def casos():
    c = []
    for nome, caminho, rotulo in PO.VETORES:
        c.append(("%s, producao" % nome, (caminho, rotulo), {}))
        c.append(("%s, uma imagem" % nome, (caminho, rotulo), {"maxImages": 1}))
    # e a variacao de u, que e onde o deficit de P1 liga
    c.append(("B colado, u alto", (PO.VETORES[0][1], PO.VETORES[0][2]),
              {"_u": 0.66}))
    c.append(("B colado, u baixo", (PO.VETORES[0][1], PO.VETORES[0][2]),
              {"_u": 0.55}))
    return c


def avalia(obs, vetor, mudancas, bandas, refino=8):
    caminho, rotulo = vetor
    base = PO.monta_pedido(obs, bandas, refino)
    base["outputPrecision"] = 17
    u_novo = mudancas.pop("_u", None) if "_u" in mudancas else None
    base.update({k: v for k, v in mudancas.items() if not k.startswith("_")})
    meta, ev = m.read_events(RAIZ / "dados/eventos.csv")
    prob = m.FitProblem(base, ev, meta, None)
    v = PO.le_vetor(prob, caminho, rotulo)
    if u_novo is not None:
        v[0] = float(u_novo)
    trab = m.EngineWorker(list(prob.worker_command()))
    t0 = time.time()
    doc = trab.evaluate(prob.worker_line(v))
    dt = time.time() - t0
    trab.close()
    if doc.get("status") != "ok":
        return None, None, dt, doc.get("message", "sem mensagem")
    return (float(prob.score_document(doc)),
            np.asarray(doc["detected_count_rate"], float), dt, None)


def richardson(hs, ys):
    """Ordem e extrapolacao de tres niveis, com h = 1/bandas."""
    h1, h2, h3 = hs
    y1, y2, y3 = ys
    d12, d23 = y1 - y2, y2 - y3
    if d23 == 0.0 or d12 / d23 <= 0.0:
        return None, None       # sem razao positiva nao ha ordem
    razao = d12 / d23
    p = np.log(razao) / np.log(h1 / h2)
    y = y3 + d23 / (razao - 1.0) if razao != 1.0 else None
    return float(p), (float(y) if y is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fatia", type=int, nargs=2, default=[0, 1],
                    help="indice e total, para rodar em paralelo")
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()
    i, n = a.fatia
    todos = casos()
    meus = [c for k, c in enumerate(todos) if k % n == i]
    obss = {o["obsid"]: o for o in PO.prepara()}
    obs = obss[OBS]
    saida = a.saida or str(RAIZ / ("docs/convergencia_producao_%d_de_%d.json"
                                   % (i + 1, n)))
    print("=== autoconvergencia do integrador de producao, fatia %d de %d ==="
          % (i + 1, n))
    print("%d casos de %d; bandas %s" % (len(meus), len(todos), BANDAS))
    rel = {"o_que_e": ("o erro do integrador DE PRODUCAO no modelo real, por "
                       "autoconvergencia em bandas"),
           "a_pergunta": ("o erro do integrador de producao cabe no orcamento? "
                          "e nao 'a candidata e boa?'"),
           "executavel_congelado": "f508ebc075a8d8eab304ee14aeea9f8637f1be1f8380e1765a316d30d535f780",
           "bandas": list(BANDAS), "observacao": OBS,
           "meta_conservadora": 0.1, "casos": []}
    for rotulo, vetor, mud in meus:
        print()
        print("--- %s ---" % rotulo, flush=True)
        lnls, tempos, falhou = [], [], None
        for b in BANDAS:
            lnl, grade, dt, erro = avalia(obs, vetor, dict(mud), b)
            tempos.append(dt)
            if lnl is None:
                falhou = "%d bandas: %s" % (b, erro)
                print("  %4d bandas: FALHOU, %s" % (b, erro), flush=True)
                break
            lnls.append(lnl)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            print("  %4d bandas: ln L = %16.6f  (%.0f s, RSS %.0f MB)"
                  % (b, lnl, dt, rss), flush=True)
            if rss > TETO_RSS_MB:
                falhou = "RSS %.0f MB passou o teto de %.0f" % (rss, TETO_RSS_MB)
                break
        caso = {"rotulo": rotulo, "mudancas": mud, "ln_L": lnls,
                "segundos": tempos, "falhou": falhou}
        if falhou is None and len(lnls) == 4:
            hs = [1.0 / b for b in BANDAS]
            p_a, y_a = richardson(hs[0:3], lnls[0:3])
            p_b, y_b = richardson(hs[1:4], lnls[1:4])
            caso["ordem_120_480"] = p_a
            caso["ordem_240_960"] = p_b
            caso["referencia_de_120_480"] = y_a
            caso["referencia_de_240_960"] = y_b
            if y_a is None or y_b is None:
                caso["veredito"] = "INCONCLUSIVO: sem razao de convergencia positiva"
            else:
                d_prod = lnls[0] - y_b
                erro_da_ref = abs(y_a - y_b)
                caso["D_lnL_producao_contra_referencia"] = d_prod
                caso["erro_da_propria_referencia"] = erro_da_ref
                caso["suspeito_de_cancelamento"] = bool(
                    abs(lnls[0] - y_b) < abs(lnls[3] - y_b))
                if erro_da_ref > abs(d_prod):
                    caso["veredito"] = (
                      "INCONCLUSIVO: as duas extrapolacoes discordam por %.4f, "
                      "mais que o efeito a medir, %.4f" % (erro_da_ref, abs(d_prod)))
                elif caso["suspeito_de_cancelamento"]:
                    caso["veredito"] = (
                      "SUSPEITO DE CANCELAMENTO: o erro a 120 bandas e menor que "
                      "a 960, o que pode ser cancelamento e nao precisao")
                elif abs(d_prod) <= 0.1:
                    caso["veredito"] = "PASSA: |D ln L| = %.4f, dentro de 0,1" % abs(d_prod)
                else:
                    caso["veredito"] = (
                      "EXIGE DECISAO: |D ln L| = %.4f, acima de 0,1" % abs(d_prod))
            print("  ordem %s / %s | D ln L producao contra referencia: %s"
                  % (("%.2f" % p_a) if p_a else "-",
                     ("%.2f" % p_b) if p_b else "-",
                     ("%.4f" % caso.get("D_lnL_producao_contra_referencia", float("nan")))))
            print("  %s" % caso["veredito"])
        rel["casos"].append(caso)
        Path(saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print()
    print("gravado em %s" % saida)


if __name__ == "__main__":
    main()
