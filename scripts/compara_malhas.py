#!/usr/bin/env python3
"""As duas malhas no alvo por observacao: o proximo teste discriminante.

O revisor, 08/10/2026, com duas restricoes que o enunciado deste script carrega:

  "Concordancia entre malhas ao refinar e evidencia util, mas nao demonstra
   sozinha que a candidata esteja correta."

  "Se as diferencas de ln L mudarem cerca de 0,9, isso confirma SENSIBILIDADE A
   QUADRATURA, nao um erro de producao de 0,9. A candidata so vira referencia
   apos validar sua propria precisao."

Logo o que este script mede e **sensibilidade da posterior a escolha de
quadratura**, e nao o erro de nenhuma das duas malhas.

O QUE SE REGISTRA, como pedido: diferencas por BIN OBSERVADO de fase e energia,
`Q_num`, diferencas de ln L entre vetores, e custo. E DOIS niveis da candidata:
se ela nao estabilizar entre eles, o veredito e inconclusivo.

Uso: python3 scripts/compara_malhas.py
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


def uma_rodada(obss, ordem, vetor, malha, bandas, refino, meta, ev,
               extra=None):
    """UMA chamada com as 13 como slots, na malha e resolucao pedidas."""
    base = PO.monta_pedido(obss[ordem[0]], bandas, refino)
    base["outputPrecision"] = 17
    prob = m.FitProblem(base, ev, meta, None)
    comando = list(prob.worker_command())
    if malha == "psi0":
        comando += ["--malha-psi0", "--psi-bandas", str(bandas)]
    if extra:
        comando += list(extra)
    for obsid in ordem:
        o = obss[obsid]
        comando += ["--obs-id", obsid, "--obs-profile", o["perfil_csv"],
                    "--obs-response", o["perfil_rmfbin"],
                    "--obs-time-resolution-us", repr(o["perfil_resolucao_us"]),
                    "--obs-dead-time-us", "0.0",
                    "--obs-period", repr(float(o["periodo"])),
                    "--obs-offset",
                    repr(PO.offset_em_ciclos(int(o["desloc_v3_bins"])))]
    trab = m.EngineWorker(comando)
    caminho, rotulo = [x[1:] for x in PO.VETORES if x[0] == vetor][0]
    v = PO.le_vetor(prob, caminho, rotulo)
    t0 = time.time()
    doc = trab.evaluate(prob.worker_line(v))
    dt = time.time() - t0
    trab.close()
    if doc.get("status") != "ok":
        raise SystemExit("motor falhou em %s/%s/%d: %s"
                         % (vetor, malha, bandas, doc))
    return {o["id"]: np.asarray(o["detected_count_rate"], float)
            for o in doc["observations"]}, dt


def pontua(probs, grades, ordem):
    """ln L por observacao e o conjunto, mais as contagens esperadas por bin."""
    total, por_obs, mus = 0.0, {}, {}
    for obsid in ordem:
        d = {"status": "ok",
             "detected_count_rate": grades[obsid].tolist()}
        l, modelo = probs[obsid].score_document(d, return_model=True)
        por_obs[obsid] = float(l)
        mus[obsid] = np.asarray(modelo, float).reshape(FASES, -1)
        total += float(l)
    return total, por_obs, mus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandas-producao", type=int, nargs="+", default=[120])
    ap.add_argument("--niveis-candidata", type=int, nargs="+", default=[120, 240])
    ap.add_argument("--refino", type=int, default=8)
    #: A TABELA DE RAIOS, fixada. Medido em docs/cruza_tabela.json: no padrao do
    #: motor, 1800 nos, ela move as diferencas de ln L em ate 0,1986, MAIS que o
    #: refino de malha de 480 para 960, que movia 0,1077. E ela converge a partir
    #: de 28800. Logo medir malha com a tabela em 1800 mede a soma dos dois e
    #: atribui ao eixo que se variou.
    ap.add_argument("--tabela", type=int, default=None)
    ap.add_argument("--saida", default=str(RAIZ / "docs/compara_malhas.json"))
    a = ap.parse_args()
    obss = {o["obsid"]: o for o in PO.prepara()}
    ordem = sorted(obss)
    meta, ev = m.read_events(RAIZ / "dados/eventos.csv")
    probs = {}
    for obsid, o in obss.items():
        b = PO.monta_pedido(o, a.bandas_producao[0], a.refino)
        b["outputPrecision"] = 17
        probs[obsid] = m.FitProblem(b, ev, meta, None)

    rel = {"o_que_e": ("as duas malhas no alvo por observacao: sensibilidade da "
                       "posterior a escolha de quadratura"),
           "o_que_isto_NAO_e": (
             "medicao do erro de nenhuma das duas malhas. Uma mudanca de 0,9 nas "
             "diferencas confirma SENSIBILIDADE A QUADRATURA, nao um erro de "
             "producao de 0,9. E concordancia entre malhas ao refinar nao "
             "demonstra sozinha que a candidata esteja correta: duas "
             "discretizacoes podem convergir para o mesmo valor errado se "
             "compartilharem uma aproximacao a montante"),
           "bins": "32 fases x 40 energias, BINS OBSERVADOS",
           "tabela_de_raios": None,
           "configuracoes": [], "vetores": {}}
    rel["tabela_de_raios"] = a.tabela or "padrao do motor, 1800"
    confs = [("producao", n) for n in a.bandas_producao] + [
        ("psi0", n) for n in a.niveis_candidata]
    print("%-12s %-10s %6s %16s %10s" % ("vetor", "malha", "bandas",
                                         "ln L conjunto", "segundos"))
    for vetor, _, _ in PO.VETORES:
        rel["vetores"][vetor] = {}
        for malha, bandas in confs:
            extra = (["--ray-table-size", str(a.tabela)]
                     if a.tabela else None)
            grades, dt = uma_rodada(obss, ordem, vetor, malha, bandas,
                                    a.refino, meta, ev, extra=extra)
            total, por_obs, mus = pontua(probs, grades, ordem)
            rel["vetores"][vetor]["%s_%d" % (malha, bandas)] = {
                "malha": malha, "bandas": bandas, "ln_L_conjunto": total,
                "por_observacao": por_obs, "segundos": dt,
                "_mus": {k: v for k, v in mus.items()}}
            print("%-12s %-10s %6d %16.6f %10.1f"
                  % (vetor, malha, bandas, total, dt), flush=True)

    # diferencas por bin e Q_num, contra a producao, por observacao
    print()
    print("=== diferencas por BIN e Q_num, contra a producao ===")
    print("%-12s %-10s %14s %14s %12s" % ("vetor", "malha", "mediana |rel|",
                                          "pior bin |rel|", "Q_num"))
    for vetor in rel["vetores"]:
        prod = rel["vetores"][vetor]["producao_%d" % a.bandas_producao[0]]
        for chave, c in rel["vetores"][vetor].items():
            if chave.startswith("producao"):
                continue
            rels, qnum = [], 0.0
            for obsid in ordem:
                mp, mc = prod["_mus"][obsid], c["_mus"][obsid]
                r = (mc - mp) / np.maximum(np.abs(mp), 1e-300)
                rels.append(np.abs(r).ravel())
                qnum += float(((mc - mp) ** 2 / np.maximum(mp, 1e-300)).sum())
            tod = np.concatenate(rels)
            c["mediana_relativa"] = float(np.median(tod))
            c["pior_bin_relativo"] = float(tod.max())
            c["Q_num_contra_producao"] = qnum
            print("%-12s %-10s %14.3e %14.3e %12.6f"
                  % (vetor, chave, c["mediana_relativa"],
                     c["pior_bin_relativo"], qnum))

    # as diferencas de ln L entre vetores, por configuracao
    print()
    print("=== DIFERENCAS de ln L entre vetores, por configuracao ===")
    vs = [v for v, _, _ in PO.VETORES]
    chaves = ["producao_%d" % n for n in a.bandas_producao] + [
        "psi0_%d" % n for n in a.niveis_candidata]
    print("%-26s %s" % ("par", "".join("%16s" % k for k in chaves)))
    pares = []
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            difs = {k: rel["vetores"][vs[i]][k]["ln_L_conjunto"]
                       - rel["vetores"][vs[j]][k]["ln_L_conjunto"]
                    for k in chaves}
            pares.append({"i": vs[i], "j": vs[j], "diferencas": difs})
            print("%-26s %s" % ("%s - %s" % (vs[i], vs[j]),
                                "".join("%16.4f" % difs[k] for k in chaves)))
    # estabilidade da CANDIDATA entre os seus dois niveis
    # OS DOIS ULTIMOS niveis, que e o criterio. Usar o primeiro e o ultimo
    # mediria a mudanca ACUMULADA e reprovaria uma candidata convergida: foi o
    # que aconteceu duas vezes, com a sequencia 240-480-960 acusando 0,2926
    # quando o passo relevante, 480 para 960, era 0,0130.
    # OS DOIS ULTIMOS NIVEIS DA MESMA FAMILIA de malha. Comparar producao_480
    # com psi0_960 nao e estabilidade de nada: sao malhas diferentes, e o rotulo
    # saiu errado assim uma vez. Se a lista nao tiver dois niveis da mesma
    # familia no fim, a estabilidade nao se calcula.
    familia = chaves[-1].split("_")[0]
    mesmos = [k for k in chaves if k.split("_")[0] == familia]
    k1, k2 = (mesmos[-2], mesmos[-1]) if len(mesmos) >= 2 else (None, None)
    if k1 is None:
        print("estabilidade NAO calculada: a lista nao tem dois niveis da "
              "familia %s" % familia)
        estab = None
    else:
        estab = max(abs(p["diferencas"][k1] - p["diferencas"][k2]) for p in pares)
    sens = max(abs(p["diferencas"][chaves[0]] - p["diferencas"][k2])
               for p in pares)
    print()
    print("sensibilidade a quadratura, producao contra candidata no nivel mais "
          "fino: %.4f (pior par)" % sens)
    print("estabilidade da CANDIDATA entre %s e %s, os dois ULTIMOS "
          "niveis: %.4f" % (k1, k2, estab))
    print("margem da referencia: 0,02")
    if estab > 0.02:
        veredito = ("INCONCLUSIVO: a candidata nao estabiliza entre os seus dois "
                    "niveis, move %.4f, acima da margem de 0,02" % estab)
    else:
        veredito = ("CANDIDATA ESTAVEL entre os seus niveis (%.4f); "
                    "sensibilidade a quadratura medida em %.4f" % (estab, sens))
    print(veredito)
    rel["pares"] = pares
    rel["sensibilidade_a_quadratura"] = sens
    rel["estabilidade_da_candidata"] = estab
    rel["margem_da_referencia"] = 0.02
    rel["veredito"] = veredito
    rel["custo"] = {v: {k: c["segundos"] for k, c in cs.items()}
                    for v, cs in rel["vetores"].items()}
    for v in rel["vetores"].values():
        for c in v.values():
            c.pop("_mus", None)
    Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print("gravado em %s" % a.saida)


if __name__ == "__main__":
    main()
