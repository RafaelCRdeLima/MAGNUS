#!/usr/bin/env python3
"""Consolida a rodada conjunta: diferencas entre os tres vetores, com procedencia.

O revisor, 07/10/2026: "A rodada conjunta deve registrar offsets, resolucao
efetiva, M, executavel e dados, e comparar DIFERENCAS de ln L entre os tres
vetores com referencia estavel. Ela decidira se o ponto medio basta ou se M=3 e
necessario."

E a ressalva dele sobre a resolucao, que entra no registro: zerar a resolucao
isola a discretizacao de fase NA IMPLEMENTACAO ATUAL. Isso nao precisa valer para
todo estudo futuro; com um operador temporal consistente entre resolucoes, ele
podera permanecer ligado.
"""
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))


def sha(c):
    h = hashlib.sha256()
    with open(c, "rb") as fh:
        for p in iter(lambda: fh.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()


def main():
    import por_observacao as PO
    arqs = sorted(glob.glob(str(RAIZ / "docs/discretizacao_conjunta_*.json")))
    dados = {}
    for f in arqs:
        d = json.loads(Path(f).read_text())
        dados[d["vetor"]] = {n["M"]: n for n in d["niveis"]}
    if len(dados) < 2:
        raise SystemExit("preciso de pelo menos dois vetores; tenho %s"
                         % list(dados))
    Ms = sorted(set.intersection(*[set(v) for v in dados.values()]))
    print("vetores: %s" % list(dados))
    print("niveis comuns: %s" % Ms)
    print()
    print("=== ln L conjunto por vetor e nivel ===")
    print("%-12s %s" % ("vetor", "".join("%18s" % ("M=%d" % M) for M in Ms)))
    for v, n in dados.items():
        print("%-12s %s" % (v, "".join("%18.4f" % n[M]["ln_L_conjunto"]
                                       for M in Ms)))
    print()
    print("=== o que decide: DIFERENCAS entre vetores, por nivel ===")
    vs = list(dados)
    pares = []
    print("%-12s %-12s %s %14s" % ("i", "j",
                                   "".join("%14s" % ("M=%d" % M) for M in Ms),
                                   "M3 - Mfino"))
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            a, b = dados[vs[i]], dados[vs[j]]
            difs = {M: a[M]["ln_L_conjunto"] - b[M]["ln_L_conjunto"] for M in Ms}
            fino = Ms[-1]
            res3 = (difs[3] - difs[fino]) if 3 in difs else None
            res1 = (difs[1] - difs[fino]) if 1 in difs else None
            pares.append({"i": vs[i], "j": vs[j], "diferencas": difs,
                          "residuo_do_ponto_medio": res1,
                          "residuo_de_M3": res3})
            print("%-12s %-12s %s %14s"
                  % (vs[i], vs[j], "".join("%14.4f" % difs[M] for M in Ms),
                     "%+.4f" % res3 if res3 is not None else "-"))
    print()
    # estabilidade da REFERENCIA na grandeza comparada, nao no lnL de um vetor
    if len(Ms) >= 3:
        est = [abs(p["diferencas"][Ms[-2]] - p["diferencas"][Ms[-1]])
               for p in pares]
        estab = max(est)
    else:
        estab = None
    pm = max(abs(p["residuo_do_ponto_medio"]) for p in pares
             if p["residuo_do_ponto_medio"] is not None)
    m3 = (max(abs(p["residuo_de_M3"]) for p in pares
              if p["residuo_de_M3"] is not None) if 3 in Ms else None)
    print("residuo do PONTO MEDIO nas diferencas: %+.4f (pior par)" % pm)
    if m3 is not None:
        print("residuo de M=3 nas diferencas:         %+.4f (pior par)" % m3)
    if estab is not None:
        print("estabilidade da DIFERENCA entre os dois niveis mais finos: %.4f"
              % estab)
    #: A MARGEM DA REFERENCIA, fixada pelo revisor em 07/10/2026: para a meta de
    #: 0,1 em diferencas de ln L, a mudanca entre os dois niveis mais finos deve
    #: ficar em <= 0,02, que e 20 por cento do orcamento. Uma referencia que
    #: varie 0,09 nao distingue com seguranca um residuo proximo de 0,1. Eu
    #: havia posto 0,1, isto e a propria meta, o que seria frouxo demais.
    META, MARGEM_DA_REFERENCIA = 0.1, 0.02
    print("meta: %.2f em diferencas; margem da referencia: %.2f"
          % (META, MARGEM_DA_REFERENCIA))
    if estab is not None and estab > MARGEM_DA_REFERENCIA:
        veredito = ("INCONCLUSIVO: a propria diferenca entre vetores ainda se "
                    "move %.4f entre os dois niveis mais finos, acima da margem "
                    "de %.2f exigida para distinguir um residuo perto de %.2f"
                    % (estab, MARGEM_DA_REFERENCIA, META))
    elif m3 is not None and abs(m3) <= META and abs(pm) > META:
        veredito = ("M=3 NECESSARIO E SUFICIENTE: o ponto medio erra %.4f nas "
                    "diferencas, acima de 0,1, e M=3 erra %.4f, dentro" % (pm, m3))
    elif abs(pm) <= META:
        veredito = ("PONTO MEDIO BASTA: ele erra %.4f nas diferencas, dentro de "
                    "0,1" % pm)
    else:
        veredito = ("EXIGE PROXIMO REFINAMENTO: o ponto medio erra %.4f e M=3 "
                    "erra %s, os dois acima de 0,1"
                    % (pm, "%.4f" % m3 if m3 is not None else "nao medido"))
    print(veredito)
    obss = PO.prepara(verbose=False)
    rel = {
      "o_que_e": "a rodada conjunta de fase, consolidada nas DIFERENCAS",
      "procedencia": {
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ,
                                 capture_output=True, text=True).stdout.strip(),
        "executavel": {"caminho": "build/magnus_engine",
                       "sha256": sha(RAIZ / "build/magnus_engine")},
        "dados": {o["obsid"]: {"npz": o["npz"],
                               "offset_bins": o["desloc_v3_bins"],
                               "offset_ciclos": PO.offset_em_ciclos(
                                   o["desloc_v3_bins"]),
                               "resolucao_declarada_us": o["resolucao_us"],
                               "resolucao_EFETIVA_us": 0.0,
                               "exposicao_s": o["exposicao"],
                               "contagens": o["contagens"]} for o in obss},
        "resolucao_efetiva_zero_porque": (
          "o motor suaviza so quando a resolucao passa de um bin de fase, e na "
          "grade fina o bin encolhe; zerar isola a discretizacao de fase NA "
          "IMPLEMENTACAO ATUAL. Com um operador temporal consistente entre "
          "resolucoes, ele podera permanecer ligado"),
        "niveis_M": Ms},
      "ln_L_conjunto": {v: {M: n[M]["ln_L_conjunto"] for M in Ms}
                        for v, n in dados.items()},
      "pares": pares,
      "residuo_do_ponto_medio_nas_diferencas": pm,
      "residuo_de_M3_nas_diferencas": m3,
      "estabilidade_da_diferenca": estab,
      "o_que_a_estabilidade_mede": (
        "a mudanca da DIFERENCA entre vetores entre os dois niveis mais finos, e "
        "nao a do ln L de um vetor so. Foi a correcao do revisor: os 0,0174 que "
        "eu citei eram a estabilidade de um vetor, nao da grandeza comparada"),
      "meta": META, "margem_da_referencia": MARGEM_DA_REFERENCIA,
      "por_que_a_margem_e_0_02": (
        "20 por cento do orcamento. Uma referencia que varie 0,09 nao distingue "
        "com seguranca um residuo proximo de 0,1; eu havia posto a margem igual "
        "a propria meta, o que seria frouxo demais"),
      "o_dominio_desta_rodada": (
        "os tres vetores e os offsets registrados. Se passar, libera a "
        "configuracao candidata; novos otimos e offsets fracionarios serao "
        "revalidados depois, sem exigir cobertura antecipada de toda a posterior"),
      "veredito": veredito,
      "custo_medido": {v: {M: n[M]["segundos"] for M in Ms}
                       for v, n in dados.items()},
    }
    Path(RAIZ / "docs/fase_consolidada.json").write_text(
        json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print("gravado em docs/fase_consolidada.json")


if __name__ == "__main__":
    main()
