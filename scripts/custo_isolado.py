#!/usr/bin/env python3
"""Custo ISOLADO das finalistas, em CPU e parede.

O revisor, 08/10/2026: "Faça a medição isolada de CPU e parede. Como a outra
finalista não passou na precisão, seu tempo serve apenas como comparação; não
poderá vencer por ser mais rápida."

POR QUE ISOLADO. Os tempos desta sessão vieram de corridas sob contenção com o
ambiente de trabalho, e a MESMA configuracao mediu 187,7 s e 1348 s em duas
corridas, fator 7. Os ln L nao sao afetados, porque sao deterministas; os tempos
sao. Aqui mede-se CPU do processo filho, por `getrusage(RUSAGE_CHILDREN)`, que
nao conta o tempo em que ele esteve esperando CPU, e tambem a parede, com a carga
do sistema registrada ao lado.

Uso: python3 scripts/custo_isolado.py [--repeticoes 3]
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import mcmc_fit as m  # noqa: E402
import por_observacao as PO  # noqa: E402
import compara_malhas as CM  # noqa: E402

CONFIGS = (("psi_0 480, tabela 115200, APROVADA", "psi0", 480, 115200),
           ("producao 120, tabela 1800, atual", "producao", 120, None),
           ("producao 480, tabela 115200, REPROVADA", "producao", 480, 115200))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeticoes", type=int, default=3)
    ap.add_argument("--vetor", default="B colado")
    ap.add_argument("--saida", default=str(RAIZ / "docs/custo_isolado.json"))
    a = ap.parse_args()
    obss = {o["obsid"]: o for o in PO.prepara()}
    ordem = sorted(obss)
    meta, ev = m.read_events(RAIZ / "dados/eventos.csv")
    rel = {"o_que_e": "custo isolado das finalistas, CPU e parede",
           "por_que": ("os tempos sob contencao oscilaram por fator 7 na mesma "
                       "configuracao; CPU do filho nao conta espera por CPU"),
           "a_outra_finalista_nao_disputa": (
             "a producao nao passou na precisao, logo o tempo dela e so "
             "comparacao e nao pode vencer por ser mais rapida"),
           "vetor": a.vetor, "repeticoes": a.repeticoes, "configs": []}
    print("carga do sistema no inicio: %s" % (os.getloadavg(),))
    print("%-42s %9s %9s %9s %7s" % ("configuracao", "CPU s", "parede s",
                                     "CPU/par", "carga"))
    for rotulo, malha, bandas, tabela in CONFIGS:
        cpus, paredes, cargas = [], [], []
        for _ in range(a.repeticoes):
            extra = ["--ray-table-size", str(tabela)] if tabela else None
            antes = resource.getrusage(resource.RUSAGE_CHILDREN)
            t0 = time.time()
            CM.uma_rodada(obss, ordem, a.vetor, malha, bandas, 8, meta, ev,
                          extra=extra)
            parede = time.time() - t0
            depois = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu = ((depois.ru_utime - antes.ru_utime)
                   + (depois.ru_stime - antes.ru_stime))
            cpus.append(cpu)
            paredes.append(parede)
            cargas.append(os.getloadavg()[0])
        c = {"rotulo": rotulo, "malha": malha, "bandas": bandas,
             "tabela": tabela, "cpu_s": cpus, "parede_s": paredes,
             "carga": cargas,
             "cpu_mediano": sorted(cpus)[len(cpus) // 2],
             "parede_mediano": sorted(paredes)[len(paredes) // 2]}
        c["razao_cpu_parede"] = c["cpu_mediano"] / c["parede_mediano"]
        rel["configs"].append(c)
        print("%-42s %9.1f %9.1f %9.3f %7.2f"
              % (rotulo, c["cpu_mediano"], c["parede_mediano"],
                 c["razao_cpu_parede"], max(cargas)), flush=True)
    base = [c for c in rel["configs"] if c["malha"] == "producao"
            and c["bandas"] == 120][0]
    aprov = rel["configs"][0]
    rel["fator_contra_a_atual"] = aprov["cpu_mediano"] / base["cpu_mediano"]
    print()
    print("a APROVADA custa %.2f vezes a configuracao atual, em CPU"
          % rel["fator_contra_a_atual"])
    print("e a razao CPU/parede diz se houve contencao: perto de 1 e limpo")
    Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print("gravado em %s" % a.saida)


if __name__ == "__main__":
    main()
