"""A grade de exploração fina: tabelas (Sigma, vácuo, conversão) em paralelo.

O construtor serial (tabela_estagio2.py) serve para uma tabela; esta grade
constrói VÁRIAS — a série de Sigma da atmosfera fina mais a semi-infinita de
controle, todas com vácuo e conversão parcial — paralelizando por ponto de
grade (T, theta_B), porque cada solução com vácuo custa ~6-9 min e a série
inteira passa de 30 h seriais.

O paralelismo é por PONTO e não por tabela: os pontos custam parecido e a
fila nunca fica com um worker segurando a tabela mais cara sozinho no fim.

Memória manda no dimensionamento: esta máquina trava com >5-6 processos
(RETOMAR.md do BWmcmc, 2026-08). Aqui: 4 workers, e o lançador deve exportar
OMP_NUM_THREADS=2 antes do python para o BLAS não sobre-inscrever.

Uso:
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 scripts/tabela_fina_grade.py \\
        --colunas 1,10,100,inf --saida-prefixo build/fina
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from atmosfera import estrutura, magnetizada                        # noqa: E402
import tabela_intensidade as formato                                # noqa: E402
from tabela_estagio2 import TABLE_ENERGIES                          # noqa: E402

# Os mesmos eixos da magnus_HB1350_completa: a comparação de lnL entre as
# tabelas só isola FÍSICA se a estrutura for idêntica.
LOG_T = [5.699, 5.799, 5.898, 6.000, 6.114, 6.204,
         6.301, 6.398, 6.505, 6.602, 6.699, 6.799]
LOG_G = 14.38
THETA_B_DEG = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0]
FIELD_LOG = 13.5
MU_NODES = 8


def _solve_point(task: tuple) -> tuple:
    sigma, log_t, theta_deg, iterations = task
    field = 10.0 ** FIELD_LOG
    ion = magnetizada.CYCLOTRON_E_PER_GAUSS * field * magnetizada.MASS_RATIO
    energies = np.unique(np.concatenate(
        [estrutura.energy_grid(1.0e-3, 60.0, 160),
         ion * (1.0 + np.linspace(-0.12, 0.12, 60))]))
    started = time.time()
    solution = magnetizada.solve(
        log_t, LOG_G, field, theta_b=float(np.radians(theta_deg)),
        energies=energies, mu_nodes=MU_NODES, iterations=iterations,
        surface_column=(None if np.isinf(sigma) else float(sigma)),
        vacuum=True, conversion="partial")
    total = (solution["intensity"][:, :MU_NODES]
             + solution["intensity"][:, MU_NODES:])
    reference = estrutura.planck_energy(TABLE_ENERGIES, 10.0 ** log_t)
    rows = np.empty((MU_NODES, TABLE_ENERGIES.size))
    for index in range(MU_NODES):
        intensity = np.interp(TABLE_ENERGIES, solution["energies"],
                              total[:, index])
        rows[index] = np.log10(np.clip(intensity / reference, 1.0e-30, 1.0e30))
    return (sigma, log_t, theta_deg, rows, solution["mu"],
            solution["flux_error"], solution["iterations"],
            time.time() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colunas", default="1,10,100,inf",
                        help="Sigmas em g/cm^2, 'inf' = semi-infinita")
    parser.add_argument("--saida-prefixo", type=Path,
                        default=ROOT / "build" / "fina")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iteracoes", type=int, default=220)
    parser.add_argument("--angulos", default=None,
                        help="theta_B em graus (vírgula); ausente = os 6 padrão")
    parser.add_argument("--temperaturas", default=None,
                        help="lg T (vírgula); ausente = os 12 padrão")
    arguments = parser.parse_args()
    global THETA_B_DEG, LOG_T
    if arguments.angulos:
        THETA_B_DEG = [float(v) for v in arguments.angulos.split(",")]
    if arguments.temperaturas:
        LOG_T = [float(v) for v in arguments.temperaturas.split(",")]
    sigmas = [float(v) for v in arguments.colunas.split(",")]

    def path_of(sigma: float) -> Path:
        tag = "inf" if np.isinf(sigma) else f"{sigma:g}"
        return arguments.saida_prefixo.with_name(
            arguments.saida_prefixo.name + f"_S{tag}.magnus")

    # Uma tabela por Sigma, ESCRITA assim que seu Sigma fecha. A campanha de
    # 288 pontos morreu uma vez por memória (guarda em 272 MB, 113/288) e
    # levou tudo junto porque a escrita era só no fim. Agora: Sigmas do mais
    # BARATO (menos profundidades) ao mais caro, cada um salvo ao terminar, e
    # os já-salvos são PULADOS na retomada. Um corte perde no máximo a tabela
    # em curso.
    ordered = sorted(sigmas, key=lambda s: (np.inf if np.isinf(s) else s))
    summary = {}
    fluxos_path = arguments.saida_prefixo.parent / "fina_grade_fluxos.json"
    if fluxos_path.exists():
        summary = json.loads(fluxos_path.read_text())
    for sigma in ordered:
        tag = "inf" if np.isinf(sigma) else f"{sigma:g}"
        if path_of(sigma).exists() and tag in summary:
            print(f"[pulando] Sigma={tag} já existe ({path_of(sigma).name})",
                  flush=True)
            continue
        tasks = [(sigma, log_t, theta, arguments.iteracoes)
                 for log_t in LOG_T for theta in THETA_B_DEG]
        table = np.zeros((len(LOG_T), 1, len(THETA_B_DEG),
                          MU_NODES, TABLE_ENERGIES.size))
        worst = 0.0
        mu = None
        done = 0
        print(f"Sigma={tag}: {len(tasks)} pontos, {arguments.workers} workers",
              flush=True)
        with mp.Pool(arguments.workers) as pool:
            for (_sig, log_t, theta_deg, rows, mu_out, flux_error,
                 iterations, cost) in pool.imap_unordered(_solve_point, tasks):
                mu = mu_out
                table[LOG_T.index(log_t), 0, THETA_B_DEG.index(theta_deg)] = rows
                worst = max(worst, flux_error)
                done += 1
                print(f"[S{tag} {done}/{len(tasks)}] lgT={log_t:.3f} "
                      f"th={theta_deg:2.0f} fluxo {flux_error:.1e} "
                      f"{iterations} it {cost:.0f} s", flush=True)
        formato.write(path_of(sigma), LOG_T, [LOG_G], THETA_B_DEG, mu,
                      np.log10(TABLE_ENERGIES), table)
        summary[tag] = worst
        fluxos_path.write_text(json.dumps(summary, indent=1))
        print(f"GRAVADA {path_of(sigma).name}  pior fluxo {worst:.1e}", flush=True)


if __name__ == "__main__":
    main()
