"""Grade de tabelas MAGNUS variando o CAMPO — para o perfil de lnL(B).

Uma tabela por lg B, todas totalmente ionizadas (sem vácuo, sem átomo — o modo
validado do desfecho-2), mesma estrutura (lgT × θ_B × μ × E). Ajustando cada uma
à mesma fonte e comparando a verossimilhança máxima, o pico de lnL(B) é o campo
que os dados preferem — sem escolher B à mão.

Paraleliza por PONTO (lgB, lgT, θ_B), como a grade de Σ; memória leve porque sem
vácuo as amplitudes independem da densidade.

Uso:
    OMP_NUM_THREADS=2 python3 scripts/tabela_campo_grade.py \\
        --campos 12.5,13.0,13.5,13.8 --saida-prefixo build/campo --workers 4
"""

from __future__ import annotations

import argparse
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

LOG_T = [5.699, 5.799, 5.898, 6.000, 6.114, 6.204,
         6.301, 6.398, 6.505, 6.602, 6.699, 6.799]
LOG_G = 14.38
THETA_B_DEG = [0.0, 30.0, 60.0]
MU_NODES = 8


def _solve_point(task: tuple) -> tuple:
    log_field, log_t, theta_deg, iterations = task
    field = 10.0 ** log_field
    ion = magnetizada.CYCLOTRON_E_PER_GAUSS * field * magnetizada.MASS_RATIO
    energies = np.unique(np.concatenate(
        [estrutura.energy_grid(1.0e-3, 60.0, 160),
         ion * (1.0 + np.linspace(-0.12, 0.12, 60))]))
    started = time.time()
    solution = magnetizada.solve(
        log_t, LOG_G, field, theta_b=float(np.radians(theta_deg)),
        energies=energies, mu_nodes=MU_NODES, iterations=iterations)  # ionizado
    total = (solution["intensity"][:, :MU_NODES]
             + solution["intensity"][:, MU_NODES:])
    reference = estrutura.planck_energy(TABLE_ENERGIES, 10.0 ** log_t)
    rows = np.empty((MU_NODES, TABLE_ENERGIES.size))
    for index in range(MU_NODES):
        intensity = np.interp(TABLE_ENERGIES, solution["energies"],
                              total[:, index])
        rows[index] = np.log10(np.clip(intensity / reference, 1.0e-30, 1.0e30))
    return (log_field, log_t, theta_deg, rows, solution["mu"],
            solution["flux_error"], time.time() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campos", default="12.5,13.0,13.5,13.8")
    parser.add_argument("--saida-prefixo", type=Path, default=ROOT / "build" / "campo")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iteracoes", type=int, default=180)
    arguments = parser.parse_args()
    fields = [float(v) for v in arguments.campos.split(",")]
    tasks = [(b, t, th, arguments.iteracoes)
             for b in fields for t in LOG_T for th in THETA_B_DEG]
    print(f"{len(tasks)} soluções, {arguments.workers} workers", flush=True)
    tables = {b: np.zeros((len(LOG_T), 1, len(THETA_B_DEG),
                           MU_NODES, TABLE_ENERGIES.size)) for b in fields}
    worst = {b: 0.0 for b in fields}
    mu = None
    done = 0
    with mp.Pool(arguments.workers) as pool:
        for (b, log_t, theta_deg, rows, mu_out, flux_error, cost) in \
                pool.imap_unordered(_solve_point, tasks):
            mu = mu_out
            tables[b][LOG_T.index(log_t), 0, THETA_B_DEG.index(theta_deg)] = rows
            worst[b] = max(worst[b], flux_error)
            done += 1
            print(f"[{done}/{len(tasks)}] lgB={b} lgT={log_t:.3f} th={theta_deg:2.0f} "
                  f"fluxo {flux_error:.1e} {cost:.0f} s", flush=True)
    for b in fields:
        tag = f"{b:g}".replace(".", "_")
        path = arguments.saida_prefixo.with_name(
            arguments.saida_prefixo.name + f"_B{tag}.magnus")
        formato.write(path, LOG_T, [LOG_G], THETA_B_DEG, mu,
                      np.log10(TABLE_ENERGIES), tables[b])
        print(f"GRAVADA {path.name}  pior fluxo {worst[b]:.1e}", flush=True)


if __name__ == "__main__":
    main()
