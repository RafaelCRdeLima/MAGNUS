"""Grade MAGNUS com eixo de log g — para QUEBRAR a degenerescência M-R.

Estende a grade de campo (tabela_campo_grade.py) com um eixo de gravidade
superficial. Com g fixo, o espectro só responde a M,R pelo redshift e pela área
(degenerado). Com o eixo de g, a FORMA do contínuo responde a g=GM(1+z)/R^2, e o
ajuste ganha um segundo vínculo (o redshift dá u=2GM/Rc^2, a forma dá g) que
separa M de R.

Grade: lg B x lg T x lg g x theta_B x mu x lg E, totalmente ionizada (modo
validado do desfecho-2). Escreve MAGNUSI2 direto (eixo de B na frente).
Embaraçosamente paralela por ponto; checkpoint a cada N solves (corrida ~2.6 h).

Uso:
    OMP_NUM_THREADS=2 python3 scripts/tabela_campo_g_grade.py \\
        --saida build/magnus_campoBg.magnus --workers 5
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

LOG_B = [12.5, 13.0, 13.5, 13.8]
LOG_T = [5.699, 5.799, 5.898, 6.000, 6.114, 6.204,
         6.301, 6.398, 6.505, 6.602, 6.699, 6.799]
LOG_G = [13.8, 14.0, 14.2, 14.4, 14.6]      # o eixo NOVO
THETA_B_DEG = [0.0, 30.0, 60.0]
MU_NODES = 8
TABLE_ENERGIES = np.logspace(np.log10(0.03), np.log10(20.0), 160)


def _solve_point(task: tuple) -> tuple:
    log_b, log_t, log_g, theta_deg, iterations = task[:5]
    atomic = bool(task[5]) if len(task) > 5 else False
    field = 10.0 ** log_b
    ion = magnetizada.CYCLOTRON_E_PER_GAUSS * field * magnetizada.MASS_RATIO
    energies = np.unique(np.concatenate(
        [estrutura.energy_grid(1.0e-3, 60.0, 160),
         ion * (1.0 + np.linspace(-0.12, 0.12, 60))]))
    started = time.time()
    solution = magnetizada.solve(
        log_t, log_g, field, theta_b=float(np.radians(theta_deg)),
        energies=energies, mu_nodes=MU_NODES, iterations=iterations,
        atomic=atomic)  # atomic=False: totalmente ionizada
    total = (solution["intensity"][:, :MU_NODES]
             + solution["intensity"][:, MU_NODES:])
    reference = estrutura.planck_energy(TABLE_ENERGIES, 10.0 ** log_t)
    rows = np.empty((MU_NODES, TABLE_ENERGIES.size))
    for index in range(MU_NODES):
        intensity = np.interp(TABLE_ENERGIES, solution["energies"], total[:, index])
        rows[index] = np.log10(np.clip(intensity / reference, 1.0e-30, 1.0e30))
    return (log_b, log_t, log_g, theta_deg, rows, solution["mu"],
            solution["flux_error"], time.time() - started)


def _shape():
    return (len(LOG_B), len(LOG_T), len(LOG_G), len(THETA_B_DEG),
            MU_NODES, TABLE_ENERGIES.size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saida", type=Path, default=ROOT / "build" / "magnus_campoBg.magnus")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--iteracoes", type=int, default=180)
    parser.add_argument("--checkpoint-cada", type=int, default=15)
    parser.add_argument("--atomic", action="store_true",
                        help="ionizacao parcial: x(H) do Ioffe + ligado-livre atomico (atomico.py)")
    arguments = parser.parse_args()
    ckpt = arguments.saida.with_suffix(".ckpt.npz")
    arguments.saida.parent.mkdir(parents=True, exist_ok=True)

    log_w = np.zeros(_shape())
    done = np.zeros(_shape()[:4], dtype=bool)   # (nB,nT,nG,nThetaB)
    mu = None
    worst = 0.0
    if ckpt.exists():
        z = np.load(ckpt)
        if z["log_w"].shape == log_w.shape:
            log_w = z["log_w"]; done = z["done"]; mu = z["mu"]
            worst = float(z["worst"])
            print(f"checkpoint retomado: {done.sum()}/{done.size} pontos ja feitos", flush=True)

    tasks = [(b, t, g, th, arguments.iteracoes, arguments.atomic)
             for b in LOG_B for t in LOG_T for g in LOG_G for th in THETA_B_DEG
             if not done[LOG_B.index(b), LOG_T.index(t),
                         LOG_G.index(g), THETA_B_DEG.index(th)]]
    total_pts = int(np.prod(_shape()[:4]))
    print(f"{len(tasks)} soluções restantes de {total_pts}, {arguments.workers} workers", flush=True)

    started = time.time()
    completed = 0
    with mp.Pool(arguments.workers) as pool:
        for (b, log_t, log_g, theta_deg, rows, mu_out, flux_error, cost) in \
                pool.imap_unordered(_solve_point, tasks):
            mu = mu_out
            ib, it_, ig, ith = (LOG_B.index(b), LOG_T.index(log_t),
                                LOG_G.index(log_g), THETA_B_DEG.index(theta_deg))
            log_w[ib, it_, ig, ith] = rows
            done[ib, it_, ig, ith] = True
            worst = max(worst, flux_error)
            completed += 1
            elapsed = time.time() - started
            eta = elapsed / completed * (len(tasks) - completed) / 3600.0
            print(f"[{done.sum()}/{total_pts}] lgB={b} lgT={log_t:.3f} lgg={log_g:.2f} "
                  f"th={theta_deg:2.0f} fluxo {flux_error:.1e} {cost:.0f}s ETA {eta:.1f}h",
                  flush=True)
            if completed % arguments.checkpoint_cada == 0:
                np.savez(ckpt, log_w=log_w, done=done, mu=mu, worst=worst)

    formato.write_with_field(arguments.saida, LOG_B, LOG_T, LOG_G, THETA_B_DEG,
                             mu, np.log10(TABLE_ENERGIES), log_w)
    print(f"GRAVADA {arguments.saida}  pior fluxo {worst:.1e}", flush=True)
    if ckpt.exists():
        ckpt.unlink()


if __name__ == "__main__":
    main()
