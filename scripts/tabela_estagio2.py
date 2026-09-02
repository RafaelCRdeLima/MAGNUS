"""A tabela magnetizada: do solucionador de dois modos ao motor — a entrega.

A pergunta que abriu o projeto era se o feixe calculado é lápis ou leque. O
solucionador respondeu: no campo e na temperatura da RBS 1223, LEQUE em toda a
banda — máximo em mu ~ 0,6-0,8, normal suprimida — com o poço de cíclotron do
próton em 0,20 keV aparecendo sozinho, de física e não de ajuste. Este script
empacota essa resposta no formato que o motor lê.

O eixo theta_B da tabela fica, por ora, com um ponto só (campo ao longo da
normal): é o caso que o solucionador já resolve. A varredura em theta_B é o
passo seguinte, e o formato já a espera.

Uso:
    python3 scripts/tabela_estagio2.py --campo 13.5 --temperaturas 6.0,6.1,6.2 \\
        --gravidades 14.4 --saida build/estagio2.magnus
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from atmosfera import estrutura, magnetizada                        # noqa: E402
import tabela_intensidade as formato                                # noqa: E402

TABLE_ENERGIES = np.logspace(np.log10(0.03), np.log10(20.0), 160)


def build(field_log: float, log_temperatures: list[float],
          log_gravities: list[float], theta_b_degrees: list[float] | None = None,
          mu_nodes: int = 8, **kwargs) -> dict:
    theta_b_degrees = [0.0] if theta_b_degrees is None else theta_b_degrees
    field = 10.0 ** field_log
    ion = magnetizada.CYCLOTRON_E_PER_GAUSS * field * magnetizada.MASS_RATIO
    energies = np.unique(np.concatenate(
        [estrutura.energy_grid(1.0e-3, 60.0, 160),
         ion * (1.0 + np.linspace(-0.12, 0.12, 60))]))
    mu = None
    values = np.zeros((len(log_temperatures), len(log_gravities), len(theta_b_degrees),
                       mu_nodes, TABLE_ENERGIES.size))
    for it, log_t in enumerate(log_temperatures):
      for ig, log_g in enumerate(log_gravities):
        for ib, theta_deg in enumerate(theta_b_degrees):
            started = time.time()
            solution = magnetizada.solve(log_t, log_g, field,
                                         theta_b=float(np.radians(theta_deg)),
                                         energies=energies,
                                         mu_nodes=mu_nodes, **kwargs)
            mu = solution["mu"]
            # A intensidade TOTAL, soma dos dois modos — o motor de hoje não é
            # polarimétrico. Os modos separados continuam no solucionador para
            # quando ele for.
            total = solution["intensity"][:, :mu_nodes] + solution["intensity"][:, mu_nodes:]
            reference = estrutura.planck_energy(TABLE_ENERGIES, 10.0 ** log_t)
            for index in range(mu_nodes):
                intensity = np.interp(TABLE_ENERGIES, solution["energies"],
                                      total[:, index])
                # Cada direção carrega I_total; o corpo negro isotrópico de
                # referência também é o total das duas polarizações. lg w = 0
                # continua devolvendo o corpo negro exato.
                values[it, ig, ib, index] = np.log10(
                    np.clip(intensity / reference, 1.0e-30, 1.0e30))
            print(f"  lg T = {log_t:.2f}  lg g = {log_g:.2f}  theta_B = {theta_deg:4.1f}  "
                  f"fluxo {solution['flux_error']:.1e}  {solution['iterations']} it  "
                  f"{time.time() - started:.0f} s", flush=True)
    return {"log_t": log_temperatures, "log_g": log_gravities,
            "theta_b": theta_b_degrees, "mu": mu,
            "log_e": np.log10(TABLE_ENERGIES), "log_w": values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campo", type=float, default=13.5, help="lg B em gauss")
    parser.add_argument("--temperaturas", default="6.0,6.1,6.2")
    parser.add_argument("--gravidades", default="14.4")
    parser.add_argument("--angulos", default="0",
                        help="theta_B em graus, separados por vírgula")
    parser.add_argument("--saida", type=Path, default=ROOT / "build" / "estagio2.magnus")
    parser.add_argument("--nos-mu", type=int, default=8)
    parser.add_argument("--iteracoes", type=int, default=220)
    arguments = parser.parse_args()
    table = build(arguments.campo,
                  [float(v) for v in arguments.temperaturas.split(",")],
                  [float(v) for v in arguments.gravidades.split(",")],
                  theta_b_degrees=[float(v) for v in arguments.angulos.split(",")],
                  mu_nodes=arguments.nos_mu, iterations=arguments.iteracoes)
    formato.write(arguments.saida, table["log_t"], table["log_g"], table["theta_b"],
                  table["mu"], table["log_e"], table["log_w"])
    print(f"{arguments.saida}  ({arguments.saida.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
