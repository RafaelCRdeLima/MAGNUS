"""Do solucionador para a tabela de intensidade: a cadeia inteira, ponta a ponta.

O estágio 0 definiu o formato justamente para isto — para que cada estágio a
partir do 1 entregue uma tabela na mesma escala, que vai direto ao ajuste sem
tradução no meio. Este é o primeiro que a produz de física calculada.

A atmosfera do estágio 1 é NÃO magnética, então o eixo theta_B tem um ponto só.
O formato já o carrega mesmo assim: o estágio 2 preenche o eixo, e quem lê a
tabela não precisa saber de qual estágio ela veio.

Uso:
    python3 scripts/tabela_estagio1.py --saida build/estagio1.magnus \\
        --temperaturas 5.9,6.0,6.1 --gravidades 14.3
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

from atmosfera import estrutura                                     # noqa: E402
import tabela_intensidade as formato                                # noqa: E402

#: Grade em energia da tabela. Log, cobrindo com folga o que o XMM enxerga da
#: RBS 1223, e mais larga que a banda de ajuste para a interpolação nunca ter de
#: extrapolar nas bordas.
TABLE_ENERGIES = np.logspace(np.log10(0.03), np.log10(20.0), 160)


def build(log_temperatures: list[float], log_gravities: list[float],
          mu_nodes: int = 12, **kwargs) -> dict:
    """Resolve a grade e devolve lg w pronto para gravar.

    w = I / B_E(T_ef) é a razão para a intensidade de corpo negro ISOTRÓPICA de
    mesma temperatura efetiva — a mesma definição do formato, para que lg w = 0
    continue sendo o corpo negro exato e o modelo continue aninhando.
    """
    mu = None
    values = np.zeros((len(log_temperatures), len(log_gravities), 1,
                       mu_nodes, TABLE_ENERGIES.size))
    for it, log_t in enumerate(log_temperatures):
        for ig, log_g in enumerate(log_gravities):
            started = time.time()
            solution = estrutura.solve(log_t, log_g, mu_nodes=mu_nodes, **kwargs)
            mu = solution["mu"]
            reference = estrutura.planck_energy(TABLE_ENERGIES, 10.0 ** log_t)
            for index in range(mu_nodes):
                intensity = np.interp(TABLE_ENERGIES, solution["energies"],
                                      solution["intensity"][:, index])
                ratio = np.clip(intensity / reference, 1.0e-30, 1.0e30)
                values[it, ig, 0, index] = np.log10(ratio)
            print(f"  lg T = {log_t:.2f}  lg g = {log_g:.2f}  "
                  f"erro de fluxo {solution['flux_error']:.1e}  "
                  f"{solution['iterations']} iterações  {time.time() - started:.0f} s",
                  flush=True)
    return {"log_t": log_temperatures, "log_g": log_gravities,
            "theta_b": [0.0], "mu": mu, "log_e": np.log10(TABLE_ENERGIES),
            "log_w": values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saida", type=Path, default=ROOT / "build" / "estagio1.magnus")
    parser.add_argument("--temperaturas", default="5.9,6.0,6.1")
    parser.add_argument("--gravidades", default="14.3")
    parser.add_argument("--nos-mu", type=int, default=12)
    parser.add_argument("--iteracoes", type=int, default=300)
    arguments = parser.parse_args()
    table = build([float(v) for v in arguments.temperaturas.split(",")],
                  [float(v) for v in arguments.gravidades.split(",")],
                  mu_nodes=arguments.nos_mu, iterations=arguments.iteracoes)
    formato.write(arguments.saida, table["log_t"], table["log_g"], table["theta_b"],
                  table["mu"], table["log_e"], table["log_w"])
    print(f"{arguments.saida}  ({arguments.saida.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
