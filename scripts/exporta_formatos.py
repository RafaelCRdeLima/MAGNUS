"""Exporta a tabela do MAGNUS nos formatos da comunidade.

Dois destinos, cada um com o que o outro não tem:

* **Estilo `.in` do Ho** — seis linhas de cabeçalho (nT, lg T, ng, lg g, nE,
  energias) e um espectro F(E) por par (T, g), integrado em ângulo, na fatia
  theta_B = 0. É o formato que o XSPEC e meio mundo leem, e o fluxo sai na MESMA
  unidade das tabelas dele: F tal que int F dE = C T^4 com C = 2,342e-22 — o
  valor medido nos arquivos originais pela auditoria do estágio 0. Um arquivo
  nosso e um dele ficam comparáveis número a número.
* **Cinco colunas do X-PSI** — (lg T, lg g, mu, lg E, lg I), texto, com I em
  erg s^-1 cm^-2 sr^-1 keV^-1. É o formato de intercâmbio de intensidade da
  comunidade de traçado de raios; também sai da fatia theta_B = 0, porque o
  formato não tem o eixo.

O que NENHUM dos dois carrega — o eixo theta_B — continua só no formato nativo,
e é por isso que ele existe.

Uso:
    python3 scripts/exporta_formatos.py build/estagio2_B135_thetab.magnus
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from atmosfera import estrutura, transporte                         # noqa: E402
import tabela_intensidade as fmt                                    # noqa: E402

#: A constante de unidade dos arquivos do Ho, medida pela auditoria: C tal que
#: int F dE = C T^4 nos 22 arquivos locais (mediana 2,342e-22 no HB1350).
HO_CONSTANT = 2.342e-22


def export(path: Path) -> None:
    table = fmt.read(path)
    log_t = np.asarray(table["log_t"], dtype=float)
    log_g = np.asarray(table["log_g"], dtype=float)
    mu = np.asarray(table["mu"], dtype=float)
    energies = 10.0 ** np.asarray(table["log_e"], dtype=float)
    slice_b = int(np.argmin(np.abs(np.asarray(table["theta_b_deg"], dtype=float))))

    # Pesos de quadratura para integrar o feixe em fluxo: os nós de mu da
    # tabela SÃO os de Gauss-Legendre do solucionador, então os pesos certos
    # existem e não precisam ser inventados por trapézio.
    nodes, weights = transporte.gauss_legendre_mu(mu.size)
    if not np.allclose(nodes, mu, atol=1.0e-9):
        raise ValueError("os nós de mu da tabela não são os de Gauss-Legendre")

    stem = path.with_suffix("")
    # ------------------------------------------------------------- estilo Ho
    lines = [f" {log_t.size}", " " + " ".join(f"{v:.6g}" for v in log_t),
             f" {log_g.size}", " " + " ".join(f"{v:.6g}" for v in log_g),
             f" {energies.size}", " " + " ".join(f"{v:.6g}" for v in energies)]
    for it in range(log_t.size):
        for ig in range(log_g.size):
            planck = estrutura.planck_energy(energies, 10.0 ** log_t[it])
            ratio = 10.0 ** np.asarray(table["log_w"][it, ig, slice_b], dtype=float)
            flux = 2.0 * np.pi * np.einsum("m,me->e", weights * mu, ratio) * planck
            lines.append(" " + " ".join(f"{v:.5g}"
                                        for v in flux * HO_CONSTANT / estrutura.STEFAN))
    ho_path = stem.parent / (stem.name + "_estilo_ho.in")
    ho_path.write_text("\n".join(lines) + "\n", encoding="ascii")

    # ------------------------------------------------------------- X-PSI
    rows = ["# lgT[K] lgg[cm/s2] mu lgE[keV] lgI[erg/s/cm2/sr/keV]",
            f"# MAGNUS, fatia theta_B = {table['theta_b_deg'][slice_b]:.0f} graus; "
            "lg I = lg(B_E(T_ef)) + lg w"]
    for it in range(log_t.size):
        planck = estrutura.planck_energy(energies, 10.0 ** log_t[it])
        for ig in range(log_g.size):
            for im in range(mu.size):
                ratio = np.asarray(table["log_w"][it, ig, slice_b, im], dtype=float)
                intensity = np.log10(np.maximum(planck, 1.0e-300)) + ratio
                for ie in range(energies.size):
                    rows.append(f"{log_t[it]:.5f} {log_g[ig]:.4f} {mu[im]:.6f} "
                                f"{np.log10(energies[ie]):.5f} {intensity[ie]:.5f}")
    xpsi_path = stem.parent / (stem.name + "_xpsi.txt")
    xpsi_path.write_text("\n".join(rows) + "\n", encoding="ascii")
    print(ho_path, f"({ho_path.stat().st_size/1e3:.0f} kB)")
    print(xpsi_path, f"({xpsi_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    export(Path(sys.argv[1]) if len(sys.argv) > 1
           else ROOT / "build" / "estagio2_B135_thetab.magnus")
