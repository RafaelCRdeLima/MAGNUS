"""Os portões do estágio 2 que já fecharam, presos para não reabrirem.

O módulo magnetizado tira os modos de um autoproblema exato do tensor
dielétrico — não há fórmula de artigo para errar de memória — e estes testes
prendem as três coisas que a auditoria já mediu: o aninhamento em B -> 0, a
geometria dos modos nos dois limites exatos, e o tensor de Rosseland contra uma
linha REAL do Potekhin em que a comparação é limpa (ionização completa, plasma
dominando o vácuo por duas ordens).
"""

from __future__ import annotations

import gzip
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atmosfera import magnetizada as mg                             # noqa: E402
from atmosfera.estrutura import opacities, rosseland_mean           # noqa: E402


class TestModos(unittest.TestCase):
    def test_normalizacao_e_limites_exatos(self) -> None:
        """theta = 0: circulares puros; theta = 90: lineares. Soma = 1."""
        energies = np.array([0.05, 0.2, 1.0])
        along = mg.mode_amplitudes(energies, 0.0, 1.0e-10, 1.0e13)
        across = mg.mode_amplitudes(energies, np.pi / 2.0, 1.0e-10, 1.0e13)
        for amplitudes in (along, across):
            self.assertLess(float(np.max(np.abs(amplitudes.sum(axis=2) - 1.0))), 1.0e-9)
        # Ao longo de B nada aponta em z, e cada modo é uma circular quase pura.
        self.assertLess(float(np.max(along[:, :, 2])), 1.0e-9)
        self.assertGreater(float(np.min(np.max(along, axis=2))), 0.999)
        # Perpendicular: um modo é linear ao longo de B (e_z = 1), o outro
        # linear no plano, que na base cíclica é meio a meio.
        ordinary = np.max(across[:, :, 2], axis=1)
        self.assertGreater(float(np.min(ordinary)), 0.999)

    def test_aninhamento_em_campo_nulo(self) -> None:
        """Com B -> 0 os dois modos degeneram no livre-livre + Thomson do
        estágio 1, em qualquer ângulo. É o análogo do 'tabela de zeros devolve
        corpo negro': mede o leitor da física, sem física nova no meio."""
        energies = np.logspace(-2.0, 1.0, 30)
        density, temperature = 1.0e-3, 1.0e6
        absorption, scattering = opacities(energies, np.full_like(energies, density),
                                           np.full_like(energies, temperature))
        worst = 0.0
        for theta in (0.0, 0.7, np.pi / 2.0):
            modes = mg.mode_opacities(energies, theta, density, temperature, 1.0)
            for mode in (0, 1):
                worst = max(
                    worst,
                    float(np.max(np.abs(modes["scattering"][:, mode] / scattering - 1.0))),
                    float(np.max(np.abs(modes["absorption"][:, mode] / absorption - 1.0))))
        self.assertLess(worst, 1.0e-5, f"pior desvio {worst:.2e}")

    def test_tensor_isotropico_reproduz_o_escalar(self) -> None:
        """Com B -> 0, K_par = K_perp = a média de Rosseland comum.

        É o teste que pegou um fator 2 de normalização angular: int dOmega/4pi
        de função par de mu é int_0^1 dmu, não a metade."""
        energies = np.logspace(-2.0, 1.0, 200)
        density, temperature = 1.0e-2, 2.0e6
        parallel, transverse = mg.rosseland_tensor(energies, density, temperature, 1.0)
        absorption, scattering = opacities(energies, np.full_like(energies, density),
                                           np.full_like(energies, temperature))
        scalar = float(rosseland_mean(energies, (absorption + scattering)[:, None],
                                      np.array([temperature]))[0])
        self.assertLess(abs(parallel / transverse - 1.0), 1.0e-6)
        self.assertLess(abs(parallel / scalar - 1.0), 1.0e-6)


class TestContraOPotekhin(unittest.TestCase):
    """A linha limpa: lg B = 12, lg T = 6,5, lg R = -1.

    Ionização completa (x(H) = 4e-4 com contínuo suprimido só por ~1e-2, então o
    átomo residual não pesa), e o plasma domina o vácuo por duas ordens. Medido
    na auditoria: razões 0,98 e 0,92. O teste prende a janela de 20% do plano.
    """

    def test_k0_e_k1_dentro_do_alvo(self) -> None:
        target = None
        current = None
        with gzip.open(ROOT / "atmosphere_data" / "potekhin_magnetic_h"
                       / "hmag12_0.dat.gz", "rt", encoding="latin-1") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) == 2:
                    try:
                        current = float(fields[0])
                    except ValueError:
                        continue
                elif len(fields) >= 14 and current is not None and \
                        abs(current - 6.5) < 1.0e-6 and abs(float(fields[0]) + 1.0) < 1e-6:
                    target = [float(v) for v in fields]
        self.assertIsNotNone(target)
        temperature, field = 10.0 ** 6.5, 1.0e12
        density = 10.0 ** target[0] * (temperature / 1.0e6) ** 3
        ion = mg.CYCLOTRON_E_PER_GAUSS * field * mg.MASS_RATIO
        energies = np.unique(np.concatenate(
            [np.logspace(-2.3, 1.5, 400),
             ion * (1.0 + np.linspace(-0.05, 0.05, 60)),
             mg.CYCLOTRON_E_PER_GAUSS * field * (1.0 + np.linspace(-0.1, 0.1, 60))]))
        parallel, transverse = mg.rosseland_tensor(energies, density, temperature, field)
        ratio_parallel = 10.0 ** target[12] / parallel
        ratio_transverse = 10.0 ** target[13] / transverse
        self.assertLess(abs(ratio_parallel - 1.0), 0.2,
                        f"K0: tabelado/nosso = {ratio_parallel:.3f}")
        self.assertLess(abs(ratio_transverse - 1.0), 0.2,
                        f"K1: tabelado/nosso = {ratio_transverse:.3f}")


if __name__ == "__main__":
    unittest.main()


class TestTransportePolarizado(unittest.TestCase):
    """O solucionador de dois modos, aferido nos seus dois limites."""

    def test_modos_identicos_reproduzem_o_coerente(self) -> None:
        """Dois modos iguais e isotrópicos = o problema de um modo, exato.

        O mapeamento é: cada canal termaliza em B/2, os pesos somam a quadratura
        inteira por modo, e J, H e I saem como TOTAIS. Dois meios-fatores já
        morderam aqui — pesos somando 1 davam fluxo na metade, e a emissão sem o
        B/2 estacionava o equilíbrio no meio do caminho — e é este teste que os
        impede de voltar."""
        from atmosfera.transporte import (coupled_feautrier, gauss_legendre_mu,
                                          optical_depth_grid, polarized_feautrier)
        mu, w = gauss_legendre_mu(6)
        n_freq, n_depth = 8, 51
        tau = np.tile(optical_depth_grid(1.0e4, n_depth), (n_freq, 1)) \
            * np.linspace(0.5, 2.0, n_freq)[:, None]
        epsilon = np.linspace(1.0e-4, 0.9, n_freq)[:, None] * np.ones((n_freq, n_depth))
        planck = np.linspace(1.0, 3.0, n_freq)[:, None] * np.ones((n_freq, n_depth))
        reference = coupled_feautrier(tau, mu, w, epsilon, planck)

        n_channel = 2 * mu.size
        mu_c = np.concatenate([mu, mu])
        w_c = np.concatenate([w, w])
        tau_c = np.repeat(tau[:, None, :], n_channel, axis=1)
        thermal = np.repeat((epsilon * planck / 2.0)[:, None, :], n_channel, axis=1)
        into = np.zeros((n_freq, 3, n_channel, n_depth))
        out = np.zeros((n_freq, 3, n_channel, n_depth))
        into[:, 0] = np.repeat((1.0 - epsilon)[:, None, :], n_channel, axis=1) / 2.0
        out[:, 0] = w_c[None, :, None] * np.ones((n_freq, n_channel, n_depth))
        solved = polarized_feautrier(tau_c, mu_c, w_c, thermal, into, out)
        self.assertLess(float(np.max(np.abs(solved["J"] / reference["J"] - 1.0))),
                        1.0e-9)
        scale = float(np.abs(reference["H_mid"]).max())
        self.assertLess(float(np.max(np.abs(solved["H_mid"] - reference["H_mid"]))
                              / scale), 1.0e-9)

    def test_a_atmosfera_magnetizada_conserva_o_fluxo(self) -> None:
        """lg B = 12: a estrutura converge e emite sigma T_ef^4, não outra coisa.

        É a armadilha de redefinir T_ef, agora com dois modos para escondê-la."""
        from atmosfera import estrutura
        solution = mg.solve(6.5, 14.0, 1.0e12, iterations=120, mu_nodes=4,
                            energies=estrutura.energy_grid(1.0e-3, 60.0, 90),
                            columns=estrutura.column_grid(1.0e-6, 1.0e5, 61))
        total = float(np.trapezoid(solution["flux_energy"], solution["energies"]))
        expected = estrutura.STEFAN * (10.0 ** 6.5) ** 4
        self.assertLess(abs(total / expected - 1.0), 2.0e-2,
                        f"int F dE / sigma T^4 = {total / expected:.4f}")
        self.assertLess(solution["flux_error"], 5.0e-2)


class TestAtmosferaFina(unittest.TestCase):
    """P3: a atmosfera fina sobre superfície emissora, nos dois limites exatos.

    A condição do fundo é a eq. (15) de Suleimanov, Pavlov & Werner (2009):
    I+(fundo) = B/2 por modo. Sigma pequeno tem de degenerar no corpo negro
    (medido: 0,6% com fluxo a 1e-7) e Sigma grande no semi-infinito (0,6%).
    Entre os dois mora a classe de modelo que já venceu na RBS 1223
    (Hambaryan et al. 2011), agora com feixe resolvido.
    """

    def test_coluna_minuscula_devolve_o_corpo_negro(self) -> None:
        from atmosfera import estrutura
        energies = estrutura.energy_grid(1.0e-3, 60.0, 70)
        solution = mg.solve(6.1, 14.4, 1.0e13, iterations=120, mu_nodes=4,
                            energies=energies, surface_column=1.0e-4)
        blackbody = np.pi * estrutura.planck_energy(energies, 10.0 ** 6.1)
        band = (energies >= 0.15) & (energies <= 2.0)
        ratio = (solution["flux_energy"] / blackbody)[band]
        self.assertLess(float(np.max(np.abs(ratio - 1.0))), 3.0e-2,
                        f"razão {ratio.min():.4f} a {ratio.max():.4f}")

    def test_coluna_intermediaria_converge(self) -> None:
        # O regime que derrubou cinco consertos errados (amortecimento x2,
        # escala de Newton, refino de grade x2): Sigma ~ 100 só converge com a
        # injeção do fundo escravizada ao gás (SPW09, eq. 15) e a âncora de
        # Newton em T[-1]. Este teste prende a cura: fluxo constante em
        # profundidade e total no alvo.
        from atmosfera import estrutura
        energies = estrutura.energy_grid(1.0e-3, 60.0, 70)
        solution = mg.solve(6.1, 14.4, 1.0e13, iterations=250, mu_nodes=4,
                            energies=energies, surface_column=100.0)
        total = float(np.trapezoid(solution["flux_energy"], energies)
                      / (estrutura.STEFAN * (10.0 ** 6.1) ** 4))
        self.assertLess(solution["flux_error"], 2.0e-2)
        self.assertLess(abs(total - 1.0), 2.0e-2, f"total {total:.4f}")

    def test_coluna_grande_devolve_o_semi_infinito(self) -> None:
        from atmosfera import estrutura
        energies = estrutura.energy_grid(1.0e-3, 60.0, 70)
        shared = dict(iterations=150, mu_nodes=4, energies=energies)
        thick = mg.solve(6.1, 14.4, 1.0e13, surface_column=1.0e5, **shared)
        infinite = mg.solve(6.1, 14.4, 1.0e13, **shared)
        band = (energies >= 0.15) & (energies <= 2.0)
        ratio = (thick["flux_energy"] / infinite["flux_energy"])[band]
        self.assertLess(float(np.max(np.abs(ratio - 1.0))), 3.0e-2,
                        f"razão {ratio.min():.4f} a {ratio.max():.4f}")
