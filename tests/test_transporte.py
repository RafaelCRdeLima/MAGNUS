"""O portão analítico — estágio 0,5 do plano de desenvolvimento.

A atmosfera cinza tem solução exata, e é ela que mede a máquina numérica
inteira antes de qualquer opacidade existir. Se um erro de Feautrier passar
daqui, ele reaparece no estágio 1 como "5% que não fecham" e some dentro de uma
discussão sobre fator de Gaunt.

Os números contra os quais se mede não são nossos: q(0) = 1/sqrt(3) é exato,
q(infinito) = 0,7104 é da literatura clássica, e a lei do raiz de epsilon é
resultado analítico. Nenhum deles depende de nada que este repositório escreveu.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atmosfera.transporte import (                                  # noqa: E402
    coherent_scattering, comptonized_feautrier, coupled_feautrier,
    gauss_legendre_mu, grey_milne, optical_depth_grid)

#: Valores exatos da função de Hopf. O da superfície é analítico; o do fundo é
#: o clássico, e o portão de 0,5% do plano tem folga de sobra para a última casa.
HOPF_SURFACE = 1.0 / math.sqrt(3.0)
HOPF_DEEP = 0.710446


class TestMilneCinza(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tau = optical_depth_grid(1.0e3, 401)
        cls.mu, cls.weights = gauss_legendre_mu(16)
        cls.solution = grey_milne(cls.tau, cls.mu, cls.weights, flux=1.0)

    def test_hopf_na_superficie(self) -> None:
        measured = float(self.solution["hopf"][0])
        error = abs(measured - HOPF_SURFACE) / HOPF_SURFACE
        self.assertLess(error, 5.0e-3, f"q(0) = {measured:.6f}, erro {error:.2e}")

    def test_hopf_no_fundo(self) -> None:
        measured = float(self.solution["hopf"][-1])
        error = abs(measured - HOPF_DEEP) / HOPF_DEEP
        self.assertLess(error, 5.0e-3, f"q(inf) = {measured:.6f}, erro {error:.2e}")

    def test_a_funcao_de_hopf_cresce_de_uma_ponta_a_outra(self) -> None:
        """Monotônica entre os dois valores exatos — a excursão é o resultado.

        Em Eddington q vale 2/3 em toda parte; se este teste passar a ver uma
        constante, a solução voltou a ser a aproximação e não a exata.
        """
        hopf = self.solution["hopf"]
        self.assertGreater(np.min(np.diff(hopf)), -1.0e-9)
        self.assertLess(hopf[0], 2.0 / 3.0)
        self.assertGreater(hopf[-1], 2.0 / 3.0)

    def test_fluxo_constante_em_profundidade(self) -> None:
        """O auto-teste padrão: H da solução FORMAL, não da equação de momentos.

        O portão do plano pede 0,1%. Trava-se aqui em 1e-8, que é a ordem em que
        a medida está: uma degradação para 1e-5 seria defeito, ainda que o
        portão do plano continuasse passando.
        """
        flux = self.solution["H_mid"]
        spread = float((flux.max() - flux.min()) / np.median(flux))
        offset = abs(float(np.median(flux)) - 1.0)
        self.assertLess(spread, 1.0e-8, f"H varia {spread:.2e} em profundidade")
        self.assertLess(offset, 1.0e-8, f"H vale {np.median(flux):.10f} e não 1")

    def test_eddington_no_fundo_e_no_topo(self) -> None:
        """K/J vai a 1/3 no fundo; na superfície vale 0,41 e não 1/3."""
        factor = self.solution["eddington_factor"]
        self.assertAlmostEqual(float(factor[-1]), 1.0 / 3.0, places=4)
        self.assertGreater(float(factor[0]), 0.40)
        self.assertLess(float(factor[0]), 0.42)

    def test_escurecimento_de_bordo(self) -> None:
        """I(0,0)/I(0,1) é 0,35 e não os 0,40 de Eddington.

        A diferença entre 0,35 e 0,40 é pequena e é justamente o tipo de coisa
        que um esquema errado acerta por acaso; por isso o teste exige o valor
        exato E rejeita o aproximado, em vez de aceitar qualquer um dos dois.
        """
        intensity = self.solution["I_surface"]
        self.assertGreater(np.min(np.diff(intensity)), 0.0)      # monotônica em mu
        grazing = float(np.polyval(np.polyfit(self.mu[:4], intensity[:4], 2), 0.0))
        ratio = grazing / float(intensity[-1])
        self.assertGreater(ratio, 0.33, f"I(0,0)/I(0,1) = {ratio:.4f}")
        self.assertLess(ratio, 0.36, f"I(0,0)/I(0,1) = {ratio:.4f}, perto do Eddington 0,40")

    def test_a_quadratura_angular_ja_convergiu(self) -> None:
        """De 8 a 16 nós o resultado não se move: o erro que sobra é da grade."""
        coarse = grey_milne(self.tau, *gauss_legendre_mu(8))
        self.assertAlmostEqual(float(coarse["hopf"][0]),
                               float(self.solution["hopf"][0]), places=5)

    def test_a_grade_em_profundidade_converge(self) -> None:
        """Refinar a grade tem de aproximar de q(0) exato, e aproxima."""
        errors = []
        for points in (101, 201, 401):
            solution = grey_milne(optical_depth_grid(1.0e3, points), self.mu, self.weights)
            errors.append(abs(float(solution["hopf"][0]) - HOPF_SURFACE))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])


class TestEspalhamentoCoerente(unittest.TestCase):
    """A lei do raiz de epsilon, e o que ela mostra sobre a iteração lambda."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mu, cls.weights = gauss_legendre_mu(8)

    def grid(self, epsilon: float) -> np.ndarray:
        # Fundo bem além da profundidade de termalização, 1/sqrt(eps).
        return optical_depth_grid(100.0 / math.sqrt(epsilon), 241, tau_min=1.0e-4)

    def test_lei_da_raiz_de_epsilon(self) -> None:
        for epsilon in (1.0e-2, 1.0e-4, 1.0e-6):
            with self.subTest(epsilon=epsilon):
                tau = self.grid(epsilon)
                surface = coherent_scattering(tau, self.mu, self.weights, epsilon,
                                              method="exact")["S"][0]
                ratio = float(surface) / math.sqrt(epsilon)
                self.assertAlmostEqual(ratio, 1.0, delta=1.0e-3,
                                       msg=f"S(0)/sqrt(eps) = {ratio:.6f}")

    def test_ali_converge_e_a_lambda_pura_nao(self) -> None:
        """O ALI chega na resposta; a lambda pura para na termalização.

        Não é questão de paciência: a lambda pura tem razão de convergência
        (1 - eps), e em eps = 1e-6 isso são centenas de milhares de iterações. O
        ALI troca essa razão por algo como (1 - sqrt(eps)). É a armadilha número
        dois do plano, medida.
        """
        epsilon = 1.0e-6
        tau = self.grid(epsilon)
        exact = float(coherent_scattering(tau, self.mu, self.weights, epsilon,
                                          method="exact")["S"][0])
        accelerated = coherent_scattering(tau, self.mu, self.weights, epsilon,
                                          method="ali", iterations=20000, ng_depth=3)
        plain = coherent_scattering(tau, self.mu, self.weights, epsilon,
                                    method="lambda", iterations=5000)
        self.assertAlmostEqual(float(accelerated["S"][0]) / exact, 1.0, delta=5.0e-3)
        self.assertLess(float(plain["S"][0]) / exact, 0.5,
                        "a lambda pura convergiu, e não devia — conferir o teste")

    def test_ng_encurta_a_iteracao(self) -> None:
        """A aceleração de Ng sobre o ALI, medida e não suposta."""
        epsilon = 1.0e-4
        tau = self.grid(epsilon)
        without = coherent_scattering(tau, self.mu, self.weights, epsilon,
                                      method="ali", iterations=20000)
        with_ng = coherent_scattering(tau, self.mu, self.weights, epsilon,
                                      method="ali", iterations=20000, ng_depth=3)
        self.assertTrue(without["converged"] and with_ng["converged"])
        self.assertLess(with_ng["iterations"], without["iterations"] / 2)
        self.assertAlmostEqual(float(with_ng["S"][0]) / float(without["S"][0]),
                               1.0, delta=1.0e-6)


if __name__ == "__main__":
    unittest.main()


class TestFeautrierEmBloco(unittest.TestCase):
    """O espalhamento resolvido de uma vez, contra o mesmo problema iterado.

    `coupled_feautrier` monta o sistema em bloco sobre os ângulos e resolve
    direto; `coherent_scattering(method="exact")` monta o operador Lambda e
    resolve o sistema em S. São dois caminhos independentes para a mesma
    equação — discretização igual, álgebra diferente —, e por isso a comparação
    entre eles pega erro de montagem de bloco, que é o tipo de defeito que não
    levanta exceção nenhuma e só devolve o número errado.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.mu, cls.weights = gauss_legendre_mu(8)

    def test_bate_com_o_operador_lambda(self) -> None:
        for epsilon in (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8):
            with self.subTest(epsilon=epsilon):
                tau = optical_depth_grid(100.0 / math.sqrt(epsilon), 161, tau_min=1.0e-4)
                reference = coherent_scattering(tau, self.mu, self.weights, epsilon,
                                                method="exact")["S"]
                block = coupled_feautrier(tau[None, :], self.mu, self.weights,
                                          np.full((1, tau.size), epsilon),
                                          np.ones((1, tau.size)))["S"][0]
                difference = float(np.max(np.abs(block / reference - 1.0)))
                # 1e-7 e não a precisão de máquina: em eps = 1e-8 a matriz
                # I - (1-eps) Lambda fica quase singular, e o caminho pelo
                # operador Lambda paga isso em condicionamento. A diferença
                # medida, 1,6e-8, é aritmética de ponto flutuante e não física —
                # nas outras três décadas ela fica abaixo de 1e-12.
                self.assertLess(difference, 1.0e-7, f"diferença máxima {difference:.2e}")

    def test_o_lote_em_frequencia_nao_mistura_nada(self) -> None:
        """Com espalhamento coerente as frequências são independentes.

        Resolver dez de uma vez tem de dar o mesmo que resolver dez vezes uma —
        e um erro de eixo no lote apareceria aqui, e em lugar nenhum antes.
        """
        tau = optical_depth_grid(1.0e4, 121, tau_min=1.0e-4)
        epsilon = np.geomspace(1.0e-6, 0.5, 10)
        stacked = coupled_feautrier(
            np.tile(tau, (10, 1)), self.mu, self.weights,
            np.repeat(epsilon[:, None], tau.size, axis=1),
            np.repeat(np.linspace(1.0, 3.0, 10)[:, None], tau.size, axis=1))
        for index, value in enumerate(epsilon):
            alone = coupled_feautrier(
                tau[None, :], self.mu, self.weights,
                np.full((1, tau.size), value),
                np.full((1, tau.size), np.linspace(1.0, 3.0, 10)[index]))
            difference = float(np.max(np.abs(stacked["S"][index] / alone["S"][0] - 1.0)))
            self.assertLess(difference, 1.0e-12, f"frequência {index}: {difference:.2e}")


class TestOperadorDeCompton(unittest.TestCase):
    """O operador de Kompaneets, isolado do esquema que ainda não o acopla.

    O acoplamento com o transporte não converge e está desligado por padrão; o
    operador em si está certo, e estes testes existem para que continue certo
    quando alguém voltar para escrever a linearização completa.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from atmosfera import estrutura
        cls.es = estrutura
        cls.energies = estrutura.energy_grid()
        cls.temperature = np.array([5.0e5, 1.0e6, 5.0e6])

    def test_numero_de_ocupacao(self) -> None:
        """n = OCCUPATION J/E^3 devolve 1/(e^x - 1) para um Planck."""
        planck = self.es.planck_energy(self.energies[:, None], self.temperature[None, :])
        occupation = self.es.OCCUPATION * planck / (self.energies ** 3)[:, None]
        x = (self.es.ERG_PER_KEV * self.energies[:, None]
             / (self.es.BOLTZMANN * self.temperature[None, :]))
        window = (x > 1.0e-3) & (x < 30.0)
        error = float(np.max(np.abs(occupation[window] * np.expm1(x[window]) - 1.0)))
        self.assertLess(error, 1.0e-12, f"erro {error:.2e}")

    def test_anula_no_equilibrio(self) -> None:
        """Um Planck na própria T_e não é mexido pelo operador.

        É o teste que o termo n^2 existe para passar: sem o espalhamento
        induzido o operador NÃO zera aqui, e a fonte espúria que sobra domina
        onde o número de ocupação é grande.
        """
        planck = self.es.planck_energy(self.energies[:, None], self.temperature[None, :])
        correction = self.es.compton_correction(self.energies, planck, self.temperature)
        x = (self.es.ERG_PER_KEV * self.energies[:, None]
             / (self.es.BOLTZMANN * self.temperature[None, :]))
        window = (x > 1.0e-3) & (x < 30.0)
        residual = float(np.max(np.abs((correction / planck)[window])))
        # 3e-4 na grade padrão, de 40 pontos por década; o resíduo é de
        # discretização e cai com o refinamento. Sem o termo n^2 ele seria de
        # ordem 1, então o teste continua sendo sobre a física e não sobre a
        # grade.
        self.assertLess(residual, 3.0e-4, f"resíduo {residual:.2e}")

    def test_conserva_numero_de_fotons(self) -> None:
        """O operador é uma divergência: espalhar não cria nem destrói fóton."""
        field = (self.es.planck_energy(self.energies[:, None], self.temperature[None, :])
                 * np.linspace(0.5, 2.0, self.energies.size)[:, None])
        correction = self.es.compton_correction(self.energies, field, self.temperature)
        for column in range(self.temperature.size):
            created = np.trapezoid(correction[:, column] / self.energies, self.energies)
            present = np.trapezoid(field[:, column] / self.energies, self.energies)
            # O operador conserva número exatamente na soma de volume finito;
            # o que sobra aqui é erro do trapézio com que o teste INTEGRA, não
            # do operador. Medido, 1e-9 na grade padrão.
            self.assertLess(abs(created / present), 1.0e-8)

    def test_o_sinal_da_troca_de_energia(self) -> None:
        """Campo mais duro que o gás aquece o gás, e não o contrário."""
        gas = np.array([1.0e6])
        hard = 0.1 * self.es.planck_energy(self.energies[:, None], np.array([5.0e6])[None, :])
        soft = 10.0 * self.es.planck_energy(self.energies[:, None], np.array([2.0e5])[None, :])
        to_radiation = lambda field: float(np.trapezoid(                      # noqa: E731
            self.es.compton_correction(self.energies, field, gas)[:, 0], self.energies))
        self.assertLess(to_radiation(hard), 0.0)      # radiação perde, gás ganha
        self.assertGreater(to_radiation(soft), 0.0)   # radiação ganha, gás esfria

    def test_a_diagonal_e_um_sumidouro(self) -> None:
        """d(dC)/dJ na própria frequência é negativa: é a rigidez do operador."""
        field = self.es.planck_energy(self.energies[:, None], self.temperature[None, :])
        diagonal = self.es.compton_diagonal(self.energies, field, self.temperature)
        self.assertLessEqual(float(np.max(diagonal)), 1.0e-6)
        self.assertLess(float(np.median(diagonal)), 0.0)


class TestLinearizacaoConjunta(unittest.TestCase):
    """O transporte comptonizado, resolvido de uma vez.

    As duas divisões mais baratas — fonte explícita, e diagonal no albedo —
    divergem neste mesmo problema; está medido nas notas de desenvolvimento. Estes
    testes prendem o que a linearização conjunta tem de entregar: convergência
    de Picard rápida, positividade, consistência interna, e o limite certo
    quando o acoplamento é desligado.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from atmosfera import estrutura
        cls.es = estrutura
        cls.mu, cls.weights = gauss_legendre_mu(8)
        cls.energies = np.logspace(-3.0, np.log10(60.0), 90)
        depth = optical_depth_grid(1.0e4, 61, tau_min=1.0e-4)
        cls.temperature = 1.0e6 * (0.75 * (depth + 0.71)) ** 0.25
        cls.temperature[0] = cls.temperature[1]
        # tau por frequência ~ E^-3 + espalhamento, como numa atmosfera real.
        shape = 1.0 + 50.0 / (cls.energies ** 3 + 0.01)
        cls.tau = depth[None, :] * shape[:, None] / shape.min()
        cls.epsilon = np.clip(1.0 - 1.0 / shape, 1.0e-6, 1.0)[:, None] * \
            np.ones_like(cls.tau)
        cls.planck = cls.es.planck_energy(cls.energies[:, None],
                                          cls.temperature[None, :])

    def test_sem_acoplamento_reproduz_o_coerente(self) -> None:
        """Com K = 0 a linearização conjunta É o espalhamento coerente."""
        zero = tuple(np.zeros_like(self.tau) for _ in range(3))
        joint = coupled = None
        joint = comptonized_feautrier(self.tau, self.mu, self.weights,
                                      self.epsilon, self.planck, zero)
        coupled = coupled_feautrier(self.tau, self.mu, self.weights,
                                    self.epsilon, self.planck)
        scale = np.abs(coupled["J"]).max()
        difference = float(np.max(np.abs(joint["J"] - coupled["J"])) / scale)
        self.assertLess(difference, 1.0e-10, f"diferença {difference:.2e}")

    def test_picard_converge_e_j_fica_positivo(self) -> None:
        """Três passadas de Picard, variação abaixo de 1e-6, nada negativo.

        É o comportamento medido que justificou a arquitetura: onde as divisões
        explícitas divergem geometricamente, a conjunta fecha de imediato.
        """
        field = np.maximum(coupled_feautrier(self.tau, self.mu, self.weights,
                                             self.epsilon, self.planck)["J"], 0.0)
        change = 1.0
        for _ in range(3):
            coupling = self.es.kompaneets_bands(self.energies, field, self.temperature)
            solved = comptonized_feautrier(self.tau, self.mu, self.weights,
                                           self.epsilon, self.planck, coupling)
            updated = np.maximum(solved["J"], 0.0)
            change = float(np.max(np.abs(updated - field)
                                  / np.maximum(updated, 1.0e-20 * updated.max())))
            field = updated
        self.assertLess(change, 1.0e-6, f"Picard ainda variava {change:.2e}")
        self.assertGreaterEqual(float(solved["J"].min()), 0.0)

    def test_consistencia_interna(self) -> None:
        """J do sistema em S e J da solução formal são o mesmo número.

        Se a montagem dos blocos divergir da solução formal por mais que
        arredondamento, há um índice trocado em algum lugar — e é assim que ele
        aparece, porque de outro jeito não aparece.
        """
        field = np.maximum(coupled_feautrier(self.tau, self.mu, self.weights,
                                             self.epsilon, self.planck)["J"], 0.0)
        coupling = self.es.kompaneets_bands(self.energies, field, self.temperature)
        solved = comptonized_feautrier(self.tau, self.mu, self.weights,
                                       self.epsilon, self.planck, coupling)
        scale = np.abs(solved["J"]).max()
        difference = float(np.max(np.abs(solved["J"] - solved["J_formal"])) / scale)
        self.assertLess(difference, 1.0e-10, f"diferença {difference:.2e}")

    def test_bandas_anulam_no_equilibrio_e_conservam_numero(self) -> None:
        """Chang-Cooper: zero sobre um Planck a T_e, e divergência exata."""
        planck = self.es.planck_energy(self.energies[:, None], np.array([1.0e6, 5.0e6])[None, :])
        coupling = self.es.kompaneets_bands(self.energies, planck, np.array([1.0e6, 5.0e6]))
        moved = self.es.apply_kompaneets(coupling, planck)
        x = (self.es.ERG_PER_KEV * self.energies[:, None]
             / (self.es.BOLTZMANN * np.array([1.0e6, 5.0e6])[None, :]))
        window = (x > 1.0e-3) & (x < 30.0)
        self.assertLess(float(np.max(np.abs(moved / planck)[window])), 1.0e-3)
        for column in range(2):
            created = np.trapezoid(moved[:, column] / self.energies, self.energies)
            present = np.trapezoid(planck[:, column] / self.energies, self.energies)
            self.assertLess(abs(created / present), 1.0e-8)
