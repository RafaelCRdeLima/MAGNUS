"""P1: verificação anti-typo do átomo de H magnetizado (Potekhin 1998, v3).

Cada valor transcrito é conferido contra uma fonte INDEPENDENTE, não contra
outra transcrição — a disciplina que o usuário pediu depois de a própria v3 do
paper ter corrigido um misprint na Eq.(21).
"""

import unittest

import numpy as np

from atmosfera import atomico as at


class TestHidrogenioMagnetizado(unittest.TestCase):
    def test_limite_campo_nulo_e_um_rydberg(self) -> None:
        # Portão analítico: sem campo, o estado fundamental é 1 Ryd = 13,6 eV.
        # A Eq.(10) tem de devolvê-lo exatamente, sem tabela nem memória.
        e0 = at.ground_binding_at_rest(1.0e-3, s=0) / at.RYDBERG_KEV
        self.assertAlmostEqual(e0, 1.0, places=4)

    def test_eq10_cruza_com_table1(self) -> None:
        # A Eq.(10) (ajuste contínuo em γ) contra a Table 1 (parametrização
        # independente do mesmo paper) em γ discretos. Concordância = nenhuma
        # das duas transcrições carrega typo.
        casos = [(0, 300, 10.722), (0, 10000, 28.286),
                 (1, 300, 7.669), (1, 1000, 11.277)]
        for s, gamma, referencia in casos:
            field = gamma * at.FIELD_GAMMA_G
            valor = at.ground_binding_at_rest(field, s=s) / at.RYDBERG_KEV
            self.assertLess(abs(valor - referencia) / referencia, 1.0e-2,
                            f"s={s} γ={gamma}: {valor:.3f} vs {referencia}")

    def test_regime_de_ionizacao_parcial(self) -> None:
        # No campo da RBS 1223, a ligação tem de ser MUITO maior que kT~0,1 keV
        # — é o que faz sobrar átomo neutro (a razão de ser do estágio 3).
        for log_field in (13.0, 13.5):
            binding = at.ground_binding_at_rest(10.0 ** log_field, s=0)
            self.assertGreater(binding, 0.25)   # keV, >> kT
            self.assertLess(binding, 1.0)

    def test_ligacao_cresce_com_o_campo(self) -> None:
        # Monotonia: campo mais forte aperta o átomo.
        fields = 10.0 ** np.array([12.0, 12.5, 13.0, 13.5])
        binding = at.ground_binding_at_rest(fields, s=0)
        self.assertTrue(np.all(np.diff(binding) > 0.0))


class TestAtomoEmMovimento(unittest.TestCase):
    """P1: E_0s0(K), Eqs.(6)-(8) — a origem do alargamento magnético."""

    def test_K_zero_recupera_o_atomo_em_repouso(self) -> None:
        # Âncora principal: em K=0 a energia é exatamente E^(0)(γ).
        for log_field in (13.0, 13.5):
            field = 10.0 ** log_field
            movimento = at.moving_binding(field, 0.0)
            repouso = at.ground_binding_at_rest(field, 0)
            self.assertAlmostEqual(float(movimento), float(repouso), places=6)

    def test_figura1_gamma1000_K1000(self) -> None:
        # Âncora da Figura 1 do paper: em γ=1000, o fundamental cai de
        # E^(0)=15,3 Ryd (K=0) para ~1-2 Ryd em K=1000. Este ponto está no
        # ramo descentrado E^(2), sem ambiguidade de unidade.
        field = 1000.0 * at.FIELD_GAMMA_G
        e_k1000 = float(at.moving_binding(field, 1000.0)) / at.RYDBERG_KEV
        self.assertGreater(e_k1000, 1.0)
        self.assertLess(e_k1000, 2.0)

    def test_cai_monotonicamente_com_K(self) -> None:
        # A ligação só decresce com o pseudomomento (átomo se descentra).
        for log_field in (13.0, 13.5):
            field = 10.0 ** log_field
            K = np.linspace(0.0, 3000.0, 200)
            E = at.moving_binding(field, K)
            self.assertTrue(np.all(np.diff(E) <= 1.0e-12))
            # e cai a uma fração pequena de E^(0) no fim
            self.assertLess(float(E[-1]) / float(E[0]), 0.2)


if __name__ == "__main__":
    unittest.main()
