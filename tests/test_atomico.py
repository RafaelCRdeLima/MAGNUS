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


if __name__ == "__main__":
    unittest.main()
