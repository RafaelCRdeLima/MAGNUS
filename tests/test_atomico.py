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


class TestEquilibrioDeIonizacao(unittest.TestCase):
    """P1: a fração neutra (Saha magnetizada, Eqs.50/54 de PCS99, 1ª passada).

    Os portões são os limites físicos e a LOCALIZAÇÃO da transição — não um
    valor absoluto, que depende do corte em K_c da 1ª passada.
    """

    def test_fica_entre_zero_e_um(self) -> None:
        for rho in (1.0e-3, 1.0, 1.0e3):
            f = at.neutral_fraction(1.0e13, 1.0e6, rho / at._MASS_H)
            self.assertGreaterEqual(f, 0.0)
            self.assertLessEqual(f, 1.0)

    def test_cresce_com_a_densidade(self) -> None:
        # Denso recombina, rarefeito ioniza — monotônico em ρ.
        rhos = np.array([1.0e-2, 1.0, 1.0e2, 1.0e4])
        f = [at.neutral_fraction(1.0e13, 1.0e6, r / at._MASS_H) for r in rhos]
        self.assertTrue(np.all(np.diff(f) > 0.0))
        self.assertLess(f[0], 0.01)     # rarefeito: quase todo ionizado
        self.assertGreater(f[-1], 0.5)  # denso: maioria neutra

    def test_cai_com_a_temperatura(self) -> None:
        # Mais quente ioniza mais, a densidade fixa.
        temps = 10.0 ** np.array([5.7, 6.0, 6.3, 6.6])
        f = [at.neutral_fraction(1.0e13, T, 1.0 / at._MASS_H) for T in temps]
        self.assertTrue(np.all(np.diff(f) < 0.0))

    def test_fotosfera_parcialmente_ionizada(self) -> None:
        # No ponto da fotosfera da RBS 1223 (ρ~1, T~1e6, lgB=13), a fração
        # neutra é de poucos % — pequena, mas não nula: o regime que imprime
        # as feições atômicas na janela mole.
        f = at.neutral_fraction(1.0e13, 1.0e6, 1.0 / at._MASS_H)
        self.assertGreater(f, 1.0e-3)
        self.assertLess(f, 0.2)


class TestAlargamentoMagnetico(unittest.TestCase):
    """P1: o perfil da feição ligada, a assinatura sem borda do estágio 3."""

    def test_distribuicao_termica_normalizada(self) -> None:
        gamma = float(at.field_to_gamma(1.0e13))
        e0 = float(at.ground_binding_at_rest(1.0e13, 0)) / at.RYDBERG_KEV
        k_c = at._table1_s0(gamma)["q0"] * np.sqrt(2.0 * at._MASS_H_ME * e0)
        K = np.linspace(0.0, k_c, 4000)
        pdf = at.thermal_pseudomomentum_pdf(1.0e13, 1.0e6, K)
        self.assertAlmostEqual(float(np.trapezoid(pdf, K)), 1.0, places=2)

    def test_largura_supera_o_doppler_por_ordens(self) -> None:
        # A marca do alargamento magnético: 10³-10⁴× o Doppler.
        e0 = float(at.ground_binding_at_rest(1.0e13, 0))
        E = np.linspace(0.001, e0 * 1.05, 800)
        g = at.magnetic_broadening_profile(1.0e13, 1.0e6, E)
        g = g / np.trapezoid(g, E)
        cum = np.cumsum(g) * (E[1] - E[0])
        largura = E[np.searchsorted(cum, 0.9)] - E[np.searchsorted(cum, 0.1)]
        doppler = at.doppler_width_kev(1.0e13, 1.0e6, e0)
        self.assertGreater(largura / doppler, 100.0)

    def test_feicao_cai_na_janela_mole(self) -> None:
        # O payoff físico: a feição alargada pousa em 0,15-0,3 keV, a janela
        # onde o ajuste da RBS 1223 perdia verossimilhança.
        e0 = float(at.ground_binding_at_rest(1.0e13, 0))
        E = np.linspace(0.001, e0 * 1.05, 800)
        g = at.magnetic_broadening_profile(1.0e13, 1.0e6, E)
        centro = float(np.trapezoid(E * g, E) / np.trapezoid(g, E))
        self.assertGreater(centro, 0.15)
        self.assertLess(centro, 0.30)
        # e o centro fica ABAIXO de E^(0) (átomos descentrados)
        self.assertLess(centro, e0)


class TestForcasDeOscilador(unittest.TestCase):
    """P1: Eq.21 v3 (a do misprint corrigido), gabaritada pela Lyman-α."""

    def test_limite_campo_nulo_e_lyman_alpha(self) -> None:
        # Portão independente: γ→0 devolve 0,4162 (1s→2p do H sem campo),
        # um número conhecido — valida a fórmula que teve o typo na v1-v2.
        for transition in ("001_par", "010_plus"):
            f0 = float(at.oscillator_strength_rest(1.0e-6, transition))
            self.assertAlmostEqual(f0, 0.4162, places=3)

    def test_longitudinal_domina_no_campo_forte(self) -> None:
        # No campo forte a σ+ vai ao cíclotron e some; a π longitudinal domina.
        for log_field in (13.0, 13.5):
            f_par = float(at.oscillator_strength_rest(10.0 ** log_field, "001_par"))
            f_plus = float(at.oscillator_strength_rest(10.0 ** log_field, "010_plus"))
            self.assertGreater(f_par, 10.0 * f_plus)

    def test_K_zero_recupera_f_em_repouso(self) -> None:
        for log_field in (13.0, 13.5):
            field = 10.0 ** log_field
            f_k0 = float(at.oscillator_strength_longitudinal(field, np.array([0.0]))[0])
            f_rest = float(at.oscillator_strength_rest(field, "001_par"))
            self.assertAlmostEqual(f_k0, f_rest, places=6)


class TestOpacidadeLigadoLivre(unittest.TestCase):
    """P1: κ_bf de 1ª passada — magnitude hidrogênica, limiar alargado."""

    def test_domina_thomson_na_janela(self) -> None:
        # Perto do limiar, mesmo poucos % de neutros tornam o ligado-livre
        # o absorvedor dominante — muito acima de Thomson.
        from atmosfera import estrutura
        n0 = 1.0 / at._MASS_H
        k = float(at.bound_free_opacity(1.0e13, 1.0e6, n0, np.array([0.28]))[0])
        self.assertGreater(k, 100.0 * estrutura.THOMSON_CM2_G)

    def test_pico_na_banda_mole(self) -> None:
        # O pico da opacidade atômica cai na banda mole (perto de E^(0)).
        E = np.linspace(0.05, 0.6, 300)
        k = at.bound_free_opacity(1.0e13, 1.0e6, 1.0 / at._MASS_H, E)
        pico = E[int(np.argmax(k))]
        self.assertGreater(pico, 0.15)
        self.assertLess(pico, 0.45)

    def test_cresce_com_a_densidade(self) -> None:
        # Mais fundo (mais denso) recombina mais → mais opaco.
        E = np.array([0.28])
        ks = [float(at.bound_free_opacity(1.0e13, 1.0e6, r / at._MASS_H, E)[0])
              for r in (0.1, 1.0, 10.0, 100.0)]
        self.assertTrue(np.all(np.diff(ks) > 0.0))

    def test_zera_abaixo_do_limiar_minimo(self) -> None:
        # Consequência declarada do corte em K_c: sem opacidade muito abaixo
        # da banda (borda espúria; os estados descentrados a preencheriam).
        k = float(at.bound_free_opacity(1.0e13, 1.0e6, 1.0 / at._MASS_H,
                                        np.array([0.08]))[0])
        self.assertEqual(k, 0.0)


if __name__ == "__main__":
    unittest.main()
