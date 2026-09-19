"""A interpolação em energia reescalada pelo cíclotron.

A feição do próton fica em E_c = 0,63 (B/1e14) keV: ela ANDA com o campo. Entre
dois nós de 0,05 dex em lg B ela se desloca 12%, mais que a própria largura, e
interpolar os dois nós na MESMA energia soma duas linhas em lugares diferentes,
produzindo uma rasa e larga em vez de uma deslocada.

Medido em 18/09/2026 contra o modelo calculado no ponto médio entre nós
(lg B = 13,875, lg T = 6,114, lg g = 14,4):

    linear em E      erro mediano na linha 66,5 %, pico 125 % (em 0,98 E_c)
    linear em E/E_c  erro mediano na linha 19,7 %, pico  47 %

e a verossimilhança do melhor ajuste de então caiu 70 no MESMO ponto de
parâmetros quando a interpolação foi corrigida — aquele ponto era, em parte,
artefato da linha borrada.

Este arquivo guarda o ALGORITMO. O elo com o motor em C++, que o implementa em
`AtmosphereTable::log_ratio`, é coberto por `tests/test_tabela_intensidade.py`.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CICLOTRON_KEV_POR_GAUSS = 1.1577e-11 / 1836.15267   # E_c do próton, keV/G
#: (E_ce = 11,577 keV em B = 1e12 G, logo 1,1577e-11 keV/G; o próton divide por m_p/m_e)


class InterpolacaoCiclotron(unittest.TestCase):
    """A feição tem de sobreviver à interpolação entre nós de campo."""

    @classmethod
    def setUpClass(cls) -> None:
        tabela = ROOT / "tabelas" / "magnus_campoBg_denso_v3.magnus"
        if not tabela.is_file():
            raise unittest.SkipTest("tabela v3 ausente")
        from mesclar_tabelas import read_table
        cls.eixos, cls.w = read_table(str(tabela))
        cls.lg_b, cls.lg_t, cls.lg_g, cls.th, cls.mu, cls.lg_e = cls.eixos
        # um par de nós vizinhos perto do campo da fonte
        cls.i = int(np.argmin(np.abs(cls.lg_b - 13.85)))
        cls.it = int(np.argmin(np.abs(cls.lg_t - 6.114)))
        cls.ig = int(np.argmin(np.abs(cls.lg_g - 14.4)))

    def _bloco(self, ib: int) -> np.ndarray:
        """lg w médio em mu, no nó de campo `ib`."""
        return self.w[ib, self.it, self.ig, 0].mean(axis=0)

    def _profundidade(self, energia: np.ndarray, lg_w: np.ndarray,
                      centro: float) -> float:
        """1 − I(centro)/I(contínuo), com o contínuo ajustado fora da feição."""
        janela = (energia > 0.7 * centro) & (energia < 1.5 * centro)
        fora = janela & ((energia < 0.85 * centro) | (energia > 1.25 * centro))
        if fora.sum() < 3:
            return float("nan")
        coef = np.polyfit(np.log(energia[fora]), lg_w[fora], 1)
        fundo = np.polyval(coef, np.log(centro))
        no_centro = float(np.interp(centro, energia, lg_w))
        return float(1.0 - 10.0 ** (no_centro - fundo))

    def test_o_reescalamento_preserva_a_profundidade_da_feicao(self) -> None:
        """No ponto médio entre nós, a feição tem de continuar tão funda quanto neles.

        Sem reescalamento a soma de duas linhas deslocadas de 12% preenche o
        fundo; com reescalamento cada nó é lido no seu próprio E/E_c e a feição
        não é borrada.
        """
        energia = 10.0 ** self.lg_e
        b0, b1 = float(self.lg_b[self.i]), float(self.lg_b[self.i + 1])
        meio = 0.5 * (b0 + b1)
        centro = lambda b: CICLOTRON_KEV_POR_GAUSS * 10.0 ** b
        w0, w1 = self._bloco(self.i), self._bloco(self.i + 1)

        p_nos = 0.5 * (self._profundidade(energia, w0, centro(b0))
                       + self._profundidade(energia, w1, centro(b1)))
        ingenua = 0.5 * (w0 + w1)
        p_ingenua = self._profundidade(energia, ingenua, centro(meio))
        reescalada = 0.5 * (
            np.interp(np.log10(energia * 10.0 ** (b0 - meio)), self.lg_e, w0)
            + np.interp(np.log10(energia * 10.0 ** (b1 - meio)), self.lg_e, w1))
        p_reescalada = self._profundidade(energia, reescalada, centro(meio))

        self.assertFalse(np.isnan(p_nos), "feição não encontrada nos nós")
        # a ingênua PERDE profundidade; a reescalada fica perto da dos nós
        self.assertLess(p_ingenua, p_nos,
                        f"ingênua {p_ingenua:.3f} não é mais rasa que os nós {p_nos:.3f}")
        self.assertGreater(p_reescalada, p_ingenua,
                           f"reescalada {p_reescalada:.3f} não melhora sobre {p_ingenua:.3f}")
        self.assertLess(abs(p_reescalada - p_nos), abs(p_ingenua - p_nos),
                        f"reescalada {p_reescalada:.3f} não fica mais perto de {p_nos:.3f} "
                        f"que a ingênua {p_ingenua:.3f}")

    def test_em_cima_de_um_no_o_reescalamento_e_identidade(self) -> None:
        """Onde não há o que corrigir, a mudança não pode alterar resultado."""
        energia = 10.0 ** self.lg_e
        b = float(self.lg_b[self.i])
        w = self._bloco(self.i)
        lido = np.interp(np.log10(energia * 10.0 ** (b - b)), self.lg_e, w)
        # não é igualdade de bits: `energia` é 10**lg_e e voltar por log10
        # arredonda na última casa, o que a interpolação amplifica um pouco onde
        # lg w varia rápido. O que o portão exige é que NADA físico mude.
        self.assertTrue(np.allclose(lido, w, rtol=1.0e-9, atol=1.0e-9),
                        f"desvio máximo {np.max(np.abs(lido - w)):.2e}")
