"""P1 — o átomo de hidrogênio magnetizado, em movimento: estrutura atômica.

A fundação do estágio 3 (ionização parcial). Tudo — equilíbrio de ionização,
limiar ligado-livre, e o "alargamento magnético" das linhas — se apoia nas
energias de ligação do átomo que se move através do campo. Este módulo começa
por elas.

FONTE, e a disciplina anti-typo:
  Potekhin 1998, J.Phys.B 31, 49 (preprint arXiv:physics/9710046 **v3**).
  A v3 corrige um misprint da Eq.(21) (força de oscilador) presente na v1-v2 —
  exatamente o risco de que o usuário advertiu. Usamos a v3, e cada fórmula
  transcrita é conferida contra um valor INDEPENDENTE, não contra outra
  transcrição:
    - o limite analítico γ→0, que devolve 1 Ryd (o H sem campo), sem apelar a
      memória nem a tabela;
    - o cruzamento entre a Eq.(10) (ajuste contínuo em γ) e a Table 1 do mesmo
      paper (parametrização independente do estado ligado), que tabela E^(0)
      em γ discretos. Se as duas concordam, a transcrição está limpa.

Unidades do paper: energias em Rydberg; γ = B / (2,35×10⁹ G) é o campo em
unidades atômicas (γ = 1 quando ħω_c = 2 Ryd). O estado é rotulado (n, s, ν);
aqui tratamos a coluna n=0 (o multipleto de menor energia), com s o número
quântico de Landau do buraco e ν a excitação longitudinal.
"""

from __future__ import annotations

import numpy as np

#: γ = B / B_gamma, com B_gamma = 2,35×10⁹ G (campo atômico: ħω_c = 2 Ryd).
#: O valor é m_e² e³ c / ħ³ = 2,3505×10⁹ G; Potekhin arredonda para 2,35e9.
FIELD_GAMMA_G = 2.3505e9

#: Rydberg em keV, para converter a saída (que sai em Ryd) ao resto do MAGNUS.
RYDBERG_KEV = 13.605693122994e-3


def field_to_gamma(field_g: np.ndarray | float) -> np.ndarray:
    """γ (adimensional) a partir de B em gauss."""
    return np.asarray(field_g, dtype=float) / FIELD_GAMMA_G


# --- Eq. (10) de Potekhin 1998: E_0s0^(0)(γ), o estado ligado EM REPOUSO ------
#
#   E_0s0^(0)(γ)/Ryd = ln{ exp[(1+s)^-2] + p1 [ln(1+p2 √γ)]^2 }
#                      + p3 [ln(1+p4 γ^p5)]^2
#
# Válida para 0,1 ≤ γ ≤ 10^4 (a RBS 1223, lgB=13,5, dá γ≈1,3×10^4 — na borda
# superior; lgB=13,0 dá γ≈4,3×10^3, folgado dentro). Coeficientes: Table 3,
# s = 0..7.  Conferidos por cruzamento com a Table 1 no teste.
_TABLE3_P = np.array([
    # p1,       p2,     p3,     p4,      p5
    [15.55,    0.378,  2.727,  0.3034,  0.4380],   # s=0
    [0.5332,   2.100,  3.277,  0.3092,  0.3784],   # s=1
    [0.1707,   4.150,  3.838,  0.2945,  0.3472],   # s=2
    [0.07924,  6.110,  4.906,  0.2748,  0.3157],   # s=3
    [0.04696,  7.640,  5.787,  0.2579,  0.2977],   # s=4
    [0.03075,  8.642,  6.669,  0.2431,  0.2843],   # s=5
    [0.02142,  9.286,  7.421,  0.2312,  0.2750],   # s=6
    [0.01589,  9.376,  8.087,  0.2209,  0.2682],   # s=7
])


def ground_binding_at_rest(field_g: np.ndarray | float, s: int = 0) -> np.ndarray:
    """|E_0s0^(0)| em keV para o átomo EM REPOUSO (K=0), estado (0, s, 0).

    É a energia de ligação do multipleto de menor energia. Para s=0 é o estado
    fundamental; s>0 são os estados de Landau do buraco, ligeiramente menos
    ligados. Devolve keV para casar com o resto do MAGNUS; internamente é Ryd.
    """
    if not 0 <= s <= 7:
        raise ValueError("Table 3 cobre s = 0..7")
    gamma = field_to_gamma(field_g)
    p1, p2, p3, p4, p5 = _TABLE3_P[s]
    term1 = np.log(np.exp((1.0 + s) ** -2)
                   + p1 * np.log1p(p2 * np.sqrt(gamma)) ** 2)
    term2 = p3 * np.log1p(p4 * gamma ** p5) ** 2
    return (term1 + term2) * RYDBERG_KEV
