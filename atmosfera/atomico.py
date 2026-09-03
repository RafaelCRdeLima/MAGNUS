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

from . import estrutura

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


# --- O átomo EM MOVIMENTO: E_0s0(K), Eqs. (6)-(8) de Potekhin 1998 -----------
#
# A física decisiva do estágio 3. Um átomo que se move através de B ganha um
# momento de dipolo, e a energia de ligação CAI com o pseudomomento transversal
# K. Como os átomos têm uma distribuição de K (térmica), a linha de absorção
# vira uma banda larga — o "alargamento magnético", ordens de grandeza acima do
# Doppler, e a razão de as feições saírem sem borda.
#
#   |E^||(K)| = E^(1)(K)/[1+(K/Kc)^(1/α)] + E^(2)(K)/[1+(Kc/K)^(1/α)]   (6)
#   E^(1)(K)  = E^(0) - K²/(2 m_eff + q1 K²/E^(0))                       (7)  centrado
#   E^(2)(K)  = 2[r*² + r*^{3/2} + q2 r*]^{-1/2} Ryd                     (8)  descentrado
#   r* = K/γ (u.a.),  Kc = q0 √(2 m_H E^(0)),  q1 = lg(γ/300) [s=0]
#
# Unidades (confirmadas no paper): E em Ryd, K em u.a. (ħ/a_B), massas em m_e,
# e o termo cinético K²/(2m) sai DIRETO em Ryd (unidades de Rydberg).
#
# Table 1, LINHA s=0: [lg(m_eff/m_H), q0, α, q2] nos γ tabelados. A coluna E^(0)
# foi conferida contra a Eq.10 (0,3%); estes parâmetros são interpolados em lg γ.
_MASS_H_ME = 1836.15267            # m_H / m_e (próton + elétron, u.a. de massa)
_TABLE1_S0_GAMMA = np.array([300., 600., 1000., 2000., 3000., 10000.])
_TABLE1_S0 = {
    "lg_meff": np.array([0.009, 0.042, 0.072, 0.141, 0.175, 0.319]),
    "q0":      np.array([0.859, 0.811, 0.823, 0.850, 0.873, 1.019]),
    # SUSPEITO: α=0,001 em γ=300 é outlier (vizinhos ~0,1) e faz a transição
    # da Eq.(6) virar um degrau — v_max artificial de 1223 km/s ali. Cheira a
    # typo de transcrição da Table 1. NÃO afeta nosso regime (lgB≥13 → γ≥4×10³,
    # onde α≈0,17-0,19, limpo); fica marcado para o portão B / correção futura
    # se alguém descer abaixo de γ~600.
    "alpha":   np.array([0.001, 0.107, 0.117, 0.178, 0.191, 0.173]),
    "q2":      np.array([0.102, 0.157, 0.189, 0.233, 0.244, 0.275]),
}


def _table1_s0(gamma: float) -> dict:
    """Parâmetros da Table 1 (s=0) interpolados em lg γ (extrapola nas bordas)."""
    lg = np.log10(gamma)
    grid = np.log10(_TABLE1_S0_GAMMA)
    return {k: float(np.interp(lg, grid, v)) for k, v in _TABLE1_S0.items()}


def moving_binding(field_g: float, pseudomomentum: np.ndarray,
                   s: int = 0) -> np.ndarray:
    """|E_0s0(K)| em keV para o átomo em MOVIMENTO, pseudomomento K em u.a.

    Só s=0 (o estado fundamental, que domina a opacidade) por ora. K é o
    pseudomomento transversal em unidades atômicas; devolve a energia de ligação
    em keV, caindo de E^(0) em K=0 para ~0 quando o átomo se descentra.
    """
    if s != 0:
        raise NotImplementedError("por ora só o estado fundamental s=0")
    gamma = float(field_to_gamma(field_g))
    K = np.asarray(pseudomomentum, dtype=float)
    e0 = float(ground_binding_at_rest(field_g, 0)) / RYDBERG_KEV     # Ryd
    par = _table1_s0(gamma)
    m_eff = _MASS_H_ME * 10.0 ** par["lg_meff"]                      # m_e
    q0, alpha, q2 = par["q0"], par["alpha"], par["q2"]
    q1 = np.log10(gamma / 300.0)                                     # s=0
    k_c = q0 * np.sqrt(2.0 * _MASS_H_ME * e0)                        # u.a.

    r_star = K / gamma
    e1 = e0 - K ** 2 / (2.0 * m_eff + q1 * K ** 2 / e0)             # centrado (7)
    with np.errstate(divide="ignore", invalid="ignore"):
        e2 = 2.0 * (r_star ** 2 + r_star ** 1.5 + q2 * r_star) ** -0.5  # (8)
        # Pesos da Eq.(6). O expoente 1/α pode ser enorme (α~1e-3): calcula em
        # log e satura, senão estoura. Em K=0 o termo descentrado é 0 (peso 0),
        # mesmo com e2 -> inf: guarda-se explicitamente.
        ratio = np.log(np.maximum(K, 1e-300) / k_c) / alpha
        w_centered = 1.0 / (1.0 + np.exp(np.clip(ratio, -700, 700)))
        e = np.where(K <= 0.0, e0,
                     e1 * w_centered + np.where(np.isfinite(e2), e2, 0.0)
                     * (1.0 - w_centered))
    return e * RYDBERG_KEV


# --- Equilíbrio de ionização: a fração neutra (Saha magnetizada, 1ª passada) --
#
# Quantos átomos sobrevivem a cada (T, ρ, B). Normaliza toda a opacidade
# atômica. FONTE: Potekhin, Chabrier & Shibanov 1999 (astro-ph/9907006),
#   Z_sν = (λ_H²/2πℏ²) ∫ w_sν(K) exp(βχ_sν(K)) K dK              (Eq.50)
#   n_H  = n_p n_e (λ_p λ_e (2π a_m²)² / λ_H³)[1−e^{−βℏω_cp}] Z_w e^Λ  (Eq.54)
#   χ_sν(K) = |E^∥_sν(K)| − s ℏω_cp                              (Eq.45)
#
# APROXIMAÇÕES DA 1ª PASSADA (declaradas, pendentes do portão B / refino):
#   · só o estado fundamental s=ν=0 no Z_w (domina; excitados vêm depois);
#   · gás ideal não-degenerado, elétron no nível de Landau fundamental → Λ≈0
#     (ℏω_ce ~ 11,6·B₁₃ keV ≫ kT, então só o nível zero conta);
#   · w≡1 e corte em K_c: a integral de Eq.50 DIVERGE no K grande (estados
#     descentrados), regularizada no tratamento completo pela probabilidade de
#     ocupação. Cortar nos estados CENTRADOS (K<K_c) é a 1ª passada; muda o
#     comportamento assintótico de baixa densidade (PCS99: f∝n^{1/3} vs n^{1/2}),
#     não a transição no regime da fotosfera.

_BOHR_CM = 5.29177210903e-9                    # a_B em cm
_HBAR = estrutura.PLANCK / (2.0 * np.pi)       # erg s
_MASS_E = estrutura.ELECTRON_REST / estrutura.LIGHT ** 2   # g
_MASS_H = estrutura.PROTON_MASS + _MASS_E      # g (próton + elétron)


def _thermal_wavelength(mass_g: float, temperature: float) -> float:
    """λ = h/√(2π m kT) em cm (comprimento de de Broglie térmico)."""
    return estrutura.PLANCK / np.sqrt(2.0 * np.pi * mass_g
                                      * estrutura.BOLTZMANN * temperature)


def _atomic_partition_ground(field_g: float, temperature: float) -> float:
    """Z_00 (Eq.50) do estado fundamental, integrando K<K_c em u.a.

    A conversão K_físico = K_ua·(ℏ/a_B) reduz o prefator a λ_H²/(2π a_B²), e a
    integral fica adimensional em u.a. — só a energia de ligação (em keV) entra
    no Boltzmann. χ = |E^∥| (s=0, o termo sℏω_cp é nulo).
    """
    gamma = float(field_to_gamma(field_g))
    kt_kev = estrutura.BOLTZMANN * temperature / estrutura.ERG_PER_KEV
    lambda_h = _thermal_wavelength(_MASS_H, temperature)
    prefactor = lambda_h ** 2 / (2.0 * np.pi * _BOHR_CM ** 2)   # adimensional

    e0 = float(ground_binding_at_rest(field_g, 0)) / RYDBERG_KEV
    k_c = _table1_s0(gamma)["q0"] * np.sqrt(2.0 * _MASS_H_ME * e0)   # u.a.
    K = np.linspace(0.0, k_c, 400)
    chi = moving_binding(field_g, K, 0)                          # keV, = |E^∥|
    integrand = np.exp(np.clip(chi / kt_kev, -700, 700)) * K
    return prefactor * float(np.trapezoid(integrand, K))


def neutral_fraction(field_g: float, temperature: float,
                     proton_density_cm3: float) -> float:
    """Fração neutra f_H = n_H/(n_H+n_p) do H puro em (T, n_total, B).

    `proton_density_cm3` é a densidade TOTAL de prótons (livres + ligados),
    n_0 = ρ/m_H. Resolve a Saha (Eq.54) com n_e = n_p (neutralidade) e
    n_p + n_H = n_0: n_H = C n_p², logo n_p = (−1+√(1+4 C n_0))/(2C).
    """
    kt_erg = estrutura.BOLTZMANN * temperature
    a_m2 = _BOHR_CM ** 2 / float(field_to_gamma(field_g))         # a_m² = a_B²/γ
    # ℏω_cp = ℏ²/(m_p a_m²), do próprio comprimento magnético (sem depender de
    # magnetizada.py). Confere: 0,063 keV em B=10¹³ G (cíclotron do próton).
    beta_hw_cp = _HBAR ** 2 / (estrutura.PROTON_MASS * a_m2) / kt_erg
    lam_e = _thermal_wavelength(_MASS_E, temperature)
    lam_p = _thermal_wavelength(estrutura.PROTON_MASS, temperature)
    lam_h = _thermal_wavelength(_MASS_H, temperature)
    z_w = _atomic_partition_ground(field_g, temperature)

    # C = n_H/(n_p n_e), em cm³ (Eq.54 com e^Λ≈1)
    c = (lam_p * lam_e * (2.0 * np.pi * a_m2) ** 2 / lam_h ** 3
         * -np.expm1(-beta_hw_cp) * z_w)
    n0 = proton_density_cm3
    n_p = (-1.0 + np.sqrt(1.0 + 4.0 * c * n0)) / (2.0 * c)
    n_h = n0 - n_p
    return float(n_h / n0)


# --- Alargamento magnético: o perfil da feição ligada ------------------------
#
# A assinatura do estágio 3. Um átomo com pseudomomento K tem energia de
# ligação ε(K) MENOR que a em repouso. Como os átomos têm distribuição térmica
# em K, a energia de qualquer feição presa ao fundamental (o LIMIAR de
# fotoionização; as linhas Lyman magnéticas) se espalha sobre a FAIXA de ε(K) —
# largura da ordem do próprio E^(0), 10³-10⁴× o Doppler, e sem borda nítida.
# PC03: "resembles a reversed bound-free profile", sem Lorentziana explícita.
#
# Distribuição térmica: p(K) dK ∝ exp(ε(K)/kT) K dK (o mesmo integrando de
# Z_w, Eq.50). O peso térmico é exp(−E_estado/kT) = exp(+ε/kT), ε>0 a ligação.

def thermal_pseudomomentum_pdf(field_g: float, temperature: float,
                               pseudomomentum: np.ndarray) -> np.ndarray:
    """p(K) normalizada em [0, K_c]: a distribuição dos átomos sobre K."""
    gamma = float(field_to_gamma(field_g))
    kt_kev = estrutura.BOLTZMANN * temperature / estrutura.ERG_PER_KEV
    e0 = float(ground_binding_at_rest(field_g, 0)) / RYDBERG_KEV
    k_c = _table1_s0(gamma)["q0"] * np.sqrt(2.0 * _MASS_H_ME * e0)
    K = np.asarray(pseudomomentum, dtype=float)
    chi = moving_binding(field_g, K, 0)
    weight = np.where(K <= k_c, np.exp(np.clip(chi / kt_kev, -700, 700)) * K, 0.0)
    grid = np.linspace(0.0, k_c, 2000)
    norm = np.trapezoid(np.exp(np.clip(moving_binding(field_g, grid, 0) / kt_kev,
                                       -700, 700)) * grid, grid)
    return weight / norm


def magnetic_broadening_profile(field_g: float, temperature: float,
                                photon_energy_kev: np.ndarray) -> np.ndarray:
    """g(E) [1/keV]: distribuição da energia de LIMIAR ε(K) sobre os átomos.

    O perfil do limiar de fotoionização magneticamente alargado. Como ε(K) é
    monótona, g(E) = p(K)/|dε/dK|. Vai de ε(K_c) até E^(0), largo e liso.
    """
    gamma = float(field_to_gamma(field_g))
    e0 = float(ground_binding_at_rest(field_g, 0)) / RYDBERG_KEV
    k_c = _table1_s0(gamma)["q0"] * np.sqrt(2.0 * _MASS_H_ME * e0)
    K = np.linspace(1.0e-3, k_c, 4000)
    eps = moving_binding(field_g, K, 0)                 # keV, decrescente em K
    pdf = thermal_pseudomomentum_pdf(field_g, temperature, K)
    jac = np.abs(np.gradient(eps, K))                   # |dε/dK|
    g_at_K = pdf / np.maximum(jac, 1.0e-300)            # g(ε(K))
    # reamostra em E crescente (ε decresce em K, então inverte)
    order = np.argsort(eps)
    return np.interp(np.asarray(photon_energy_kev, dtype=float),
                     eps[order], g_at_K[order], left=0.0, right=0.0)


def doppler_width_kev(field_g: float, temperature: float,
                      line_energy_kev: float) -> float:
    """Largura Doppler ΔE_D = E √(kT/m_H c²), para comparar com o magnético."""
    kt_erg = estrutura.BOLTZMANN * temperature
    return line_energy_kev * np.sqrt(kt_erg / (_MASS_H * estrutura.LIGHT ** 2))
