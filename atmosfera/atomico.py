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
    # q1 = lg(γ/300) é definido para γ≥300; abaixo, ficar ≥0 evita o flip de
    # sinal do denominador (F3 da auditoria). Nosso regime é γ≥4×10³.
    q1 = max(0.0, np.log10(gamma / 300.0))                           # s=0
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
# APROXIMAÇÕES DECLARADAS (pendentes do portão B):
#   · só o estado fundamental s=ν=0 no Z_w (domina; excitados vêm depois);
#   · gás ideal não-degenerado, elétron no nível de Landau fundamental → Λ≈0
#     (ℏω_ce ~ 11,6·B₁₃ keV ≫ kT, então só o nível zero conta).
#
# A PROBABILIDADE DE OCUPAÇÃO regulariza os estados descentrados (o que antes
# era um corte cru em K_c). A integral de Eq.50 divergiria no K grande porque
# ali χ→0 (átomo quase livre): um átomo com pseudomomento K tem seus centros de
# carga separados por r_c = (K/γ) a_B, e some quando r_c passa da distância
# média entre partículas d = (3/4π n)^{1/3} — pressão-ionização / superposição
# com o vizinho. Peso w(K) = exp[−(r_c/d)³] (Poisson: prob. de não haver
# perturbador dentro de r_c). Vai a 1 para o átomo centrado (r_c≪d) e a 0 para o
# descentrado, tornando a integral convergente E dando a dependência correta com
# a densidade (PCS99: f∝n^{1/3} no limite diluído). O perturbador é a densidade
# TOTAL n_0 (≈ n_p na fotosfera, onde f é de poucos %).

_BOHR_CM = 5.29177210903e-9                    # a_B em cm
_HBAR = estrutura.PLANCK / (2.0 * np.pi)       # erg s
_MASS_E = estrutura.ELECTRON_REST / estrutura.LIGHT ** 2   # g
_MASS_H = estrutura.PROTON_MASS + _MASS_E      # g (próton + elétron)


def _thermal_wavelength(mass_g: float, temperature: float) -> float:
    """λ = h/√(2π m kT) em cm (comprimento de de Broglie térmico)."""
    return estrutura.PLANCK / np.sqrt(2.0 * np.pi * mass_g
                                      * estrutura.BOLTZMANN * temperature)


def _interparticle_distance(density_cm3: float) -> float:
    """d = (3/4π n)^{1/3} em cm — distância média entre perturbadores."""
    return (3.0 / (4.0 * np.pi * density_cm3)) ** (1.0 / 3.0)


def _occupation(field_g: float, pseudomomentum: np.ndarray,
                density_cm3: float) -> np.ndarray:
    """w(K) = exp[−(r_c/d)³], r_c = (K/γ) a_B — a probabilidade de ocupação."""
    gamma = float(field_to_gamma(field_g))
    r_c = (np.asarray(pseudomomentum, dtype=float) / gamma) * _BOHR_CM
    ratio = r_c / _interparticle_distance(density_cm3)
    return np.exp(-np.clip(ratio ** 3, 0.0, 700.0))


def _k_grid(field_g: float, density_cm3: float) -> np.ndarray:
    """Grade em K: densa no pico térmico + cauda log até a ocupação zerar.

    O corte cru em K_c virou o ponto onde w→0 (r_c ~ 2,4 d, w~1e-6); nunca menos
    que cobrir bem a região centrada (3 K_c).
    """
    gamma = float(field_to_gamma(field_g))
    e0 = float(ground_binding_at_rest(field_g, 0)) / RYDBERG_KEV
    k_c = _table1_s0(gamma)["q0"] * np.sqrt(2.0 * _MASS_H_ME * e0)
    k_occ = 2.4 * (_interparticle_distance(density_cm3) / _BOHR_CM) * gamma
    k_max = max(3.0 * k_c, k_occ)
    near = np.linspace(0.0, min(6.0 * k_c, k_max), 2000)
    if k_max > near[-1] * 1.001:
        far = np.logspace(np.log10(near[-1] + 1.0), np.log10(k_max), 1200)
        return np.unique(np.concatenate([near, far]))
    return near


def _partition_integrand(field_g: float, temperature: float,
                         density_cm3: float, K: np.ndarray) -> np.ndarray:
    """w(K) exp(χ/kT) K — o integrando de Z_00 (Eq.50) com a ocupação."""
    kt_kev = estrutura.BOLTZMANN * temperature / estrutura.ERG_PER_KEV
    chi = moving_binding(field_g, K, 0)                          # keV, = |E^∥|
    boltz = np.exp(np.clip(chi / kt_kev, -700, 700))
    return _occupation(field_g, K, density_cm3) * boltz * K


def _atomic_partition_ground(field_g: float, temperature: float,
                             density_cm3: float) -> float:
    """Z_00 (Eq.50) do estado fundamental, com a probabilidade de ocupação.

    A conversão K_físico = K_ua·(ℏ/a_B) reduz o prefator a λ_H²/(2π a_B²), e a
    integral fica adimensional em u.a. A ocupação w(K) corta os descentrados —
    a integral converge sem corte cru, e depende da densidade.
    """
    lambda_h = _thermal_wavelength(_MASS_H, temperature)
    prefactor = lambda_h ** 2 / (2.0 * np.pi * _BOHR_CM ** 2)   # adimensional
    K = _k_grid(field_g, density_cm3)
    integrand = _partition_integrand(field_g, temperature, density_cm3, K)
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
    z_w = _atomic_partition_ground(field_g, temperature, proton_density_cm3)

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
                               pseudomomentum: np.ndarray,
                               proton_density_cm3: float) -> np.ndarray:
    """p(K) normalizada: a distribuição dos átomos sobre K, com a ocupação.

    p(K) ∝ w(K) exp(χ/kT) K — o mesmo integrando de Z_00. A normalização é sobre
    a grade própria (cauda incluída), então ∫p dK numa grade que a cubra dá 1.
    """
    K = np.asarray(pseudomomentum, dtype=float)
    weight = _partition_integrand(field_g, temperature, proton_density_cm3, K)
    grid = _k_grid(field_g, proton_density_cm3)
    norm = np.trapezoid(_partition_integrand(field_g, temperature,
                                             proton_density_cm3, grid), grid)
    return weight / norm


def magnetic_broadening_profile(field_g: float, temperature: float,
                                photon_energy_kev: np.ndarray,
                                proton_density_cm3: float) -> np.ndarray:
    """g(E) [1/keV]: distribuição da energia de LIMIAR ε(K) sobre os átomos.

    O perfil do limiar de fotoionização magneticamente alargado. Como ε(K) é
    monótona, g(E) = p(K)/|dε/dK|. Com a ocupação, os estados descentrados
    (limiares baixos) preenchem a faixa mole — sem a borda espúria do corte cru.
    """
    kt_kev = estrutura.BOLTZMANN * temperature / estrutura.ERG_PER_KEV
    K = _k_grid(field_g, proton_density_cm3)
    eps = moving_binding(field_g, K, 0)                 # keV, decrescente em K
    pdf = thermal_pseudomomentum_pdf(field_g, temperature, K, proton_density_cm3)
    # peso óptico (mesmo da seção de choque): é a feição OBSERVÁVEL, não a
    # distribuição termodinâmica crua — que seria dominada pelos descentrados.
    optical = eps ** 4 / (eps ** 4 + (2.0 * kt_kev) ** 4)
    jac = np.abs(np.gradient(eps, K))                   # |dε/dK|
    g_at_K = pdf * optical / np.maximum(jac, 1.0e-300)  # g(ε(K))
    # reamostra em E crescente (ε decresce em K, então inverte)
    order = np.argsort(eps)
    return np.interp(np.asarray(photon_energy_kev, dtype=float),
                     eps[order], g_at_K[order], left=0.0, right=0.0)


def doppler_width_kev(field_g: float, temperature: float,
                      line_energy_kev: float) -> float:
    """Largura Doppler ΔE_D = E √(kT/m_H c²), para comparar com o magnético."""
    kt_erg = estrutura.BOLTZMANN * temperature
    return line_energy_kev * np.sqrt(kt_erg / (_MASS_H * estrutura.LIGHT ** 2))


# --- Forças de oscilador das transições principais (Potekhin 1998, Eq.21 v3) --
#
# Pesam as transições ligado-ligado e normalizam o ligado-livre (regra da soma).
# Eq.21 é JUSTO a fórmula cujo misprint a v3 corrigiu — o alerta de typo do
# usuário materializado. Portão independente: no limite γ→0 ambas devem dar
# 0,4162, a força de oscilador Lyman-α (1s→2p) do H SEM campo — um número
# conhecido, não uma tabela.
#
#   f(0) = (1 − 0,584/(1+u1 γ^u2)) (1+u3 γ)/(1+u4 γ^u5)                (21)
#
# f_001^|| : transição longitudinal (π), |000⟩→|001⟩ — DOMINA na banda mole.
# f_010^+  : transição σ+, |000⟩→|010⟩ — some no campo forte (vai ao cíclotron).
_OSC_U = {
    "001_par": (2.64, 1.076, 6.0e-6, 0.247, 0.381),   # π longitudinal
    "010_plus": (12.0, 1.43, 9.8e-5, 1.585, 0.713),   # σ+
}
_LYMAN_ALPHA_F = 0.4162    # 1s→2p do H sem campo — o gabarito de γ→0


def oscillator_strength_rest(field_g: np.ndarray | float,
                             transition: str = "001_par") -> np.ndarray:
    """f(0)(γ) da transição (Eq.21 v3). γ→0 devolve 0,4162 (Lyman-α)."""
    gamma = field_to_gamma(field_g)
    u1, u2, u3, u4, u5 = _OSC_U[transition]
    return ((1.0 - 0.584 / (1.0 + u1 * gamma ** u2))
            * (1.0 + u3 * gamma) / (1.0 + u4 * gamma ** u5))


def oscillator_strength_longitudinal(field_g: float,
                                     pseudomomentum: np.ndarray) -> np.ndarray:
    """f_001^||(K) da transição longitudinal (Eq.22-23, 300≤γ≤10⁴).

    K→0 recupera f(0). É a força de oscilador da linha que domina a banda mole,
    modulada pelo movimento do átomo.
    """
    gamma = float(field_to_gamma(field_g))
    K = np.asarray(pseudomomentum, dtype=float)
    e0 = float(ground_binding_at_rest(field_g, 0)) / RYDBERG_KEV
    k_c = _table1_s0(gamma)["q0"] * np.sqrt(2.0 * _MASS_H_ME * e0)
    f0 = float(oscillator_strength_rest(field_g, "001_par"))
    a = 0.877 * np.log(13100.0 / gamma)
    # b>0 é preciso: base negativa a potência fracionária vira NaN (F2 da
    # auditoria) para γ>15130. A fórmula vale até γ=10⁴; acima é extrapolação,
    # e manter b num piso positivo evita o NaN silencioso.
    b = max(1.0e-6, 0.89 - gamma / 17000.0)
    beta = 0.61 * (1.0 + 2410.0 / gamma) ** 1.5
    x = np.maximum(K / k_c, 1.0e-12)      # piso evita overflow no ramo K=0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        term1 = f0 * np.exp(-(a * x) ** 2)
        term2 = np.exp(-np.clip((b * x) ** (-beta), 0.0, 700.0)) \
            / (1.0 + 0.5 * np.sqrt(1.0 / x))
        return np.where(np.asarray(K) <= 0.0, f0, term1 + term2)


# --- Opacidade ligado-livre (fotoionização), 1ª passada ----------------------
#
# κ_bf = n_H σ_bf, com n_H = f_neutra n_0 e σ_bf a seção de choque de
# fotoionização do fundamental, MEDIADA sobre a distribuição térmica em K —
# o que espalha o limiar (o alargamento magnético). A física nova está aqui;
# a MAGNITUDE é de 1ª passada (ver PLANO.md, "A rota do ligado-livre"):
#
#   σ_bf(E; K) = σ₀ (ε(K)/E)³  para E ≥ ε(K),  0 abaixo               (Kramers)
#
# MAGNITUDE (achado A da auditoria): a seção NO LIMIAR escala como 1/ε. σ₀ é a
# hidrogênica para limiar de 1 Ryd; para o átomo magnetizado (limiar ε~0,3 keV)
# a seção é σ₀·(Ryd/ε), ~20× menor. A regra da soma confirmava que sem isso a
# integral ficava ~16× inflada. Polarização paralela a B (a que domina; a menos
# modificada pelo campo). O REFINO, se o portão B pedir: trocar por σ^bf(ω,K,B)
# tabelado do PC03 (numérica exata de PP97). Só o fundamental, só paralela.

_SIGMA0_BF_CM2 = 6.30e-18      # seção hidrogênica no limiar de 1 Ryd (H sem campo)


def bound_free_cross_section(field_g: float, temperature: float,
                             photon_energy_kev: np.ndarray,
                             proton_density_cm3: float) -> np.ndarray:
    """σ_bf(E) [cm²] por átomo neutro, mediada em K (limiar alargado).

    ∫ p(K) σ₀(Ryd/ε) (ε/E)³ Θ(E−ε) dK. Cada átomo tem limiar ε(K) e seção no
    limiar σ₀·Ryd/ε(K); a média sobre p(K) (com a ocupação) alarga a borda e,
    pelos estados descentrados, preenche a faixa mole sem borda espúria.
    """
    kt_kev = estrutura.BOLTZMANN * temperature / estrutura.ERG_PER_KEV
    K = _k_grid(field_g, proton_density_cm3)
    eps = moving_binding(field_g, K, 0)                     # keV, limiar de cada K
    pdf = thermal_pseudomomentum_pdf(field_g, temperature, K, proton_density_cm3)
    # OCUPAÇÃO ÓPTICA (distinção de PC03): a ocupação termodinâmica conta todo
    # estado ligado — certo para a fração neutra. Mas só os átomos ligados bem
    # ACIMA do térmico são absorvedores ópticos DISCRETOS; os descentrados
    # (ε≲kT) estão termicamente desfeitos e, sem esta supressão, formariam um
    # pico espúrio de opacidade abaixo da janela (a distribuição termodinâmica é
    # dominada por eles). Peso ε⁴/(ε⁴+(2kT)⁴): →1 bem ligado, →0 em ε≲2kT, com
    # transição SUAVE (sem a borda dura do corte cru). Calibrado para o pico
    # cair em E^(0), na janela, como a física manda. A forma exata é do
    # tratamento de ocupação de PC03 (refino, se o portão B pedir).
    optical = eps ** 4 / (eps ** 4 + (2.0 * kt_kev) ** 4)
    E = np.atleast_1d(np.asarray(photon_energy_kev, dtype=float))[:, None]
    # σ no limiar ∝ 1/ε (hidrogênico), tetada em σ₀ (não passa do pico hidrogênico).
    sigma_threshold = _SIGMA0_BF_CM2 * np.minimum(RYDBERG_KEV / eps[None, :], 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = np.where(E >= eps[None, :],
                          sigma_threshold * (eps[None, :] / E) ** 3, 0.0)
    return np.trapezoid(kernel * (pdf * optical)[None, :], K, axis=1)


def bound_free_opacity(field_g: float, temperature: float,
                       proton_density_cm3: float,
                       photon_energy_kev: np.ndarray) -> np.ndarray:
    """κ_bf [cm²/g] do ligado-livre atômico: f_neutra n_0 σ_bf / ρ.

    ρ = n_0 m_H, então κ_bf = f_neutra σ_bf / m_H — a densidade entra pela
    fração neutra E pela ocupação (que molda σ_bf).
    """
    f_neutral = neutral_fraction(field_g, temperature, proton_density_cm3)
    sigma = bound_free_cross_section(field_g, temperature, photon_energy_kev,
                                     proton_density_cm3)
    return f_neutral * sigma / _MASS_H


def bound_free_opacity_profile(field_g: float, temperatures: np.ndarray,
                               densities_g_cm3: np.ndarray,
                               energies: np.ndarray) -> np.ndarray:
    """κ_bf [cm²/g] em (nE, nD) sobre um perfil de atmosfera.

    `densities_g_cm3` é a densidade de MASSA ρ (g/cm³); converte para
    n_0 = ρ/m_H. É o que o solucionador soma à componente α=0 (paralela a B) da
    absorção cíclica. Loop em profundidade (a grade em K depende da densidade).
    """
    T = np.atleast_1d(np.asarray(temperatures, dtype=float))
    n0 = np.atleast_1d(np.asarray(densities_g_cm3, dtype=float)) / _MASS_H
    E = np.asarray(energies, dtype=float)
    out = np.empty((E.size, T.size))
    for d in range(T.size):
        out[:, d] = bound_free_opacity(field_g, float(T[d]),
                                       float(max(n0[d], 1.0e-30)), E)
    return out


def atomic_opacity_table(field_g: float, log_t_grid: np.ndarray,
                         log_rho_grid: np.ndarray,
                         energies: np.ndarray) -> np.ndarray:
    """κ_bf[nE, nT, nρ] pré-computada — cara, roda UMA vez fora do laço.

    O solucionador tabula em (lgT, lgρ) e interpola por iteração, porque o
    perfil por profundidade custa ~2 s e recalculá-lo a cada passo dobraria o
    tempo do solve. κ_bf varia devagar com T,ρ, então a interpolação basta.
    """
    lt = np.atleast_1d(np.asarray(log_t_grid, dtype=float))
    lr = np.atleast_1d(np.asarray(log_rho_grid, dtype=float))
    E = np.asarray(energies, dtype=float)
    # Guarda lg(κ_bf): κ varia por ordens de grandeza (borda de fotoionização),
    # e interpolar em linear dava 115% de erro. O piso mapeia κ=0 (abaixo do
    # limiar) para um lg muito negativo, que a exp devolve a ~0.
    table = np.full((E.size, lt.size, lr.size), -300.0)
    for it in range(lt.size):
        for ir in range(lr.size):
            kappa = bound_free_opacity(field_g, 10.0 ** lt[it],
                                       10.0 ** lr[ir] / _MASS_H, E)
            table[:, it, ir] = np.log10(np.maximum(kappa, 1.0e-300))
    return table


def interpolate_atomic_opacity(table: np.ndarray, log_t_grid: np.ndarray,
                               log_rho_grid: np.ndarray, log_t: np.ndarray,
                               log_rho: np.ndarray) -> np.ndarray:
    """Bilinear de lg(κ_bf) em (lgT, lgρ) por profundidade → κ_bf (nE, nD)."""
    lt = np.clip(np.asarray(log_t, dtype=float), log_t_grid[0], log_t_grid[-1])
    lr = np.clip(np.asarray(log_rho, dtype=float), log_rho_grid[0], log_rho_grid[-1])
    it = np.clip(np.searchsorted(log_t_grid, lt) - 1, 0, len(log_t_grid) - 2)
    ir = np.clip(np.searchsorted(log_rho_grid, lr) - 1, 0, len(log_rho_grid) - 2)
    ft = (lt - log_t_grid[it]) / (log_t_grid[it + 1] - log_t_grid[it])
    fr = (lr - log_rho_grid[ir]) / (log_rho_grid[ir + 1] - log_rho_grid[ir])
    c00 = table[:, it, ir];       c10 = table[:, it + 1, ir]
    c01 = table[:, it, ir + 1];   c11 = table[:, it + 1, ir + 1]
    log_kappa = ((c00 * (1 - ft) + c10 * ft) * (1 - fr)
                 + (c01 * (1 - ft) + c11 * ft) * fr)
    return 10.0 ** log_kappa
