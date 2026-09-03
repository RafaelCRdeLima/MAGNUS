"""Opacidades magnetizadas, resolvidas por modo e por ângulo — o estágio 2.

A peça que o plano diz que é a cara do projeto: kappa_j(E, theta_B, rho, T, B)
para os dois modos normais do plasma magnetizado. A Ioffe publica só as médias
de Rosseland; o monocromático sai daqui.

**A decisão de arquitetura, e por que ela é a segurança do módulo.** Os vetores
de polarização dos modos NÃO vêm de fórmulas lembradas de artigo — vêm de um
autoproblema exato: monta-se o tensor dielétrico frio de duas espécies
(elétron + próton), escreve-se a equação de onda

    eps E = n^2 (I - k k) E

e os dois modos eletromagnéticos são os autovetores de autovalor finito (o
terceiro, longitudinal, sai com autovalor infinito e é descartado). Não há
sinal de alpha para errar, não há convenção de beta para trocar, e a
polarização do vácuo do estágio 5 entra depois como um termo a mais no MESMO
tensor, sem tocar em nada disto.

A decomposição cíclica em relação a B — alpha = -1, 0, +1 — só entra depois,
para montar a opacidade:

    kappa_j(E, theta) = soma_alpha |e_alpha^j(theta)|^2 kappa^alpha(E)

com as ressonâncias morando nos kappa^alpha: o cíclotron do elétron numa
componente circular, o do próton na OUTRA — os dois têm carga oposta e giram em
sentidos opostos, e é o autoproblema que garante a consistência dos sinais.

**Portões, na ordem do plano.** Aninhamento em B -> 0 (os kappa_j degeneram no
livre-livre + Thomson do estágio 1, para os dois modos e qualquer ângulo); o
piso de espalhamento do Potekhin com a geometria certa; K0 e K1 a 20%; espectro
contra o nsmaxg quente.
"""

from __future__ import annotations

import numpy as np

from .estrutura import (BOLTZMANN, ERG_PER_KEV, PROTON_MASS, THOMSON_CM2_G,
                        opacities as unmagnetized_opacities)

#: Energia de cíclotron do elétron, keV por gauss: hbar e B / (m_e c).
CYCLOTRON_E_PER_GAUSS = 1.157672e-11
#: m_e/m_p — o cíclotron do próton é o do elétron vezes isto, e a seção de
#: choque Thomson do próton é o quadrado disto.
MASS_RATIO = 5.446170214889e-4
ELECTRON_REST_KEV = 510.99895
FINE_STRUCTURE = 7.2973525693e-3
#: Campo crítico da QED, m_e^2 c^3 / (e hbar), em gauss.
CRITICAL_FIELD_G = 4.41405e13
#: Amortecimento radiativo clássico: Gamma(E) = (2 alfa/3) E^2 / (m c^2).
def _radiative_damping(energy_kev: np.ndarray, mass_kev: float) -> np.ndarray:
    return (2.0 * FINE_STRUCTURE / 3.0) * energy_kev ** 2 / mass_kev


# --------------------------------------------------------------------------- #
# O tensor dielétrico e os modos


def dielectric_cyclic(energy_kev: np.ndarray, density: float, field_g: float,
                      damping: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(eps_+, eps_-, eps_z) do plasma frio de elétrons e prótons.

    Na base cíclica o tensor é diagonal, e cada espécie soma

        eps_pm -= v / (1 -/+ s (E_c/E) + i Gamma/E)

    com v = (E_p/E)^2 o termo de plasma, E_c o cíclotron da espécie e s o SINAL
    da carga — que é o que põe a ressonância do elétron numa componente e a do
    próton na outra, sem ninguém escolher isso à mão. O amortecimento radiativo
    regulariza as ressonâncias; sem ele o autoproblema fica singular em cima
    delas.
    """
    energy = np.asarray(energy_kev, dtype=float)
    plasma_e = 28.7135 * np.sqrt(density) * 1.0e-3          # hbar omega_pe, keV
    plasma_p = plasma_e * np.sqrt(MASS_RATIO)
    cyclotron_e = CYCLOTRON_E_PER_GAUSS * field_g
    cyclotron_p = cyclotron_e * MASS_RATIO

    def species(plasma, cyclotron, mass_kev, sign):
        v = (plasma / energy) ** 2
        gamma = (1j * _radiative_damping(energy, mass_kev) / energy) if damping else 0.0
        plus = -v / (1.0 - sign * cyclotron / energy + gamma)
        minus = -v / (1.0 + sign * cyclotron / energy + gamma)
        along = -v / (1.0 + gamma)
        return plus, minus, along

    e_plus, e_minus, e_along = species(plasma_e, cyclotron_e, ELECTRON_REST_KEV, +1.0)
    p_plus, p_minus, p_along = species(plasma_p, cyclotron_p,
                                       ELECTRON_REST_KEV / MASS_RATIO, -1.0)
    return 1.0 + e_plus + p_plus, 1.0 + e_minus + p_minus, 1.0 + e_along + p_along


def vacuum_delta(field_g: float) -> float:
    """O parâmetro do vácuo de Euler-Heisenberg: delta = (alfa/45pi)(B/B_Q)^2.

    Regime de campo fraco, B abaixo do crítico (4,41e13 G). Em lg B = 13,5 o
    B/B_Q vale 0,72 e a expansão já é marginal — fica declarado: os coeficientes
    exatos de campo forte (funções especiais de Heyl & Hernquist) são o
    refinamento, e trocam só três números aqui dentro.
    """
    return FINE_STRUCTURE / (45.0 * np.pi) * (field_g / CRITICAL_FIELD_G) ** 2


def mode_amplitudes(energy_kev: np.ndarray, theta_b: float, density: float,
                    field_g: float, vacuum: bool = False,
                    details: bool = False):
    """|e_alpha^j|^2 dos dois modos: forma (n_E, 2 modos, 3 componentes).

    Resolve a equação de onda COMPLETA, com permeabilidade anisotrópica,

        n^2 k x (mu^-1 (k x E)) + eps E = 0,

    reescrita como autoproblema ordinário A E = lambda E com A = eps^-1 M,
    M = -[k]x mu^-1 [k]x e lambda = 1/n^2: dois autovalores não nulos são os
    modos eletromagnéticos (n^2 = 1/lambda) e o nulo é o longitudinal,
    descartado. Com mu = I isso degenera exatamente em M = I - kk, que era a
    forma anterior. Tudo vetorizado sobre a energia — o laço por ponto com eig
    generalizado custava a viabilidade do vácuo dentro da atmosfera, onde as
    amplitudes deixam de ser independentes da densidade.

    **O vácuo entra aqui e em mais lugar nenhum** (a aposta de arquitetura do
    arranque, agora paga): com `vacuum=True`,

        eps_± += -2 delta      eps_z += +5 delta      (I - 2d + 7d bb)
        mu^-1 = (1 - 2 delta) I - 4 delta bb

    os coeficientes -2, 7, -4 de Euler-Heisenberg em campo fraco. A conversão
    adiabática de modos NA ressonância de vácuo não é tratada — os modos são
    calculados dos dois lados dela, e o transporte os acopla por espalhamento;
    o colchete com/sem conversão continua sendo o teste do estágio 5.

    Os modos saem ordenados por |n^2| decrescente, como antes.
    """
    plus, minus, along = dielectric_cyclic(energy_kev, density, field_g)
    if vacuum:
        delta = vacuum_delta(field_g)
        plus = plus - 2.0 * delta
        minus = minus - 2.0 * delta
        along = along + 5.0 * delta
        inverse_mu = np.diag([1.0 - 2.0 * delta, 1.0 - 2.0 * delta,
                              1.0 - 6.0 * delta])
    else:
        inverse_mu = np.eye(3)

    sin_t, cos_t = np.sin(theta_b), np.cos(theta_b)
    cross = np.array([[0.0, -cos_t, 0.0],
                      [cos_t, 0.0, -sin_t],
                      [0.0, sin_t, 0.0]])
    propagation = -cross @ inverse_mu @ cross

    # Tudo na BASE CÍCLICA, onde eps é diagonal exata. Não é estética: em
    # theta = 0 a matriz inteira fica diagonal e os modos saem como vetores da
    # base por construção. Na base cartesiana os dois autovalores transversos
    # coincidem ao nível do termo de plasma — 1e-14 em densidade baixa — e o
    # eig devolve uma base ARBITRÁRIA do subespaço quase degenerado, misturando
    # as circulares por ~1%: exatamente as componentes que separam a
    # ressonância do próton da do elétron.
    n_energy = len(np.atleast_1d(energy_kev))
    root_half = 1.0 / np.sqrt(2.0)
    to_cyclic = np.array([[root_half, 1j * root_half, 0.0],
                          [root_half, -1j * root_half, 0.0],
                          [0.0, 0.0, 1.0]])
    cyclic_propagation = to_cyclic @ propagation.astype(complex) @ to_cyclic.conj().T
    epsilon_diag = np.stack([plus, minus, along], axis=-1)      # (n_E, 3)
    system = cyclic_propagation[None, :, :] / epsilon_diag[:, :, None]
    values, vectors = np.linalg.eig(system)

    # Dois maiores |lambda| = os modos; ordena por |n^2| = 1/|lambda| decrescente,
    # ou seja |lambda| CRESCENTE entre os dois escolhidos.
    order = np.argsort(-np.abs(values), axis=1)[:, :2]
    order = np.take_along_axis(order, np.argsort(
        np.take_along_axis(np.abs(values), order, axis=1), axis=1), axis=1)
    rows = np.arange(n_energy)[:, None]
    chosen = vectors[rows, :, order]                            # (n_E, 2, 3) cíclico
    chosen = chosen / np.linalg.norm(chosen, axis=2, keepdims=True)
    amplitudes = np.abs(chosen) ** 2
    if details:
        refractive = 1.0 / np.take_along_axis(values, order, axis=1)
        return amplitudes, refractive
    return amplitudes


# --------------------------------------------------------------------------- #
# As opacidades cíclicas


def cyclic_scattering(energy_kev: np.ndarray, field_g: float,
                      density: np.ndarray | float = 0.0,
                      temperature: np.ndarray | float = 0.0) -> np.ndarray:
    """sigma^alpha de espalhamento, em cm^2/g: forma (n_E, 3), alpha=(+1,-1,0).

    Oscilador carregado amortecido em campo: cada componente circular vê

        sigma^alpha = sigma_T E^2 / ((E + s alpha E_c)^2 + Gamma^2)

    para cada espécie, com o sinal da carga decidindo qual componente ressoa.
    O próton entra com (m_e/m_p)^2 na escala — seis ordens abaixo — mas ao longo
    de B, onde o elétron está suprimido por (E/E_Be)^2, ele pode mandar: é a
    conta de guardanapo do plano, que aqui vira número.
    """
    energy = np.asarray(energy_kev, dtype=float)
    cyclotron_e = CYCLOTRON_E_PER_GAUSS * field_g
    cyclotron_p = cyclotron_e * MASS_RATIO
    # As mesmas larguras do livre-livre: com densidade e temperatura em mãos, a
    # ressonância do próton ganha Doppler e colisão; sem elas (o padrão), fica a
    # radiativa de antes — e os testes de aninhamento continuam valendo.
    broaden = np.any(np.asarray(density) > 0.0)
    extra_p = (MASS_RATIO * collision_width(energy_kev, density, temperature)
               + doppler_width(energy_kev, temperature)) if broaden else 0.0

    def one(cyclotron, mass_kev, scale, sign, extra):
        gamma = _radiative_damping(energy, mass_kev) + extra
        columns = []
        for alpha in (+1.0, -1.0, 0.0):
            shifted = energy + sign * alpha * cyclotron
            columns.append(scale * energy ** 2 / (shifted ** 2 + gamma ** 2))
        return np.stack(columns, axis=-1)

    electron = one(cyclotron_e, ELECTRON_REST_KEV, THOMSON_CM2_G, +1.0, 0.0)
    proton = one(cyclotron_p, ELECTRON_REST_KEV / MASS_RATIO,
                 THOMSON_CM2_G * MASS_RATIO ** 2, -1.0, extra_p)
    return electron + proton


def collision_width(energy_kev: np.ndarray, density: np.ndarray,
                    temperature: np.ndarray) -> np.ndarray:
    """Largura colisional do oscilador, em keV, por autoconsistência.

    A frequência efetiva de colisão é a MESMA grandeza que gera o livre-livre —
    então em vez de recalculá-la de fórmulas próprias, inverte-se o kappa_ff já
    aferido do estágio 1 pela relação de Drude do plasma sem campo:

        alpha_ff = nu_eff omega_p^2 / (c omega^2)
        =>  hbar nu_eff = kappa_ff rho (hbar c) (E/E_pe)^2

    Uma fonte só para a força E para a largura, e nenhuma para errar duas vezes.
    """
    base, _ = unmagnetized_opacities(energy_kev, density, temperature)
    plasma = 28.7135e-3 * np.sqrt(np.maximum(density, 1.0e-300))    # E_pe, keV
    hbar_c = 1.973269804e-8                                          # keV cm
    return base * density * hbar_c * (np.asarray(energy_kev, float) / plasma) ** 2


def doppler_width(energy_kev: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """Largura Doppler térmica do PRÓTON, em keV: E sqrt(2kT/m_p c^2)."""
    thermal_kev = BOLTZMANN * np.asarray(temperature, float) / ERG_PER_KEV
    return np.asarray(energy_kev, float) * np.sqrt(
        2.0 * thermal_kev / (ELECTRON_REST_KEV / MASS_RATIO))


def cyclic_free_free(energy_kev: np.ndarray, density: np.ndarray,
                     temperature: np.ndarray, field_g: float) -> np.ndarray:
    """kappa^alpha livre-livre, em cm^2/g: forma como a entrada, mais (3,).

    O movimento relativo elétron-próton dirigido pela onda, com o freio das
    colisões. O denominador vem FATORADO em duas lorentzianas, cada ressonância
    com a sua largura:

        E^4 / [ ((E + alpha E_Be)^2 + Gamma_e^2) ((E - alpha E_ci)^2 + Gamma_p^2) ]

    A FORÇA da absorção é colisional — está no kappa^0 da frente, que é o
    livre-livre do estágio 1 com Elwert-Born. A LARGURA é a soma do que
    interrompe a fase do oscilador: no elétron, radiativa + colisional; no
    próton, radiativa + colisional/1836 (o atrito por unidade de massa) +
    **Doppler térmico**, que é quem manda na fotosfera — medido, a linha só com
    largura radiativa saía 2 a 3 vezes funda demais contra o nsmaxg, com os
    piores pontos do portão inteiro concentrados nas asas dela.

    Degenera em 1 - O(Gamma^2/E^2) quando B -> 0: o aninhamento continua.
    """
    base, _ = unmagnetized_opacities(energy_kev, density, temperature)
    energy = np.asarray(energy_kev, dtype=float)
    cyclotron_e = CYCLOTRON_E_PER_GAUSS * field_g
    cyclotron_p = cyclotron_e * MASS_RATIO
    collision = collision_width(energy_kev, density, temperature)
    gamma_e = _radiative_damping(energy, ELECTRON_REST_KEV) + collision
    gamma_p = (_radiative_damping(energy, ELECTRON_REST_KEV / MASS_RATIO)
               + MASS_RATIO * collision + doppler_width(energy_kev, temperature))
    columns = []
    for alpha in (+1.0, -1.0, 0.0):
        electron_part = (energy + alpha * cyclotron_e) ** 2 + gamma_e ** 2
        proton_part = (energy - alpha * cyclotron_p) ** 2 + gamma_p ** 2
        columns.append(base * energy ** 4 / (electron_part * proton_part))
    return np.stack(columns, axis=-1)


def mode_opacities(energy_kev: np.ndarray, theta_b: float, density: float,
                   temperature: float, field_g: float, vacuum: bool = False) -> dict:
    """kappa_j(E) dos dois modos, absorção e espalhamento separados.

    A montagem final: amplitudes cíclicas do autoproblema vezes as opacidades
    cíclicas. Devolve também as amplitudes, porque o traçado de raios do futuro
    vai querer a polarização e não só a opacidade.
    """
    amplitudes = mode_amplitudes(energy_kev, theta_b, density, field_g, vacuum=vacuum)
    scattering = cyclic_scattering(energy_kev, field_g)
    absorption = cyclic_free_free(energy_kev, np.full_like(np.asarray(energy_kev, float),
                                                           density),
                                  np.full_like(np.asarray(energy_kev, float),
                                               temperature), field_g)
    return {
        "amplitudes": amplitudes,
        "scattering": np.einsum("eja,ea->ej", amplitudes, scattering),
        "absorption": np.einsum("eja,ea->ej", amplitudes, absorption),
    }


# --------------------------------------------------------------------------- #
# A média de Rosseland de dois modos


def rosseland_two_modes(energy_kev: np.ndarray, theta_b: float, density: float,
                        temperature: float, field_g: float,
                        vacuum: bool = False) -> float:
    """K(theta) em cm^2/g, com os dois modos conduzindo o fluxo em paralelo.

        1/K = < (1/2)(1/chi_1 + 1/chi_2) >_Rosseland

    — a definição que a auditoria 3 verificou contra o piso do Potekhin: com
    média aritmética o piso sairia sete ordens de grandeza acima do tabelado.
    """
    from .estrutura import planck_temperature_derivative
    modes = mode_opacities(energy_kev, theta_b, density, temperature, field_g,
                           vacuum=vacuum)
    total = modes["scattering"] + modes["absorption"]
    weight = planck_temperature_derivative(np.asarray(energy_kev, float),
                                           np.full(len(np.atleast_1d(energy_kev)),
                                                   temperature))
    inverse = 0.5 * (1.0 / total[:, 0] + 1.0 / total[:, 1])
    numerator = np.trapezoid(weight * inverse, energy_kev)
    denominator = np.trapezoid(weight, energy_kev)
    return float(denominator / numerator)


def rosseland_tensor(energy_kev: np.ndarray, density: float, temperature: float,
                     field_g: float, angle_nodes: int = 12,
                     vacuum: bool = False) -> tuple[float, float]:
    """(K_paralelo, K_perpendicular) em cm^2/g — as duas colunas do Potekhin.

    **K0 não é a opacidade em theta = 0.** O fluxo difusivo ao longo de B soma
    fótons propagando em TODAS as direções, pesados pela projeção mu^2 — e é
    isso que a relação dos próprios autores,

        1/K(theta) = cos^2(theta)/K0 + sin^2(theta)/K1,

    diz: ela é a contração do tensor de difusão, e K0 e K1 são as componentes
    dele. Confundir K0 com chi(theta = 0) custa o fator ~3 que o portão mediu
    antes desta correção: ao longo de B os dois modos são circulares e cegos à
    componente alpha = 0, mas o fluxo paralelo também carrega fótons oblíquos,
    que a enxergam.

        1/K_par  = < 3   int_0^1 mu^2     (1/2)(1/chi_1 + 1/chi_2) dmu >_R
        1/K_perp = < 3/2 int_0^1 (1-mu^2) (1/2)(1/chi_1 + 1/chi_2) dmu >_R

    com a média de Rosseland em energia por fora. Os coeficientes conferem no
    isotrópico — 3·(1/3) = 1 e (3/2)·(2/3) = 1 — e o caminho errado já foi
    trilhado uma vez: int dOmega/4pi de função par de mu É int_0^1 dmu, não a
    metade, e o fator 2 perdido saiu daqui como "razão 0,45" no portão.
    """
    from .estrutura import planck_temperature_derivative
    mu, weight = np.polynomial.legendre.leggauss(angle_nodes)
    mu = 0.5 * (mu + 1.0)
    weight = 0.5 * weight
    energy = np.atleast_1d(np.asarray(energy_kev, dtype=float))
    inverse = np.zeros((energy.size, mu.size))
    for index, cosine in enumerate(mu):
        modes = mode_opacities(energy, float(np.arccos(cosine)), density,
                               temperature, field_g, vacuum=vacuum)
        total = modes["scattering"] + modes["absorption"]
        inverse[:, index] = 0.5 * (1.0 / total[:, 0] + 1.0 / total[:, 1])
    parallel = 3.0 * inverse @ (weight * mu ** 2)
    transverse = 1.5 * inverse @ (weight * (1.0 - mu ** 2))
    planck_weight = planck_temperature_derivative(energy, np.full(energy.size,
                                                                  temperature))
    norm = np.trapezoid(planck_weight, energy)
    return (float(norm / np.trapezoid(planck_weight * parallel, energy)),
            float(norm / np.trapezoid(planck_weight * transverse, energy)))


# --------------------------------------------------------------------------- #
# A atmosfera magnetizada


def channel_geometry(energy_kev: np.ndarray, mu: np.ndarray, field_g: float,
                     theta_b: float = 0.0, phi_nodes: int = 8,
                     reference_density: float = 1.0e-2) -> dict:
    """Amplitudes cíclicas de todos os canais (modo x ângulo), uma vez só.

    No plasma frio sem vácuo os autovetores não dependem da densidade — todos os
    desvios do tensor são proporcionais a (omega_p/omega)² e cancelam nas razões
    — então a geometria se calcula UMA vez por (E, mu) e vale em toda a coluna.

    **Campo inclinado, com a aproximação declarada.** Com B a um ângulo theta_b
    da normal, o ângulo de um raio (mu, phi) com o campo é

        cos(theta) = mu cos(theta_b) + sqrt(1 - mu^2) sin(theta_b) cos(phi)

    e a simetria azimutal quebra: a opacidade passa a depender de phi, e também
    deixa de ser igual para cima e para baixo. Aqui as amplitudes são MEDIADAS
    em phi e simetrizadas em ±mu — o transporte continua azimutalmente simétrico
    e o Feautrier continua valendo, ao custo de o feixe sair phi-médio, que é
    exatamente o que o formato de tabela (sem eixo phi) e o motor de hoje
    consomem. O feixe resolvido em phi é refinamento com endereço: pediria um
    eixo a mais na tabela e outro no traçado de raios.

    theta_b = 0 degenera exatamente no caso do campo normal: um nó em phi e a
    média não mediam nada.
    """
    n_mu = mu.size
    energy = np.atleast_1d(energy_kev)
    amplitudes = np.zeros((energy.size, 2 * n_mu, 3))
    if abs(theta_b) < 1.0e-12:
        angles_by_mu = [np.array([float(np.arccos(c))]) for c in mu]
    else:
        phi = np.pi * (np.arange(phi_nodes) + 0.5) / phi_nodes
        angles_by_mu = []
        for cosine in mu:
            sine = np.sqrt(max(0.0, 1.0 - cosine ** 2))
            with_field = cosine * np.cos(theta_b) + sine * np.sin(theta_b) * np.cos(phi)
            # ±mu juntos: simetriza a opacidade para cima e para baixo, que é o
            # que o esquema de Feautrier exige.
            angles_by_mu.append(np.arccos(np.clip(
                np.concatenate([with_field, -with_field]), -1.0, 1.0)))
    for index, angles in enumerate(angles_by_mu):
        first = np.zeros((energy.size, 3))
        second = np.zeros((energy.size, 3))
        for angle in angles:
            block = mode_amplitudes(energy, float(angle), reference_density, field_g)
            first += block[:, 0]
            second += block[:, 1]
        amplitudes[:, index] = first / len(angles)
        amplitudes[:, n_mu + index] = second / len(angles)
    return {
        "amplitudes": amplitudes,
        "mu_channel": np.concatenate([mu, mu]),
    }


def solve(log_t_eff: float, log_g: float, field_g: float, theta_b: float = 0.0,
          energies: np.ndarray | None = None, columns: np.ndarray | None = None,
          mu_nodes: int = 6, iterations: int = 200, tolerance: float = 1.0e-5,
          damping: float = 0.25, vacuum: bool = False,
          surface_column: float | None = None,
          conversion: str = "full", trace: list | None = None,
          atomic: bool = False) -> dict:
    """Atmosfera magnetizada, campo ao longo da normal: o caso dos `ThB00`.

    A mesma máquina do estágio 1 — hidrostática P = g·y, Unsöld–Lucy com
    amortecimento adaptativo — com o transporte trocado pelo polarizado: dois
    modos, opacidade dependente do ângulo, espalhamento acoplando tudo pelas
    componentes cíclicas. Cada modo emite kappa_c B/2, e a soma sobre canais
    com os pesos w̃ (que somam 1 sobre os DOIS modos) devolve o balanço de
    energia sem meio fator perdido.

    Sem comptonização e sem ionização parcial — as duas dívidas continuam as do
    estágio 1, e valem aqui o que valem lá.
    """
    from . import transporte, estrutura, atomico
    energies = estrutura.energy_grid(1.0e-3, 60.0, 220) if energies is None else energies
    # ATMOSFERA FINA (P3): coluna truncada em surface_column, com uma superfície
    # emissora embaixo — I+(fundo) = B/2 por modo, a eq. (15) do Suleimanov,
    # Pavlov & Werner (2009). É a classe de modelo que já venceu na RBS 1223
    # (Hambaryan et al. 2011). Sigma pequeno degenera no corpo negro; Sigma
    # grande, no semi-infinito — os dois limites são portões de graça.
    if surface_column is not None and columns is None:
        y = estrutura.column_grid(1.0e-6, surface_column,
                                  max(41, int(24 * np.log10(surface_column * 1.0e6))))
    else:
        y = estrutura.column_grid() if columns is None else columns
    mu, weights = transporte.gauss_legendre_mu(mu_nodes)
    geometry = channel_geometry(energies, mu, field_g, theta_b=theta_b)
    # Amplitudes SEMPRE com o eixo de profundidade, por difusão de forma: sem
    # vácuo elas não dependem da densidade e o eixo é broadcast; com vácuo a
    # razão plasma/vácuo varia ao longo da coluna — a ressonância de vácuo
    # cruza a atmosfera — e as amplitudes são recalculadas por profundidade a
    # cada iteração (o autoproblema vetorizado é o que paga essa conta).
    amplitudes = geometry["amplitudes"][:, :, :, None]      # (nE, nC, 3, 1)
    mu_channel = geometry["mu_channel"]
    # Os pesos somam DOIS — a quadratura angular inteira por modo — para que J
    # e H saiam como totais (soma das duas polarizações) e não como médias por
    # polarização. Com soma 1 tudo fecha bonito e o fluxo sai exatamente pela
    # metade: medido, int F dE / sigma T^4 = 0,5008 antes deste comentário.
    weight_channel = np.concatenate([weights, weights])
    n_channel = mu_channel.size

    gravity, effective = 10.0 ** log_g, 10.0 ** log_t_eff
    pressure = gravity * y
    target_flux = estrutura.STEFAN * effective ** 4 / (4.0 * np.pi)
    temperature = estrutura._initial_temperature(effective, estrutura.THOMSON_CM2_G * y)
    grid = energies[:, None]
    history = []
    previous_step = np.zeros(y.size)
    relaxation = np.ones(y.size)
    # A temperatura da SUPERFÍCIE de baixo é estado próprio, separado do gás da
    # última célula: são papéis diferentes — o gás obedece ao equilíbrio
    # radiativo local, a superfície carrega o fluxo que falta. Amarrá-las numa
    # variável só fazia o UL local re-esfriar o que a correção de fluxo subia.
    surface_temperature = effective
    surface_relaxation = 1.0
    surface_previous = 0.0

    # IONIZAÇÃO PARCIAL (P1): a opacidade ligado-livre atômica entra como
    # absorção na componente α=0 (paralela a B) da base cíclica. Cara por
    # profundidade, então tabulada UMA vez em (lgT, lgρ) e interpolada por
    # iteração (ver atomico.py). Ranges generosos que cobrem a atmosfera.
    atomic_table = atomic_lt = atomic_lr = None
    if atomic:
        atomic_lt = np.linspace(log_t_eff - 0.8, log_t_eff + 0.9, 16)
        atomic_lr = np.linspace(-7.0, 4.0, 24)
        atomic_table = atomico.atomic_opacity_table(field_g, atomic_lt,
                                                    atomic_lr, energies)

    for step in range(iterations):
        density = pressure * estrutura.PROTON_MASS / (2.0 * estrutura.BOLTZMANN * temperature)
        if vacuum:
            stack = np.zeros((energies.size, 2 * mu.size, 3, y.size))
            for im, cosine in enumerate(mu):
                for id_ in range(y.size):
                    block = mode_amplitudes(energies, float(np.arccos(cosine)),
                                            max(float(density[id_]), 1.0e-30),
                                            field_g, vacuum=True)
                    stack[:, im, :, id_] = block[:, 0]
                    stack[:, mu.size + im, :, id_] = block[:, 1]
            amplitudes = stack
        absorption_cyclic = cyclic_free_free(grid, density[None, :],
                                             temperature[None, :], field_g)  # (nE,nD,3)
        if atomic:
            # κ_bf atômico RESOLVIDO EM POLARIZAÇÃO. A fotoionização paralela a B
            # (α=0) é a que calculamos; as componentes perpendiculares (α=±1) são
            # a resposta de dipolo do elétron ligado ao campo, [E/(E±E_Be)]², que
            # PP97 mostra fortemente SUPRIMIDA abaixo do cíclotron. Como E_Be=366
            # keV em lgB=13,5, a banda mole (0,2 keV) está muito abaixo e a
            # perpendicular é ~(E/E_Be)²~10⁻⁷ — quase nula. Em θ_B=0 só as
            # circulares (α=±1) se propagam, então o ligado-livre atômico mal
            # toca o raio vertical — a física, não uma escolha.
            kappa_bf = atomico.interpolate_atomic_opacity(
                atomic_table, atomic_lt, atomic_lr,
                np.log10(temperature), np.log10(np.maximum(density, 1.0e-30)))
            e_be = CYCLOTRON_E_PER_GAUSS * field_g
            factor_plus = (grid / (grid + e_be)) ** 2               # α=+1
            factor_minus = (grid / (grid - e_be)) ** 2              # α=−1
            absorption_cyclic = absorption_cyclic.copy()
            absorption_cyclic[:, :, 0] = absorption_cyclic[:, :, 0] + kappa_bf * factor_plus
            absorption_cyclic[:, :, 1] = absorption_cyclic[:, :, 1] + kappa_bf * factor_minus
            absorption_cyclic[:, :, 2] = absorption_cyclic[:, :, 2] + kappa_bf
        # O espalhamento agora também é por profundidade: a ressonância do
        # próton carrega as larguras colisional e Doppler locais.
        scattering_cyclic = cyclic_scattering(grid, field_g, density[None, :],
                                              temperature[None, :])          # (nE,nD,3)
        # Por canal: absorção e espalhamento projetados nas amplitudes.
        wide = np.broadcast_to(amplitudes,
                               (energies.size, mu_channel.size, 3, y.size))
        absorption = np.einsum("ecad,eda->ecd", wide, absorption_cyclic)
        scattering = np.einsum("ecad,eda->ecd", wide, scattering_cyclic)
        extinction = absorption + scattering
        # Piso no INCREMENTO de tau, e é numérico declarado, não física: com o
        # vácuo dominante o modo X fica com e_z exatamente zero em todo ângulo e
        # a extinção do canal cai a 1e-12 cm²/g — o fóton é quase livre, o que é
        # verdade, mas o bloco de Feautrier passa a misturar escalas separadas
        # por 13 ordens e devolve J negativo enorme. Um passo mínimo de 1e-8 em
        # tau (1e-6 na coluna inteira: transparente do mesmo jeito) devolve o
        # condicionamento sem tocar em nada observável.
        increments = np.maximum(0.5 * (extinction[:, :, 1:] + extinction[:, :, :-1])
                                * np.diff(y), 1.0e-8)
        optical_depth = np.concatenate(
            [np.zeros((energies.size, n_channel, 1)),
             np.cumsum(increments, axis=2)], axis=2)
        # CONVERSÃO PARCIAL DE MODOS (P2), van Adelsberg & Lai (2006): na
        # célula em que a densidade cruza rho_V = 0,96 E_1^2 B_14^2 g/cm^3, os
        # dois modos trocam com probabilidade P_jump = exp[-(pi/2)(E/E_ad)^3].
        # Aqui a troca vira um espalhamento de TROCA localizado na célula do
        # cruzamento, de espessura por travessia t = -ln(1 - P_jump): os canais
        # já seguem os ramos ADIABÁTICOS (ordenação contínua por n^2), então o
        # salto não adiabático é o evento de troca. conversion='full' (P=0) é o
        # adiabático puro que já tínhamos; 'none' força a troca completa.
        exchange = None
        if vacuum and conversion != "full":
            exchange = np.zeros((energies.size, n_channel, n_channel, y.size))
            rho_v = 0.96 * energies ** 2 * (field_g / 1.0e14) ** 2
            crossing = np.searchsorted(density, rho_v)
            log_rho = np.log(np.maximum(density, 1.0e-300))
            for ie in range(energies.size):
                d = int(crossing[ie])
                if d < 2 or d >= y.size - 1:
                    continue
                if conversion == "none":
                    thickness = 30.0
                else:
                    dz = (y[d] - y[d - 1]) / max(float(density[d]), 1.0e-30)
                    scale = dz / max(float(log_rho[d] - log_rho[d - 1]), 1.0e-12)
                    for im, cosine in enumerate(mu):
                        tan_kb = np.sqrt(max(0.0, 1.0 - cosine ** 2)) / max(cosine, 0.02)
                        ion = CYCLOTRON_E_PER_GAUSS * field_g * MASS_RATIO
                        e_ad = 2.52 * (tan_kb * abs(1.0 - (ion / energies[ie]) ** 2)) \
                            ** (2.0 / 3.0) * max(scale / cosine, 1.0e-6) ** (-1.0 / 3.0)
                        p_jump = np.exp(-0.5 * np.pi
                                        * min((energies[ie] / max(e_ad, 1.0e-12)), 10.0) ** 3)
                        t_cross = min(30.0, -np.log(max(1.0 - p_jump, 1.0e-13)))
                        chi_ex = t_cross * cosine / max(float(y[d] - y[d - 1]), 1.0e-30)
                        for mode in (0, 1):
                            c = mode * mu.size + im
                            partner = (1 - mode) * mu.size + im
                            extinction[ie, c, d] += chi_ex
                            exchange[ie, c, partner, d] = chi_ex
                    continue
                # conversion == 'none': troca forte para os dois modos, todo mu
                for im, cosine in enumerate(mu):
                    chi_ex = 30.0 * cosine / max(float(y[d] - y[d - 1]), 1.0e-30)
                    for mode in (0, 1):
                        c = mode * mu.size + im
                        partner = (1 - mode) * mu.size + im
                        extinction[ie, c, d] += chi_ex
                        exchange[ie, c, partner, d] = chi_ex
            # normaliza o acoplamento pela extinção total (unidades de S)
            exchange = exchange / extinction[:, :, None, :]

        planck = estrutura.planck_energy(grid, temperature[None, :])
        thermal = absorption / extinction * planck[:, None, :] / 2.0

        # Acoplamento de posto 3, conservando fóton por construção.
        norm = np.einsum("c,ecad->ead", weight_channel, wide)
        turned = wide.transpose(0, 2, 1, 3)                     # (nE, 3, nC, nD)
        into = (turned * scattering_cyclic.transpose(0, 2, 1)[:, :, None, :]
                / (extinction[:, None, :, :] * norm[:, :, None, :]))
        out_of = weight_channel[None, None, :, None] * turned

        surface = None
        if surface_column is not None:
            # B na temperatura do GÁS do fundo (SPW09, eq. 15) — não num
            # estado separado. Com injeção escravizada a T[-1], o fluxo
            # profundo responde ao T local como na difusão do semi-infinito e
            # os déficits do Unsöld-Lucy fecham; com injeção independente eles
            # persistem e a integral de constância infla (medido: dJ/J até
            # ±300 nos nós do fundo com o equilíbrio local exato).
            surface = np.repeat(
                (estrutura.planck_energy(energies, temperature[-1]) / 2.0)[:, None],
                n_channel, axis=1)
        field = transporte.polarized_feautrier(optical_depth, mu_channel,
                                               weight_channel, thermal, into, out_of,
                                               surface_intensity=surface,
                                               exchange=exchange)

        mean_intensity = np.trapezoid(field["J"], energies, axis=0)
        flux_mid = np.trapezoid(field["H_mid"], energies, axis=0)
        flux_surface = float(np.trapezoid(field["H_surface"], energies))
        integrated_planck = np.trapezoid(planck, energies, axis=0)
        y_mid = 0.5 * (y[1:] + y[:-1])
        flux = np.interp(y, y_mid, flux_mid)
        flux[0] = flux_surface

        # Momentos com os pesos de canal; K/J para o fator de Eddington.
        second = np.trapezoid(np.einsum("c,ecd->ed", weight_channel * mu_channel ** 2,
                                        field["u"]), energies, axis=0)
        eddington = second / mean_intensity
        surface_factor = flux_surface / mean_intensity[0]

        channel_j = field["u"]
        absorbed = np.einsum("c,ecd->ed", weight_channel, absorption * channel_j)
        absorption_j = np.trapezoid(absorbed, energies, axis=0) / mean_intensity
        # Cada canal emite kappa_c B/2 — o meio é a intensidade de equilíbrio de
        # UM modo. Esquecê-lo conta a emissão em dobro, e aí o equilíbrio local
        # e a constância do fluxo puxam para lados opostos: medido, a iteração
        # estaciona com o fluxo na METADE do alvo.
        emitted = np.einsum("c,ecd->ed", weight_channel, absorption) / 2.0
        absorption_b = np.trapezoid(emitted * planck, energies, axis=0) / integrated_planck
        extinction_h = np.interp(
            y, y_mid,
            np.trapezoid(np.einsum("c,ecd->ed", weight_channel * mu_channel ** 2,
                                   0.5 * (extinction[:, :, 1:] + extinction[:, :, :-1])
                                   * np.diff(field["u"], axis=2)
                                   / np.diff(optical_depth, axis=2)),
                         energies, axis=0) / np.where(np.abs(flux_mid) > 0.0,
                                                      flux_mid, 1.0))

        deficit = target_flux - flux
        if surface_column is not None:
            # Onde o gás NÃO tem autoridade sobre o fluxo, o acelerador de
            # Unsöld-Lucy não entra. Na camada colada à fronteira emissora o
            # fluxo é da superfície (condutor de Newton), não do T local; a
            # integral de constância com opacidade enorme ali é windup de
            # integrador — medido, ciclo-limite em Sigma = 100 imune a
            # amortecimento, e REFINAR a grade só piorou (mais células sem
            # autoridade). O funil é o acoplamento radiativo à fronteira:
            # w = 1 - exp(-tau_h até o fundo), com tau_h da própria opacidade
            # pesada pelo fluxo. Longe do fundo, w = 1 e nada muda.
            # Em módulo: a opacidade pesada pelo fluxo troca de sinal quando
            # H cruza zero, e uma distância óptica negativa estoura o expm1
            # (NaN medido em Sigma = 100). Distância é comprimento, não saldo.
            reach_step = np.abs(0.5 * (extinction_h[1:] + extinction_h[:-1])
                                * np.diff(y))
            reach = np.concatenate(
                [np.cumsum(reach_step[::-1])[::-1], [0.0]])
            authority = -np.expm1(-np.minimum(reach, 700.0))
            deficit = deficit * authority
        integrand = extinction_h * deficit
        accumulated = np.concatenate(
            [[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(y))])
        delta_j = (eddington[0] * (target_flux - flux_surface) / surface_factor
                   + accumulated) / eddington
        source_function = estrutura.STEFAN * temperature ** 4 / np.pi
        delta_b = ((absorption_j * (mean_intensity + delta_j)
                    - absorption_b * integrated_planck)
                   / np.maximum(absorption_b, 1.0e-300))
        delta_b[0] = 0.0
        delta_t = delta_b * temperature / (4.0 * source_function)
        relaxation = np.clip(np.where(delta_t * previous_step < 0.0,
                                      0.5 * relaxation, 1.1 * relaxation), 0.05, 1.0)
        previous_step = delta_t
        delta_t = np.clip(relaxation * delta_t, -damping * temperature,
                          damping * temperature)
        if surface_column is not None:
            delta_t[-1] = 0.0    # o nó do fundo pertence à âncora de Newton
        temperature = np.maximum(temperature + delta_t, 0.05 * effective)
        temperature[0] = temperature[1]
        # Atmosfera fina: a superfície de baixo é quem carrega o fluxo, e a
        # correção de Unsöld-Lucy não a alcança quando a coluna é transparente —
        # não há opacidade por onde o déficit se propagar (medido: com
        # Sigma = 1e-4 o fluxo estaciona em 0,5 do alvo, que é exatamente
        # (0,84)^4 do palpite cinza). Então T do fundo é dirigida DIRETAMENTE
        # pelo fluxo emergente: sigma T_N^4 sobe ou desce pelo que falta.
        if surface_column is not None:
            # E só a cada 5 iterações: o Newton abaixo assume o gás congelado,
            # mas o gás do fundo reequilibra a cada passo e cancela parte da
            # resposta — o ping-pong resultante tem período > 2 e escapa da
            # relaxação por troca de sinal (medido em Sigma = 100, erro preso
            # em ~4 com tudo o mais convergindo). Separar as escalas de tempo
            # deixa o gás assentar entre passos da superfície.
            surface_turn = True
        if surface_column is not None:
            flux_bottom = float(flux[-1])
        if surface_column is not None and surface_turn:
            # O observável do condutor é o fluxo líquido NO FUNDO, não no topo.
            # Mirando o topo, o calor que o gás despeja para baixo some num
            # sorvedouro (a superfície absorve sem reemitir) e o Unsöld-Lucy
            # compensa aquecendo o interior sem freio — medido, erro de fluxo
            # até 72 no regime intermediário. Ancorado no fundo, a malha é
            # conservativa: entrada = alvo, equilíbrio radiativo no gás, e a
            # saída no topo converge ao alvo por conservação. Nos limites, o
            # fundo coincide com o topo (transparente) ou com a luminosidade
            # interior (espessa) — o mesmo observável serve aos três regimes.
            # E com a MESMA relaxação adaptativa do Unsöld-Lucy: sem memória,
            # o ganho do laço superfície-interior passa de 1 em Sigma ~ 100 e a
            # iteração entra em ciclo-limite (medido: erro de fluxo quicando
            # entre 0,8 e 20 por 1500 iterações). Troca de sinal corta o passo
            # pela metade; persistência o deixa crescer de volta.
            # Passo de Newton com a derivada física dF/dT_sup ~ sigma T_sup^3:
            # quando o fundo é opticamente espesso, a injeção líquida responde a
            # T_sup como contato térmico, ~30x mais ríspida em T_sup ~ 3 T_ef do
            # que o passo multiplicativo assumia — 1% em T_sup arremessava o
            # fluxo do fundo por múltiplos do alvo (ciclo-limite medido em
            # Sigma = 100). O Newton encolhe sozinho conforme T_sup sobe.
            derivative = estrutura.STEFAN * temperature[-1] ** 3 / np.pi
            newton_step = (target_flux - flux_bottom) / derivative
            if newton_step * surface_previous < 0.0:
                surface_relaxation = max(0.05, 0.5 * surface_relaxation)
            else:
                surface_relaxation = min(1.0, 1.1 * surface_relaxation)
            surface_previous = newton_step
            newton_step = float(np.clip(surface_relaxation * newton_step,
                                        -0.1 * temperature[-1],
                                        0.12 * temperature[-1]))
            temperature[-1] = max(temperature[-1] + newton_step,
                                  0.05 * effective)

        change = float(np.max(np.abs(delta_t[1:]) / temperature[1:]))
        if trace is not None:
            j = 1 + int(np.argmax(np.abs(delta_t[1:]) / temperature[1:]))
            trace.append((j, float(y[j]), float(temperature[j] / effective),
                          float(temperature[-1] / effective),
                          float(delta_j[j] / max(mean_intensity[j], 1e-300)),
                          float(absorption_j[j] * mean_intensity[j]
                                / max(absorption_b[j] * integrated_planck[j], 1e-300)),
                          float(flux[j] / target_flux)))
        if surface_column is not None:
            # O erro conta onde há autoridade, mais o desvio da âncora do fundo
            # (no transparente, w ~ 0 em toda parte e a âncora é o que resta).
            flux_error = float(max(
                np.max(np.abs(flux - target_flux) * authority),
                abs(target_flux - flux_bottom)) / target_flux)
        else:
            flux_error = float(np.max(np.abs(flux - target_flux)) / target_flux)
        history.append((change, flux_error))
        if change < tolerance and flux_error < 1.0e-3:
            break

    # A função fonte por canal, reconstruída do estado convergido: térmica mais
    # espalhada. É o que o feixe resolvido em phi consome na solução formal.
    gathered = np.einsum("eacd,ecd->ead", out_of, np.maximum(field["u"], 0.0))
    source_channel = thermal + np.einsum("eacd,ead->ecd", into, gathered)
    return {
        "energies": energies, "columns": y, "mu": mu, "mu_channel": mu_channel,
        "weight_channel": weight_channel, "temperature": temperature,
        "density": density, "intensity": field["I_surface"],
        "intensity_modes": (field["I_surface"][:, :mu.size],
                            field["I_surface"][:, mu.size:]),
        "source_channel": source_channel,
        "theta_b": theta_b, "field_g": field_g, "vacuum": vacuum,
        "flux_energy": 4.0 * np.pi * field["H_surface"],
        "flux_error": history[-1][1], "iterations": len(history), "history": history,
        "flux_depth": flux, "surface_temperature": temperature[-1],
    }


def phi_resolved_intensity(solution: dict, phi: np.ndarray) -> np.ndarray:
    """I(E, mu, phi) emergente, com a atenuação exata e a fonte phi-média.

    O item que faltava do feixe: o transporte resolve a ESTRUTURA com opacidades
    mediadas em phi (aproximação declarada em `channel_geometry`), mas o feixe
    que sai não precisa herdar a média — a solução formal ao longo de cada raio
    (mu, phi) usa a opacidade EXATA daquele azimute,

        I_j(0; mu, phi) = int S_j(tau') e^-tau' dtau',
        dtau' = chi_j(theta(mu, phi)) dy / mu,

    com S_j por canal vinda do estado convergido. O que continua phi-médio é a
    fonte; a atenuação e a geometria dos modos são as do raio. Devolve a soma
    dos dois modos, forma (n_E, n_mu, n_phi).
    """
    energies = solution["energies"]
    y = solution["columns"]
    mu = solution["mu"]
    density = solution["density"]
    temperature = solution["temperature"]
    field_g = solution["field_g"]
    theta_b = solution["theta_b"]
    vacuum = solution["vacuum"]
    source = solution["source_channel"]
    n_mu = mu.size

    absorption_cyclic = cyclic_free_free(energies[:, None], density[None, :],
                                         temperature[None, :], field_g)
    scattering_cyclic = cyclic_scattering(energies[:, None], field_g,
                                          density[None, :], temperature[None, :])
    cyclic = absorption_cyclic + scattering_cyclic                  # (nE, nD, 3)

    result = np.zeros((energies.size, n_mu, phi.size))
    sin_b, cos_b = np.sin(theta_b), np.cos(theta_b)
    for im, cosine in enumerate(mu):
        sine = np.sqrt(max(0.0, 1.0 - cosine ** 2))
        for ip, azimuth in enumerate(phi):
            ray = np.arccos(np.clip(cosine * cos_b
                                    + sine * sin_b * np.cos(azimuth), -1.0, 1.0))
            for mode in (0, 1):
                # Amplitudes do raio, por profundidade se o vácuo estiver
                # ligado; uma vez só se não estiver.
                if vacuum:
                    extinction = np.zeros((energies.size, y.size))
                    for depth in range(y.size):
                        amp = mode_amplitudes(energies, float(ray),
                                              max(float(density[depth]), 1.0e-30),
                                              field_g, vacuum=True)[:, mode]
                        extinction[:, depth] = np.einsum(
                            "ea,ea->e", amp, cyclic[:, depth])
                else:
                    amp = mode_amplitudes(energies, float(ray), 1.0e-2,
                                          field_g)[:, mode]
                    extinction = np.einsum("ea,eda->ed", amp, cyclic)
                slant = np.concatenate(
                    [np.zeros((energies.size, 1)),
                     np.cumsum(np.maximum(0.5 * (extinction[:, 1:] + extinction[:, :-1])
                                          * np.diff(y), 1.0e-10) / cosine, axis=1)],
                    axis=1)
                weight = np.exp(-slant)
                channel = source[:, mode * n_mu + im, :]
                result[:, im, ip] += np.sum(
                    0.5 * (channel[:, 1:] * weight[:, 1:]
                           + channel[:, :-1] * weight[:, :-1])
                    * np.diff(slant, axis=1), axis=1)
    return result
