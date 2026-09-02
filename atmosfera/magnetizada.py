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


def mode_amplitudes(energy_kev: np.ndarray, theta_b: float, density: float,
                    field_g: float) -> np.ndarray:
    """|e_alpha^j|^2 dos dois modos: forma (n_E, 2 modos, 3 componentes).

    Resolve o autoproblema generalizado eps E = n^2 (I - kk) E por energia. A
    matriz (I - kk) tem posto 2, então saem dois autovalores finitos — os modos
    eletromagnéticos — e um infinito, o longitudinal, que é descartado. Os
    autovetores voltam projetados na base cíclica (e_+, e_-, e_z) em relação a
    B, normalizados a 1.

    Os modos saem ordenados por |n^2| DECRESCENTE, o que abaixo do cíclotron do
    elétron põe o ordinário antes do extraordinário. A ordenação é convenção de
    saída; a física está nas amplitudes.
    """
    plus, minus, along = dielectric_cyclic(energy_kev, density, field_g)
    sin_t, cos_t = np.sin(theta_b), np.cos(theta_b)
    direction = np.array([sin_t, 0.0, cos_t])
    projector = np.eye(3) - np.outer(direction, direction)

    # Base cíclica -> cartesiana: eps_xx = (eps_+ + eps_-)/2 etc.
    result = np.zeros((len(np.atleast_1d(energy_kev)), 2, 3))
    from scipy.linalg import eig
    for index in range(result.shape[0]):
        p, m, z = plus[index], minus[index], along[index]
        tensor = np.array([[(p + m) / 2.0, -1j * (p - m) / 2.0, 0.0],
                           [1j * (p - m) / 2.0, (p + m) / 2.0, 0.0],
                           [0.0, 0.0, z]])
        values, vectors = eig(tensor, projector)
        finite = np.argsort(np.where(np.isfinite(values), -np.abs(values), np.inf))[:2]
        for slot, which in enumerate(finite):
            e = vectors[:, which]
            e = e / np.linalg.norm(e)
            cyclic = np.array([(e[0] + 1j * e[1]) / np.sqrt(2.0),
                               (e[0] - 1j * e[1]) / np.sqrt(2.0), e[2]])
            result[index, slot] = np.abs(cyclic) ** 2
    return result


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
                   temperature: float, field_g: float) -> dict:
    """kappa_j(E) dos dois modos, absorção e espalhamento separados.

    A montagem final: amplitudes cíclicas do autoproblema vezes as opacidades
    cíclicas. Devolve também as amplitudes, porque o traçado de raios do futuro
    vai querer a polarização e não só a opacidade.
    """
    amplitudes = mode_amplitudes(energy_kev, theta_b, density, field_g)
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
                        temperature: float, field_g: float) -> float:
    """K(theta) em cm^2/g, com os dois modos conduzindo o fluxo em paralelo.

        1/K = < (1/2)(1/chi_1 + 1/chi_2) >_Rosseland

    — a definição que a auditoria 3 verificou contra o piso do Potekhin: com
    média aritmética o piso sairia sete ordens de grandeza acima do tabelado.
    """
    from .estrutura import planck_temperature_derivative
    modes = mode_opacities(energy_kev, theta_b, density, temperature, field_g)
    total = modes["scattering"] + modes["absorption"]
    weight = planck_temperature_derivative(np.asarray(energy_kev, float),
                                           np.full(len(np.atleast_1d(energy_kev)),
                                                   temperature))
    inverse = 0.5 * (1.0 / total[:, 0] + 1.0 / total[:, 1])
    numerator = np.trapezoid(weight * inverse, energy_kev)
    denominator = np.trapezoid(weight, energy_kev)
    return float(denominator / numerator)


def rosseland_tensor(energy_kev: np.ndarray, density: float, temperature: float,
                     field_g: float, angle_nodes: int = 12) -> tuple[float, float]:
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
                               temperature, field_g)
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
          damping: float = 0.25) -> dict:
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
    from . import transporte, estrutura
    energies = estrutura.energy_grid(1.0e-3, 60.0, 220) if energies is None else energies
    y = estrutura.column_grid() if columns is None else columns
    mu, weights = transporte.gauss_legendre_mu(mu_nodes)
    geometry = channel_geometry(energies, mu, field_g, theta_b=theta_b)
    amplitudes = geometry["amplitudes"]                     # (nE, nC, 3)
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

    for step in range(iterations):
        density = pressure * estrutura.PROTON_MASS / (2.0 * estrutura.BOLTZMANN * temperature)
        absorption_cyclic = cyclic_free_free(grid, density[None, :],
                                             temperature[None, :], field_g)  # (nE,nD,3)
        # O espalhamento agora também é por profundidade: a ressonância do
        # próton carrega as larguras colisional e Doppler locais.
        scattering_cyclic = cyclic_scattering(grid, field_g, density[None, :],
                                              temperature[None, :])          # (nE,nD,3)
        # Por canal: absorção e espalhamento projetados nas amplitudes.
        absorption = np.einsum("eca,eda->ecd", amplitudes, absorption_cyclic)
        scattering = np.einsum("eca,eda->ecd", amplitudes, scattering_cyclic)
        extinction = absorption + scattering
        optical_depth = np.concatenate(
            [np.zeros((energies.size, n_channel, 1)),
             np.cumsum(0.5 * (extinction[:, :, 1:] + extinction[:, :, :-1])
                       * np.diff(y), axis=2)], axis=2)
        planck = estrutura.planck_energy(grid, temperature[None, :])
        thermal = absorption / extinction * planck[:, None, :] / 2.0

        # Acoplamento de posto 3, conservando fóton por construção.
        norm = np.einsum("c,eca->ea", weight_channel, amplitudes)
        into = (amplitudes.transpose(0, 2, 1)[:, :, :, None]
                * scattering_cyclic.transpose(0, 2, 1)[:, :, None, :]
                / (extinction[:, None, :, :] * norm[:, :, None, None]))
        out_of = (weight_channel[None, None, :, None]
                  * amplitudes.transpose(0, 2, 1)[:, :, :, None]
                  * np.ones((1, 1, 1, y.size)))

        field = transporte.polarized_feautrier(optical_depth, mu_channel,
                                               weight_channel, thermal, into, out_of)

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
        temperature = np.maximum(temperature + delta_t, 0.05 * effective)
        temperature[0] = temperature[1]

        change = float(np.max(np.abs(delta_t[1:]) / temperature[1:]))
        flux_error = float(np.max(np.abs(flux - target_flux)) / target_flux)
        history.append((change, flux_error))
        if change < tolerance and flux_error < 1.0e-3:
            break

    return {
        "energies": energies, "columns": y, "mu": mu, "mu_channel": mu_channel,
        "weight_channel": weight_channel, "temperature": temperature,
        "density": density, "intensity": field["I_surface"],
        "flux_energy": 4.0 * np.pi * field["H_surface"],
        "flux_error": history[-1][1], "iterations": len(history), "history": history,
    }
