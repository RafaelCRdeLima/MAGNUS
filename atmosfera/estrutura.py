"""Atmosfera plano-paralela sem campo, hidrogênio totalmente ionizado.

O estágio 1 do PLANO.md. O objetivo não é o resultado — é validar a máquina
numérica inteira num regime em que ela não pode se esconder atrás da
complicação, e onde há gabarito: as tabelas `nsx` de B = 0 do Ho, com 25
temperaturas e 15 gravidades.

**A física, e o que ela deliberadamente não tem.** Hidrogênio totalmente
ionizado, gás ideal, equilíbrio hidrostático com P = g y — a pressão de radiação
é desprezível a estas luminosidades, seis ordens de grandeza abaixo de
Eddington. Opacidade: livre-livre com fator de Gaunt na aproximação de Born,
mais espalhamento de Thomson coerente. NÃO há ionização parcial, ligado-livre,
ligado-ligado, condução, nem Comptonização — a última é a que mais deve pesar na
ponta quente da grade, e o portão vai dizer quanto.

**A estrutura de temperatura** sai por correção de Unsöld–Lucy, que junta duas
condições: o equilíbrio radiativo local, `int kappa (J - B) dnu = 0`, e a
constância do fluxo, que é a integral. Só a primeira converge devagar demais em
profundidade, onde J e B diferem por muito pouco e o erro de fluxo não aparece.

**O transporte** é o Feautrier em bloco de `transporte.coupled_feautrier`: o
espalhamento coerente resolvido de uma vez, sem iteração lambda. Ver lá por quê.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import k0e

from . import transporte

#: cgs, e nada de unidade escondida.
BOLTZMANN = 1.380649e-16                     # erg/K
PLANCK = 6.62607015e-27                      # erg s
LIGHT = 2.99792458e10                        # cm/s
PROTON_MASS = 1.67262192369e-24              # g
STEFAN = 5.670374419e-5                      # erg cm^-2 s^-1 K^-4
ERG_PER_KEV = 1.602176634e-9
ELECTRON_REST = 8.1871057769e-7              # m_e c^2, erg
#: Fração da intensidade local que a correção de Compton pode mover numa
#: passada. Ver o comentário em `solve`: é um limitador de estabilidade, não
#: física, e a dívida que ele representa está declarada lá.
COMPTON_LIMIT = 0.3
#: Espalhamento de Thomson por grama de hidrogênio totalmente ionizado.
THOMSON_CM2_G = 0.6652458e-24 / PROTON_MASS
#: A constante de Rybicki & Lightman para o livre-livre, em cgs.
FREE_FREE = 3.7e8


def planck_energy(energy_kev: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """B_E em erg cm^-2 s^-1 sr^-1 keV^-1, com int B_E dE = sigma T^4 / pi."""
    x = np.clip(ERG_PER_KEV * energy_kev / (BOLTZMANN * temperature), 1.0e-12, 700.0)
    energy_erg = energy_kev * ERG_PER_KEV
    return (2.0 * energy_erg ** 3 * ERG_PER_KEV
            / (PLANCK ** 3 * LIGHT ** 2 * np.expm1(x)))


def planck_temperature_derivative(energy_kev: np.ndarray,
                                  temperature: np.ndarray) -> np.ndarray:
    """dB_E/dT, o peso da média de Rosseland."""
    x = np.clip(ERG_PER_KEV * energy_kev / (BOLTZMANN * temperature), 1.0e-12, 700.0)
    # x e^x/(e^x - 1) escrito como x/(1 - e^-x): a primeira forma transborda na
    # cauda de Wien, onde e^x passa de 1e300 antes de a divisão desfazer isso.
    return planck_energy(energy_kev, temperature) * x / (-np.expm1(-x)) / temperature


def _elwert_born_gaunt(u: np.ndarray, gamma_squared: np.ndarray,
                       nodes: int = 96) -> np.ndarray:
    """Fator de Gaunt térmico de Born com correção de Elwert.

    Para UM elétron de energia inicial E_i emitindo hnu, o Born não
    relativístico dá

        g(x, u) = (sqrt(3)/pi) ln[(sqrt(x) + sqrt(x-u)) / (sqrt(x) - sqrt(x-u))]

    com x = E_i/kT. A correção de Elwert repõe o foco coulombiano que o Born
    ignora, pelo parâmetro de Sommerfeld eta = Z sqrt(Ry/E):

        f = (eta_f/eta_i) [1 - exp(-2 pi eta_i)] / [1 - exp(-2 pi eta_f)]

    e a média térmica é a integral em exp(-y) dy sobre y = (E_i - hnu)/kT, feita
    por Gauss-Laguerre, que é a quadratura com exatamente esse peso.

    **Por que não bastava o Born puro.** Ele não depende de gamma^2 = Ry/kT, e
    portanto tem a FORMA em frequência errada: em u pequeno o valor verdadeiro
    cresce como ln(1/(gamma u)) e o de Born só como ln(1/u). Medido em
    gamma^2 = 0,18 e u = 1, a diferença é de 0,84 para cerca de 1,2.

    E a forma é o que importa, não a escala: o resfriamento da superfície é
    governado pela RAZÃO kappa_J/kappa_B, e um fator constante multiplicando a
    opacidade cancela nela exatamente. Foi o que a auditoria mostrou — escalar o
    livre-livre por até 4 não mexeu no espectro emergente pelo lado que
    interessava. O que mexe é a dependência em frequência.
    """
    y, weight = np.polynomial.laguerre.laggauss(nodes)
    initial = np.sqrt(y[None, :] + np.asarray(u)[..., None])
    final = np.sqrt(y)[None, :]
    born = (np.sqrt(3.0) / np.pi) * np.log((initial + final) / (initial - final))
    gamma = np.sqrt(np.asarray(gamma_squared))[..., None]
    eta_initial, eta_final = gamma / initial, gamma / final
    elwert = (eta_final / eta_initial) * (np.expm1(-2.0 * np.pi * eta_initial)
                                          / np.expm1(-2.0 * np.pi * eta_final))
    return (born * elwert) @ weight


#: Ry/k, em kelvin: gamma^2 = RYDBERG_K / T. Vale 1,579e5 K.
RYDBERG_K = 13.605693122994e-3 * ERG_PER_KEV / BOLTZMANN


def _gaunt_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A média térmica, tabelada uma vez em (lg u, lg gamma^2) e interpolada.

    Refazer a quadratura de Laguerre a cada iteração de temperatura, em duzentas
    frequências vezes cento e vinte profundidades, seria repetir cem vezes um
    cálculo que depende só de duas variáveis. A grade cobre com folga tudo o que
    a banda e a grade de temperatura pedem.
    """
    log_u = np.linspace(-6.0, 4.0, 201)
    log_gamma = np.linspace(-5.0, 1.5, 79)
    grid_u, grid_gamma = np.meshgrid(10.0 ** log_u, 10.0 ** log_gamma, indexing="ij")
    return log_u, log_gamma, _elwert_born_gaunt(grid_u, grid_gamma)


_GAUNT_LOG_U, _GAUNT_LOG_GAMMA, _GAUNT_VALUES = _gaunt_table()


def _van_hoof_table():
    """A tabela exata de van Hoof et al. (2014), se estiver no disco.

    O portão do estágio 1 mediu o resíduo de +4–5% em temperatura de cor como
    degenerado com a INCLINAÇÃO do Gaunt em frequência — e inclinação é
    exatamente o que separa a aproximação de Elwert–Born da resposta exata.
    Ver atmosphere_data/van_hoof/PROVENIENCIA.json. Sem o arquivo, o código cai
    de volta no Elwert–Born e continua funcionando — mais um grau aproximado,
    como era.
    """
    path = Path(__file__).resolve().parents[1] / "atmosphere_data" / "van_hoof" / "gauntff.dat"
    if not path.is_file():
        return None
    numbers, header = [], []
    for line in path.read_text().splitlines():
        bare = line.split("#")[0].strip()
        if not bare:
            continue
        if len(header) < 5:
            header.append(float(bare.split()[0]))
            if len(header) == 2:
                header.append(float(bare.split()[1]) if len(bare.split()) > 1 else None)
            continue
        numbers.extend(float(v) for v in bare.split())
    # Cabeçalho: magia, (n_gam2 n_u), inicio lg gam2, inicio lg u, passo.
    n_gamma, n_u = 81, 146
    start_gamma, start_u, step = -6.0, -16.0, 0.2
    values = np.array(numbers[:n_gamma * n_u]).reshape(n_u, n_gamma)
    return (start_u + step * np.arange(n_u),
            start_gamma + step * np.arange(n_gamma), values)


_VAN_HOOF = _van_hoof_table()


def gaunt_free_free(energy_kev: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """Fator de Gaunt livre-livre térmico: van Hoof exato, Elwert-Born reserva."""
    log_u = np.log10(np.clip(ERG_PER_KEV * energy_kev / (BOLTZMANN * temperature),
                             1.0e-15, 1.0e12))
    log_gamma = np.log10(np.clip(RYDBERG_K / temperature, 1.0e-5, 1.0e9))
    if _VAN_HOOF is not None:
        axis_u, axis_g, table = _VAN_HOOF
        iu = np.clip(np.searchsorted(axis_u, log_u) - 1, 0, axis_u.size - 2)
        ig = np.clip(np.searchsorted(axis_g, log_gamma) - 1, 0, axis_g.size - 2)
        fu = np.clip((log_u - axis_u[iu]) / 0.2, 0.0, 1.0)
        fg = np.clip((log_gamma - axis_g[ig]) / 0.2, 0.0, 1.0)
        return ((1 - fu) * (1 - fg) * table[iu, ig]
                + fu * (1 - fg) * table[iu + 1, ig]
                + (1 - fu) * fg * table[iu, ig + 1]
                + fu * fg * table[iu + 1, ig + 1])
    u = np.clip(log_u, -6.0, 4.0)
    gamma = np.clip(log_gamma, -5.0, 1.5)
    iu = np.clip(np.searchsorted(_GAUNT_LOG_U, u) - 1, 0, len(_GAUNT_LOG_U) - 2)
    ig = np.clip(np.searchsorted(_GAUNT_LOG_GAMMA, gamma) - 1, 0,
                 len(_GAUNT_LOG_GAMMA) - 2)
    fu = (u - _GAUNT_LOG_U[iu]) / (_GAUNT_LOG_U[iu + 1] - _GAUNT_LOG_U[iu])
    fg = (gamma - _GAUNT_LOG_GAMMA[ig]) / (_GAUNT_LOG_GAMMA[ig + 1] - _GAUNT_LOG_GAMMA[ig])
    return ((1 - fu) * (1 - fg) * _GAUNT_VALUES[iu, ig]
            + fu * (1 - fg) * _GAUNT_VALUES[iu + 1, ig]
            + (1 - fu) * fg * _GAUNT_VALUES[iu, ig + 1]
            + fu * fg * _GAUNT_VALUES[iu + 1, ig + 1])


def opacities(energy_kev: np.ndarray, density: np.ndarray,
              temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(absorção verdadeira, espalhamento) em cm^2/g.

    O livre-livre já traz o fator de emissão estimulada (1 - e^-u), então é a
    absorção LÍQUIDA — que é a que entra em epsilon e na função fonte.
    """
    frequency = ERG_PER_KEV * energy_kev / PLANCK
    u = ERG_PER_KEV * energy_kev / (BOLTZMANN * temperature)
    number = density / PROTON_MASS
    absorption = (FREE_FREE * temperature ** -0.5 * number * number
                  * frequency ** -3.0 * (-np.expm1(-u))
                  * gaunt_free_free(energy_kev, temperature) / np.maximum(density, 1.0e-300))
    return absorption, np.full_like(absorption, THOMSON_CM2_G)


#: n = OCCUPATION * J_E / E^3 devolve o número de ocupação de fótons, com J_E em
#: erg cm^-2 s^-1 sr^-1 keV^-1 e E em keV. Confere: para B_E de Planck dá
#: exatamente 1/(exp(x) - 1).
OCCUPATION = LIGHT ** 2 * PLANCK ** 3 / (2.0 * ERG_PER_KEV ** 4)


def compton_correction(energy_kev: np.ndarray, mean_intensity: np.ndarray,
                       temperature: np.ndarray) -> np.ndarray:
    """S_espalhada - J na aproximação de Kompaneets. Forma (n_E, n_prof).

    **É a física que o portão do estágio 1 achou faltando.** O espalhamento
    coerente conserva a energia de cada fóton; o de verdade não. O elétron
    térmico devolve o fóton com energia deslocada, e o efeito acumulado ao longo
    de mil espalhamentos na cauda dura é o que limita o endurecimento — sem ele
    o fator de cor sai alto, e sai mais alto quanto mais quente a atmosfera, que
    é exatamente o que se mediu: 0,1% de erro em lg T = 5,7 e 7,6% em 6,4.

    Da equação de Kompaneets para o número de ocupação n(x), com
    x = hnu/kT_e e theta = kT_e/(m_e c^2):

        S_esp - J = (E_T^3 / OCCUPATION) theta (1/x^3) d/d(ln E) [ G ]
        G = x^4 (n + n^2 + dn/dx)

    **O termo n^2 não é opcional**, embora o espalhamento induzido seja
    desprezível em número: sem ele o operador NÃO se anula para um Planck na
    própria temperatura do elétron — some 1/(e^x-1) + 1/(e^x-1)^2 e a derivada
    dão zero, e as duas primeiras parcelas sozinhas não. Fora do equilíbrio isso
    seria uma fonte espúria, e em x pequeno, onde n passa de mil, seria a
    parcela dominante dela.

    O operador é uma divergência em x, então conserva NÚMERO de fótons desde que
    G se anule nas duas bordas da grade. Em x grande n cai exponencialmente; em
    x pequeno n ~ 1/x e x^4 n^2 ~ x^2. As duas se anulam, e é por isso que a
    grade em energia tem de ser larga: truncá-la cedo cria ou destrói fótons.

    Entra no transporte como fonte conhecida, avaliada na iteração anterior — o
    operador acopla frequências e não cabe no bloco angular, que é resolvido de
    uma vez.
    """
    thermal_kev = BOLTZMANN * temperature / ERG_PER_KEV
    theta = BOLTZMANN * temperature / ELECTRON_REST
    occupation = np.maximum(OCCUPATION * mean_intensity / (energy_kev ** 3)[:, None],
                            1.0e-300)
    log_energy = np.log(energy_kev)
    step = np.diff(log_energy)[:, None]

    # n + n^2 + dn/dx reescrito como n(1+n)[1 + d ln(n/(1+n))/dx]. As duas formas
    # são a mesma álgebra, mas a segunda anula EXATAMENTE num Planck a T_e: ali
    # ln(n/(1+n)) = -x, a derivada vale -1 e o colchete zera, sem depender de as
    # parcelas se cancelarem numericamente. Na forma direta elas se cancelam a
    # 1/x^2, e em x = 0,01 isso são quatro dígitos perdidos — uma fonte espúria
    # de fótons justamente onde o número deles é maior.
    potential = np.log(occupation) - np.log1p(occupation)

    # Volume finito: o fluxo é avaliado nas INTERFACES entre células e a fonte é
    # a diferença de fluxos. Assim o operador é uma divergência exata na grade
    # discreta, e não só no papel — o número de fótons se conserva até o último
    # dígito. Com o fluxo posto a zero nas duas bordas, nada entra nem sai pelas
    # pontas, que é a condição de contorno certa e também a que impede a
    # derivada de um só lado de inventar uma fonte enorme no primeiro ponto.
    middle = np.exp(0.5 * (log_energy[1:] + log_energy[:-1]))[:, None] / thermal_kev[None, :]
    face = 0.5 * (occupation[1:] + occupation[:-1])
    bracket = 1.0 + np.diff(potential, axis=0) / step / middle
    flow = middle ** 4 * face * (1.0 + face) * bracket
    divergence = np.zeros_like(occupation)
    width = np.gradient(log_energy)[:, None]
    divergence[1:-1] = (flow[1:] - flow[:-1]) / width[1:-1]
    divergence[0] = flow[0] / width[0]
    divergence[-1] = -flow[-1] / width[-1]
    correction = (thermal_kev ** 3 / OCCUPATION)[None, :] * theta[None, :] * divergence

    # **Onde não há fóton, não há correção.** Na ponta extrema de Wien — x = 100,
    # que a grade alcança nas camadas fundas — o número de ocupação vale 1e-44 e
    # a correção RELATIVA explode, embora a absoluta seja nada. Deixada solta,
    # ela leva a fonte a negativo e a iteração inteira junto: medido, J negativo
    # na segunda passada. O corte é em n, e não em energia, porque é o número de
    # fótons que decide se a região importa. O erro de conservação que ele
    # introduz é da ordem do que foi cortado, ou seja 1e-25 do total.
    return np.where(occupation > 1.0e-25, correction, 0.0)


def kompaneets_bands(energy_kev: np.ndarray, mean_intensity: np.ndarray,
                     temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """O operador de Kompaneets como coeficientes tridiagonais em frequência.

    É a forma que a linearização conjunta consome: dC_k = sub_k J_{k-1} +
    dia_k J_k + sup_k J_{k+1}, com cada coeficiente dependendo da profundidade.
    Duas escolhas fazem dele um operador decente e não só uma discretização:

    **Chang-Cooper nas interfaces.** O fluxo em x é G = x^4 [A n + dn/dx], com
    A = 1 + n_médio carregando o termo induzido (atrasado da iteração anterior,
    que é o que o torna linear). O valor de n na interface entra pesado por
    delta = 1/w - 1/(e^w - 1), com w = A vezes o passo em x — o peso exato que
    faz o fluxo discreto se anular sobre a exponencial de equilíbrio. Sem isso,
    o equilíbrio só se anula no contínuo, e a diferença vira fonte espúria.

    **Volume finito com fluxo nulo nas bordas.** O operador é uma divergência na
    grade discreta, então conserva número de fótons por construção, e nada entra
    nem sai pelas pontas da grade de energia.
    """
    thermal_kev = BOLTZMANN * temperature / ERG_PER_KEV            # (n_prof,)
    theta = BOLTZMANN * temperature / ELECTRON_REST
    occupation = np.maximum(OCCUPATION * mean_intensity / (energy_kev ** 3)[:, None],
                            0.0)
    log_energy = np.log(energy_kev)
    width = np.gradient(log_energy)[:, None]                       # (n_freq, 1)
    du = np.diff(log_energy)[:, None]

    face_x = (np.exp(0.5 * (log_energy[1:] + log_energy[:-1]))[:, None]
              / thermal_kev[None, :])                              # (n_freq-1, n_prof)
    induced = 1.0 + 0.5 * (occupation[1:] + occupation[:-1])
    step_x = face_x * du
    drift = induced * step_x
    # delta -> 1/2 quando o passo é pequeno; a série evita 0/0 ali.
    small = drift < 1.0e-6
    delta = np.where(small, 0.5 - drift / 12.0,
                     1.0 / np.maximum(drift, 1.0e-300)
                     - 1.0 / np.maximum(np.expm1(drift), 1.0e-300))

    body = face_x ** 4
    to_lower = body * (induced * delta - 1.0 / step_x)             # coef. de n_i
    to_upper = body * (induced * (1.0 - delta) + 1.0 / step_x)     # coef. de n_{i+1}

    # Converte n -> J e monta a divergência. O fator por célula é
    # E_T^3 theta / (largura da célula), e cada n_k traz OCC/E_k^3 — que junto
    # com o E_T^3/OCC do operador dá E_T^3 theta / E_k^3.
    scale = (thermal_kev ** 3 * theta)[None, :]
    cube = (energy_kev ** 3)[:, None]
    sub = np.zeros_like(mean_intensity)
    dia = np.zeros_like(mean_intensity)
    sup = np.zeros_like(mean_intensity)
    # +G_{i+1/2} para i = 0..n-2
    dia[:-1] += scale / width[:-1] * to_lower / cube[:-1]
    sup[:-1] += scale / width[:-1] * to_upper / cube[1:]
    # -G_{i-1/2} para i = 1..n-1
    dia[1:] -= scale / width[1:] * to_upper / cube[1:]
    sub[1:] -= scale / width[1:] * to_lower / cube[:-1]
    return sub, dia, sup


def apply_kompaneets(coupling: tuple, mean_intensity: np.ndarray) -> np.ndarray:
    """dC = K J, com os coeficientes tridiagonais de `kompaneets_bands`."""
    sub, dia, sup = coupling
    result = dia * mean_intensity
    result[1:] += sub[1:] * mean_intensity[:-1]
    result[:-1] += sup[:-1] * mean_intensity[1:]
    return result


def compton_diagonal(energy_kev: np.ndarray, mean_intensity: np.ndarray,
                     temperature: np.ndarray, relative: float = 1.0e-5) -> np.ndarray:
    """d(S_esp - J)/dJ na própria frequência: a parte implícita do operador.

    O operador de Kompaneets discretizado é TRIDIAGONAL em frequência — cada
    célula fala só com as vizinhas. Então três perturbações coloridas, nos
    índices 0, 1 e 2 módulo 3, dão a diagonal EXATA sem que uma perturbação
    contamine a outra, e a três avaliações do operador em vez de duzentas.

    A diagonal é da ordem de theta/(passo em ln E)^2 e portanto de ordem 1: é
    exatamente essa a rigidez que derruba o tratamento explícito. Passada ao
    albedo, ela vira parte do sistema resolvido de uma vez.
    """
    base = compton_correction(energy_kev, mean_intensity, temperature)
    scale = np.maximum(mean_intensity, 1.0e-30 * np.max(mean_intensity))
    diagonal = np.zeros_like(mean_intensity)
    for colour in range(3):
        index = np.arange(colour, energy_kev.size, 3)
        perturbed = mean_intensity.copy()
        step = relative * scale[index]
        perturbed[index] = perturbed[index] + step
        moved = compton_correction(energy_kev, perturbed, temperature)
        diagonal[index] = (moved[index] - base[index]) / step
    return diagonal


def rosseland_mean(energy_kev: np.ndarray, extinction: np.ndarray,
                   temperature: np.ndarray) -> np.ndarray:
    """1/K = <1/chi> com o peso dB/dT, por profundidade."""
    weight = planck_temperature_derivative(energy_kev[None, :], temperature[:, None])
    numerator = np.trapezoid(weight / extinction.T, energy_kev, axis=1)
    denominator = np.trapezoid(weight, energy_kev, axis=1)
    return denominator / numerator


def energy_grid(minimum: float = 1.0e-3, maximum: float = 60.0,
                points: int = 200) -> np.ndarray:
    """Log em energia. O plano nomeia a grade como armadilha; sem campo não há
    ressonância estreita para resolver, e o que a grade tem de conter é o corpo
    negro inteiro — do Rayleigh-Jeans frio à cauda de Wien das camadas fundas,
    que são dez vezes mais quentes que T_ef."""
    return np.logspace(np.log10(minimum), np.log10(maximum), points)


def column_grid(minimum: float = 1.0e-6, maximum: float = 1.0e5,
                points: int = 121) -> np.ndarray:
    """Densidade de coluna y em g/cm^2, com o zero na frente.

    O zero é necessário e não é enfeite: a condição de contorno de cima do
    Feautrier é imposta em tau = 0, e ali a densidade é nula, o livre-livre
    desaparece e sobra só espalhamento — que é exatamente o vácuo por cima da
    atmosfera. O fundo em 1e5 g/cm^2 dá tau_espalhamento de 4e4, e mesmo na
    energia mais TRANSPARENTE da banda a profundidade efetiva
    sqrt(tau_abs tau_total) passa de 1 com folga. É a armadilha da profundidade
    máxima, que se desvia pela energia mais transparente e não pela média.
    """
    return np.concatenate(([0.0], np.logspace(np.log10(minimum), np.log10(maximum),
                                              points - 1)))


# --------------------------------------------------------------------------- #
# A estrutura


def _initial_temperature(effective: float, tau: np.ndarray) -> np.ndarray:
    """T(tau) cinza com a função de Hopf — o palpite de partida.

    Não é aproximação de Eddington: é a q(tau) que o estágio 0,5 calculou, e
    reusá-la aqui é o que o portão analítico compra além de si mesmo. Começar
    perto encurta a correção de temperatura de dezenas de iterações a poucas.
    """
    reference = transporte.optical_depth_grid(1.0e3, 201)
    hopf = transporte.grey_milne(reference, *transporte.gauss_legendre_mu(8))["hopf"]
    return effective * (0.75 * (tau + np.interp(tau, reference, hopf))) ** 0.25


def solve(log_t_eff: float, log_g: float, energies: np.ndarray | None = None,
          columns: np.ndarray | None = None, mu_nodes: int = 12,
          iterations: int = 200, tolerance: float = 1.0e-5,
          damping: float = 0.25, compton: bool = False) -> dict:
    """A atmosfera inteira: estrutura, transporte e a intensidade emergente.

    Devolve I(E, mu) na superfície, em erg cm^-2 s^-1 sr^-1 keV^-1, além da
    estrutura convergida e do histórico do erro de fluxo — que é o que diz se a
    coisa convergiu ou apenas parou.

    **`compton` liga a linearização conjunta.** O operador de Kompaneets entra
    como coeficientes tridiagonais de Chang-Cooper DENTRO do sistema de
    transporte — denso em profundidade, tridiagonal em frequência, resolvido por
    bloco-Thomas sobre a frequência. Só o termo induzido (1 + n) fica atrasado
    uma passada, e Picard sobre ele converge à precisão de máquina em três
    iterações, medido com a temperatura congelada. As duas divisões mais baratas
    — fonte explícita, e diagonal no albedo — foram testadas no mesmo problema e
    divergem; a história e os números estão no PLANO.md.

    O caminho comptonizado custa ~1,4 s por iteração de temperatura, contra
    ~0,15 s do coerente.
    """
    energies = energy_grid() if energies is None else energies
    y = column_grid() if columns is None else columns
    mu, weights = transporte.gauss_legendre_mu(mu_nodes)
    gravity, effective = 10.0 ** log_g, 10.0 ** log_t_eff
    pressure = gravity * y
    target_flux = STEFAN * effective ** 4 / (4.0 * np.pi)          # H_0 = sigma T^4 / 4pi

    # Partida: cinza na escala de Thomson, que é o piso da opacidade.
    temperature = _initial_temperature(effective, THOMSON_CM2_G * y)
    grid = energies[:, None]
    history = []
    # O termo de Compton entra no transporte como fonte conhecida, e se atualiza
    # junto com a temperatura em vez de num laço próprio: as duas coisas
    # convergem para o mesmo ponto fixo, e iterar uma dentro da outra só
    # multiplicaria o custo.
    correction = np.zeros((energies.size, y.size))
    previous_field = None
    # Amortecimento adaptativo, ponto a ponto. Em T_ef baixa o passo de
    # Unsöld-Lucy sobressalta e a iteração entra em oscilação de período 2 —
    # medido: o erro de fluxo alterna entre 0,79 e 0,33 para sempre, com e sem
    # Compton. Onde o sinal da correção vira, o passo local cai pela metade;
    # onde persiste, recupera devagar. É o de sempre para relaxação rígida, e
    # não mexe no ponto fixo — só no caminho até ele.
    previous_step = np.zeros(y.size)
    relaxation = np.ones(y.size)

    for step in range(iterations):
        density = pressure * PROTON_MASS / (2.0 * BOLTZMANN * temperature)
        absorption, scattering = opacities(grid, density[None, :], temperature[None, :])
        extinction = absorption + scattering
        # tau por frequência, integrando a extinção na coluna. O primeiro ponto
        # tem y = 0 e portanto tau = 0, que é onde o contorno de cima vale.
        optical_depth = np.concatenate(
            [np.zeros((energies.size, 1)),
             np.cumsum(0.5 * (extinction[:, 1:] + extinction[:, :-1]) * np.diff(y), axis=1)],
            axis=1)
        planck = planck_energy(grid, temperature[None, :])
        epsilon = absorption / extinction

        # **A linearização conjunta.** O operador de Kompaneets, tridiagonal em
        # frequência com pesos de Chang-Cooper, entra DENTRO do sistema — denso
        # em profundidade, tridiagonal em frequência, bloco-Thomas sobre a
        # frequência. Só o termo induzido (1 + n) e a estrutura de temperatura
        # ficam de fora, atrasados uma passada: Picard sobre eles converge à
        # precisão de máquina em três iterações, medido com T congelada. As duas
        # divisões mais baratas — fonte explícita, e diagonal no albedo — foram
        # testadas no mesmo problema e divergem; a história está no PLANO.md.
        if compton and previous_field is not None:
            coupling = kompaneets_bands(energies, previous_field, temperature)
            field = transporte.comptonized_feautrier(optical_depth, mu, weights,
                                                     epsilon, planck, coupling)
            correction = apply_kompaneets(coupling, np.maximum(field["J"], 0.0))
        else:
            field = transporte.coupled_feautrier(optical_depth, mu, weights,
                                                 epsilon, planck)
        previous_field = np.maximum(field["J"], 0.0)

        # Momentos integrados em frequência.
        mean_intensity = np.trapezoid(field["J"], energies, axis=0)
        second_moment = np.trapezoid(field["K"], energies, axis=0)
        flux_mid = np.trapezoid(field["H_mid"], energies, axis=0)
        flux_surface = float(np.trapezoid(field["H_surface"], energies))
        integrated_planck = np.trapezoid(planck, energies, axis=0)

        y_mid = 0.5 * (y[1:] + y[:-1])
        flux = np.interp(y, y_mid, flux_mid)
        flux[0] = flux_surface
        eddington = second_moment / mean_intensity
        surface_factor = flux_surface / mean_intensity[0]

        # Médias de opacidade: absorção verdadeira nas duas primeiras, extinção
        # total na do fluxo — é essa que entra na equação do primeiro momento.
        absorption_j = np.trapezoid(absorption * field["J"], energies, axis=0) / mean_intensity
        absorption_b = np.trapezoid(absorption * planck, energies, axis=0) / integrated_planck
        extinction_h = np.interp(
            y, y_mid,
            np.trapezoid(0.5 * (extinction[:, 1:] + extinction[:, :-1]) * field["H_mid"],
                         energies, axis=0) / flux_mid)

        # A troca de energia com o gás sai do PRÓPRIO operador de Kompaneets, e
        # não de uma fórmula à parte: o que a radiação ganha o gás perde, e usar
        # duas expressões diferentes para os dois lados seria abrir uma fresta
        # por onde a energia some sem ninguém ver.
        # A correção entra em rampa nas primeiras passadas e mistura devagar: a
        # estrutura de temperatura ainda está longe no começo, e o operador de
        # Kompaneets avaliado sobre um campo errado é uma fonte errada grande.
        # **A parte diagonal do operador de Compton, implícita.** O operador de
        # Kompaneets é uma difusão em energia, e tratá-lo todo como fonte
        # conhecida é tratamento explícito de operador rígido: J vai a negativo
        # na terceira passada. A saída é a mesma do ALI, um andar acima — separar
        #
        #     (1-eps) dC = [(1-eps) a] J  +  resto
        #
        # com a = d(dC)/dJ na própria frequência, e passar o primeiro pedaço ao
        # ALBEDO do bloco angular, onde ele é resolvido junto com J. O resto,
        # que é só o acoplamento com as duas frequências vizinhas, fica
        # explícito e é manso.
        #
        # Prender a em [-1, 0] não muda o ponto fixo: o que o corte tira da
        # parte implícita o `resto` devolve, e na convergência
        # S = eps B + (1-eps)(J + dC) vale exatamente, com qualquer a. O corte
        # só governa a velocidade — e evita albedo negativo, que não é física.
        ramp = min(1.0, (step + 1) / 10.0)
        positive = np.maximum(field["J"], 0.0)
        if compton:
            delta = ramp * compton_correction(energies, positive, temperature)
            slope = ramp * compton_diagonal(energies, positive, temperature)
            slope = np.clip(slope, -1.0, 0.0)
            target_albedo = (1.0 - epsilon) * (1.0 + slope)
            target_explicit = (1.0 - epsilon) * (delta - slope * positive)
            # Sem mistura: albedo e fonte são recalculados inteiros a cada
            # passada. Misturá-los seria misturar grandezas ABSOLUTAS que mudam
            # de ordem de grandeza junto com a temperatura, e o valor velho não
            # é uma aproximação do novo — é outro problema. A estabilidade vem
            # da parte implícita, não de amortecer.
            scattering_albedo = target_albedo
            explicit = target_explicit
            correction = delta

        # A troca de energia com o gás sai do PRÓPRIO operador: o que a
        # radiação ganha o gás perde, e usar duas expressões diferentes para os
        # dois lados abriria uma fresta por onde a energia some sem ninguém ver.
        gas_gain = -THOMSON_CM2_G * np.trapezoid(correction, energies, axis=0)
        # Só o denominador de Newton continua com a forma simples: ele governa o
        # tamanho do passo, não o ponto para onde ele vai.
        compton_slope = (4.0 * BOLTZMANN * THOMSON_CM2_G / ELECTRON_REST) * mean_intensity

        # Unsöld-Lucy. A primeira parcela vem da constância do fluxo, integrada
        # na coluna; a segunda, do equilíbrio radiativo local. Sozinha, a
        # segunda converge devagar demais em profundidade, onde J e B diferem
        # por muito pouco e o erro de fluxo não aparece nessa diferença.
        deficit = target_flux - flux
        integrand = extinction_h * deficit
        accumulated = np.concatenate(
            [[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(y))])
        delta_j = (eddington[0] * (target_flux - flux_surface) / surface_factor
                   + accumulated) / eddington
        # A derivada do termo Compton em T entra no denominador: ele é o que
        # segura a correção onde a absorção é fraca demais para definir
        # temperatura sozinha — e no primeiro ponto, de coluna nula, é o único
        # que sobra. Ali a solução é a própria temperatura de Compton, que é
        # física e não remendo.
        source_function = STEFAN * temperature ** 4 / np.pi
        delta_b = ((absorption_j * (mean_intensity + delta_j) + gas_gain
                    - absorption_b * integrated_planck)
                   / (absorption_b + compton_slope * temperature / (4.0 * source_function)))

        delta_t = delta_b * temperature / (4.0 * source_function)
        relaxation = np.clip(np.where(delta_t * previous_step < 0.0,
                                      0.5 * relaxation, 1.1 * relaxation), 0.05, 1.0)
        previous_step = delta_t
        delta_t = relaxation * delta_t
        delta_t = np.clip(delta_t, -damping * temperature, damping * temperature)
        temperature = np.maximum(temperature + delta_t, 0.05 * effective)
        temperature[0] = temperature[1]

        change = float(np.max(np.abs(delta_t[1:]) / temperature[1:]))
        flux_error = float(np.max(np.abs(flux - target_flux)) / target_flux)
        history.append((change, flux_error))
        if change < tolerance and flux_error < 1.0e-3:
            break

    return {
        "energies": energies, "columns": y, "mu": mu, "weights": weights,
        "temperature": temperature, "density": density,
        "optical_depth": optical_depth, "rosseland": rosseland_mean(
            energies, extinction, temperature),
        # I(E, mu) emergente, em erg cm^-2 s^-1 sr^-1 keV^-1.
        "intensity": field["I_surface"],
        # F_E = 4 pi H_E(0), com int F_E dE = sigma T_ef^4.
        "flux_energy": 4.0 * np.pi * field["H_surface"],
        "compton": correction, "compton_applied": compton,
        "flux_error": history[-1][1], "iterations": len(history), "history": history,
        "target_flux": target_flux,
    }
