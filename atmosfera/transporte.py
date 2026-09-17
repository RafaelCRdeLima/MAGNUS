"""Transporte radiativo plano-paralelo: Feautrier, quadratura angular e ALI.

**Por que este módulo vem antes de qualquer opacidade.** O plano diz que não há
teste de unidade que pegue física errada. Para o TRANSPORTE isso não é verdade:
a atmosfera cinza tem solução exata, e ela mede a máquina numérica inteira —
esquema de diferenças, quadratura em ângulo, condições de contorno, aceleração
do espalhamento — num regime em que nada pode se esconder atrás da complicação.
Um erro de Feautrier que passe daqui vira, no estágio 1, "5% que não fecham", e
some dentro de uma discussão sobre fator de Gaunt.

Três soluções exatas, e cada uma mede o que as outras não medem:

* **Milne cinza.** Em equilíbrio radiativo com opacidade única, S = J, e a
  solução é S(tau) = 3 H (tau + q(tau)) com q a função de Hopf: q(0) = 1/sqrt(3)
  exatamente, e q(infinito) = 0,7104. Mede o esquema e a quadratura.
* **Constância do fluxo.** H(tau) calculado da SOLUÇÃO FORMAL, e não da equação
  de momentos, tem de ser constante. É o auto-teste padrão dos códigos de
  atmosfera e pega quase todo erro de transporte — inclusive o de redefinir
  T_ef sem perceber, que já mordeu o modelo cinza do PULSARIS.
* **Lei do raiz de epsilon.** Numa atmosfera isotérmica semi-infinita com
  espalhamento coerente, S = (1-eps) J + eps B, a superfície vale
  S(0) = sqrt(eps) B. É o teste canônico do ALI, e é onde se vê a olho que a
  iteração lambda pura não converge: ela para na profundidade de termalização,
  1/sqrt(eps), em vez de descer até a superfície.

**Convenção dos momentos.** Pesos de Gauss-Legendre em mu de 0 a 1, somando 1:

    J = soma w u          H = soma w mu v          K = soma w mu^2 u

com u = (I+ + I-)/2 e v = (I+ - I-)/2 = mu du/dtau, as variáveis de Feautrier.
Nessa convenção o fluxo de energia é F = 4 pi H e a atmosfera cinza tem
T^4(tau)/T_ef^4 = (3/4)(tau + q(tau)).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Quadratura


def gauss_legendre_mu(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Nós e pesos em mu de 0 a 1, com os pesos somando 1.

    Gauss-Legendre e não trapézio: a intensidade emergente é suave em mu, e a
    quadratura gaussiana com 8 a 16 nós erra menos que o trapézio com cem. O
    plano nomeia a quadratura angular como armadilha justamente porque poucos
    nós arruínam a região rasante, que é a saída do projeto.
    """
    x, w = np.polynomial.legendre.leggauss(nodes)
    return 0.5 * (x + 1.0), 0.5 * w


def optical_depth_grid(tau_max: float, points: int, tau_min: float = 1.0e-5) -> np.ndarray:
    """tau = 0 e depois logarítmico: a superfície é onde tudo acontece.

    O zero exato é necessário porque a condição de contorno de cima é imposta em
    tau = 0, e o espaçamento logarítmico depois porque a intensidade emergente
    vem de tau ~ mu, ou seja de todas as décadas entre 1e-5 e 1.
    """
    return np.concatenate(([0.0], np.logspace(np.log10(tau_min), np.log10(tau_max),
                                              points - 1)))


# --------------------------------------------------------------------------- #
# Feautrier

def _feautrier_bands(tau: np.ndarray, mu: np.ndarray) -> tuple:
    """As três diagonais de A, e os coeficientes de D, para todos os ângulos.

    A equação é mu^2 d2u/dtau2 = u - S, e o sistema A u = D S é TRIDIAGONAL. As
    duas condições de contorno são de segunda ordem, obtidas eliminando a
    derivada segunda pela própria equação — de primeira ordem elas custariam um
    erro que aparece como deriva no fluxo, e seria confundido com erro de física.

    Em cima, sem radiação entrando: mu du/dtau = u.
    Embaixo, difusão: mu du/dtau = S + mu dS/dtau - u.

    D é quase a identidade: só a primeira linha, que carrega o contorno de cima,
    e a última, que carrega o de baixo junto com a derivada de S.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    n = tau.size
    delta = np.diff(tau)
    lower = np.zeros((mu.size, n))
    diag = np.zeros((mu.size, n))
    upper = np.zeros((mu.size, n))

    first = delta[0]
    diag[:, 0] = 1.0 + mu / first + first / (2.0 * mu)
    upper[:, 0] = -mu / first

    low, high = delta[:-1], delta[1:]
    mean = 0.5 * (low + high)
    below = mu[:, None] ** 2 / (low * mean)
    above = mu[:, None] ** 2 / (high * mean)
    lower[:, 1:-1] = -below
    diag[:, 1:-1] = 1.0 + below + above
    upper[:, 1:-1] = -above

    last = delta[-1]
    lower[:, -1] = -mu / last
    diag[:, -1] = 1.0 + mu / last + last / (2.0 * mu)

    top = first / (2.0 * mu)
    bottom_self = 1.0 + last / (2.0 * mu) + mu / last
    bottom_previous = -mu / last
    return lower, diag, upper, top, bottom_self, bottom_previous


def _thomas(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray,
            rhs: np.ndarray) -> np.ndarray:
    """Resolve A x = rhs para A tridiagonal, em lote sobre o eixo dos ângulos.

    Thomas é O(n) por ângulo, contra O(n^3) de uma inversão densa. Não é
    otimização prematura: no estágio 1 são centenas de energias vezes centenas
    de profundidades vezes dois modos, e a diferença entre O(n) e O(n^3) é a
    diferença entre segundos e horas. O laço em Python percorre a PROFUNDIDADE,
    com os ângulos e as colunas vetorizados por dentro.
    """
    single = rhs.ndim == 2
    values = rhs[:, :, None] if single else rhs
    count, n, columns = values.shape
    scratch = np.zeros((count, n))
    forward = np.zeros((count, n, columns))
    scratch[:, 0] = upper[:, 0] / diag[:, 0]
    forward[:, 0] = values[:, 0] / diag[:, 0, None]
    for i in range(1, n):
        denominator = diag[:, i] - lower[:, i] * scratch[:, i - 1]
        if i < n - 1:
            scratch[:, i] = upper[:, i] / denominator
        forward[:, i] = ((values[:, i] - lower[:, i, None] * forward[:, i - 1])
                         / denominator[:, None])
    solution = np.zeros((count, n, columns))
    solution[:, -1] = forward[:, -1]
    for i in range(n - 2, -1, -1):
        solution[:, i] = forward[:, i] - scratch[:, i, None] * solution[:, i + 1]
    return solution[:, :, 0] if single else solution


def lambda_operator(tau: np.ndarray, mu: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """A matriz Lambda inteira, que leva S em J.

    Construída explicitamente: com algumas centenas de profundidades custa
    O(n^2) e é instantânea, e ter Lambda na mão dá de graça o operador
    aproximado do ALI — a diagonal — E a solução exata da discretização, contra
    a qual as iterativas têm de convergir. Num módulo cuja função é ser o padrão
    de aferição, clareza vale mais que ciclos.
    """
    lower, diag, upper, top, bottom_self, bottom_previous = _feautrier_bands(tau, mu)
    n = tau.size
    d_matrix = np.tile(np.eye(n), (len(mu), 1, 1))
    d_matrix[:, 0, 0] = top
    d_matrix[:, -1, -1] = bottom_self
    d_matrix[:, -1, -2] = bottom_previous
    return np.einsum("m,mij->ij", weights, _thomas(lower, diag, upper, d_matrix))


def formal_solution(tau: np.ndarray, mu: np.ndarray, weights: np.ndarray,
                    source: np.ndarray) -> dict:
    """Resolve o transporte para uma função fonte dada e devolve os momentos.

    H sai da própria solução das equações de raio, por diferença centrada nos
    pontos médios — e NÃO da equação de momentos. É essa independência que faz a
    constância do fluxo ser teste de alguma coisa.
    """
    lower, diag, upper, top, bottom_self, bottom_previous = _feautrier_bands(tau, mu)
    rhs = np.tile(source, (len(mu), 1))
    rhs[:, 0] = top * source[0]
    rhs[:, -1] = bottom_self * source[-1] + bottom_previous * source[-2]
    u = _thomas(lower, diag, upper, rhs)

    mean_intensity = weights @ u
    second_moment = (weights * mu * mu) @ u
    # v = mu du/dtau nos pontos médios, onde a diferença é de segunda ordem.
    slope = np.diff(u, axis=1) / np.diff(tau)
    flux_mid = (weights * mu) @ (mu[:, None] * slope)
    # Na superfície a condição de contorno dá v = u, exatamente.
    flux_surface = float((weights * mu) @ u[:, 0])
    return {
        "u": u,
        "J": mean_intensity,
        "K": second_moment,
        "H_mid": flux_mid,
        "H_surface": flux_surface,
        # Sem radiação entrando, I+(0, mu) = 2 u(0, mu).
        "I_surface": 2.0 * u[:, 0],
    }


# --------------------------------------------------------------------------- #
# Milne cinza, por fator de Eddington variável


def grey_milne(tau: np.ndarray, mu: np.ndarray, weights: np.ndarray,
               flux: float = 1.0, iterations: int = 40,
               tolerance: float = 1.0e-12) -> dict:
    """A atmosfera cinza em equilíbrio radiativo. Devolve a função de Hopf.

    **Por que fator de Eddington variável e não iteração lambda.** Com S = J o
    albedo é 1, e a iteração lambda pura não converge — o mesmo defeito que a
    lei do raiz de epsilon exibe adiante. A saída padrão é fechar as equações de
    momento com f = K/J e h = H(0)/J(0) vindos da solução formal:

        dK/dtau = H  =>  f J = H tau + f(0) J(0)  =>  J = H (tau + f(0)/h) / f

    e iterar f e h. Em Eddington, f = 1/3 e h = 1/2, isso devolve
    S = 3H(tau + 2/3) — a aproximação clássica, com q identicamente 2/3. A
    solução exata move q de 0,5774 na superfície a 0,7104 no fundo, e é essa
    excursão que o portão mede.
    """
    eddington_factor = np.full(len(tau), 1.0 / 3.0)
    surface_factor = 0.5
    source = 3.0 * flux * (tau + 2.0 / 3.0)
    moments: dict = {}
    for _ in range(iterations):
        moments = formal_solution(tau, mu, weights, source)
        eddington_factor = moments["K"] / moments["J"]
        surface_factor = moments["H_surface"] / moments["J"][0]
        updated = flux * (tau + eddington_factor[0] / surface_factor) / eddington_factor
        change = np.max(np.abs(updated - source) / np.abs(updated))
        source = updated
        if change < tolerance:
            break
    moments = formal_solution(tau, mu, weights, source)
    moments["S"] = source
    moments["hopf"] = source / (3.0 * flux) - tau
    moments["eddington_factor"] = eddington_factor
    moments["surface_factor"] = surface_factor
    return moments


# --------------------------------------------------------------------------- #
# Espalhamento coerente: o teste do ALI


def _anderson_step(iterates: list[np.ndarray], mapped: list[np.ndarray]) -> np.ndarray:
    """Aceleração de Ng, na forma de Anderson: combina os últimos passos.

    A iteração acelerada ainda é uma série geométrica, e o ALI só encurta a
    razão dela — de (1 - eps) para algo como (1 - sqrt(eps)). Continua sendo uma
    série, e uma combinação linear dos últimos iterados a soma de uma vez. Os
    coeficientes saem de minimizar o resíduo da combinação, com a soma presa em
    1 para não mexer no ponto fixo.

    Se o sistema pequeno for mal condicionado — o que acontece quando os últimos
    passos já são quase paralelos, ou seja perto da convergência — devolve o
    último iterado e a iteração segue sem aceleração naquele passo.
    """
    residuals = np.array([m - i for m, i in zip(mapped, iterates)])
    difference = residuals[:-1] - residuals[-1]
    try:
        coefficients, *_ = np.linalg.lstsq(difference.T, -residuals[-1], rcond=1.0e-10)
    except np.linalg.LinAlgError:
        return mapped[-1]
    weights_full = np.append(coefficients, 1.0 - coefficients.sum())
    if not np.all(np.isfinite(weights_full)) or np.abs(weights_full).sum() > 1.0e3:
        return mapped[-1]
    return weights_full @ np.array(mapped)


def coherent_scattering(tau: np.ndarray, mu: np.ndarray, weights: np.ndarray,
                        epsilon: float, planck: float = 1.0,
                        method: str = "ali", iterations: int = 20000,
                        tolerance: float = 1.0e-10, ng_depth: int = 0) -> dict:
    """S = (1-eps) J + eps B numa atmosfera isotérmica semi-infinita.

    `method` escolhe como se resolve, e a diferença entre as três é o ponto:

    * `exact`   — resolve (I - (1-eps) Lambda) S = eps B de uma vez. É a
                  resposta EXATA da discretização, e serve de alvo para as duas
                  iterativas: quem não chegar nela tem defeito de iteração, não
                  de grade.
    * `lambda`  — iteração lambda pura, S <- (1-eps) J + eps B. Não converge em
                  profundidade óptica grande, e é isso que se quer exibir.
    * `ali`     — iteração lambda acelerada, com operador aproximado diagonal:

                      S <- S + [(1-eps) J + eps B - S] / [1 - (1-eps) diag(Lambda)]

                  o denominador é a soma da série geométrica que a iteração pura
                  teria de percorrer termo a termo.
    """
    operator = lambda_operator(tau, mu, weights)
    thermal = epsilon * planck
    if method == "exact":
        matrix = np.eye(len(tau)) - (1.0 - epsilon) * operator
        source = np.linalg.solve(matrix, np.full(len(tau), thermal))
        return {"S": source, "iterations": 0, "converged": True}

    source = np.full(len(tau), thermal)                  # começa opticamente fino
    diagonal = np.diag(operator) if method.startswith("ali") else np.zeros(len(tau))
    acceleration = 1.0 - (1.0 - epsilon) * diagonal
    history_in: list[np.ndarray] = []
    history_out: list[np.ndarray] = []
    converged = False
    used = iterations
    for step in range(iterations):
        residual = (1.0 - epsilon) * (operator @ source) + thermal - source
        mapped_source = source + residual / acceleration
        if np.max(np.abs(residual) / np.abs(source)) < tolerance:
            converged, used = True, step + 1
            source = mapped_source
            break
        if ng_depth > 0:
            history_in.append(source)
            history_out.append(mapped_source)
            if len(history_in) > ng_depth + 1:
                history_in.pop(0)
                history_out.pop(0)
            source = (_anderson_step(history_in, history_out)
                      if len(history_in) > 1 else mapped_source)
        else:
            source = mapped_source
    return {"S": source, "iterations": used, "converged": converged}


# --------------------------------------------------------------------------- #
# Transporte com espalhamento, em bloco e sem iteração


def _bands_by_frequency(tau: np.ndarray, mu: np.ndarray) -> tuple:
    """As diagonais de Feautrier quando cada frequência tem sua escala de tau.

    Mesma discretização de `_feautrier_bands`, com um eixo a mais: a opacidade
    depende da frequência, e portanto a profundidade óptica também. Formas
    (n_freq, n_mu, n_prof).
    """
    delta = np.diff(tau, axis=1)
    n_freq, n_depth = tau.shape
    shape = (n_freq, mu.size, n_depth)
    lower, diag, upper = np.zeros(shape), np.zeros(shape), np.zeros(shape)

    first = delta[:, 0][:, None]
    diag[:, :, 0] = 1.0 + mu / first + first / (2.0 * mu)
    upper[:, :, 0] = -mu / first

    low, high = delta[:, :-1], delta[:, 1:]
    mean = 0.5 * (low + high)
    below = mu[None, :, None] ** 2 / (low * mean)[:, None, :]
    above = mu[None, :, None] ** 2 / (high * mean)[:, None, :]
    lower[:, :, 1:-1] = -below
    diag[:, :, 1:-1] = 1.0 + below + above
    upper[:, :, 1:-1] = -above

    last = delta[:, -1][:, None]
    lower[:, :, -1] = -mu / last
    diag[:, :, -1] = 1.0 + mu / last + last / (2.0 * mu)

    top = first / (2.0 * mu)
    bottom_self = 1.0 + last / (2.0 * mu) + mu / last
    bottom_previous = -mu / last
    return lower, diag, upper, top, bottom_self, bottom_previous


def coupled_feautrier(tau: np.ndarray, mu: np.ndarray, weights: np.ndarray,
                      epsilon: np.ndarray, planck: np.ndarray,
                      extra: np.ndarray | None = None,
                      albedo: np.ndarray | None = None) -> dict:
    """Espalhamento coerente resolvido DE UMA VEZ, sem iteração lambda.

    Com S = eps B + (1 - eps) J e J = soma w u, a função fonte é linear nos
    próprios u — então o sistema de Feautrier deixa de ser tridiagonal escalar e
    passa a ser tridiagonal EM BLOCO, com o bloco correndo sobre os ângulos. O
    acoplamento é de posto um, `1 (x) w`, e resolver o bloco custa n_prof vezes
    n_mu^3, o que para dezesseis nós é barato.

    **Por que assim e não por ALI.** O plano previa iteração lambda acelerada,
    com linearização completa como contingência. Medido no estágio 0,5, o ALI
    diagonal precisa de milhares de iterações quando eps é pequeno — e eps é
    pequeno exatamente onde a atmosfera é interessante, na banda dura em que o
    espalhamento domina o livre-livre. A linearização completa, aqui, é DIRETA e
    custa uma fração do que custaria iterar: não há razão para deixá-la de
    reserva. E a estrutura em bloco é a mesma de que o estágio 2 precisa, com o
    bloco dobrado para os dois modos.

    `extra` é a fonte que não depende de J na própria frequência, e `albedo`
    substitui o (1 - eps) padrão. Os dois juntos são por onde entra a
    comptonização: o operador em energia acopla frequências vizinhas, e a parte
    dele que depende da PRÓPRIA frequência vai no albedo — implícita, dentro do
    bloco —, enquanto o que sobra vai em `extra`, avaliado na passada anterior.
    É a mesma divisão do ALI, aplicada ao acoplamento em energia em vez de ao
    acoplamento em profundidade.

    `tau`, `epsilon` e `planck` têm forma (n_freq, n_prof); `tau` é a escala de
    profundidade óptica TOTAL, absorção mais espalhamento, de cada frequência.
    """
    if extra is None:
        extra = np.zeros_like(planck)
    lower, diag, upper, top, bottom_self, bottom_previous = _bands_by_frequency(tau, mu)
    n_freq, n_mu, n_depth = lower.shape
    identity = np.eye(n_mu)
    # O albedo é (1 - eps) por padrão. Passá-lo à parte é o que deixa a parte
    # DIAGONAL do operador de Compton entrar aqui dentro, implícita, em vez de
    # ficar de fora como fonte explícita: ali ela é rígida e desestabiliza.
    scattering = (1.0 - epsilon) if albedo is None else albedo

    def block(index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Os três blocos e o termo independente, na profundidade `index`."""
        sub = lower[:, :, index, None] * identity
        main = diag[:, :, index, None] * identity
        sup = upper[:, :, index, None] * identity
        if index == 0:
            main = main - scattering[:, 0, None, None] * top[:, :, None] * weights
            free = epsilon[:, 0] * planck[:, 0] + extra[:, 0]
            return sub, main, sup, free[:, None] * top
        if index == n_depth - 1:
            main = main - scattering[:, -1, None, None] * bottom_self[:, :, None] * weights
            sub = sub - scattering[:, -2, None, None] * bottom_previous[:, :, None] * weights
            free = (epsilon[:, -1] * planck[:, -1] + extra[:, -1])[:, None] * bottom_self + \
                   (epsilon[:, -2] * planck[:, -2] + extra[:, -2])[:, None] * bottom_previous
            return sub, main, sup, free
        main = main - scattering[:, index, None, None] * np.ones((n_mu, 1)) * weights
        free = epsilon[:, index] * planck[:, index] + extra[:, index]
        return sub, main, sup, np.repeat(free[:, None], n_mu, axis=1)

    forward = np.zeros((n_depth, n_freq, n_mu))
    carry = np.zeros((n_depth, n_freq, n_mu, n_mu))
    previous = None
    for index in range(n_depth):
        sub, main, sup, free = block(index)
        if index == 0:
            reduced, right = main, free
        else:
            reduced = main + sub @ carry[index - 1]
            right = free - np.einsum("fjk,fk->fj", sub, forward[index - 1])
        carry[index] = np.linalg.solve(reduced, -sup)
        forward[index] = np.linalg.solve(reduced, right[..., None])[..., 0]
        previous = index

    u = np.zeros((n_depth, n_freq, n_mu))
    u[previous] = forward[previous]
    for index in range(n_depth - 2, -1, -1):
        u[index] = forward[index] + np.einsum("fjk,fk->fj", carry[index], u[index + 1])
    u = np.moveaxis(u, 0, 2)                                  # (n_freq, n_mu, n_prof)

    mean_intensity = np.einsum("j,fjd->fd", weights, u)
    second_moment = np.einsum("j,fjd->fd", weights * mu * mu, u)
    source = epsilon * planck + extra + scattering * mean_intensity
    slope = np.diff(u, axis=2) / np.diff(tau, axis=1)[:, None, :]
    flux_mid = np.einsum("j,fjd->fd", weights * mu, mu[None, :, None] * slope)
    return {
        "u": u,
        "J": mean_intensity,
        "K": second_moment,
        "S": source,
        "H_mid": flux_mid,
        "H_surface": np.einsum("j,fj->f", weights * mu, u[:, :, 0]),
        "I_surface": 2.0 * u[:, :, 0],
    }


# --------------------------------------------------------------------------- #
# Transporte comptonizado: a linearização conjunta


def frequency_lambda_operators(tau: np.ndarray, mu: np.ndarray,
                               weights: np.ndarray) -> np.ndarray:
    """A matriz Lambda em profundidade, uma por frequência: (n_freq, n_d, n_d).

    É o ingrediente que torna a linearização conjunta barata: com Lambda_k na
    mão, o sistema comptonizado fica denso em PROFUNDIDADE mas tridiagonal em
    FREQUÊNCIA — o operador de Kompaneets só fala com as vizinhas — e um
    bloco-Thomas sobre a frequência resolve tudo de uma vez, com blocos do
    tamanho da grade de profundidade.
    """
    lower, diag, upper, top, bottom_self, bottom_previous = _bands_by_frequency(tau, mu)
    n_freq, n_mu, n_depth = lower.shape
    operators = np.zeros((n_freq, n_depth, n_depth))
    for index in range(n_mu):
        rhs = np.tile(np.eye(n_depth), (n_freq, 1, 1))
        rhs[:, 0, :] = 0.0
        rhs[:, 0, 0] = top[:, index]
        rhs[:, -1, :] = 0.0
        rhs[:, -1, -1] = bottom_self[:, index]
        rhs[:, -1, -2] = bottom_previous[:, index]
        operators += weights[index] * _thomas(lower[:, index], diag[:, index],
                                              upper[:, index], rhs)
    return operators


def comptonized_feautrier(tau: np.ndarray, mu: np.ndarray, weights: np.ndarray,
                          epsilon: np.ndarray, planck: np.ndarray,
                          coupling: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict:
    """Espalhamento com troca de energia, resolvido DE UMA VEZ.

    A função fonte é S = eps B + (1 - eps)(J + dC), com dC = K J e K o operador
    de Kompaneets já linearizado em coeficientes tridiagonais `coupling` =
    (sub, dia, sup), cada um de forma (n_freq, n_prof):

        dC_k = sub_k J_{k-1} + dia_k J_k + sup_k J_{k+1}

    Com J_k = Lambda_k S_k, o sistema em S é bloco-tridiagonal na frequência,
    com blocos densos do tamanho da profundidade — e isso se resolve por
    bloco-Thomas em O(n_freq · n_prof^3), que para duzentas frequências e cento
    e vinte profundidades é um segundo.

    **Por que de uma vez e não por iteração.** Foi medido, com a temperatura
    congelada: qualquer divisão que deixe parte do operador do lado explícito
    diverge — a perturbação é amplificada pelo número de espalhamentos antes de
    ser amortecida, e em tau ~ 1e4 esse ganho engole o resto. Aqui não há resto:
    a amplificação está dentro da matriz.
    """
    operators = frequency_lambda_operators(tau, mu, weights)
    n_freq, n_depth = epsilon.shape
    identity = np.eye(n_depth)
    scattering = 1.0 - epsilon
    sub, dia, sup = coupling

    carry = np.zeros((n_freq, n_depth, n_depth))
    forward = np.zeros((n_freq, n_depth))
    for k in range(n_freq):
        block = identity - (scattering[k] * (1.0 + dia[k]))[:, None] * operators[k]
        right = epsilon[k] * planck[k]
        if k > 0:
            behind = -(scattering[k] * sub[k])[:, None] * operators[k - 1]
            block = block - behind @ carry[k - 1]
            right = right - behind @ forward[k - 1]
        if k < n_freq - 1:
            ahead = -(scattering[k] * sup[k])[:, None] * operators[k + 1]
            carry[k] = np.linalg.solve(block, -ahead)
        forward[k] = np.linalg.solve(block, right)

    source = np.zeros((n_freq, n_depth))
    source[-1] = forward[-1]
    for k in range(n_freq - 2, -1, -1):
        source[k] = forward[k] + carry[k] @ source[k + 1]

    mean_intensity = np.einsum("kij,kj->ki", operators, source)

    # Solução formal com a S encontrada, para os momentos angulares e I(0, mu).
    lower, diag, upper, top, bottom_self, bottom_previous = _bands_by_frequency(tau, mu)
    n_mu = mu.size
    u = np.zeros((n_freq, n_mu, n_depth))
    for index in range(n_mu):
        rhs = source.copy()
        rhs[:, 0] = top[:, index] * source[:, 0]
        rhs[:, -1] = bottom_self[:, index] * source[:, -1] + \
            bottom_previous[:, index] * source[:, -2]
        u[:, index, :] = _thomas(lower[:, index], diag[:, index], upper[:, index], rhs)

    slope = np.diff(u, axis=2) / np.diff(tau, axis=1)[:, None, :]
    return {
        "u": u,
        "J": mean_intensity,
        "J_formal": np.einsum("j,fjd->fd", weights, u),
        "K": np.einsum("j,fjd->fd", weights * mu * mu, u),
        "S": source,
        "H_mid": np.einsum("j,fjd->fd", weights * mu, mu[None, :, None] * slope),
        "H_surface": np.einsum("j,fj->f", weights * mu, u[:, :, 0]),
        "I_surface": 2.0 * u[:, :, 0],
    }


# --------------------------------------------------------------------------- #
# Transporte polarizado: dois modos, opacidade dependente do ângulo


def _bands_by_channel(tau: np.ndarray, mu: np.ndarray) -> tuple:
    """Feautrier quando cada CANAL (modo x ângulo) tem sua própria escala.

    No plasma magnetizado a opacidade depende do ângulo com B, então tau deixa
    de ser compartilhado entre os ângulos: cada canal carrega o seu. As formas
    são (n_lote, n_prof) para tau e (n_lote,) para mu, com o lote correndo sobre
    (frequência x canal); a álgebra é a mesma de sempre.
    """
    delta = np.diff(tau, axis=1)
    batch, n_depth = tau.shape
    lower = np.zeros((batch, n_depth))
    diag = np.zeros((batch, n_depth))
    upper = np.zeros((batch, n_depth))

    first = delta[:, 0]
    diag[:, 0] = 1.0 + mu / first + first / (2.0 * mu)
    upper[:, 0] = -mu / first

    low, high = delta[:, :-1], delta[:, 1:]
    mean = 0.5 * (low + high)
    below = mu[:, None] ** 2 / (low * mean)
    above = mu[:, None] ** 2 / (high * mean)
    lower[:, 1:-1] = -below
    diag[:, 1:-1] = 1.0 + below + above
    upper[:, 1:-1] = -above

    last = delta[:, -1]
    lower[:, -1] = -mu / last
    diag[:, -1] = 1.0 + mu / last + last / (2.0 * mu)

    top = first / (2.0 * mu)
    bottom_self = 1.0 + last / (2.0 * mu) + mu / last
    bottom_previous = -mu / last
    return lower, diag, upper, top, bottom_self, bottom_previous


def polarized_feautrier(tau: np.ndarray, mu_channel: np.ndarray,
                        weight_channel: np.ndarray, thermal: np.ndarray,
                        into: np.ndarray, out_of: np.ndarray,
                        surface_intensity: np.ndarray | None = None,
                        exchange: np.ndarray | None = None) -> dict:
    """Os dois modos acoplados por espalhamento, resolvidos de uma vez.

    A peça central do estágio 2. Os canais são (modo, ângulo) — 2 x n_mu — e a
    função fonte do canal c é

        S_c = thermal_c + soma_alpha into[alpha, c] · soma_c' out_of[alpha, c'] u_c'

    O acoplamento é de POSTO TRÊS: o fóton espalhado passa por uma das três
    componentes cíclicas alpha, e é a componente que lembra para onde ele pode
    ir — |e_alpha^c|^2 na saída, |e_alpha^c'|^2 pesado pela quadratura na
    entrada. É a estrutura de Ho & Lai com a redistribuição média por
    componente, e ela conserva fóton por construção quando

        into[alpha, c] = |e_alpha^c|^2 sigma^alpha / (chi_c N_alpha)
        out_of[alpha, c'] = w_c' |e_alpha^c'|^2
        N_alpha = soma_c w_c |e_alpha^c|^2

    Formas: tau, thermal (n_E, n_C, n_prof); into, out_of (n_E, 3, n_C, n_prof);
    mu_channel, weight_channel (n_C,), com os pesos somando DOIS — a quadratura
    inteira por modo — para que J e H sejam TOTAIS, soma das duas polarizações.
    Cada canal termaliza em B/2, e a soma devolve B.

    O mesmo bloco-tridiagonal em profundidade de `coupled_feautrier`, com o
    bloco agora sobre canais e o acoplamento de posto 3 em vez de 1.

    **`surface_intensity`** troca o contorno de baixo: em vez de difusão
    (atmosfera semi-infinita), impõe I+(fundo) = valor dado por canal — a
    condição de ATMOSFERA FINA sobre superfície emissora, literalmente a
    eq. (15) de Suleimanov, Pavlov & Werner (2009): I(mu>0, m_max) = B_nu/2.
    A álgebra: u + mu du/dtau = I+_sup vira, em segunda ordem,

        (1 + mu/D + D/2mu) u_N - (mu/D) u_{N-1} = I+_sup + (D/2mu) S_N

    — mesmas diagonais do caso difusivo; muda só o lado direito, que perde o
    termo em S_{N-1} e ganha a fonte da superfície.

    **`exchange`** (n_E, n_C, n_C, n_prof) soma um acoplamento extra por célula
    ao bloco — é por onde entra a conversão parcial de modos na ressonância de
    vácuo, como espalhamento de troca localizado na célula do cruzamento.
    """
    n_freq, n_channel, n_depth = tau.shape
    flat_tau = tau.reshape(n_freq * n_channel, n_depth)
    flat_mu = np.tile(mu_channel, n_freq)
    lower, diag, upper, top, bottom_self, bottom_previous = _bands_by_channel(flat_tau,
                                                                              flat_mu)
    shape = (n_freq, n_channel, n_depth)
    lower, diag, upper = (a.reshape(shape) for a in (lower, diag, upper))
    top = top.reshape(n_freq, n_channel)
    bottom_self = bottom_self.reshape(n_freq, n_channel)
    bottom_previous = bottom_previous.reshape(n_freq, n_channel)

    identity = np.eye(n_channel)

    def coupling(index: int) -> np.ndarray:
        # soma_alpha into (x) out_of, o posto 3 do bloco — mais a troca local,
        # se houver.
        block = np.einsum("fac,fad->fcd", into[:, :, :, index], out_of[:, :, :, index])
        if exchange is not None:
            block = block + exchange[:, :, :, index]
        return block

    def block(index: int):
        sub = lower[:, :, index, None] * identity
        main = diag[:, :, index, None] * identity
        sup = upper[:, :, index, None] * identity
        if index == 0:
            main = main - top[:, :, None] * coupling(0)
            return sub, main, sup, top * thermal[:, :, 0]
        if index == n_depth - 1:
            if surface_intensity is not None:
                # Atmosfera fina: I+(fundo) imposto. O coeficiente de S_N é
                # D/(2 mu), que sai das três quantidades já montadas.
                bottom_thermal = bottom_self - 1.0 + bottom_previous
                main = main - bottom_thermal[:, :, None] * coupling(index)
                free = bottom_thermal * thermal[:, :, -1] + surface_intensity
                return sub, main, sup, free
            main = main - bottom_self[:, :, None] * coupling(index)
            sub = sub - bottom_previous[:, :, None] * coupling(index - 1)
            free = bottom_self * thermal[:, :, -1] + \
                bottom_previous * thermal[:, :, -2]
            return sub, main, sup, free
        main = main - coupling(index)
        return sub, main, sup, thermal[:, :, index]

    carry = np.zeros((n_depth, n_freq, n_channel, n_channel))
    forward = np.zeros((n_depth, n_freq, n_channel))
    for index in range(n_depth):
        sub, main, sup, free = block(index)
        if index == 0:
            reduced, right = main, free
        else:
            reduced = main + sub @ carry[index - 1]
            right = free - np.einsum("fjk,fk->fj", sub, forward[index - 1])
        carry[index] = np.linalg.solve(reduced, -sup)
        forward[index] = np.linalg.solve(reduced, right[..., None])[..., 0]

    u = np.zeros((n_depth, n_freq, n_channel))
    u[-1] = forward[-1]
    for index in range(n_depth - 2, -1, -1):
        u[index] = forward[index] + np.einsum("fjk,fk->fj", carry[index], u[index + 1])
    u = np.moveaxis(u, 0, 2)                                # (n_E, n_C, n_prof)

    slope = np.diff(u, axis=2) / np.diff(tau, axis=2)
    flux_mid = np.einsum("c,fcd->fd", weight_channel * mu_channel,
                         mu_channel[None, :, None] * slope)
    return {
        "u": u,
        "J": np.einsum("c,fcd->fd", weight_channel, u),
        "H_mid": flux_mid,
        "H_surface": np.einsum("c,fc->f", weight_channel * mu_channel, u[:, :, 0]),
        "I_surface": 2.0 * u[:, :, 0],
    }


def _pesos_caracteristica(delta: np.ndarray) -> tuple:
    """Pesos da característica curta com fonte LINEAR na célula.

    Integrando a equação de transporte ao longo do raio, com S linear entre os
    dois nós e passo óptico d,

        I_chegada = I_partida e^-d + S_chegada (e0 - e1/d) + S_partida (e1/d),
        e0 = 1 - e^-d,   e1 = e0 - d e^-d.

    Em d grande o peso local vai a 1 e em d pequeno os dois pesos viram d/2, o
    trapézio. Os limites são escritos à mão porque e1/d é 0/0 numericamente
    abaixo de 1e-4. Serve nas pontas da varredura, onde falta um vizinho para a
    parábola.
    """
    d = np.maximum(delta, 0.0)
    decaimento = np.exp(-d)
    e0 = -np.expm1(-d)
    e1 = e0 - d * decaimento
    pequeno = d < 1.0e-4
    seguro = np.where(pequeno, 1.0, d)
    partida = np.where(pequeno, 0.5 * d, e1 / seguro)
    chegada = np.where(pequeno, 0.5 * d, e0 - e1 / seguro)
    return decaimento, partida, chegada


def _pesos_parabolicos(passo_acima: np.ndarray, passo_abaixo: np.ndarray) -> tuple:
    """Pesos da característica curta com fonte PARABÓLICA (Olson & Kunasz 1987).

    A fonte é interpolada por uma parábola nos três nós que o raio encosta — o
    de onde ele vem (`passo_acima` de distância óptica), o de chegada, e o
    seguinte (`passo_abaixo`) —, e a integral fica

        I_chegada = I_partida e^-D + psi_p S_partida + psi_c S_chegada
                    + psi_s S_seguinte,
        e0 = 1 - e^-D,   e1 = D - e0,   e2 = D^2 - 2 e1,
        psi_p = e0 + [e2 - (d + 2D) e1] / [D (D + d)],
        psi_c = [(D + d) e1 - e2] / (D d),
        psi_s = (e2 - D e1) / [d (D + d)],

    com D = passo_acima e d = passo_abaixo. NÃO é refinamento cosmético: em
    célula opticamente espessa a versão linear só acerta o limite difusivo em
    primeira ordem em 1/D, e a iteração de espalhamento amplifica esse erro.
    Medido numa atmosfera semi-infinita de albedo 0,99 com 100 pontos em
    profundidade, a linear erra 6,5 por cento em J e só converge para a resposta
    certa com 800 pontos; a parabólica acerta na grade grossa. No limite D
    grande ela devolve I = S - dS/ds, que é a difusão exata.

    e1 e e2 saem de subtrações que se cancelam quando D é pequeno, então abaixo
    de 1e-4 entram as séries, e os pesos viram o trapézio.
    """
    grande = np.maximum(passo_acima, 0.0)
    pequena = np.maximum(passo_abaixo, 1.0e-300)
    decaimento = np.exp(-grande)
    e0 = -np.expm1(-grande)
    curto = grande < 1.0e-4
    e1 = np.where(curto, grande ** 2 / 2.0 - grande ** 3 / 6.0, grande - e0)
    e2 = np.where(curto, grande ** 3 / 3.0 - grande ** 4 / 12.0,
                  grande * grande - 2.0 * e1)
    soma = grande + pequena
    seguro_grande = np.where(curto, 1.0, grande)
    psi_p = np.where(curto, 0.5 * grande,
                     e0 + (e2 - (pequena + 2.0 * grande) * e1) / (seguro_grande * soma))
    psi_c = np.where(curto, 0.5 * grande,
                     (soma * e1 - e2) / (seguro_grande * pequena))
    psi_s = np.where(curto, 0.0, (e2 - grande * e1) / (pequena * soma))
    # LIMITADOR (Auer & Paletou 1994): a parabola so vale quando as duas celulas
    # tem espessuras opticas comparaveis. Onde elas nao tem — e e o caso no pico
    # de opacidade da ressonancia de vacuo, com uma celula de 3e10 profundidades
    # opticas colada numa de 1e-8 — os pesos crescem como 1/d_abaixo e chegam a
    # 1e8, o que estoura a recursao. A soma psi_p + psi_c + psi_s = e0 vale
    # sempre, entao o teste e so de magnitude: acima de 1,5 volta para a
    # caracteristica linear, que nunca tem peso negativo nem maior que 1.
    exagero = ((np.abs(psi_p) > 1.5) | (np.abs(psi_c) > 1.5) | (np.abs(psi_s) > 1.5)
               | ~np.isfinite(psi_p) | ~np.isfinite(psi_c) | ~np.isfinite(psi_s))
    if np.any(exagero):
        _, linear_p, linear_c = _pesos_caracteristica(passo_acima)
        psi_p = np.where(exagero, linear_p, psi_p)
        psi_c = np.where(exagero, linear_c, psi_c)
        psi_s = np.where(exagero, 0.0, psi_s)
    return decaimento, psi_p, psi_c, psi_s


def polarized_direct(tau: np.ndarray, mu_channel: np.ndarray,
                     weight_channel: np.ndarray, thermal: np.ndarray,
                     into: np.ndarray, out_of: np.ndarray,
                     surface_intensity: np.ndarray | None = None,
                     jump_probability: np.ndarray | None = None,
                     jump_index: np.ndarray | None = None,
                     iterations: int = 400, tolerance: float = 1.0e-7,
                     guess: np.ndarray | None = None, ng_every: int = 4) -> dict:
    """Integração DIRETA raio a raio, com o salto de conversão imposto exato.

    Alternativa ao `polarized_feautrier` com as mesmas entradas e as mesmas
    saídas, pela razão que van Adelsberg & Lai (2006), §3.1.1, dão:

        "since the resonance density depends on photon energy, the standard
        Feautrier procedure for integrating the radiative transfer equation
        cannot be used here, as there is no simple way to incorporate
        eqs. (33)-(34) into the method of forward and backward substitution
        employed by Feautrier. Instead, we use the standard Runga-Kutta method
        to integrate the transfer eq. (6) in the upward and downward directions
        ... The Runga-Kutta integration is stopped at the resonance, where
        eqs. (33) and (34) are used to convert the mode intensities."

    O Feautrier resolve em u = (I+ + I-)/2 por substituição para frente e para
    trás, e só sabe representar uma função CONTÍNUA em profundidade. A conversão
    de modos é uma descontinuidade: ao cruzar a ressonância,

        I_1 -> P I_1 + (1 - P) I_2,   I_2 -> P I_2 + (1 - P) I_1.

    Imposta dentro do Feautrier como espalhamento de troca, ela vira uma
    opacidade falsa de até 4,6 profundidades ópticas numa única célula, que
    engorda a extinção, dilui a emissão térmica daquela célula e esquenta as
    camadas externas (medido contra os perfis publicados: fator 1,4 a 2,2 de
    aquecimento a mais do que a literatura). Aqui a troca não gera opacidade
    nenhuma: é uma redistribuição da intensidade entre os dois modos no ponto em
    que o raio cruza a ressonância, que é o que ela fisicamente é.

    O preço é que o espalhamento deixa de ser implícito. Isso é pago com
    iteração lambda ACELERADA (Olson, Auer & Buchler 1986): o operador local
    aproximado é a diagonal da característica curta, `lstar`, e a correção por
    iteração resolve (I - Lambda* C) dx = residuo em cada profundidade, com C o
    acoplamento de posto 3 entre canais. A taxa de convergência deixa de
    depender da profundidade óptica, que é o ponto do método.

    `jump_index` (n_E,) é o índice de profundidade em que a ressonância cai, com
    valor negativo onde não há; `jump_probability` (n_E, n_mu) é o P de cada
    ângulo. O salto acontece na interface entre `jump_index - 1` e `jump_index`,
    nos dois sentidos de propagação.
    """
    n_freq, n_channel, n_depth = tau.shape
    n_mu = n_channel // 2
    passo = np.diff(tau, axis=2) / mu_channel[None, :, None]
    # Parabolica no miolo, linear nas pontas (onde falta um vizinho).
    dec_desce, pp_desce, pc_desce, ps_desce = _pesos_parabolicos(passo[:, :, :-1],
                                                                 passo[:, :, 1:])
    dec_sobe, pp_sobe, pc_sobe, ps_sobe = _pesos_parabolicos(passo[:, :, 1:],
                                                             passo[:, :, :-1])
    dec_lin, pp_lin, pc_lin = _pesos_caracteristica(passo)

    tem_salto = jump_index is not None and jump_probability is not None
    if tem_salto:
        jump_index = np.asarray(jump_index, int)
        jump_probability = np.asarray(jump_probability, float)
        alvos = {int(k): np.nonzero(jump_index == k)[0]
                 for k in np.unique(jump_index[jump_index > 0])}
    else:
        alvos = {}

    # C[c, g] = soma_alpha into[alpha, c] out_of[alpha, g], o posto 3 do Ho & Lai
    acoplamento = np.einsum("facd,fagd->fcgd", into, out_of)

    def troca(plano: np.ndarray, k: int) -> None:
        indices = alvos.get(k)
        if indices is None or indices.size == 0:
            return
        bloco = plano[indices].reshape(-1, 2, n_mu)
        p = jump_probability[indices]
        misturado = np.empty_like(bloco)
        misturado[:, 0] = p * bloco[:, 0] + (1.0 - p) * bloco[:, 1]
        misturado[:, 1] = p * bloco[:, 1] + (1.0 - p) * bloco[:, 0]
        plano[indices] = misturado.reshape(-1, n_channel)

    def formal(fonte: np.ndarray) -> tuple:
        desce = np.zeros((n_freq, n_channel, n_depth))
        for k in range(1, n_depth):
            if k < n_depth - 1:
                # chega em k vindo de k-1, com k+1 fechando a parabola
                j = k - 1
                desce[:, :, k] = (desce[:, :, j] * dec_desce[:, :, j]
                                  + fonte[:, :, j] * pp_desce[:, :, j]
                                  + fonte[:, :, k] * pc_desce[:, :, j]
                                  + fonte[:, :, k + 1] * ps_desce[:, :, j])
            else:
                j = k - 1
                desce[:, :, k] = (desce[:, :, j] * dec_lin[:, :, j]
                                  + fonte[:, :, j] * pp_lin[:, :, j]
                                  + fonte[:, :, k] * pc_lin[:, :, j])
            if tem_salto:
                troca(desce[:, :, k], k)
        sobe = np.zeros((n_freq, n_channel, n_depth))
        if surface_intensity is not None:
            # Atmosfera fina: I+(fundo) vem da superficie emissora.
            sobe[:, :, -1] = surface_intensity
        else:
            # Semi-infinita: difusao, I+ = S + mu dS/dtau.
            # I+ = S + mu dS/dtau, com o termo de gradiente preso a |S|: onde a
            # ultima celula e opticamente fina a difusao nao vale, e o gradiente
            # cru manda a intensidade para qualquer lugar.
            gradiente = (fonte[:, :, -1] - fonte[:, :, -2]) \
                / np.maximum(passo[:, :, -1], 1.0e-300)
            sobe[:, :, -1] = fonte[:, :, -1] + np.clip(gradiente, -fonte[:, :, -1],
                                                       fonte[:, :, -1])
        for k in range(n_depth - 2, -1, -1):
            if k > 0:
                # chega em k vindo de k+1, com k-1 fechando a parabola
                sobe[:, :, k] = (sobe[:, :, k + 1] * dec_sobe[:, :, k - 1]
                                 + fonte[:, :, k + 1] * pp_sobe[:, :, k - 1]
                                 + fonte[:, :, k] * pc_sobe[:, :, k - 1]
                                 + fonte[:, :, k - 1] * ps_sobe[:, :, k - 1])
            else:
                sobe[:, :, 0] = (sobe[:, :, 1] * dec_lin[:, :, 0]
                                 + fonte[:, :, 1] * pp_lin[:, :, 0]
                                 + fonte[:, :, 0] * pc_lin[:, :, 0])
            if tem_salto:
                troca(sobe[:, :, k], k + 1)
        return sobe, desce

    # Operador local aproximado: o peso da fonte NA PROPRIA celula, somado sobre
    # os dois sentidos e dividido por dois porque u e a media deles.
    lstar = np.zeros((n_freq, n_channel, n_depth))
    lstar[:, :, 1:-1] += 0.5 * pc_desce
    lstar[:, :, -1] += 0.5 * pc_lin[:, :, -1]
    lstar[:, :, 1:-1] += 0.5 * pc_sobe
    lstar[:, :, 0] += 0.5 * pc_lin[:, :, 0]
    if surface_intensity is None:
        lstar[:, :, -1] += 0.5          # o contorno difusivo e local no fundo
    lstar = np.clip(lstar, 0.0, 0.999)

    identidade = np.eye(n_channel)
    precondicionador = (identidade[None, None]
                        - lstar.transpose(0, 2, 1)[..., None]
                        * acoplamento.transpose(0, 3, 1, 2))

    u = thermal.copy() if guess is None else np.array(guess, float)
    residuo = np.inf
    # O ALI diagonal sozinho AINDA e uma serie geometrica, de razao ~1-sqrt(eps):
    # medido, com albedo 0,999 ele estaciona 30% abaixo da resposta depois de 600
    # passos. A aceleracao de Ng soma a serie de uma vez, e e o que a literatura
    # de transporte usa junto com o ALI desde Olson, Auer & Buchler (1986).
    entradas: list[np.ndarray] = []
    saidas: list[np.ndarray] = []
    for passo_interno in range(iterations):
        fonte = thermal + np.einsum("fcgd,fgd->fcd", acoplamento, u)
        sobe, desce = formal(fonte)
        diferenca = 0.5 * (sobe + desce) - u
        correcao = np.linalg.solve(
            precondicionador, diferenca.transpose(0, 2, 1)[..., None])[..., 0]
        seguinte = np.maximum(u + correcao.transpose(0, 2, 1), 0.0)
        escala = max(float(np.max(np.abs(seguinte))), 1.0e-300)
        residuo = float(np.max(np.abs(seguinte - u))) / escala
        if ng_every:
            entradas.append(u.ravel().copy())
            saidas.append(seguinte.ravel().copy())
            if len(entradas) > 4:
                entradas.pop(0)
                saidas.pop(0)
            if len(entradas) >= 3 and (passo_interno + 1) % ng_every == 0:
                seguinte = np.maximum(
                    _anderson_step(entradas, saidas).reshape(u.shape), 0.0)
                entradas.clear()
                saidas.clear()
        u = seguinte
        if residuo < tolerance:
            break

    fonte = thermal + np.einsum("fcgd,fgd->fcd", acoplamento, u)
    sobe, desce = formal(fonte)
    u = 0.5 * (sobe + desce)
    fluxo = 0.5 * np.einsum("c,fcd->fd", weight_channel * mu_channel, sobe - desce)
    return {
        "u": u,
        "J": np.einsum("c,fcd->fd", weight_channel, u),
        "H_mid": 0.5 * (fluxo[:, 1:] + fluxo[:, :-1]),
        "H_surface": fluxo[:, 0],
        "I_surface": sobe[:, :, 0],
        "inner_iterations": passo_interno + 1,
        "inner_residual": residuo,
    }
