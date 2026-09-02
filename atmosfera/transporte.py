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
