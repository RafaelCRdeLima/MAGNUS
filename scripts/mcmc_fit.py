#!/usr/bin/env python3
"""Affine-invariant ensemble MCMC for PULSARIS phase-energy event data.

Persistent C++ workers own the physical forward evaluations and cache the
instrument response.  This adapter owns event binning, priors, ensemble moves
and diagnostics.
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import csv
import io
import json
import os
import struct
import math
import queue
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: O BINARIO, com uma porta para trocar de binario sem editar codigo. Existe
#: porque o motor com OpenMP e compilado em separado (build/magnus_engine_omp) e
#: tem de ser COMPARADO com o validado, sob o mesmo adaptador e o mesmo pedido.
#: Sem a variavel, nada muda: o padrao continua sendo build/magnus_engine.
ENGINE = Path(os.environ.get("MAGNUS_ENGINE",
                             str(ROOT / "build" / "magnus_engine")))
#: Os dados de instrumento (perfis, respostas, TBabs) vivem fora do MAGNUS e são
#: usados SOMENTE PARA LEITURA. O diretório vem da variável de ambiente
#: MAGNUS_INSTRUMENT_DIR; sem ela, o padrão é a instalação do PULSARIS do autor.
PULSARIS = Path(os.environ.get("MAGNUS_INSTRUMENT_DIR",
                               str(Path.home() / "Codes" / "PULSARIS")))
#: Anisotropia da opacidade na fotosfera, derivada das tabelas de
#: Potekhin & Chabrier (2003) fora do MAGNUS (script do PULSARIS). Opcional: so
#: o modelo cinza de dois modos a usa; a tabela de atmosfera nao precisa dela.
ANISOTROPY_TABLE = ROOT / "atmosphere_data" / "magnetic_anisotropy.csv"
#: Espectro real de hidrogênio magnetizado parcialmente ionizado, interpolado
#: das tabelas NSMAXG de Ho, Potekhin & Chabrier (2008).
NSMAXG_TABLE = ROOT / "atmosphere_data" / "nsmaxg_hydrogen.bin"
#: Faixa de lg(B) da tabela, e o ponto onde ela deixa de ser tabelada.
#:
#: O Ho publica até 13,5. Acima disso a tabela é extrapolada aqui, alinhando a
#: linha de cíclotron pelo E_cp = 0,00632 B_12 e estendendo a tendência do
#: contínuo — o que se justifica porque o contínuo quase não depende de B e
#: toda a variação está na linha. O custo foi medido por exclusão: fator 1,27
#: em w a 0,2 dex de distância, 1,67 a 0,65 dex.
#:
#: A priori vai até o limite extrapolado, e não até o tabelado, porque parar em
#: 13,5 imporia uma resposta: a gaussiana ajustada pede a feição em 0,355 keV,
#: que por E_cp exige lg B = 13,75. Um teto abaixo do valor que os dados
#: sugerem não mede nada — só devolve o teto.
NSMAXG_FIELD_TABULATED = 13.5
NSMAXG_FIELD_RANGE = (10.0, 13.9)
MANIFEST = PULSARIS / "instrument_data" / "profiles" / "manifest.json"
ABSORPTION = PULSARIS / "instrument_data" / "absorption" / "tbabs_wilm.csv"
KEV_PER_MK = 0.08617333262145
#: GM_sol/c² em km: u = 2·GM/(Rc²) = 2·(1,476625)·(M/M_sol)/R[km].
_GM_SUN_C2_KM = 1.4766250385   # identico ao gm_sun_over_c2_km do motor


def read_events(path: Path) -> tuple[dict[str, str], list[tuple[float, float]]]:
    metadata: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    data_lines = []
    for line in lines:
        if line.startswith("#"):
            key, sep, value = line[1:].strip().partition("=")
            if sep:
                metadata[key.strip()] = value.strip()
        elif line.strip():
            data_lines.append(line)
    reader = csv.DictReader(io.StringIO("\n".join(data_lines)))
    if not reader.fieldnames or "TIME" not in reader.fieldnames:
        raise ValueError("event list must contain a TIME column")
    energy_key = next((key for key in ("DETECTED_ENERGY_KEV", "ENERGY_KEV", "ENERGY")
                       if key in reader.fieldnames), None)
    if not energy_key:
        raise ValueError("event list must contain detected energy in keV")
    events = [(float(row["TIME"]), float(row[energy_key])) for row in reader]
    if not events:
        raise ValueError("event list contains no events")
    return metadata, events


def read_background(path: Path) -> list[tuple[float, float]]:
    """Lê a taxa de fundo por keV, já escalada para a região da fonte.

    Formato: ``energy_keV,rate_per_keV_per_s`` com comentários em ``#``. É o que
    o XREDUX escreve a partir do espectro OGIP de fundo e do ``BACKSCAL``, que é
    a receita padrão para levar o fundo da sua região de extração para a da fonte.
    """
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            energy, rate = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if energy > 0.0 and rate >= 0.0:
            rows.append((energy, rate))
    if len(rows) < 2:
        raise ValueError(f"background table has too few usable rows: {path}")
    return sorted(rows)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(math.floor(position))
    hi = min(len(ordered) - 1, lo + 1)
    fraction = position - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def superficie_efetiva(request) -> dict:
    """Resolucao da esfera com que o modelo foi integrado, para o artefato.

    Espelha a regra do motor: bandas = sphereBands se positivo, senao
    clamp(2*rings, 16, 60); e por banda max(4, ceil(2*nlat*sin(theta_meio)))
    azimutes, amostrados no ponto medio. Validado contra o relato do proprio
    motor com refino 1: 120 bandas dao 18394 pontos nos dois.

    `pontosBase` conta a malha ANTES do refino de fronteira. Com
    boundaryRefine > 1 o motor subdivide as celulas cortadas por n.eixo = 0, e
    quantas sao depende da COLATITUDE MAGNETICA do vetor avaliado: medido em
    26 bandas com refino 8, os tres vetores do incidente dao 8560, 8749 e 8182
    pontos contra os 874 da malha base. Nenhuma funcao do pedido pode prever
    esse numero, e por isso ele nao e reportado aqui; o motor imprime o valor
    efetivo por processo em stderr.

    Existe porque o artefato tem de carregar a resolucao com que foi produzido.
    O incidente de 04/10/2026 so foi pego por um teste a parte; nenhum
    resultado dizia em que resolucao havia sido integrado.
    """
    pedido = int(request.get("sphereBands", 0))
    rings = int(request.get("rings", 13))
    if pedido > 0:
        bandas, origem = max(8, min(pedido, 4000)), "explicita (sphereBands)"
    else:
        bandas, origem = max(16, min(2 * rings, 60)), "herdada de rings"
    nlat = max(8, bandas)
    pontos = sum(max(4, math.ceil(2.0 * nlat * math.sin(math.pi * (i + 0.5) / nlat)))
                 for i in range(nlat))
    refino = max(1, int(request.get("boundaryRefine", 1)))
    return {"bandasSolicitadas": pedido if pedido > 0 else 2 * rings,
            "bandasEfetivas": bandas, "pontosBase": pontos,
            "refinoFronteira": refino,
            "pontosEfetivos": pontos if refino == 1 else
                              "depende da geometria do vetor; ver stderr do motor",
            "origem": origem,
            "quadratura": "ponto medio em malha lat-lon alinhada ao eixo de "
                          "coordenadas; azimutes por banda = max(4, ceil(2 nlat "
                          "sin theta)); celulas cortadas por n.eixo = 0 "
                          "subdivididas em refino x refino"}


#: Teto de iterações por caminhante. Existe para o tempo de execução não
#: explodir sem aviso, não por limitação de memória — as trajetórias já são
#: subamostradas para ~250 pontos. Pedir mais que isto é podado, e a poda é
#: reportada: truncar em silêncio faz alguém acreditar que rodou o que pediu.
MAX_ITERATIONS = 100000

#: Fração da posteriori que precisa encostar num limite da priori para que ele
#: seja considerado ativo. Um décimo das amostras coladas na parede já indica
#: que a priori está definindo o resultado, e não os dados.
BOUND_FRACTION = 0.10

#: Largura da faixa que conta como "encostado", em fração do intervalo da
#: priori. Tem de acompanhar BOUND_FRACTION: uma margem de 1% com limiar de 10%
#: nunca dispara, porque a posteriori se acumula PERTO da parede sem tocá-la.
BOUND_MARGIN = 0.10


def beaming_is_physical(a: float, b: float) -> bool:
    """Se 1 + a*mu + b*mu^2 fica positivo em todo o hemisfério visível.

    A intensidade emitida não pode ser negativa em ângulo nenhum, e a
    normalização do padrão foi derivada supondo que não se corta nada — cortar
    em zero dentro do motor invalidaria a integral em silêncio, mudando o fluxo
    bolométrico junto com a forma. Então a restrição vive aqui.

    Para ``mu`` em [0,1]: o valor em 0 é sempre 1; basta checar a borda em 1 e,
    quando a parábola tem mínimo interior (``b > 0`` com vértice dentro do
    intervalo), o valor no vértice.
    """
    if 1.0 + a + b < 0.0:
        return False
    if b > 0.0:
        vertex = -a / (2.0 * b)
        if 0.0 < vertex < 1.0 and 1.0 - a * a / (4.0 * b) < 0.0:
            return False
    # A normalização é a integral de I(mu)*mu sobre o hemisfério; nula ou
    # negativa não descreve emissão nenhuma.
    return 1.0 + 2.0 * a / 3.0 + b / 2.0 > 0.0


#: Parâmetros que vivem numa circunferência. Tratá-los como reta é o erro que
#: produz R-hat na casa das dezenas sem nenhuma divergência física por trás.
CYCLIC_PERIOD_DEG = 360.0


def is_cyclic(name: str) -> bool:
    return name == "phaseOffset" or name.startswith("relativePhi")


def circular_mean(values: list[float]) -> float:
    """Média de ângulos, feita como deve: pela soma dos vetores unitários."""
    if not values:
        return 0.0
    radians = [math.radians(value) for value in values]
    return math.degrees(math.atan2(
        sum(math.sin(angle) for angle in radians),
        sum(math.cos(angle) for angle in radians)))


def unwrap_around(values: list[float], centre: float) -> list[float]:
    """Traz cada ângulo para o ramo mais próximo de ``centre``.

    Sem isto, duas cadeias no MESMO lugar físico — uma em +178, outra em -178 —
    aparecem separadas por 356 graus, e o R-hat, que compara a variância entre
    cadeias com a de dentro delas, dispara. Medido: 125,9 contra 1,64 depois de
    desenrolar. O diagnóstico estava medindo a emenda do sistema de
    coordenadas, não a convergência.
    """
    return [centre + ((value - centre + CYCLIC_PERIOD_DEG / 2.0)
                      % CYCLIC_PERIOD_DEG) - CYCLIC_PERIOD_DEG / 2.0
            for value in values]


def at_bound(values: list[float], bounds: tuple[float, float]) -> str:
    """Diz se a posteriori está encostada num limite da priori.

    O R-hat sozinho não pega isto, e é aí que ele mais engana: caminhantes
    empilhados contra a mesma parede concordam perfeitamente entre si e o R-hat
    dá 1,00, que se lê como convergência quando é o contrário — o resultado é a
    priori, não o dado.

    **Duas correções sobre a versão anterior, ambas de casos medidos.**

    A primeira: a margem era 1% do intervalo enquanto o limiar de amostras era
    10%, e a incoerência tinha consequência. Medido no endurecimento da
    atmosfera, com priori (1, 3): 62% da posteriori no quinto inferior, pico
    encostado no piso, e nenhuma amostra chegando a 1,02 — então a contagem
    dava zero e nada era reportado. A margem agora acompanha o limiar.

    A segunda: a comparação passa a ser contra o que uma posteriori UNIFORME
    daria. Numa faixa que ocupa 10% da priori, o esperado por acaso é 10% das
    amostras; exigir exatamente isso dispararia em qualquer posteriori plana.
    Exigir o dobro flagra acúmulo de verdade.

    E devolve "both" quando as duas paredes acumulam. Não é caso de escola: o
    mesmo endurecimento tinha 52% no piso E 20% no teto, bimodal com um vale
    vazio no meio — a posteriori era a priori com massa nas duas pontas, e a
    versão anterior só sabia nomear uma parede.
    """
    low, high = bounds
    span = high - low
    if span <= 0.0 or not values:
        return ""
    margin = BOUND_MARGIN * span
    total = len(values)
    below = sum(1 for value in values if value <= low + margin) / total
    above = sum(1 for value in values if value >= high - margin) / total
    # O que uma posteriori uniforme poria na faixa, e o dobro disso como piso.
    threshold = max(BOUND_FRACTION, 2.0 * BOUND_MARGIN)
    if above >= threshold and below >= threshold:
        return "both"
    if above >= threshold:
        return "upper"
    if below >= threshold:
        return "lower"
    return ""


#: O passo de esticamento move cada caminhante ao longo da reta que o liga a
#: outro do enxame, então o tamanho do enxame limita as direções que ele explora.
#: A orientação usual é 4x a dimensão, mas ela NÃO se confirmou aqui: medindo na
#: RBS 1223 com 11 parâmetros, 24 caminhantes deram R-hat 1,39 e 64 deram 4,18.
#: Parte disso é que mais cadeias tornam o próprio R-hat um teste mais severo.
#: Por isso o número recomendado é reportado como sugestão a comparar, e não
#: como diagnóstico.
WALKERS_PER_DIMENSION_MINIMUM = 2
WALKERS_PER_DIMENSION_RECOMMENDED = 4


def required_walkers(dimensions: int) -> int:
    """Mínimo para o enxame conseguir gerar o espaço de parâmetros."""
    return max(4, 2 * (WALKERS_PER_DIMENSION_MINIMUM * dimensions + 1) // 2 * 2)


def affine_rank(states: list[list[float]], tolerance: float = 1.0e-9) -> int:
    """Dimensão do espaço que o enxame consegue alcançar.

    O passo de esticamento de Goodman-Weare move um caminhante SÓ ao longo da
    reta que o liga ao parceiro: a proposta é p + z (x - p). Daí sai um fato
    que não é sutileza numérica e sim uma limitação exata do método — **o
    conjunto alcançável é a envoltória afim das posições iniciais**, para
    sempre. Se o enxame nasce sem gerar as N direções, as que faltam ficam
    congeladas em todos os passos, e a posteriori que sai é a de um problema
    de dimensão menor.

    Pior: o diagnóstico não via isso. Uma coordenada congelada tem variância
    interna nula, e o rhat acima devolvia 1,0 — "convergiu". Medido: 400
    passos, amplitude explorada exatamente 0, R-hat reportado 1,00.
    """
    if not states:
        return 0
    origin = states[0]
    basis: list[list[float]] = []
    for state in states[1:]:
        vector = [a - b for a, b in zip(state, origin)]
        for row in basis:                       # ortogonaliza contra a base
            projection = sum(x * y for x, y in zip(vector, row))
            vector = [x - projection * y for x, y in zip(vector, row)]
        length = math.sqrt(sum(x * x for x in vector))
        if length > tolerance:
            basis.append([x / length for x in vector])
    return len(basis)


def rhat(chain_values: list[list[float]]) -> float | None:
    if len(chain_values) < 2 or min(map(len, chain_values)) < 2:
        return None
    n = min(map(len, chain_values))
    trimmed = [values[-n:] for values in chain_values]
    means = [statistics.fmean(values) for values in trimmed]
    within = statistics.fmean(statistics.variance(values) for values in trimmed)
    if within <= 0.0:
        # Nenhuma cadeia se moveu neste parâmetro. Se todas pararam no MESMO
        # ponto ele está congelado; se pararam em pontos DIFERENTES a
        # discordância é total. Nos dois casos "1,0" — o valor da convergência
        # perfeita — é a resposta exatamente invertida, e era o que saía daqui.
        # Ver affine_rank: num enxame afim-invariante isto não é hipótese
        # remota, é o modo de falha que pontos iniciais repetidos produzem.
        return None
    between = n * statistics.variance(means)
    variance_hat = ((n - 1) * within + between) / n
    return max(1.0, math.sqrt(max(0.0, variance_hat / within)))


#: Lags guardados no artefato. A soma de Geyer percorre a ACF INTEIRA; isto
#: limita so o que vai para o resultado.json, que com 1e5 lags por parametro
#: ficaria inutilizavel.
LAGS_NO_ARTEFATO = 100

#: Passos por tau abaixo dos quais a propria estimativa de tau nao e confiavel.
#: Regra de bolso usual para a sequencia positiva inicial; a auditoria de
#: 29/09/2026 observou que 40 mil passos cobriam so 14 a 20 tau.
PASSOS_POR_TAU_MINIMO = 50.0


def autocorrelation_ess(chain_values: list[list[float]],
                        max_lag: int | None = None) -> dict:
    """ACF media do enxame e ESS, com a soma de Geyer SEM truncamento.

    A versao anterior cortava em 100 lags, e isso nao era uma escolha de
    estimador: era o preco de uma ACF calculada por soma dupla em Python puro,
    que custa O(n^2) por cadeia. Medido na segunda metade da corrida de
    setembro, tau fica entre 1200 e 2600 passos em TODOS os parametros, e a
    auditoria de 29/09/2026 estimou 2028, 2842 e 2192 para u, T1 e NH pelo
    mesmo caminho. Com a ACF em lag 100 ainda valendo 0,855 para u, a soma
    truncada devolvia tau de ordem 185 e inflava o ESS por mais de uma ordem
    de grandeza.

    Agora a ACF sai de uma FFT, o que torna a soma completa barata, e o
    resultado carrega `passosPorTau`: com menos de 50 passos por tau a
    propria estimativa de tau e pouco confiavel, e o campo `confiavel` diz
    isso em vez de deixar o leitor supor. ATENCAO: `confiavel` falso sinaliza
    pouca confiabilidade, e NAO transforma o tau devolvido em limite inferior.

    Conferido contra uma realizacao de AR(1) com rho = 0,99, cujo tau vale
    (1+rho)/(1-rho) = 199: estimou 184,6, ou seja 7 % abaixo. Isso mostra
    subestimacao NAQUELA realizacao, nao um vies unilateral garantido, e uma
    unica realizacao nao estabelece o comportamento do estimador. Portanto o
    ESS daqui e ESTIMATIVA POTENCIALMENTE OTIMISTA, nao limite superior
    certificado. Separar tres coisas que este numero nao distingue: precisao do
    estimador, mistura do enxame e exploracao dos modos.

    `sequenciaTerminou` falso significa que os pares nunca ficaram negativos
    dentro da cadeia, isto e, a soma foi interrompida pelo FIM DA CADEIA e nao
    pelo criterio de Geyer. Nesse caso o tau nao foi estimado, foi cortado.

    `max_lag` fica aceito para nao quebrar quem chama com ele, e limita apenas
    o trecho de ACF guardado.

    O limiar de 50 passos por tau vive em PASSOS_POR_TAU_MINIMO; se mudar la,
    mudar esta frase tambem. O numero esta escrito a mao de proposito: aplicar
    o operador de formatacao ao literal faz dele uma EXPRESSAO, e a funcao fica
    sem docstring, com __doc__ igual a None. Foi o que aconteceu aqui antes.
    """
    import numpy as _np

    vazio = {"autocorrelation": [], "tau": None, "ess": None,
             "passosPorTau": None, "confiavel": False}
    if not chain_values:
        return vazio
    length = min(map(len, chain_values))
    if length < 4:
        return dict(vazio, autocorrelation=[1.0])
    x = _np.asarray([list(values[-length:]) for values in chain_values],
                    dtype=float)
    # cadeias paradas nao entram: variancia zero nao tem ACF definida
    movem = x.var(axis=1) > 0.0
    if not movem.any():
        return dict(vazio, autocorrelation=[1.0])
    x = x[movem]

    tamanho = 1
    while tamanho < 2 * length:
        tamanho *= 2
    centrado = x - x.mean(axis=1, keepdims=True)
    espectro = _np.fft.rfft(centrado, n=tamanho, axis=1)
    acf = _np.fft.irfft(espectro * _np.conj(espectro), n=tamanho,
                        axis=1)[:, :length].real
    acf /= acf[:, :1]
    media = acf.mean(axis=0)

    # sequencia positiva inicial de Geyer, sobre a ACF inteira
    tau = 1.0
    terminou = False      # a sequencia positiva acabou sozinha, como deve
    lag = 1
    while lag < length:
        par = media[lag] + (media[lag + 1] if lag + 1 < length else 0.0)
        if par <= 0.0:
            terminou = True
            break
        tau += 2.0 * par
        lag += 2
    tau = max(1.0, float(tau))
    total = x.shape[0] * length
    por_tau = length / tau
    guardar = min(int(max_lag) if max_lag else LAGS_NO_ARTEFATO, length - 1)
    return {"autocorrelation": [float(v) for v in media[:guardar + 1]],
            "tau": tau, "ess": min(float(total), total / tau),
            "passosPorTau": float(por_tau),
            "sequenciaTerminou": bool(terminou),
            "confiavel": bool(terminou and por_tau >= PASSOS_POR_TAU_MINIMO)}


#: Mensagens com que o motor recusa um vetor alegando razao FISICA, de
#: `run_fit_worker` em engine/main.cpp. ELAS NAO VALEM COMO PROBABILIDADE ZERO.
#: Uma recusa so pode ser lida como probabilidade nula quando corresponde a uma
#: restricao fisica EXPLICITA do alvo, e as restricoes explicitas deste alvo
#: estao todas em `FitProblem.in_prior`: caixa, compacidade abaixo de 0,985,
#: banda de massa, ordenacao de simetria, fisicalidade do beaming. Logo um vetor
#: que passou pela priori e foi recusado pelo motor revela DESACORDO entre a
#: priori declarada e o dominio do motor, que e defeito e nao medida. A lista
#: serve para nomear a causa no relato, nao para absolver a ocorrencia.
#:
#: Verificado em 06/10/2026, para o caso que mais preocupava: a priori checa
#: `u >= 0.985` na compacidade amostrada e o motor recheca em `2 C M / R` com a
#: massa reconstruida. Em 1,2 milhao de vetores de fronteira, seis raios de 7,5 a
#: 13 km, o percurso de ida e volta NUNCA cruzou o limite. As duas constantes sao
#: o mesmo double, 1,4766250385.
MOTIVOS_FISICOS = (
    "parameters are outside physical bounds",
    "compactness is outside physical bounds",
    "received a negative nH",
)


def gama_de_tres_cumulantes(k1: float, k2: float, k3: float):
    """Gamma DESLOCADA casada em tres cumulantes: (forma, escala, deslocamento).

    E o ajuste classico para soma de Gammas com escalas diferentes, que e o que a
    posterior do fundo latente e. Com k3 <= 0 cai para o casamento em dois
    cumulantes, sem deslocamento. Validada contra 200 mil sorteios da soma
    exata: distancia de Kolmogorov-Smirnov 0,0012 a 0,0027 contra piso de ruido
    0,00304, logo indistinguivel da exata nessa medida. Ver
    scripts/marginaliza_fundo.py.
    """
    if not (k2 > 0.0):
        return 1.0, 0.0, k1
    if k3 <= 0.0:
        return k1 * k1 / k2, k2 / k1, 0.0
    escala = k3 / (2.0 * k2)
    forma = 4.0 * k2 ** 3 / (k3 * k3)
    return forma, escala, k1 - forma * escala


def densidade_gama(b, forma: float, escala: float, deslocamento: float):
    """Densidade da Gamma deslocada, em array. Zero onde o suporte nao chega."""
    import numpy as _np
    if not (escala > 0.0):
        return _np.zeros_like(_np.asarray(b, float))
    x = (_np.asarray(b, float) - deslocamento) / escala
    saida = _np.zeros_like(x)
    bom = x > 0.0
    saida[bom] = _np.exp((forma - 1.0) * _np.log(x[bom]) - x[bom]
                         - math.lgamma(forma) - math.log(escala))
    return saida


class FalhaDoMotor(RuntimeError):
    """Interrompe a corrida na primeira falha do motor que nao se recuperou.

    Nao herda de RuntimeError por acidente: `FitProblem.evaluate` captura
    RuntimeError do protocolo do worker, e esta excecao tem que ATRAVESSAR esse
    except, entao ele a relanca explicitamente.
    """


class DiagnosticoDoMotor:
    """Separa PROPOSTA FORA DO SUPORTE de FALHA DO MOTOR, e nao deixa a segunda
    virar rejeicao.

    HISTORICO DESTA CLASSE, porque a primeira versao estava errada. Antes dela,
    tres causas devolviam o mesmo `-inf` sem registro: `in_prior` falso, o motor
    devolvendo `{"status":"error"}` e excecao do lado do Python. A primeira
    correcao contou as tres em separado e passou a abortar acima de um teto, 1%
    das avaliacoes e nunca antes de 20 falhas. O revisor recusou esse desenho, e
    com razao:

        falha numerica nao pode ser rejeicao de proposta ATE o teto. Cada
        rejeicao dessas ja altera o nucleo de transicao e pode excluir uma
        regiao valida. O teto controla alerta, nao torna a cadeia correta
        abaixo dele.

    A politica agora e:

      1. falha do motor dispara RECUPERACAO DETERMINISTICA: o mesmo vetor e
         reavaliado num worker NOVO. O motor e deterministico, verificado, de
         modo que a reavaliacao ou devolve o mesmo resultado, e entao a falha era
         do estado do processo, ou devolve o mesmo erro, e entao a falha e do
         vetor. Nada e perturbado, nada e substituido: recuperar nao pode mudar o
         alvo.
      2. se a recuperacao nao der avaliacao valida, a corrida GRAVA CHECKPOINT E
         PARA, na primeira ocorrencia. Nao existe teto para isso.
      3. recusa FISICA do motor dentro da priori tambem para, porque as
         restricoes explicitas do alvo estao todas em `in_prior`: se o motor
         recusa o que a priori aceitou, a priori declarada e o dominio do modelo
         discordam, e qual dos dois corrigir e decisao humana.
      4. o teto sobrevive apenas como ALERTA sobre falhas RECUPERADAS, que nao
         enviesam a cadeia mas denunciam processo doente.
    """

    #: Reavaliacoes do MESMO vetor em worker novo antes de desistir.
    TENTATIVAS_DE_RECUPERACAO = 1

    def __init__(self, tentativas: int = TENTATIVAS_DE_RECUPERACAO,
                 alerta_de_recuperadas: float = 0.001, guardar: int = 5):
        self.tentativas = max(0, int(tentativas))
        self.alerta_de_recuperadas = float(alerta_de_recuperadas)
        self.guardar = int(guardar)
        self.avaliacoes = 0
        self.fora_do_suporte = 0
        self.recuperadas = 0
        self.nao_recuperadas = 0
        self.recusa_fisica = 0
        self.mensagens: dict[str, int] = {}
        self.excecoes: dict[str, int] = {}
        self.exemplos: list[dict] = []
        self.parada: dict | None = None
        self._anunciadas: set[str] = set()
        self._alertou = False
        self._trava = threading.Lock()

    # -- registro ---------------------------------------------------------
    def conta_avaliacao(self) -> None:
        with self._trava:
            self.avaliacoes += 1

    def fora(self, valores) -> None:
        with self._trava:
            self.fora_do_suporte += 1

    def tentativa_falhou(self, valores, causa: str, detalhe: str,
                         tentativa: int, nomes=None) -> None:
        """Uma tentativa falhou. Ainda pode haver recuperacao; nao decide nada."""
        with self._trava:
            if causa == "excecao":
                tipo = detalhe.split(":", 1)[0]
                self.excecoes[tipo] = self.excecoes.get(tipo, 0) + 1
                chave = tipo
            else:
                self.mensagens[detalhe] = self.mensagens.get(detalhe, 0) + 1
                chave = detalhe
            self._guarda(valores, causa, detalhe, nomes, tentativa)
            anuncia = chave not in self._anunciadas
            if anuncia:
                self._anunciadas.add(chave)
        if anuncia:
            print("AVISO: o motor falhou na tentativa %d (%s): %s"
                  % (tentativa, causa, detalhe), file=sys.stderr, flush=True)

    def recuperada(self, tentativas_gastas: int) -> None:
        with self._trava:
            self.recuperadas += 1
            self.pior_recuperacao = max(getattr(self, "pior_recuperacao", 0),
                                        int(tentativas_gastas))
            fracao = self.recuperadas / max(1, self.avaliacoes)
            alerta = (not self._alertou
                      and fracao > self.alerta_de_recuperadas
                      and self.recuperadas >= 10)
            if alerta:
                self._alertou = True
        if alerta:
            print("AVISO: %d de %d avaliacoes precisaram de reavaliacao em worker "
                  "novo (%.3f%%). A cadeia nao fica enviesada por isso, porque o "
                  "valor usado e o do motor, mas a taxa indica processo doente."
                  % (self.recuperadas, self.avaliacoes, 100.0 * fracao),
                  file=sys.stderr, flush=True)

    def sem_recuperacao(self, valores, causa: str, detalhe: str, nomes=None):
        """Nao houve avaliacao valida. A corrida PARA aqui, na primeira vez."""
        fisica = any(m in detalhe for m in MOTIVOS_FISICOS)
        with self._trava:
            self.nao_recuperadas += 1
            if fisica:
                self.recusa_fisica += 1
            self.parada = {
                "causa": causa, "detalhe": detalhe,
                "recusa_fisica_alegada": fisica,
                "vetor": dict(zip(list(nomes), [float(v) for v in valores]))
                         if nomes else None,
                "valores": [float(v) for v in valores],
                "tentativas": self.tentativas + 1,
                "avaliacoes_ate_aqui": self.avaliacoes,
            }
        if fisica:
            motivo = (
                "O motor alega limite FISICO, mas isso nao vale como "
                "probabilidade zero: as restricoes explicitas deste alvo estao "
                "todas em in_prior, e este vetor passou por ela. Ou a priori "
                "admite o que o modelo nao cobre, e entao a priori tem de "
                "declarar a restricao, ou o limite e do motor e nao do alvo, e "
                "entao ele tem de ser afrouxado ou documentado como limitacao.")
        else:
            motivo = (
                "Devolver -inf aqui excluiria da cadeia uma regiao que talvez "
                "tenha probabilidade, e o nucleo de transicao ja teria sido "
                "alterado pela rejeicao.")
        return FalhaDoMotor(
            "o motor nao produziu avaliacao valida para um vetor DENTRO da "
            "priori, nem apos %d reavaliacao(oes) em worker novo. Causa: %s. "
            "Detalhe: %s. %s A corrida para aqui, com checkpoint gravado."
            % (self.tentativas, causa, detalhe, motivo))

    def _guarda(self, valores, causa, detalhe, nomes, tentativa) -> None:
        if len(self.exemplos) >= self.guardar:
            return
        registro = {"causa": causa, "detalhe": detalhe, "tentativa": tentativa,
                    "valores": [float(v) for v in valores]}
        if nomes:
            registro["vetor"] = dict(zip(list(nomes), [float(v) for v in valores]))
        self.exemplos.append(registro)

    # -- relato -----------------------------------------------------------
    def relatorio(self) -> dict:
        with self._trava:
            return {
                "avaliacoes": self.avaliacoes,
                "foraDoSuporte": self.fora_do_suporte,
                "falhasRecuperadas": self.recuperadas,
                "piorRecuperacao": getattr(self, "pior_recuperacao", 0),
                "falhasNaoRecuperadas": self.nao_recuperadas,
                "recusaFisicaAlegada": self.recusa_fisica,
                "fracaoRecuperada": self.recuperadas / max(1, self.avaliacoes),
                "tentativasDeRecuperacao": self.tentativas,
                "mensagensDoMotor": dict(self.mensagens),
                "excecoes": dict(self.excecoes),
                "exemplos": list(self.exemplos),
                "parada": self.parada,
                "politica": (
                    "falha do motor NAO e rejeicao de proposta. Ela dispara "
                    "reavaliacao deterministica do MESMO vetor em worker novo; "
                    "sem avaliacao valida, a corrida grava checkpoint e para na "
                    "primeira ocorrencia. Nao ha teto abaixo do qual a falha "
                    "seja aceitavel. Recusa fisica dentro da priori tambem para, "
                    "porque as restricoes explicitas do alvo estao em in_prior."),
                "nota": ("foraDoSuporte e a unica rejeicao legitima; "
                         "falhasRecuperadas nao enviesam a cadeia, porque o "
                         "valor usado e o do motor, e so denunciam processo "
                         "doente; falhasNaoRecuperadas e sempre 0 numa corrida "
                         "que terminou."),
            }


class FitProblem:
    def __init__(self, request: dict, events: list[tuple[float, float]],
                 metadata: dict[str, str] | None = None,
                 background: list[tuple[float, float]] | None = None):
        self.request = request
        self.metadata = metadata or {}
        #: Contabilidade das causas de `-inf`. Existe por padrao: antes, falha do
        #: motor e proposta fora da priori eram indistinguiveis, e o relatorio
        #: nao dizia se alguma regiao fora excluida por defeito numerico.
        self.diagnostico = DiagnosticoDoMotor(
            tentativas=int(request.get("engineRecoveryAttempts",
                                       DiagnosticoDoMotor.TENTATIVAS_DE_RECUPERACAO)))
        self.period = float(request["period"])
        self.distance = float(request["distance"])
        #: Distancia livre (kpc) com priori uniforme distanceRange; padrao: fixa.
        self.fit_distance = bool(request.get("fitDistance", False))
        self.distance_range = tuple(request.get("distanceRange", [0.1, 2.0]))
        # A exposição multiplica a taxa prevista, então errá-la desloca
        # diretamente a área emissora inferida. O intervalo entre o primeiro e o
        # último evento não serve: com lacunas de GTI ele superestima o tempo
        # vivo — numa observação real do EPIC-pn, em quase 50%.
        self.exposure = float(request.get("exposure")
                              or self.metadata.get("exposure_s")
                              or (max(t for t, _ in events) - min(t for t, _ in events)))
        self.energy_min = float(request["energyMin"])
        self.energy_max = float(request["energyMax"])
        self.energy_bins = int(request["energyBins"])
        self.phase_bins = int(request["phaseBins"])
        self.max_images = int(request.get("maxImages", 2))
        # Modo "curva de luz": a verossimilhança soma sobre a energia e ajusta só
        # a FORMA do pulso dobrado, com a normalização da fonte perfilada
        # analiticamente (o fundo entra como termo aditivo fixo). Assim a
        # GEOMETRIA é constrangida pelo formato de dois picos, e não pelo
        # espectro — que domina o grid fase-energia e achata o pulso. É a receita
        # padrão de pulse-profile modeling (a forma fixa a geometria; o nível, a
        # normalização, vem do espectro). Ver Hambaryan et al. 2011.
        self.fit_light_curve = bool(request.get("fitLightCurve", False))
        # Spots de corpo negro (superfície condensada) sobre o fundo de atmosfera.
        self.blackbody_spots = bool(request.get("blackbodySpots", False))
        self.spot_overlay = bool(request.get("spotOverlay", False))
        self.line_cyclotron = bool(request.get("lineCyclotron", False))
        # |B| variavel na superficie (lei dipolar, a_B=0.25 = dipolo centrado):
        # B passa a ser o campo POLAR e a linha ciclotron vira blend. None = uniforme.
        self.field_peaking = request.get("fieldPeaking", None)
        # Dipolo de Schwarzschild: a_B e alpha_B do u corrente (B_p = campo polar proprio).
        self.field_gr = bool(request.get("fieldGR", False))
        # Geometria antiga do campo, intensidade pelo polo mais proximo. Falsa
        # por padrao: o dipolo coerente e o certo. Existe para reproduzir as
        # corridas publicadas antes de 01/10/2026.
        self.field_nearest_pole = bool(request.get("fieldNearestPole", False))
        # Resolucao da esfera, independente dos spots. Zero mantem a regra antiga.
        self.sphere_bands = int(request.get("sphereBands", 0))
        # Refino das celulas cortadas pelas fronteiras dos lobos termicos.
        self.boundary_refine = int(request.get("boundaryRefine", 1))
        # Modo de camadas: contínuo de corpo negro + feixe da atmosfera, em toda a
        # superfície (atmosfera fina sobre condensada, à la Hambaryan).
        self.layered_atmosphere = bool(request.get("layeredAtmosphere", False))
        #: OFFSET DE FASE OBSERVACIONAL, em CICLOS, repassado ao motor. E o
        #: operador de alinhamento EXPLICITO, que substitui o uso de `magAzim`
        #: para esse fim. `magAzim` e um parametro FISICO e, medido, nao e um
        #: operador de fase exato: rolar a grade e mover o azimute diferem por
        #: 0,061 em ln L e 3,0e-04 por celula, porque o azimute move a
        #: configuracao magnetica dentro da malha, que e fixa no referencial da
        #: estrela. O offset explicito nao move nada dentro da malha, e por isso
        #: e exato em deslocamento inteiro.
        #:
        #: Convencao MEDIDA, nao suposta (docs/valida_offset.json):
        #:     m_offset[p] = m_0[p + offset . N]
        #: isto e, offset positivo move o padrao para bins MENORES, por
        #: offset . N bins, com razao 1,000000.
        #:
        #: A EXPOSICAO NAO E DESLOCADA: ela fica nas fases observadas.
        self.observation_phase_offset = float(
            request.get("observationPhaseOffsetCycles", 0.0))
        #: QUADRATURA DE FASE: M sub-amostras por bin OBSERVADO. O motor e
        #: pedido com `phase_bins * M` fases e a pontuacao faz a MEDIA de cada
        #: grupo de M antes de comparar com o dado, que continua em phase_bins.
        #:
        #: POR QUE ISTO VIVE AQUI, e nao no lancador. O alvo escolhido usa M = 5
        #: (docs/frenteA_fechada.json). Se a reotimizacao avaliasse com M = 1 ela
        #: acharia o maximo de uma FUNCAO DIFERENTE da que os pilotos amostram, e
        #: eu havia escrito exatamente isso no lancador da frente C, por engano.
        #: Pondo M no adaptador, reotimizador e amostrador herdam a convencao.
        #:
        #: Medido: o ponto medio, M = 1, erra 0,63 nas diferencas de ln L contra
        #: M = 15, e M = 5 erra 0,0908.
        self.media_de_fase = max(1, int(request.get("phaseAverage", 1)))
        if self.media_de_fase % 2 == 0 and self.media_de_fase > 1:
            raise ValueError(
                "phaseAverage = %d e PAR, e a convencao de centro de bin exige "
                "M IMPAR: o centro do bin grosso cai na fronteira de dois finos "
                "e as fases pedidas nao sao membros da grade fina. Verificado "
                "com fracoes para M de 1 a 8" % self.media_de_fase)
        #: Digitos significativos da saida do motor. None mantem o padrao dele,
        #: 10, e portanto a saida de sempre. Vale 17 nas VALIDACOES, onde o piso
        #: de impressao de 1e-9 relativo seria confundido com diferenca de
        #: conta: medido, duas rotas que calculam o mesmo double por ordens
        #: diferentes imprimem valores separados por um ou dois quanta do ultimo
        #: digito.
        self.output_precision = request.get("outputPrecision")
        self.instrument = str(request["instrument"])
        # Uma resposta enviada com os dados vence o catálogo: ela é *daquela*
        # observação, enquanto o perfil do manifesto é de uma configuração
        # representativa do instrumento.
        # Quem monta responseProfile a partir de arfData/rmfData e o SERVIDOR.
        # Um pedido que traz os dados crus e nao o perfil veio de codigo que
        # pulou esse passo, e o silencio custa caro: medido, o adaptador caia
        # no perfil generico do manifesto, o modelo saia 40 vezes abaixo do
        # dado e a verossimilhanca da melhor amostra despencava de 52076 para
        # 6391 — com o grafico parecendo um ajuste catastrofico que nao existia.
        if (request.get("arfData") or request.get("rmfData")) and \
                not isinstance(request.get("responseProfile"), dict):
            raise ValueError(
                "o pedido traz arfData/rmfData mas nao responseProfile: monte o "
                "perfil com scripts/response_profile.cached() antes de construir "
                "o FitProblem, senao a resposta do instrumento vira a generica "
                "do manifesto e a area efetiva fica errada")
        uploaded = request.get("responseProfile")
        if isinstance(uploaded, dict) and uploaded.get("response_file"):
            self.profile = uploaded
            self.instrument = str(uploaded.get("id", self.instrument))
        else:
            profiles = {p["id"]: p for p in json.loads(MANIFEST.read_text())["profiles"]}
            if self.instrument not in profiles:
                raise ValueError(f"unknown instrument profile: {self.instrument}")
            self.profile = profiles[self.instrument]
        self.energy_width = (self.energy_max - self.energy_min) / self.energy_bins
        # Recorte da banda dura: so entram na verossimilhanca os bins de energia
        # cujo CENTRO fica abaixo de energyFitMax (default = energy_max, ou seja,
        # sem recorte). Mantem o grid e as dimensoes intactos, entao o resumeFrom
        # continua valido e nao e preciso re-binar os dados combinados.
        self.energy_fit_max = float(request.get("energyFitMax", self.energy_max))
        self._fit_bins = [ip * self.energy_bins + ie
                          for ip in range(self.phase_bins)
                          for ie in range(self.energy_bins)
                          if self.energy_min + (ie + 0.5) * self.energy_width
                          < self.energy_fit_max]
        #: Quais bins de ENERGIA entram, que e o que o alvo marginalizado precisa:
        #: ele integra por canal de energia sobre todas as fases, e _fit_bins
        #: recorta bins de energia inteiros, nunca fases isoladas.
        self._bins_de_energia_ajustados = [
            ie for ie in range(self.energy_bins)
            if self.energy_min + (ie + 0.5) * self.energy_width
            < self.energy_fit_max]
        self.observed = [0] * (self.phase_bins * self.energy_bins)
        selected = 0
        phase_reference = float(request.get("phaseReference", 0.0))
        for arrival, energy in events:
            if not self.energy_min <= energy < self.energy_max:
                continue
            phase = ((arrival - phase_reference) / self.period) % 1.0
            ip = min(self.phase_bins - 1, int(phase * self.phase_bins))
            ie = min(self.energy_bins - 1,
                     int((energy - self.energy_min) / self.energy_width))
            self.observed[ip * self.energy_bins + ie] += 1
            selected += 1
        if selected == 0:
            raise ValueError("no events remain in the selected energy band")
        self.selected = selected
        self.background = self._bin_background(background)
        # Injeção de um grid JÁ combinado (co-adição de observações): substitui o
        # observado, a exposição e o fundo por arrays prontos, mantendo toda a
        # verossimilhança intacta. É como se fosse uma observação só, mais longa.
        #: Exposicao por fase MEDIDA, quando o produto a traz; None usa a
        #: formula analitica. Ver `tempo_representado`.
        self.exposicao_por_fase = None
        prepared = request.get("preparedDataNpz")
        if prepared:
            import numpy as _np
            data = _np.load(prepared)
            self.observed = [int(x) for x in data["observed"]]
            self.exposure = float(data["exposure"])
            self.selected = int(sum(self.observed))
            self.background = [float(x) for x in data["background"]]
            if len(self.observed) != self.phase_bins * self.energy_bins:
                raise ValueError("preparedDataNpz: grid observado com forma errada")
            if len(self.background) != self.energy_bins:
                raise ValueError("preparedDataNpz: fundo com número de bins errado")
            # O produto v2 traz os PESOS e as CONTAGENS BRUTAS de cada par
            # observacao-canal por bin, o que permite escolher a priori do fundo
            # latente aqui, na definicao do alvo, em vez de embuti-la no dado.
            # Ver scripts/preparar_fundo_v2.py e scripts/priori_do_fundo.py.
            if "fundo_pesos" in data.files:
                self.fundo_pesos = _np.asarray(data["fundo_pesos"], float)
                self.fundo_contagens = _np.asarray(data["fundo_contagens"], float)
                self.fundo_deslocamento = _np.asarray(data["fundo_deslocamento"], int)
                if "fundo_largura_do_canal" not in data.files:
                    raise ValueError(
                        "o produto traz fundo_pesos mas nao "
                        "fundo_largura_do_canal, e a priori proporcional a "
                        "largura precisa dela; regere com "
                        "scripts/preparar_fundo_v2.py")
                self.fundo_largura_do_canal = _np.asarray(
                    data["fundo_largura_do_canal"], float)
                if "fundo_canal_do_termo" in data.files:
                    self.fundo_canal_do_termo = _np.asarray(
                        data["fundo_canal_do_termo"], int)
                self.produto_do_fundo = str(prepared)
                self.versao_do_produto = (str(data["versao"])
                                          if "versao" in data.files else "1")
            # EXPOSICAO POR FASE MEDIDA, dos quadros (extensao EXPOSUnn), em vez
            # da formula analitica de `represented_time`. A formula supoe
            # cobertura uniforme a menos do ultimo ciclo parcial; os quadros
            # dizem outra coisa. Medido: a correcao desloca diferencas de ln L
            # em ate 1,0105, chegando a 15,1 por cento numa comparacao de
            # perfis, e a amplitude por OBSERVACAO chega a 1,23 por cento,
            # contra 0,17 por cento no coadicionado, porque as amplitudes
            # individuais se cancelam parcialmente ao somar com alinhamentos
            # diferentes. Logo isto pesa MAIS na forma por observacao.
            if "exposicao_por_fase" in data.files:
                pe = _np.asarray(data["exposicao_por_fase"], float)
                if pe.shape != (self.phase_bins,):
                    raise ValueError(
                        "exposicao_por_fase com %d valores, e a grade tem %d "
                        "bins de fase" % (pe.size, self.phase_bins))
                if not _np.all(pe > 0.0):
                    raise ValueError("exposicao_por_fase com bin nao positivo: "
                                     "um bin sem exposicao nao tem modelo")
                # A soma TEM de bater com a exposicao, senao uma das duas esta
                # em outra convencao e o modelo sai reescalado em silencio.
                razao = float(pe.sum()) / self.exposure
                if abs(razao - 1.0) > 1.0e-6:
                    raise ValueError(
                        "soma da exposicao_por_fase e %.6f vezes a exposicao; "
                        "as duas precisam estar na mesma convencao" % razao)
                self.exposicao_por_fase = [float(x) for x in pe]
        # ALVO NOVO, com o fundo MARGINALIZADO, selecionavel; o alvo antigo,
        # com o fundo fixo, segue como CONTROLE CONDICIONAL. A decisao de
        # definir os dois e do revisor, 06/10/2026. O que muda e o tratamento do
        # fundo, nao as contagens observadas.
        #
        # CONVENCOES DE NORMALIZACAO, que NAO sao comparaveis entre os alvos:
        #   controle: ln L = soma_b [ n_b ln mu_b - mu_b ], sem o termo ln n_b!
        #   novo:     ln L = soma_e ln integral pi_e(b) prod_p Poisson(...) db,
        #             tambem sem ln n!, e com pi_e normalizada na janela de
        #             integracao
        # Os dois omitem constantes diferentes. Dentro de um alvo elas cancelam
        # na comparacao entre pontos; ENTRE alvos o valor absoluto de ln L nao
        # significa nada.
        self.marginalizar_fundo = bool(request.get("marginalizeBackground", False))
        #: Forma da priori do fundo latente, em pseudo-contagens por keV por
        #: observacao. PROPORCIONAL A LARGURA do canal, que e a unica escolha
        #: coerente sob reagrupamento em energia: somar canais adjacentes soma as
        #: formas e da a0 vezes a largura do grupo, de modo que a afirmacao a
        #: priori depende so do intervalo. Uma priori fixa POR CANAL seria
        #: afirmacao sobre a grade, e muda a posterior ao reagrupar, medido em
        #: 3,89 por cento num intervalo de 30 canais. Ver
        #: scripts/priori_do_fundo.py.
        #: CONVENCAO FIDUCIAL, escolhida pelo revisor em 06/10/2026:
        #: a0 = 0,01 por keV, com beta = 0. O fundamento e OPERACIONAL, nao
        #: fisico: qualquer a0 > 0 evita componentes posteriores improprias, e
        #: 0,01 custou 3e-7 em lnL e 0,0019 por cento no fundo do bin 39 nos
        #: casos medidos. a0 = 0,001 e 0,1 ficam como sensibilidades declaradas.
        #:
        #: DUAS DISTINCOES QUE O REVISOR PEDIU PARA REGISTRAR AQUI:
        #:
        #: 1. POSTERIOR PROPRIA NAO TORNA A PRIORI PROPRIA. Com beta = 0 a
        #:    priori lambda^(a0 w - 1) continua impropria, e portanto a
        #:    normalizacao da posterior fica definida a menos de uma constante
        #:    arbitraria. Consequencia pratica: EVIDENCIA ABSOLUTA e FATOR DE
        #:    BAYES entre modelos nao podem ser usados sem resolver essa
        #:    constante. Pesos de REGIOES dentro do mesmo alvo normalizado sao
        #:    outra questao, e seguem validos, porque a constante cancela na
        #:    razao. A finitude da normalizacao esta demonstrada em
        #:    docs/normalizacao_do_alvo_v2.md.
        #:
        #: 2. CONTAGEM ZERO NAO DEMONSTRA FUNDO EXATAMENTE NULO. A massa pontual
        #:    em zero era convencao do limite a0 -> 0. A forma positiva pequena a
        #:    substitui, mas ainda atribui FORTE CONCENTRACAO perto de zero: com
        #:    a0 = 0,01 a forma de um canal vazio e 4,8e-5, cuja media e 4,8e-5
        #:    contagens. Continuidade numerica em a0 nao demonstra adequacao
        #:    fisica dessa concentracao.
        self.fundo_a0 = float(request.get("backgroundPriorPerKeV", 0.01))
        #: Taxa da priori Gama. FIXA EM ZERO por convencao declarada: e a unica
        #: escolha em que a media a posteriori reproduz a taxa medida quando
        #: a0 -> 0, porque beta encolhe o fundo por 1/(1+beta) uniformemente.
        #: Exposta para que a convencao fique visivel e auditavel, nao para ser
        #: variada sem justificativa propria.
        self.fundo_beta = float(request.get("backgroundPriorRate", 0.0))
        self.fundo_nos = int(request.get("backgroundQuadratureNodes", 64))
        self.absorption_table = ABSORPTION if ABSORPTION.is_file() else None
        self.nh = max(0.0, float(request.get("nh", 0.0)))
        self.nh_max = max(self.nh, float(request.get("nhMax", 5.0)))
        # N_H entra no ajuste quando pedido explicitamente ou quando o valor
        # inicial não é nulo; sem tabela de absorção não há como aplicá-lo.
        self.fit_nh = bool(request.get("fitNh", self.nh > 0.0)) and \
            self.absorption_table is not None
        if self.nh > 0.0 and self.absorption_table is None:
            raise ValueError("nH was requested but instrument_data/absorption/"
                             "tbabs_wilm.csv is missing; run "
                             "scripts/build_absorption_table.py")

        self.names = ["mass", "radius", "inclination", "phaseOffset"]
        self.bounds = [(0.8, 2.8), (7.0, 20.0), (0.0, 180.0), (-180.0, 180.0)]
        self.steps = [0.025, 0.12, 1.2, 2.0]
        self.initial = [float(request["mass"]), float(request["radius"]),
                        float(request["inclination"]), float(request.get("phaseOffset", 0.0))]
        # Faixa da INCLINACAO por pedido. Entrou em 25/09/2026: a posterior da
        # RBS 1223 tem duas bacias de inclinacao, em ~91,5 e ~95,5 graus,
        # separadas por uma barreira MEDIDA de DlnL = 60 no caminho entre as
        # medianas dos dois modos. Um amostrador de ensemble nao atravessa isso
        # (pediria da ordem de e^60 tentativas), entao o R-hat global fica em ~2,5
        # por mais iteracoes que se de. Restringir a faixa a UMA bacia converge
        # dentro dela e transforma a separacao em sistematico declarado, em vez de
        # escondido no diagnostico. Sem a chave, o comportamento e o de antes.
        faixa_i = request.get("inclinationRange")
        if faixa_i:
            lo_i, hi_i = float(faixa_i[0]), float(faixa_i[1])
            self.bounds[2] = (lo_i, hi_i)
            self.initial[2] = min(max(self.initial[2], lo_i + 1.0e-3), hi_i - 1.0e-3)
        # Ajustar a COMPACIDADE u = 2GM/(Rc²) no lugar da massa: é u que o desvio
        # de luz e o redshift constrangem, então (u, R) desemaranha o que (M, R)
        # deixava degenerado — u fixado pela forma do pulso, R pela normalização
        # do fluxo. O motor recebe M e R; a conversão u,R→M mora no worker_line.
        self.fit_compactness = bool(request.get("fitCompactness", False))
        if self.fit_compactness:
            self.names[0] = "compactness"
            self.bounds[0] = (0.05, 0.7)          # u físico de estrela de nêutrons
            self.steps[0] = 0.01
            self.initial[0] = (2.0 * _GM_SUN_C2_KM * self.initial[0]
                               / self.initial[1])  # M inicial → u inicial
        # Priors gaussianos informados, somados ao log-posterior (não são caixa).
        # Ex.: {"mass": [1.4, 0.2]} impõe M ~ N(1.4, 0.2) da população de estrelas
        # de nêutrons — quebra a degenerescência M-R quando o dado só prende z.
        self.gaussian_priors = dict(request.get("gaussianPriors", {}))
        # Piso de massa: exclui solucoes sub-solares (M < massFloor Msun), que nao
        # correspondem a estrelas de neutrons reais. Nao molda a forma do posterior
        # dentro do fisico; so remove o inexistente.
        self.mass_floor = float(request.get("massFloor", 0.0))
        #: Teto de massa (Msun) do vinculo duro em in_prior; 2.5 por padrao (era fixo).
        self.mass_cap = float(request.get("massCap", 2.5))
        self.spot_count = int(request["spotCount"])
        # Ligado por omissão: sem isto o R-hat não desce, por mais iterações
        # que se dê. Medido nesta observação, com 400 iterações: 9,19 sem a
        # restrição, 1,02 com ela.
        self.break_symmetry = bool(request.get("breakSymmetry", True))
        spots = request["spots"]
        if len(spots) != self.spot_count:
            raise ValueError("spot count does not match the supplied spot parameters")
        for index, spot in enumerate(spots):
            suffix = str(index + 1)
            self.names.extend([f"theta{suffix}", f"radius{suffix}", f"kT{suffix}"])
            self.bounds.extend([(0.0, 180.0), (0.5, 90.0), (0.01, 5.0)])
            self.steps.extend([1.5, 0.8, 0.008])
            self.initial.extend([float(spot["theta"]), float(spot["radius"]),
                                 float(spot["temperatureKev"])])
            if index > 0:
                self.names.append(f"relativePhi{suffix}")
                self.bounds.append((-180.0, 180.0))
                self.steps.append(2.0)
                reference = float(spots[0].get("phi", 0.0))
                relative = ((float(spot.get("phi", 0.0)) - reference + 180.0) % 360.0) - 180.0
                self.initial.append(relative)

        # A restrição i <= theta dobra o espaço sobre uma das metades; o ponto
        # inicial precisa ser dobrado junto, e não rejeitado. Trocar os dois não
        # muda a física — é justamente por serem indistinguíveis que a restrição
        # é legítima. Sem isto, o estado padrão da interface (i = 80, theta = 30)
        # nasce fora da priori e o ajuste morre com "none of the initial walkers
        # has finite likelihood", uma mensagem que não aponta para a causa.
        if self.break_symmetry and self.spot_count == 1 and \
                self.initial[2] > self.initial[4]:
            self.initial[2], self.initial[4] = self.initial[4], self.initial[2]

        # A linha e o feixe entram no vetor só quando escolhidos. São extensões
        # do corpo negro, não substitutos: com fitLine e fitBeaming desligados o
        # modelo é exatamente o de antes, e a comparação entre os dois é direta.
        self.fit_line = bool(request.get("fitLine", False))
        self.line_energy = float(request.get("lineEnergy", 0.3))
        self.line_width = float(request.get("lineWidth", 0.1))
        self.line_depth = float(request.get("lineDepth", 0.0))
        if self.fit_line:
            self.names.extend(["lineEnergy", "lineWidth", "lineDepth"])
            # O centro da linha tem de cair DENTRO da banda ajustada. Com o piso
            # em 0,1 keV e a banda começando em 0,15, o ajuste levava a linha
            # para fora — 0,168 keV com o piso em 0,1, profundidade 1,74 — onde
            # dado nenhum a constrange e ela vira só uma torção do contínuo na
            # borda. Uma feição que não se vê não é uma feição.
            self.bounds.extend([(self.energy_min, self.energy_max),
                                (0.02, 0.6), (0.0, 5.0)])
            self.steps.extend([0.01, 0.01, 0.05])
            # 0,1 nasceria colado no piso da priori, e um enxame que começa na
            # parede leva metade da corrida para sair dela. 0,5 é uma linha
            # rasa mas presente — se os dados não a quiserem, o ajuste a leva a
            # zero por conta própria, que é a resposta correta.
            # E o ponto inicial junto, senão nasce fora da própria priori.
            self.line_energy = min(max(self.line_energy, self.energy_min),
                                   self.energy_max)
            self.initial.extend([self.line_energy, self.line_width,
                                 self.line_depth if self.line_depth > 0.05 else 0.5])

        # Segunda linha de absorção: estrutura complexa da RBS 1223 e/ou cíclotron
        # de próton (~0,9 keV no campo ajustado), na região do excesso a alta
        # energia. Vem no vetor LOGO APÓS a primeira linha, antes da atmosfera.
        self.fit_line2 = bool(request.get("fitLine2", False))
        self.line2_energy = float(request.get("line2Energy", 0.6))
        self.line2_width = float(request.get("line2Width", 0.1))
        self.line2_depth = float(request.get("line2Depth", 0.0))
        if self.fit_line2:
            self.names.extend(["line2Energy", "line2Width", "line2Depth"])
            self.bounds.extend([(self.energy_min, self.energy_max), (0.02, 0.6), (0.0, 5.0)])
            self.steps.extend([0.01, 0.01, 0.05])
            self.line2_energy = min(max(self.line2_energy, self.energy_min), self.energy_max)
            self.initial.extend([self.line2_energy, self.line2_width,
                                 self.line2_depth if self.line2_depth > 0.05 else 0.5])

        self.fit_beaming = bool(request.get("fitBeaming", False))
        self.beaming = float(request.get("beaming", 0.0))
        # A atmosfera de dois modos substitui corpo negro + feixe: ela deriva o
        # padrão angular de transporte radiativo em vez de ajustá-lo. Um
        # parâmetro no lugar de dois, e com significado físico.
        self.fit_atmosphere = bool(request.get("fitAtmosphere", False))
        self.atmosphere = float(request.get("atmosphere", 0.0))
        if self.fit_atmosphere:
            self.names.append("atmosphere")
            # Endurecimento de cor T_X/T_ef, não a razão microscópica de
            # opacidades — o motor explica por quê. A priori 1 a 3 cobre com
            # folga o 1,5 a 2 que Ho & Lai (2001), van Adelsberg & Lai (2006) e
            # Suleimanov et al. (2009) obtêm para hidrogênio magnetizado. O
            # piso em 1 quer dizer "os dois modos saem juntos", medido contra o
            # modo ordinário — NÃO é o corpo negro, e valor nenhum de h o é: a
            # atmosfera cinza já é escurecida no bordo. A comparação com o
            # corpo negro é não aninhada, por AIC ou BIC.
            self.bounds.append((1.0, 3.0))
            self.steps.append(0.1)
            self.initial.append(self.atmosphere if self.atmosphere > 0 else 1.5)

        # A atmosfera SUBSTITUI o feixe empírico no motor, não se soma a ele.
        # Deixar os dois ligados ajustaria a e b sem que eles entrassem no
        # fluxo: dois parâmetros livres sem efeito nenhum, cuja posteriori seria
        # a priori inteira e cujo R-hat nunca desceria — e nada no resultado
        # diria que o problema era esse. Recusar é melhor do que devolver um
        # ajuste que parece ter falhado em convergir.
        # Vale para a atmosfera FIXA também: o motor usa
        # cfg.atmosphere_hardening >= 1 como chave, e não se ela está sendo
        # ajustada. Olhar só fit_atmosphere deixaria passar o caso em que o
        # usuário fixa o endurecimento e ajusta a e b — mesmo silêncio.
        # Campo magnético da superfície, em gauss. É ele que escolhe a linha da
        # tabela de anisotropia de opacidade; sem ele a atmosfera fica
        # isotrópica, e nesse caso ela não consegue produzir leque nenhum.
        # Para uma XDINS o valor de spin-down serve: B = 3,2e19 sqrt(P Pdot).
        self.magnetic_field = float(request.get("magneticField", 0.0))
        # O espectro do NSMAXG entra no lugar do corpo negro; o feixe continua
        # sendo o do modelo de dois modos, porque a tabela não carrega ângulo.
        # Precisa do campo, que é o que escolhe a linha da tabela.
        self.use_nsmaxg = bool(request.get("useNsmaxg", False))
        # A tabela de intensidade do MAGNUS: quando presente, ela substitui
        # espectro E feixe de uma vez, e as outras famílias de atmosfera ficam
        # atrás dela na cadeia do motor. Ajustar hardening ou feixe junto seria
        # contar o feixe duas vezes — o construtor recusa em vez de deixar
        # passar calado.
        self.atmosphere_table = request.get("atmosphereTable")
        self.magnetic_colatitude = float(request.get("magneticColatitude", 0.0))
        # Tabela MAGNUSI2 (com eixo de B): habilita ajustar o campo pela própria
        # atmosfera do MAGNUS, e a faixa vem do eixo da tabela.
        self.atmosphere_field_range = None
        if self.atmosphere_table:
            with open(self.atmosphere_table, "rb") as _fh:
                if _fh.read(8) == b"MAGNUSI2":
                    _n = struct.unpack("<I", _fh.read(4))[0]
                    _lb = struct.unpack(f"<{_n}f", _fh.read(4 * _n))
                    self.atmosphere_field_range = (float(_lb[0]), float(_lb[-1]))
        if self.atmosphere_table and (self.fit_atmosphere or self.fit_beaming
                                      or self.atmosphere > 0.0):
            raise ValueError("a tabela de intensidade já carrega espectro e feixe; "
                             "não se ajusta hardening nem feixe por cima dela")
        self.atmosphere_active = self.fit_atmosphere or self.atmosphere > 1.0
        if self.use_nsmaxg and not self.atmosphere_active:
            raise ValueError(
                "o espectro do NSMAXG substitui o corpo negro, mas não traz o "
                "feixe: o fluxo é um momento da intensidade e não determina a "
                "forma angular. Ligue também a atmosfera de dois modos, que é "
                "de onde o feixe vem.")
        if self.use_nsmaxg and self.magnetic_field <= 0.0:
            raise ValueError(
                "o NSMAXG precisa do campo magnético da superfície, em gauss: é "
                "ele que escolhe a tabela e põe a linha de cíclotron do próton "
                "em E = 0,0063 B_12 keV.")
        if self.atmosphere_active and self.fit_beaming:
            raise ValueError(
                "a atmosfera de dois modos e o padrão de feixe empírico são "
                "alternativas: a atmosfera deriva o feixe do transporte "
                "radiativo, e com ela ligada os parâmetros a e b não entram no "
                "modelo. Escolha um dos dois.")

        # lg(B) entra logo depois da atmosfera, e antes do feixe, que é a
        # ordem em que o motor lê. Ajusta-se o LOGARITMO porque o campo varre
        # ordens de grandeza.
        #
        # O teto da priori é a cobertura da tabela NSMAXG, e é de propósito: se
        # a posteriori encostar nele, at_bound avisa, e o aviso quer dizer uma
        # coisa concreta — os dados pedem campo mais forte do que existe tabela
        # calculada. Isso é medida, não falha.
        self.fit_log_field = bool(request.get("fitMagneticField", False))
        if self.fit_log_field:
            if self.atmosphere_field_range is not None:
                field_range = self.atmosphere_field_range   # da tabela MAGNUSI2
            elif self.use_nsmaxg:
                field_range = NSMAXG_FIELD_RANGE
            else:
                raise ValueError(
                    "ajustar o campo magnético precisa OU de uma tabela de "
                    "atmosfera MAGNUSI2 (com eixo de B), OU do espectro do NSMAXG "
                    "ligado — algo que dependa de B de forma mensurável. Só com "
                    "a anisotropia, B mal move a verossimilhança.")
            self.names.append("logMagneticField")
            self.bounds.append(field_range)
            self.steps.append(0.05)
            start = math.log10(self.magnetic_field) if self.magnetic_field > 0 \
                else 0.5 * (field_range[0] + field_range[1])
            self.initial.append(min(max(start, field_range[0] + 0.02),
                                    field_range[1] - 0.02))

        # Temperatura de fundo da estrela inteira (keV). O modelo é: a superfície
        # toda emite a essa temperatura — a MÉDIA da emissão — e os spots entram
        # por cima, descontados do fundo na sua área. O fundo sempre visível dá o
        # pulso SUAVE; um spot sobre estrela escura só sabe fazer liga-desliga.
        # Posição no vetor e na linha do worker: DEPOIS de lg B, ANTES do feixe.
        self.base_temperature = float(request.get("baseTemperature", 0.0))
        self.fit_base_temperature = bool(request.get("fitBaseTemperature", False))
        if self.fit_base_temperature:
            lo, hi = request.get("baseTemperatureRange", (0.02, 0.5))
            self.names.append("baseKT")
            self.bounds.append((float(lo), float(hi)))
            self.steps.append(min(0.01, 0.25 * (hi - lo)))
            start = self.base_temperature if self.base_temperature > 0 \
                else 0.5 * (lo + hi)
            self.initial.append(min(max(start, lo + 1.0e-3), hi - 1.0e-3))

        # Concentração `a` da lei T(θ): polos quentes, equador frio. a=0 uniforme,
        # a=1/4 dipolo clássico, a>1 calota apertada. Posição: DEPOIS de baseKT,
        # ANTES da colatitude. Precisa de baseKT (T_pole) para ter efeito.
        self.temperature_peaking = float(request.get("temperaturePeaking", 0.0))
        self.temperature_min_frac = float(request.get("temperatureMinFrac", 0.3))
        self.fit_temperature_peaking = bool(request.get("fitTemperaturePeaking", False))
        if self.fit_temperature_peaking:
            lo, hi = request.get("temperaturePeakingRange", (0.0, 4.0))
            self.names.append("peaking")
            self.bounds.append((float(lo), float(hi)))
            self.steps.append(min(0.05, 0.1 * (hi - lo)))
            start = self.temperature_peaking if lo < self.temperature_peaking < hi else 0.25
            self.initial.append(min(max(start, lo + 1.0e-3), hi - 1.0e-3))

        # SEGUNDO polo de Perez-Azorin (modelo de dois polos de Hambaryan): a
        # hemisfera oposta recebe seu proprio (T_p2, a2). Posicao: DEPOIS de
        # peaking, ANTES de atmFraction (a mesma no motor e em worker_line).
        self.base_temperature2 = float(request.get("baseTemperature2", 0.0))
        self.fit_base_temperature2 = bool(request.get("fitBaseTemperature2", False))
        if self.fit_base_temperature2:
            lo, hi = request.get("baseTemperature2Range", (0.02, 0.5))
            self.names.append("baseKT2")
            self.bounds.append((float(lo), float(hi)))
            self.steps.append(min(0.01, 0.25 * (hi - lo)))
            start = self.base_temperature2 if self.base_temperature2 > 0 else 0.5 * (lo + hi)
            self.initial.append(min(max(start, lo + 1.0e-3), hi - 1.0e-3))
        self.temperature_peaking2 = float(request.get("temperaturePeaking2", 0.0))
        self.fit_temperature_peaking2 = bool(request.get("fitTemperaturePeaking2", False))
        if self.fit_temperature_peaking2:
            lo, hi = request.get("temperaturePeaking2Range", (0.0, 4.0))
            self.names.append("peaking2")
            self.bounds.append((float(lo), float(hi)))
            self.steps.append(min(0.05, 0.1 * (hi - lo)))
            start = self.temperature_peaking2 if lo < self.temperature_peaking2 < hi else 0.25
            self.initial.append(min(max(start, lo + 1.0e-3), hi - 1.0e-3))
        # beta: desvio da antipodalidade do polo 2 (graus), no plano magneto-rotacional.
        # Posicao: DEPOIS de peaking2, ANTES de atmFraction (igual no motor e worker_line).
        self.pole2_tilt = float(request.get("pole2Tilt", 0.0))
        self.fit_pole2_tilt = bool(request.get("fitPole2Tilt", False))
        if self.fit_pole2_tilt:
            lo, hi = request.get("pole2TiltRange", (-90.0, 90.0))
            self.names.append("poleTilt2")
            self.bounds.append((float(lo), float(hi)))
            self.steps.append(min(3.0, 0.1 * (hi - lo)))
            start = self.pole2_tilt if lo < self.pole2_tilt < hi else 0.0
            self.initial.append(min(max(start, lo + 1.0e-3), hi - 1.0e-3))

        # gamma: segundo deslocamento do polo 2 (azimute do desvio em torno do
        # eixo magnetico). DEPOIS de poleTilt2, ANTES de atmFraction (igual no
        # motor e no worker_line). gamma=0 recupera o modelo de um so beta.
        self.pole2_tilt2 = float(request.get("pole2Tilt2", 0.0))
        self.fit_pole2_tilt2 = bool(request.get("fitPole2Tilt2", False))
        if self.fit_pole2_tilt2:
            lo, hi = request.get("pole2Tilt2Range", (-90.0, 90.0))
            self.names.append("poleTilt2b")
            self.bounds.append((float(lo), float(hi)))
            self.steps.append(min(3.0, 0.1 * (hi - lo)))
            start = self.pole2_tilt2 if lo < self.pole2_tilt2 < hi else 0.0
            self.initial.append(min(max(start, lo + 1.0e-3), hi - 1.0e-3))

        # Espessura efetiva f da atmosfera fina (0=condensada/corpo negro, 1=
        # atmosfera cheia). Une a medida de B (no endurecimento) com o contínuo
        # mole. Posição: DEPOIS de peaking, ANTES da colatitude.
        self.atmosphere_fraction = float(request.get("atmosphereFraction", 1.0))
        self.fit_atmosphere_fraction = bool(request.get("fitAtmosphereFraction", False))
        if self.fit_atmosphere_fraction:
            self.names.append("atmFraction")
            self.bounds.append((0.0, 1.0))
            self.steps.append(0.03)
            start = self.atmosphere_fraction if 0.0 < self.atmosphere_fraction < 1.0 else 0.5
            self.initial.append(min(max(start, 1.0e-3), 1.0 - 1.0e-3))

        # Inclinação do eixo do dipolo em relação ao de rotação. Livre, é ela que
        # faz o fundo pulsar suavemente sozinho (a atmosfera é anisotrópica em
        # theta_B), sem custo espectral — o mecanismo de pulso das XDINS. Posição
        # no vetor e na linha: DEPOIS de baseKT, ANTES do feixe.
        self.fit_magnetic_colatitude = bool(request.get("fitMagneticColatitude", False))
        if self.fit_magnetic_colatitude:
            lo, hi = request.get("magneticColatitudeRange", (0.0, 90.0))
            self.names.append("magColat")
            self.bounds.append((float(lo), float(hi)))
            self.steps.append(min(3.0, 0.25 * (hi - lo)))
            start = self.magnetic_colatitude if lo < self.magnetic_colatitude < hi \
                else 0.5 * (lo + hi)
            self.initial.append(min(max(start, lo + 1.0e-3), hi - 1.0e-3))

        # Azimute do dipolo — o knob de FASE do pulso do fundo. Cíclico em 360°.
        # Posição no vetor e na linha: DEPOIS de magColat, ANTES do feixe.
        self.magnetic_azimuth = float(request.get("magneticAzimuth", 0.0))
        self.fit_magnetic_azimuth = bool(request.get("fitMagneticAzimuth", False))
        if self.fit_magnetic_azimuth:
            self.names.append("magAzim")
            self.bounds.append((-180.0, 180.0))
            self.steps.append(5.0)
            self.initial.append(((self.magnetic_azimuth + 180.0) % 360.0) - 180.0)

        self.beaming2 = float(request.get("beaming2", 0.0))
        if self.fit_beaming:
            self.names.extend(["beaming", "beaming2"])
            # Limites largos: quem restringe de verdade é a positividade em
            # in_prior, e ela permite combinações que nenhum limite retangular
            # descreveria. Com a = -3 e b = 2.25, por exemplo, o padrão é
            # (1 - 1.5 mu)^2, que é positivo e tem um nulo em mu = 2/3 — o tipo
            # de estrutura de dois lóbulos que a forma linear não alcança.
            self.bounds.extend([(-4.0, 4.0), (-4.0, 4.0)])
            self.steps.extend([0.05, 0.05])
            self.initial.extend([self.beaming, self.beaming2])

        # Distância livre (kpc): entra depois do feixe e ANTES de N_H, que tem de
        # continuar sendo o último campo da linha do worker.
        if self.fit_distance:
            self.names.append("distance")
            self.bounds.append((float(self.distance_range[0]), float(self.distance_range[1])))
            self.steps.append(0.02 * max(self.distance, 0.1))
            self.initial.append(self.distance)

        # N_H vai no fim do vetor de propósito: unpack_spots percorre a partir do
        # índice 4 e pararia cedo se algo fosse inserido no meio.
        if self.fit_nh:
            self.names.append("nh")
            self.bounds.append((0.0, self.nh_max))
            self.steps.append(max(1.0e-4, 0.02 * max(self.nh, 0.05)))
            self.initial.append(self.nh)

        # Overrides por requisição, aplicados DEPOIS de o vetor estar montado —
        # assim pegam qualquer parâmetro pelo nome, sem depender da posição.
        #
        # `priors`: {nome: [lo, hi]} estreita (ou alarga) a caixa de um
        #  parâmetro. Uso: kT nasce com priori (0,01, 5) keV, larga demais para
        #  uma XDINS de ~0,1 keV — a posteriori ocupa só o fundo dela e at_bound
        #  dispara um falso "lower". Estreitar para ~(0,03, 0,6) tira o alarme
        #  falso sem cortar o valor físico.
        for name, (lo, hi) in dict(request.get("priors", {})).items():
            if name not in self.names:
                raise ValueError(f"priors: parâmetro '{name}' não existe")
            index = self.names.index(name)
            self.bounds[index] = (float(lo), float(hi))
            self.initial[index] = min(hi, max(lo, self.initial[index]))
            self.steps[index] = min(self.steps[index], 0.25 * (hi - lo))

        # `fixed`: {nome: valor} CONGELA um parâmetro. Não removemos do vetor
        #  (isso deslocaria o protocolo posicional do worker_line); em vez
        #  disso, damos a mesma constante a TODOS os caminhantes e passo zero.
        #  O passo de esticamento de Goodman-Weare propõe X_j + z (X_i − X_j):
        #  se a dimensão é idêntica em todo o enxame, X_i − X_j = 0 e ela nunca
        #  se move. Fica exatamente fixa, in_prior passa, e como a posteriori é
        #  um pico no MEIO da caixinha, at_bound não a marca. Padrão para
        #  ajuste de espectro de atmosfera: M e R vêm de vínculos independentes,
        #  não do pulso, e soltá-los aqui só criava modos e encostava a massa
        #  no limite inferior (medido: 6 de 11 parâmetros no limite).
        self.frozen = {}
        for name, value in dict(request.get("fixed", {})).items():
            if name not in self.names:
                raise ValueError(f"fixed: parâmetro '{name}' não existe")
            index = self.names.index(name)
            value = float(value)
            self.bounds[index] = (value - 1.0e-6, value + 1.0e-6)
            self.initial[index] = value
            self.steps[index] = 0.0
            self.frozen[name] = value

    def _bin_background(self, background) -> list[float]:
        """Taxa de fundo por bin de energia, em contagens por segundo e por keV.

        Sem esta componente o ajuste atribui à estrela todo evento que caiu na
        região de extração, inclusive o fundo — o que puxa a temperatura e a
        área emissora para valores maiores do que os reais.
        """
        if not background:
            return [0.0] * self.energy_bins
        energies = [row[0] for row in background]
        rates = [row[1] for row in background]
        binned: list[float] = []
        for index in range(self.energy_bins):
            centre = self.energy_min + (index + 0.5) * self.energy_width
            if centre <= energies[0]:
                binned.append(rates[0])
            elif centre >= energies[-1]:
                binned.append(rates[-1])
            else:
                position = bisect.bisect_right(energies, centre) - 1
                lo, hi = position, min(position + 1, len(energies) - 1)
                span = energies[hi] - energies[lo]
                fraction = (centre - energies[lo]) / span if span > 0 else 0.0
                binned.append(rates[lo] + fraction * (rates[hi] - rates[lo]))
        return binned

    def in_prior(self, values: list[float]) -> bool:
        if any(not lo <= value <= hi for value, (lo, hi) in zip(values, self.bounds)):
            return False
        if self.fit_compactness:
            # values[0] é u = 2GM/Rc²; a massa implicada tem de ser física, ou o
            # fit foge para R grande + u alto (M ~ 4 M☉). Sem este vínculo, o
            # prior de massa que segurava o modo (M,R) desaparece e a
            # degenerescência da geometria escapa para o impossível.
            u, radius = values[0], values[1]
            mass = u * radius / (2.0 * _GM_SUN_C2_KM)
            if u >= 0.985 or not (0.8 <= mass <= self.mass_cap):
                return False
        else:
            mass, radius = values[0], values[1]
            if 2.0 * _GM_SUN_C2_KM * mass / radius >= 0.985:
                return False
        # Quebra a degenerescência inclinação <-> colatitude. O problema é
        # simétrico sob a troca: medido nesta observação, permutar os dois muda
        # logL em 1 a 20 sobre valores da ordem de 1e5, ou seja, 1e-4 relativo.
        # Sem impor uma ordem, os caminhantes se assentam em modos igualmente
        # bons e o R-hat fica alto para sempre — não por falta de iterações, mas
        # porque a posteriori é mesmo multimodal.
        # Só para um ponto quente: com dois, a estrutura de simetrias é outra
        # e impor esta ordem recortaria a posteriori sem justificativa.
        if self.break_symmetry and self.spot_count == 1:
            if values[2] > values[4]:
                return False
        if self.fit_beaming:
            index = self.names.index("beaming")
            if not beaming_is_physical(values[index], values[index + 1]):
                return False
        return True

    def extra_log_prior(self, values: list[float]) -> float:
        """Termos extra do log-posterior: piso de massa (duro) e prior gaussiano."""
        total = 0.0
        mp = self.gaussian_priors.get("mass")
        if mp or self.mass_floor > 0.0:
            if self.fit_compactness:
                mass = values[0] * values[1] / (2.0 * _GM_SUN_C2_KM)
            else:
                mass = values[0]
            if self.mass_floor > 0.0 and mass < self.mass_floor:
                return -math.inf
            if mp:
                mu, sigma = float(mp[0]), float(mp[1])
                total += -0.5 * ((mass - mu) / sigma) ** 2
        return total

    def unpack_spots(self, values: list[float]) -> list[dict]:
        phase_offset = values[3]
        cursor = 4
        spots = []
        for index in range(self.spot_count):
            theta, radius, kt = values[cursor:cursor + 3]
            cursor += 3
            relative_phi = 0.0 if index == 0 else values[cursor]
            if index > 0:
                cursor += 1
            spots.append({"theta": theta, "phi": phase_offset + relative_phi,
                          "radius": radius, "temperature": kt / KEV_PER_MK})
        return spots

    def worker_command(self) -> list[str]:
        p = self.profile
        base_mass = (self.initial[0] * self.initial[1] / (2.0 * _GM_SUN_C2_KM)
                     if self.fit_compactness else self.initial[0])
        command = [str(ENGINE), "--fit-worker", "--mass", str(base_mass),]
        if self.atmosphere_table:
            command.extend(["--atmosphere-table", str(self.atmosphere_table),
                            "--magnetic-colatitude", str(self.magnetic_colatitude),
                            "--magnetic-azimuth", str(self.magnetic_azimuth)])
            if self.fit_log_field and self.atmosphere_field_range is not None:
                command.append("--fit-log-field")
        command.extend([
                   "--radius", str(self.initial[1]),
                   "--distance", str(self.distance), "--period", str(self.period),
                   "--inclination", str(self.initial[2]), "--position-angle", "0",
                   "--samples", "81", "--max-images", str(self.max_images),
                   "--energy-min", str(self.energy_min), "--energy-max", str(self.energy_max),
                   "--energy-bins", str(self.energy_bins),
                   "--time-bins", str(self.phase_bins * self.media_de_fase),
                   "--instrument", self.instrument,
                   "--instrument-label", str(p["label"]), "--time-resolution-us",
                   str(p.get("time_resolution_us", 0.0)), "--dead-time-us",
                   str(p.get("dead_time_us", 0.0))])
        if self.output_precision is not None:
            command.extend(["--output-precision", str(int(self.output_precision))])
        if self.observation_phase_offset != 0.0:
            # Passado SO quando nao nulo, para que o caso sem offset produza o
            # mesmo comando de antes e o mesmo resultado bit a bit.
            command.extend(["--observation-phase-offset",
                            repr(self.observation_phase_offset)])
        if self.absorption_table is not None:
            command.extend(["--absorption", str(self.absorption_table),
                            "--nh", str(self.nh)])
        if self.atmosphere_active:
            command.extend(["--atmosphere", str(self.atmosphere or 1.0)])
            if self.fit_atmosphere:
                command.append("--fit-atmosphere")
            if self.magnetic_field > 0.0:
                # O campo vai SEMPRE ao motor (a tabela de atmosfera e a linha
                # ciclotron dependem dele); a anisotropia do modelo cinza e o
                # NSMAXG sao opcionais e so entram se os arquivos existirem.
                command.extend(["--magnetic-field", str(self.magnetic_field)])
                if ANISOTROPY_TABLE.is_file():
                    command.extend(["--anisotropy-table", str(ANISOTROPY_TABLE)])
                if self.use_nsmaxg and NSMAXG_TABLE.is_file():
                    command.extend(["--nsmaxg-table", str(NSMAXG_TABLE)])
                if self.fit_log_field:
                    command.append("--fit-log-field")
        if self.base_temperature > 0.0 or self.fit_base_temperature:
            start = (self.initial[self.names.index("baseKT")]
                     if self.fit_base_temperature else self.base_temperature)
            command.extend(["--base-kt-kev", str(start)])
            if self.fit_base_temperature:
                command.append("--fit-base-temperature")
        if self.temperature_peaking > 0.0 or self.fit_temperature_peaking:
            command.extend(["--temperature-peaking", str(self.temperature_peaking),
                            "--temperature-min-frac", str(self.temperature_min_frac)])
            if self.fit_temperature_peaking:
                command.append("--fit-temperature-peaking")
        if self.base_temperature2 > 0.0 or self.fit_base_temperature2:
            start = (self.initial[self.names.index("baseKT2")]
                     if self.fit_base_temperature2 else self.base_temperature2)
            command.extend(["--base-kt2-kev", str(start)])
            if self.fit_base_temperature2:
                command.append("--fit-base-temperature2")
        if self.temperature_peaking2 > 0.0 or self.fit_temperature_peaking2:
            command.extend(["--temperature-peaking2", str(self.temperature_peaking2)])
            if self.fit_temperature_peaking2:
                command.append("--fit-temperature-peaking2")
        if self.pole2_tilt != 0.0 or self.fit_pole2_tilt:
            command.extend(["--pole2-tilt", str(self.pole2_tilt)])
            if self.fit_pole2_tilt:
                command.append("--fit-pole2-tilt")
        if self.pole2_tilt2 != 0.0 or self.fit_pole2_tilt2:
            command.extend(["--pole2-tilt2", str(self.pole2_tilt2)])
            if self.fit_pole2_tilt2:
                command.append("--fit-pole2-tilt2")
        if self.atmosphere_fraction < 1.0 or self.fit_atmosphere_fraction:
            command.extend(["--atmosphere-fraction", str(self.atmosphere_fraction)])
            if self.fit_atmosphere_fraction:
                command.append("--fit-atmosphere-fraction")
        if self.fit_magnetic_colatitude:
            command.append("--fit-magnetic-colatitude")
        if self.fit_magnetic_azimuth:
            command.append("--fit-magnetic-azimuth")
        if self.fit_distance:
            command.append("--fit-distance")
        if self.line_depth > 0.0 or self.fit_line:
            command.extend(["--line-energy", str(self.line_energy),
                            "--line-width", str(self.line_width),
                            "--line-depth", str(self.line_depth)])
            if self.fit_line:
                command.append("--fit-line")
        if self.line2_depth > 0.0 or self.fit_line2:
            command.extend(["--line2-energy", str(self.line2_energy),
                            "--line2-width", str(self.line2_width),
                            "--line2-depth", str(self.line2_depth)])
            if self.fit_line2:
                command.append("--fit-line2")
        if self.beaming != 0.0 or self.beaming2 != 0.0 or self.fit_beaming:
            command.extend(["--beaming", str(self.beaming),
                            "--beaming2", str(self.beaming2)])
            if self.fit_beaming:
                command.append("--fit-beaming")
        for field, flag in (("profile_file", "--instrument-profile"),
                            ("response_file", "--instrument-response")):
            if p.get(field):
                # Os perfis moram no PULSARIS; os caminhos do manifesto são
                # relativos à raiz DELE.
                base = ROOT if (ROOT / p[field]).is_file() else PULSARIS
                command.extend([flag, str(base / p[field])])
        spot_list = self.unpack_spots(self.initial)
        for spot in spot_list:
            command.extend(["--spot", ",".join(str(spot[key]) for key in
                                                ("theta", "phi", "radius", "temperature"))])
        if not spot_list:
            # sem spots: impede o motor de criar o spot default (senao o
            # fit-worker esperaria 4 valores de spot e o protocolo desalinha).
            command.append("--no-spots")
        if self.blackbody_spots:
            command.append("--blackbody-spots")
        if self.spot_overlay:
            command.append("--spot-overlay")
        if self.line_cyclotron:
            command.append("--line-cyclotron")
        if self.field_peaking is not None:
            command.extend(["--field-peaking", str(float(self.field_peaking))])
        if self.boundary_refine > 1:
            command.extend(["--boundary-refine", str(self.boundary_refine)])
        if self.sphere_bands > 0:
            command.extend(["--sphere-bands", str(self.sphere_bands)])
        if self.field_nearest_pole:
            command.append("--field-nearest-pole")
        if self.field_gr:
            command.append("--field-gr")
        if self.layered_atmosphere:
            command.append("--layered-atmosphere")
        return command

    def worker_line(self, values: list[float]) -> str:
        # Compacidade → massa: o motor recebe M. M = u·R/(2·GM_sol/c²).
        mass0 = (values[0] * values[1] / (2.0 * _GM_SUN_C2_KM)
                 if self.fit_compactness else values[0])
        fields = [mass0, values[1], values[2], values[3]]
        for spot in self.unpack_spots(values):
            fields.extend((spot["theta"], spot["phi"] - values[3],
                           spot["radius"], spot["temperature"]))
        # Ordem fixa dos opcionais, a mesma que o motor lê: linha, atmosfera,
        # feixe, N_H — e atmosfera e feixe nunca aparecem juntos.
        # A ordem tem de ser a MESMA em três lugares: aqui, na construção de
        # self.names, e na leitura do motor. É um protocolo posicional, então
        # divergir não dá erro — manda o valor errado no campo certo.
        cursor = 4 + self.spot_vector_length()
        if self.fit_line:
            fields.extend(values[cursor:cursor + 3])
            cursor += 3
        if self.fit_line2:
            fields.extend(values[cursor:cursor + 3])
            cursor += 3
        if self.fit_atmosphere:
            fields.append(values[cursor])
            cursor += 1
        if self.fit_log_field:
            fields.append(values[cursor])
            cursor += 1
        if self.fit_base_temperature:
            fields.append(values[cursor])
            cursor += 1
        if self.fit_temperature_peaking:
            fields.append(values[cursor])
            cursor += 1
        if self.fit_base_temperature2:
            fields.append(values[cursor])
            cursor += 1
        if self.fit_temperature_peaking2:
            fields.append(values[cursor])
            cursor += 1
        if self.fit_pole2_tilt:
            fields.append(values[cursor])
            cursor += 1
        if self.fit_pole2_tilt2:
            fields.append(values[cursor])
            cursor += 1
        if self.fit_atmosphere_fraction:
            fields.append(values[cursor])
            cursor += 1
        if self.fit_magnetic_colatitude:
            fields.append(values[cursor])
            cursor += 1
        if self.fit_magnetic_azimuth:
            fields.append(values[cursor])
            cursor += 1
        if self.fit_beaming:
            fields.extend(values[cursor:cursor + 2])
            cursor += 2
        if self.fit_distance:
            fields.append(values[cursor])
            cursor += 1
        if self.absorption_table is not None:
            # Sempre no fim da linha, ajustada ou fixa: é onde o motor a espera.
            fields.append(values[-1] if self.fit_nh else self.nh)
        return ",".join(format(value, ".17g") for value in fields)

    def spot_vector_length(self) -> int:
        """Quantas posições os pontos ocupam no vetor de parâmetros."""
        return 3 * self.spot_count + max(0, self.spot_count - 1)

    def prepara_fundo_marginal(self) -> None:
        """Cumulantes de pi_e e os nos da quadratura, UMA vez por problema.

        pi_e e a posterior do fundo latente no bin e, derivada das contagens
        brutas e dos pesos:

            b_e = soma_j a_j lambda_j ,  lambda_j | N_j ~ Gamma(N_j + a0 w_j, 1)

        Isto NAO e uma Gamma: as escalas a_j diferem entre observacoes e nos
        canais de borda. Os cumulantes saem fechados, e a Gamma DESLOCADA casada
        em tres deles fica no piso de ruido de Monte Carlo com 200 mil sorteios
        da soma exata. Ver scripts/marginaliza_fundo.py.
        """
        import numpy as _np
        if getattr(self, "_fundo_pronto", False):
            return
        if not hasattr(self, "fundo_pesos"):
            raise ValueError(
                "marginalizeBackground pede um preparedDataNpz que traga "
                "fundo_pesos e fundo_contagens; o produto v1 nao os tem. Gere o "
                "v2 com scripts/preparar_fundo_v2.py.")
        desl = self.fundo_deslocamento
        if len(desl) != self.energy_bins + 1:
            raise ValueError(
                "fundo_deslocamento traz %d limites para %d bins de energia"
                % (len(desl), self.energy_bins))
        # A priori entra como forma PROPORCIONAL A LARGURA do canal, a0 * w_j,
        # a unica coerente sob reagrupamento em energia; a largura vem do
        # produto preparado, nao de um valor tipico.
        self._fundo_k = []
        for ie in range(self.energy_bins):
            a = self.fundo_pesos[desl[ie]:desl[ie + 1]]
            n = self.fundo_contagens[desl[ie]:desl[ie + 1]]
            w = self.fundo_largura_do_canal[desl[ie]:desl[ie + 1]]
            forma = n + self.fundo_a0 * w
            # PROPRIEDADE EXIGIDA, nao suposta: a posterior de cada componente e
            # Gamma(forma, taxa 1 + beta), que so e propria com forma > 0. Com
            # a0 = 0 os 336 canais de contagem zero, que carregam 14,1 por cento
            # do peso total e 48 por cento do peso do bin 39, dariam forma zero,
            # e Gamma(0, 1) nao e integravel na origem. Entao isto e erro, e nao
            # aviso: um alvo com componente improprio nao pode rodar em silencio.
            if forma.size and float(forma.min()) <= 0.0:
                vazios = int((n == 0).sum())
                raise ValueError(
                    "o bin de energia %d tem %d componente(s) de forma nao "
                    "positiva, logo posterior IMPROPRIA: %d dos %d canais tem "
                    "contagem zero e a0 = %g nao os corrige. Use a0 > 0; a "
                    "convencao fiducial e 0,01 por keV."
                    % (ie, int((forma <= 0.0).sum()), vazios, forma.size,
                       self.fundo_a0))
            escala = 1.0 / (1.0 + self.fundo_beta)
            k1 = float((a * forma).sum()) * escala
            k2 = float((a ** 2 * forma).sum()) * escala ** 2
            k3 = float(2.0 * (a ** 3 * forma).sum()) * escala ** 3
            self._fundo_k.append((k1, k2, k3))
        x, w = _np.polynomial.legendre.leggauss(self.fundo_nos)
        self._gl_x, self._gl_w = x, w
        # Nos e pesos por bin, no intervalo [max(0, k1-8s), k1+12s]: o integrando
        # e suave e concentrado, e a cauda direita e mais longa porque a soma de
        # Gammas e assimetrica a direita.
        nos = _np.zeros((self.energy_bins, self.fundo_nos))
        pesos = _np.zeros_like(nos)
        densidade = _np.zeros_like(nos)
        for ie, (k1, k2, k3) in enumerate(self._fundo_k):
            s = math.sqrt(k2) if k2 > 0 else 0.0
            lo, hi = max(0.0, k1 - 8.0 * s), k1 + 12.0 * s
            if not (hi > lo):
                lo, hi = max(0.0, k1 * 0.5), max(1e-30, k1 * 1.5)
            nos[ie] = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
            pesos[ie] = 0.5 * (hi - lo) * w
            forma, escala, desvio = gama_de_tres_cumulantes(k1, k2, k3)
            densidade[ie] = densidade_gama(nos[ie], forma, escala, desvio)
        # pi normalizada NA JANELA: constante em theta, logo cancela entre
        # pontos, mas e declarada porque muda o valor absoluto de ln L.
        norma = (pesos * densidade).sum(axis=1, keepdims=True)
        self._fundo_nos = nos
        self._fundo_peso_pi = (pesos * densidade) / _np.maximum(norma, 1e-300)
        self._fundo_norma_da_janela = norma.ravel().tolist()
        self._fundo_pronto = True

    def _score_marginal(self, grid, represented_time, return_model: bool = False):
        """ln L com o fundo MARGINALIZADO, uma integral por canal de energia.

        O mesmo B[e] entra nos 32 bins de fase, logo tem de ser integrado UMA
        VEZ sobre a verossimilhanca conjunta dessas fases. Integrar celula por
        celula criaria 32 fundos independentes artificiais e estreitaria a
        incerteza por um fator espurio de raiz de 32.
        """
        import numpy as _np
        self.prepara_fundo_marginal()
        t = _np.asarray(represented_time, float)            # (fases,)
        passo = self.energy_width * t                       # (fases,)
        S = _np.asarray(grid, float)                        # (fases, energias)
        n = _np.asarray(self.observed, float).reshape(self.phase_bins,
                                                      self.energy_bins)
        total = 0.0
        por_bin = []
        for ie in self._bins_de_energia_ajustados:
            b = self._fundo_nos[ie]                         # (nos,)
            mu = (S[:, ie][None, :] + b[:, None]) * passo[None, :]
            mu = _np.maximum(mu, 1.0e-300)
            ll = (n[:, ie][None, :] * _np.log(mu) - mu).sum(axis=1)
            pico = float(ll.max())
            integral = float((self._fundo_peso_pi[ie] * _np.exp(ll - pico)).sum())
            valor = pico + math.log(max(integral, 1.0e-300))
            total += valor
            por_bin.append(valor)
        if not return_model:
            return total
        # Para os graficos: o modelo no fundo ESPERADO a posteriori, que e o
        # valor central de pi_e. Nao e o modelo que maximiza a integral.
        esperado = [max(1.0e-14, (grid[ip][ie] + self._fundo_k[ie][0])
                        * self.energy_width * represented_time[ip])
                    for ip in range(self.phase_bins)
                    for ie in range(self.energy_bins)]
        return total, esperado

    def tempo_representado(self) -> list[float]:
        """Segundos de tempo vivo em cada bin de fase.

        Duas receitas, e a MEDIDA vence quando existe. A analitica supoe
        cobertura uniforme a menos do ultimo ciclo parcial, o que seria exato
        se a observacao fosse um intervalo continuo; ela nao e, porque os
        quadros tem FRACEXP e as GTIs recortam. A medida vem da extensao
        EXPOSUnn, quadro por quadro, que e mais fina que as GTIs.
        """
        if getattr(self, "exposicao_por_fase", None) is not None:
            return list(self.exposicao_por_fase)
        complete_cycles = math.floor(self.exposure / self.period)
        partial = self.exposure - complete_cycles * self.period
        phase_bin_duration = self.period / self.phase_bins
        return [complete_cycles * phase_bin_duration + max(
            0.0, min(phase_bin_duration, partial - ip * phase_bin_duration))
            for ip in range(self.phase_bins)]

    def score_document(self, document: dict, return_model: bool = False):
        if document.get("status") != "ok":
            return (-math.inf, None) if return_model else -math.inf
        grid = document["detected_count_rate"]
        if self.media_de_fase > 1:
            # MEDIA e nao soma: a saida e densidade de TAXA, em ct/s/keV, logo o
            # representante do bin observado e a media das M amostras finas, e a
            # exposicao multiplica depois.
            import numpy as _np
            g = _np.asarray(grid, float)
            if g.shape[0] != self.phase_bins * self.media_de_fase:
                raise ValueError(
                    "o motor devolveu %d fases, e phaseAverage = %d com %d bins "
                    "observados pede %d" % (g.shape[0], self.media_de_fase,
                                            self.phase_bins,
                                            self.phase_bins * self.media_de_fase))
            grid = g.reshape(self.phase_bins, self.media_de_fase,
                             -1).mean(axis=1).tolist()
        represented_time = self.tempo_representado()
        expected = [max(1.0e-14,
                        (grid[ip][ie] + self.background[ie])
                        * self.energy_width * represented_time[ip])
                    for ip in range(self.phase_bins) for ie in range(self.energy_bins)]
        if self.fit_light_curve:
            return self._score_light_curve(grid, represented_time, return_model)
        if self.marginalizar_fundo:
            return self._score_marginal(grid, represented_time, return_model)
        log_likelihood = sum(self.observed[b] * math.log(expected[b]) - expected[b]
                             for b in self._fit_bins)
        return (log_likelihood, expected) if return_model else log_likelihood

    def _score_light_curve(self, grid, represented_time,
                           return_model: bool = False):
        """Verossimilhança da FORMA do pulso: soma em energia e perfila a
        normalização da fonte. Fonte e fundo entram separados, para que só a
        forma dos dois picos constranja a geometria — o nível é livre.

        Modelo por fase: mu_p = alpha * S_p + B_p, com S_p a soma em energia da
        fonte e B_p a do fundo (fixo). O alpha ótimo (Poisson) resolve
        sum_p [N_p S_p/(alpha S_p + B_p) - S_p] = 0, monótona em alpha —
        acha-se por bisseção. Substituído, sobra só a forma.
        """
        source_p = [sum(grid[ip][ie] * self.energy_width * represented_time[ip]
                        for ie in range(self.energy_bins))
                    for ip in range(self.phase_bins)]
        background_p = [sum(self.background[ie] * self.energy_width
                            * represented_time[ip]
                            for ie in range(self.energy_bins))
                        for ip in range(self.phase_bins)]
        observed_p = self.phase_pulse([float(n) for n in self.observed])
        total_source = sum(source_p)
        if not (total_source > 0.0):
            return (-math.inf, None) if return_model else -math.inf

        def derivative(alpha: float) -> float:
            return sum(n * s / (alpha * s + b) - s
                       for n, s, b in zip(observed_p, source_p, background_p)
                       if s > 0.0)

        # Bisseção em alpha: a derivada da log-verossimilhança é decrescente.
        lo, hi = 1.0e-9, 1.0e9
        if derivative(lo) <= 0.0:
            alpha = lo
        elif derivative(hi) >= 0.0:
            alpha = hi
        else:
            for _ in range(80):
                mid = math.sqrt(lo * hi)
                if derivative(mid) > 0.0:
                    lo = mid
                else:
                    hi = mid
            alpha = math.sqrt(lo * hi)

        model_p = [max(1.0e-14, alpha * s + b)
                   for s, b in zip(source_p, background_p)]
        log_likelihood = sum(n * math.log(mu) - mu
                             for n, mu in zip(observed_p, model_p))
        if not return_model:
            return log_likelihood
        # Devolve o grid com a fonte já normalizada por alpha, para os gráficos.
        scaled = [max(1.0e-14, (alpha * grid[ip][ie] + self.background[ie])
                      * self.energy_width * represented_time[ip])
                  for ip in range(self.phase_bins) for ie in range(self.energy_bins)]
        return log_likelihood, scaled

    def phase_pulse(self, expected: list[float]) -> list[float]:
        return [sum(expected[ip * self.energy_bins:(ip + 1) * self.energy_bins])
                for ip in range(self.phase_bins)]

    def evaluate(self, worker: "EngineWorker", values: list[float],
                 return_model: bool = False, return_pulse: bool = False):
        """Log-posteriori do vetor, `-inf` SO fora da priori.

        Falha do motor nao devolve `-inf`: ela tenta recuperacao deterministica e,
        se nao houver avaliacao valida, levanta `FalhaDoMotor`. Ver
        `DiagnosticoDoMotor` para a politica e por que a versao anterior, que
        tolerava falha abaixo de um teto, estava errada.

        A recuperacao e reavaliar o MESMO vetor num worker NOVO. Nada e
        perturbado e nada e substituido, porque recuperar nao pode mudar o alvo.
        O diagnostico e opcional para nao quebrar quem monta um FitProblem a mao;
        sem ele o comportamento antigo volta, inclusive o silencio.
        """
        diagnostico = getattr(self, "diagnostico", None)
        if diagnostico is not None:
            diagnostico.conta_avaliacao()
        if not self.in_prior(values):
            if diagnostico is not None:
                diagnostico.fora(values)
            if return_model or return_pulse:
                return -math.inf, None
            return -math.inf
        extra = self.extra_log_prior(values)
        tentativas = 1 + (diagnostico.tentativas if diagnostico is not None else 0)
        for tentativa in range(1, tentativas + 1):
            causa = detalhe = None
            documento = None
            try:
                documento = worker.evaluate(self.worker_line(values))
            except (BrokenPipeError, json.JSONDecodeError, OSError,
                    RuntimeError) as erro:
                causa, detalhe = "excecao", "%s: %s" % (type(erro).__name__, erro)
            if documento is not None and documento.get("status") != "ok":
                causa = "status"
                detalhe = str(documento.get("message", "sem mensagem"))
            if causa is None:
                if diagnostico is not None and tentativa > 1:
                    diagnostico.recuperada(tentativa - 1)
                break
            if diagnostico is None:
                # Sem diagnostico, o comportamento historico: -inf silencioso.
                if return_model or return_pulse:
                    return -math.inf, None
                return -math.inf
            diagnostico.tentativa_falhou(values, causa, detalhe, tentativa,
                                         self.names)
            if tentativa == tentativas:
                raise diagnostico.sem_recuperacao(values, causa, detalhe,
                                                  self.names)
            # Worker NOVO: distingue estado corrompido do processo de recusa do
            # vetor. O motor e deterministico, entao se o novo tambem falhar a
            # falha e do vetor.
            worker.reiniciar()
        if return_model or return_pulse:
            log_likelihood, expected = self.score_document(documento, True)
            log_likelihood += extra
            if return_model:
                return log_likelihood, expected
            return log_likelihood, self.phase_pulse(expected) if expected else None
        return self.score_document(documento) + extra


class EngineWorker:
    def __init__(self, command: list[str]):
        #: Guardado para `reiniciar`: o worker novo tem de ser o MESMO motor.
        self.command = list(command)
        self.reinicios = 0
        self.process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE,
                                        # stderr HERDADO, nao DEVNULL: o motor so
                                        # escreve ali duas coisas, o relato de
                                        # resolucao da superficie, uma vez por
                                        # processo, e a mensagem de erro fatal. O
                                        # DEVNULL engolia as duas, inclusive os
                                        # erros, que e o mesmo tipo de canal
                                        # silencioso do teto de iteracoes.
                                        stdout=subprocess.PIPE, stderr=None,
                                        text=True, bufsize=1)

    def evaluate(self, line: str) -> dict:
        if self.process.poll() is not None or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("persistent C++ worker stopped unexpectedly")
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError("persistent C++ worker returned no result")
        return json.loads(response)

    def reiniciar(self) -> None:
        """Mata o processo e sobe outro com o MESMO comando.

        Existe para a recuperacao deterministica de `FitProblem.evaluate`: o
        comando e identico, logo o motor novo e o mesmo motor, e reavaliar o
        mesmo vetor nele nao muda o alvo. O que muda e so o estado do processo,
        que e justamente o suspeito quando o primeiro falhou.
        """
        try:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        self.process = subprocess.Popen(self.command, cwd=ROOT,
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=None,
                                        text=True, bufsize=1)
        self.reinicios += 1

    def close(self) -> None:
        if self.process.poll() is None and self.process.stdin:
            try:
                self.process.stdin.write("QUIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=3)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.terminate()


class WorkerPool:
    def __init__(self, problem: FitProblem, size: int):
        self.workers = [EngineWorker(problem.worker_command()) for _ in range(size)]
        self.available: queue.Queue[EngineWorker] = queue.Queue()
        for worker in self.workers:
            self.available.put(worker)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=size)

    def evaluate_one(self, problem: FitProblem, values: list[float], with_pulse: bool = False):
        worker = self.available.get()
        try:
            return problem.evaluate(worker, values, return_pulse=with_pulse)
        finally:
            self.available.put(worker)

    def evaluate(self, problem: FitProblem, states: list[list[float]],
                 with_pulse: bool = False):
        futures = [self.executor.submit(self.evaluate_one, problem, state, with_pulse)
                   for state in states]
        return [future.result() for future in futures]

    def model(self, problem: FitProblem, state: list[float]):
        worker = self.available.get()
        try:
            return problem.evaluate(worker, state, return_model=True)
        finally:
            self.available.put(worker)

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        for worker in self.workers:
            worker.close()


def draw_initial_state(problem: FitProblem, rng: random.Random) -> list[float]:
    """Um ponto inicial dentro da priori, e **distinto** dos outros.

    A versão anterior desistia depois de 200 tentativas e devolvia
    ``problem.initial`` — o mesmo ponto para todos os caminhantes que
    falhassem. Com priori apertada (a ordem i <= theta, a positividade do
    feixe, a compacidade) isso acontece em série, e cada repetição rouba uma
    direção da envoltória afim que o enxame nunca mais recupera. Ver
    affine_rank.

    Então, em vez de desistir no mesmo ponto, o passo aumenta: se a vizinhança
    do ponto central é estreita demais, a busca se espalha até achar espaço, e
    o último recurso é um sorteio uniforme na caixa da priori.
    """
    for attempt in range(400):
        spread = 3.0 * (1.0 + attempt / 50.0)   # alarga quando a priori aperta
        state = [min(hi - 1e-12, max(lo + 1e-12, value + rng.gauss(0, step * spread)))
                 for value, step, (lo, hi) in
                 zip(problem.initial, problem.steps, problem.bounds)]
        if problem.in_prior(state):
            return state
    for _ in range(400):
        state = [lo + rng.random() * (hi - lo) for lo, hi in problem.bounds]
        if problem.in_prior(state):
            return state
    return [value + rng.gauss(0, 1e-6 * max(1.0, abs(value)))
            for value in problem.initial]


def draw_stretch(rng: random.Random, scale: float) -> float:
    low, high = 1.0 / math.sqrt(scale), math.sqrt(scale)
    return (low + rng.random() * (high - low)) ** 2


def run_ensemble(problem: FitProblem, pool: WorkerPool, walkers: int, iterations: int,
                 burn_in: int, seed: int, stretch_scale: float = 2.0,
                 checkpoint_path: str | None = None,
                 checkpoint_every: int = 50,
                 resume_states: list[list[float]] | None = None,
                 coletor=None, com_pulso: bool = True) -> dict:
    """Goodman-Weare red/blue stretch move with parallel half-ensemble updates.

    `com_pulso=False` desliga o reservatorio de perfis posteriores. O padrao
    True mantem o comportamento historico, bit a bit. O desligamento existe por
    uma razao concreta: no alvo POR OBSERVACAO ha treze previsoes e portanto
    treze perfis de pulso, e qual deles registrar e decisao cientifica, nao de
    implementacao. Um piloto cujo objetivo e medir amostras efetivas por hora
    nao precisa das bandas de credibilidade do pulso, e desligar o reservatorio
    evita tomar aquela decisao por omissao. Quando a producao precisar das
    bandas, a escolha sera explicita.

    Se `resume_states` vier (as posições finais de uma corrida anterior), o
    enxame nasce EXATAMENTE onde a corrida parou, em vez de ser sorteado da
    priori. Goodman-Weare é sem memória — só o estado atual dos caminhantes
    importa — então isso continua a mesma cadeia, sem desperdiçar o que já
    misturou. Use com burn_in=0 (já está equilibrado)."""
    # COLETOR: alternativa a retencao das trajetorias inteiras. Sem ele o
    # comportamento e identico ao de sempre. Com ele, `trajectories` e
    # `trajectory_logps` NAO sao acumulados: a cada passo registrado o coletor
    # recebe (states, logps) e guarda o que o chamador precisa.
    #
    # Existe porque a retencao completa custa 584 bytes por amostra em 19
    # dimensoes, MEDIDO, o que da 5,60 GB em 96 x 100000 e nao cabe numa maquina
    # de 15 GB com aplicativos abertos. Isso e limitacao da implementacao e nao
    # exigencia do metodo: quem so precisa de uma coordenada, de um indicador ou
    # de estatisticas correntes nao tem por que pagar o vetor inteiro.
    #
    # Com coletor, checkpoint fica proibido: ele e escrito a partir das
    # trajetorias, e gravar um checkpoint vazio seria pior que nao gravar.
    if coletor is not None and checkpoint_path:
        raise RuntimeError(
            "coletor e checkpoint_path sao incompativeis: o checkpoint e escrito "
            "a partir das trajetorias, que o coletor substitui")
    rng = random.Random(seed)
    dimensions = len(problem.names)
    if resume_states is not None:
        if len(resume_states) != walkers:
            raise RuntimeError(
                f"resumeFrom traz {len(resume_states)} caminhantes, mas a corrida "
                f"pede {walkers}; use o mesmo numero de walkers ao retomar.")
        states = [list(s) for s in resume_states]
    else:
        states = [draw_initial_state(problem, rng) for _ in range(walkers)]
    # As coordenadas congeladas precisam ser EXATAMENTE o valor pedido em todos
    # os caminhantes, senao o expoente do passo de esticamento abaixo esta
    # errado. Nao basta confiar na checagem de posto: ela recusa posto MENOR
    # que o esperado, e aceita posto maior, que e o caso em que uma coordenada
    # declarada congelada se move. E ha tres maneiras de ela se mover:
    #   (1) draw_initial_state cai no sorteio uniforme da caixa da priori,
    #       que ignora steps=0 e espalha a coordenada em valor +- 1e-6;
    #   (2) cai no ultimo recurso, que soma jitter de 1e-6 a tudo;
    #   (3) resume_states vem de uma corrida com OUTRO valor congelado.
    # Os dois primeiros disparam quando a priori aperta, que e o regime das
    # corridas de bacia restrita. Entao: conferir contra o pedido, recusar
    # divergencia real, e zerar o residuo de arredondamento.
    for name, value in dict(getattr(problem, "frozen", {})).items():
        index = problem.names.index(name)
        alvo = float(value)
        tolerancia = 1.0e-6 * max(1.0, abs(alvo))
        for walker, state in enumerate(states):
            if abs(state[index] - alvo) > tolerancia:
                raise RuntimeError(
                    f"o caminhante {walker} traz '{name}' = {state[index]!r}, "
                    f"mas o pedido congela esse parametro em {alvo!r}. Se isto "
                    f"veio de resumeFrom, a retomada e de uma corrida com outra "
                    f"configuracao e os dois conjuntos nao podem ser misturados.")
            state[index] = alvo

    initial_results = pool.evaluate(problem, states, with_pulse=com_pulso)
    logps = [result[0] for result in initial_results]
    pulses = [result[1] for result in initial_results]
    if not any(math.isfinite(value) for value in logps):
        raise RuntimeError("none of the initial walkers has finite likelihood")
    missing = [index for index, value in enumerate(logps) if not math.isfinite(value)]
    if missing:
        # Perto de um ponto que já se sabe bom, mas nunca EM cima dele: dar o
        # mesmo vetor a vários caminhantes colapsaria a envoltória afim.
        anchor = states[logps.index(max(logps, key=lambda v: v if math.isfinite(v) else -math.inf))]
        for index in missing:
            states[index] = [value + rng.gauss(0, step) for value, step
                             in zip(anchor, problem.steps)]
            for _ in range(200):
                if problem.in_prior(states[index]):
                    break
                states[index] = [value + rng.gauss(0, step) for value, step
                                 in zip(anchor, problem.steps)]
        replacements = pool.evaluate(problem, [states[index] for index in missing],
                                     with_pulse=com_pulso)
        for index, (value, pulse) in zip(missing, replacements):
            logps[index] = value
            pulses[index] = pulse
    # O enxame tem de gerar as N direções ANTES de começar: depois é tarde,
    # porque o passo de esticamento nunca sai da envoltória afim inicial.
    # Os parâmetros DELIBERADAMENTE fixos (request "fixed") são congelados de
    # propósito — não contam como degeneração acidental, que é o que esta
    # checagem existe para pegar. Descontá-los, senão a fixação legítima de M,R
    # dispararia o próprio guarda que protege contra congelamento não intencional.
    frozen = len(getattr(problem, "frozen", {}))
    rank = affine_rank(states)
    degenerate_axes = (dimensions - frozen) - rank
    if degenerate_axes < 0:
        raise RuntimeError(
            f"o enxame gera {rank} direcoes, mais que as {dimensions - frozen} "
            f"livres: alguma coordenada declarada congelada esta se movendo, e "
            f"o expoente do passo de esticamento ficaria menor que a dimensao "
            f"real do suporte.")
    if degenerate_axes > 0:
        raise RuntimeError(
            f"o enxame inicial gera só {rank} das {dimensions - frozen} "
            f"direções livres do espaço de parâmetros, e o passo de "
            f"esticamento nunca sai da envoltória afim onde nasce: "
            f"{degenerate_axes} parâmetro(s) ficariam congelados a corrida "
            f"inteira, com R-hat reportado como 1,00. Aumente o número de "
            f"caminhantes ou afrouxe a priori.")

    # DIMENSAO EFETIVA do suporte, nao o numero de nomes. O passo de
    # Goodman-Weare propoe Y = X_j + z (X_i - X_j) e so e exato com
    # aceitacao z^(n-1), onde n e a dimensao do espaco em que o enxame
    # realmente se move. Uma coordenada congelada tem o mesmo valor em
    # todos os caminhantes, logo Y herda esse valor e ela nunca se move:
    # o enxame vive num subespaco afim de dimensao len(names) - frozen.
    # Usar len(names) punha um fator z^frozen a mais na aceitacao, que
    # favorece propostas de expansao e infla a dispersao amostrada (medido
    # em alvo gaussiano de 17 dimensoes livres com 2 congeladas e a=1,3:
    # +11 por cento na variancia, +5,5 por cento em sigma). A checagem de
    # posto acima garante que o enxame gera exatamente essas direcoes.
    free_dimensions = dimensions - frozen

    trajectories: list[list[list[float]]] = [[] for _ in range(walkers)]
    trajectory_logps: list[list[float]] = [[] for _ in range(walkers)]
    accepted = [0] * walkers
    posterior_pulses: list[list[float]] = []
    total = burn_in + iterations
    halves = [list(range(0, walkers, 2)), list(range(1, walkers, 2))]
    # FalhaDoMotor para a corrida na primeira falha nao recuperada (ver
    # DiagnosticoDoMotor). O checkpoint TEM de ser gravado aqui, nao no chamador:
    # as trajetorias sao locais desta funcao, e parar sem gravar jogaria fora
    # tudo o que foi amostrado desde o ultimo checkpoint periodico.
    try:
        _laco_do_enxame(
            problem, pool, rng, states, logps, pulses, accepted, halves,
            trajectories, trajectory_logps, posterior_pulses, free_dimensions,
            stretch_scale, burn_in, total, coletor, checkpoint_path,
            checkpoint_every, com_pulso=com_pulso)
    except FalhaDoMotor:
        if checkpoint_path:
            amostrado = len(trajectories[0]) if trajectories else 0
            if amostrado:
                _save_checkpoint(checkpoint_path, trajectories,
                                 trajectory_logps, problem.names, amostrado)
                print("checkpoint gravado com %d amostras por caminhante antes "
                      "de parar" % amostrado, file=sys.stderr, flush=True)
            else:
                print("nada amostrado ainda: nenhum checkpoint a gravar",
                      file=sys.stderr, flush=True)
        raise
    return {"trajectories": trajectories, "trajectory_logps": trajectory_logps,
            "acceptance": [count / max(1, total) for count in accepted],
            "posterior_pulses": posterior_pulses}


def _laco_do_enxame(problem, pool, rng, states, logps, pulses, accepted, halves,
                    trajectories, trajectory_logps, posterior_pulses,
                    free_dimensions, stretch_scale, burn_in, total, coletor,
                    checkpoint_path, checkpoint_every, com_pulso=True):
    """O laco de passos, extraido para que `run_ensemble` possa gravar o
    checkpoint se ele for interrompido. Nao muda nada do algoritmo."""
    walkers = len(states)
    pulse_samples_seen = 0
    pulse_reservoir_size = 512
    for step_index in range(total):
        for half_index, active in enumerate(halves):
            complement = halves[1 - half_index]
            proposals, stretch_values = [], []
            for walker in active:
                partner = states[rng.choice(complement)]
                stretch = draw_stretch(rng, stretch_scale)
                proposals.append([p + stretch * (x - p)
                                  for x, p in zip(states[walker], partner)])
                stretch_values.append(stretch)
            proposed_results = pool.evaluate(problem, proposals,
                                             with_pulse=com_pulso)
            for walker, proposal, proposed_result, stretch in zip(
                    active, proposals, proposed_results, stretch_values):
                proposal_logp, proposal_pulse = proposed_result
                log_acceptance = ((free_dimensions - 1) * math.log(stretch) +
                                  proposal_logp - logps[walker])
                if (math.isfinite(proposal_logp) and
                        math.log(max(rng.random(), 1e-300)) < log_acceptance):
                    states[walker] = proposal
                    logps[walker] = proposal_logp
                    pulses[walker] = proposal_pulse
                    accepted[walker] += 1
        if step_index >= burn_in:
            if coletor is not None:
                coletor(states, logps)
            for walker in range(walkers):
                if coletor is None:
                    trajectories[walker].append(states[walker].copy())
                    trajectory_logps[walker].append(logps[walker])
                if pulses[walker] is not None:
                    pulse_samples_seen += 1
                    if len(posterior_pulses) < pulse_reservoir_size:
                        posterior_pulses.append(pulses[walker].copy())
                    else:
                        replacement = rng.randrange(pulse_samples_seen)
                        if replacement < pulse_reservoir_size:
                            posterior_pulses[replacement] = pulses[walker].copy()
        # CHECKPOINT à prova de crash: a cada `checkpoint_every` passos, grava o
        # que já foi amostrado (atômico: .tmp + rename, para nunca deixar um
        # arquivo meio-escrito). Um crash perde no máximo os últimos passos.
        if (checkpoint_path and step_index >= burn_in
                and (step_index - burn_in + 1) % checkpoint_every == 0):
            _save_checkpoint(checkpoint_path, trajectories, trajectory_logps,
                             problem.names, step_index - burn_in + 1)


def _save_checkpoint(path: str, trajectories, trajectory_logps, names,
                     samples_per_walker: int) -> None:
    """Grava as trajetórias amostradas até agora, de forma atômica."""
    import numpy as _np
    final = path if path.endswith(".npz") else path + ".npz"
    tmp = final + ".tmp"
    with open(tmp, "wb") as handle:
        _np.savez_compressed(
            handle, samples=_np.asarray(trajectories, dtype=_np.float32),
            logps=_np.asarray(trajectory_logps, dtype=_np.float64),
            names=_np.asarray(names), samples_per_walker=samples_per_walker)
    os.replace(tmp, final)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--background", type=Path,
                        help="taxa de fundo por keV já escalada para a região "
                             "da fonte (energy_keV,rate_per_keV_per_s)")
    parser.add_argument("--recuperacoes-do-motor", type=int,
                        default=DiagnosticoDoMotor.TENTATIVAS_DE_RECUPERACAO,
                        help="quantas vezes reavaliar o MESMO vetor num worker "
                             "novo antes de parar, quando o motor falha. Não "
                             "existe opção de tolerar a falha: sem avaliação "
                             "válida a corrida grava checkpoint e para, porque "
                             "devolver -inf alteraria o núcleo de transição e "
                             "poderia excluir uma região válida "
                             "(padrão: %d)" % DiagnosticoDoMotor.TENTATIVAS_DE_RECUPERACAO)
    parser.add_argument("--alerta-de-recuperadas", type=float, default=0.001,
                        help="fração de avaliações recuperadas acima da qual o "
                             "stderr avisa. Falha recuperada não enviesa a "
                             "cadeia, porque o valor usado é o do motor; a taxa "
                             "só denuncia processo doente (padrão: 0,001)")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    metadata, events = read_events(args.events)
    background = read_background(args.background) if args.background else None
    problem = FitProblem(request, events, metadata, background)
    problem.diagnostico = DiagnosticoDoMotor(
        tentativas=args.recuperacoes_do_motor,
        alerta_de_recuperadas=args.alerta_de_recuperadas)
    dimensions = len(problem.names)
    requested_walkers = int(request.get("walkers", request.get("chains", 24)))
    walkers = max(2 * dimensions, min(128, requested_walkers))
    if walkers % 2:
        walkers += 1
    requested = int(request.get("iterations", 200))
    # O teto pode ser levantado de proposito pelo pedido, com maxIterations, mas
    # NUNCA e aplicado em silencio: o comentario de MAX_ITERATIONS sempre
    # prometeu que a poda seria reportada e o codigo nao reportava. Em 04/10/2026
    # isso custou um ajuste de 70 h que pediu 140000 e rodou 100000 sem aviso,
    # e so foi notado ao conferir a forma do checkpoint.
    teto = max(20, int(request.get("maxIterations", MAX_ITERATIONS)))
    iterations = max(20, min(teto, requested))
    if iterations < requested:
        print("AVISO: pediu %d iteracoes, o teto e %d, vai rodar %d. "
              "Use a chave maxIterations do pedido para levantar o teto."
              % (requested, teto, iterations), file=sys.stderr, flush=True)
    burn_in = max(0, min(MAX_ITERATIONS, int(request.get("burnIn", 80))))
    workers = max(1, min(walkers, int(request.get("workers", min(4, walkers)))))
    seed = int(request.get("seed", 2026))
    stretch_scale = max(1.1, min(5.0, float(request.get("stretchScale", 2.0))))
    # Retomada: carrega as posições finais dos walkers de um checkpoint anterior
    # e continua a mesma cadeia (ver run_ensemble). Reordena as colunas para os
    # nomes desta corrida, por segurança.
    resume_states = None
    resume_from = request.get("resumeFrom")
    if resume_from:
        import sys
        import numpy as _np
        ck = _np.load(resume_from, allow_pickle=True)
        ck_names = [str(x) for x in ck["names"]]
        ck_samples = ck["samples"]  # (walkers, iters, dims)
        last = ck_samples[:, -1, :]
        index = [ck_names.index(n) for n in problem.names]
        resume_states = [[float(last[w, j]) for j in index]
                         for w in range(last.shape[0])]
        walkers = len(resume_states)
        if walkers % 2:
            walkers = walkers - 1
            resume_states = resume_states[:walkers]
        print(f"retomando de {resume_from}: {walkers} caminhantes, "
              f"iter de partida {ck_samples.shape[1]}", file=sys.stderr, flush=True)
    started = time.monotonic()
    pool = WorkerPool(problem, workers)
    try:
        try:
            result = run_ensemble(problem, pool, walkers, iterations, burn_in,
                                  seed, stretch_scale,
                                  checkpoint_path=request.get("checkpointPath"),
                                  checkpoint_every=int(request.get("checkpointEvery", 50)),
                                  resume_states=resume_states)
        except FalhaDoMotor as parada:
            # Interrupcao DELIBERADA na PRIMEIRA falha nao recuperada: ver
            # DiagnosticoDoMotor. O checkpoint ja foi gravado dentro do
            # run_ensemble; aqui vai o relato, para o stderr e para um JSON ao
            # lado dele, porque uma corrida que para por falha do motor precisa
            # deixar registro de QUAL vetor falhou e por que.
            print("CORRIDA INTERROMPIDA: %s" % parada, file=sys.stderr, flush=True)
            relato = problem.diagnostico.relatorio()
            destino = request.get("checkpointPath")
            if destino:
                caminho = str(destino).removesuffix(".npz") + "_falha_do_motor.json"
                Path(caminho).write_text(
                    json.dumps(relato, indent=2, ensure_ascii=False) + "\n")
                print("diagnostico gravado em %s" % caminho,
                      file=sys.stderr, flush=True)
            else:
                print(json.dumps(relato, indent=2, ensure_ascii=False),
                      file=sys.stderr, flush=True)
            raise SystemExit(3)
        all_samples = [sample for trajectory in result["trajectories"]
                       for sample in trajectory]
        all_logp = [value for trajectory in result["trajectory_logps"]
                    for value in trajectory]
        # Só entre os finitos. max() com NaN na lista devolve o NaN: a comparação
        # com NaN é sempre falsa, então o primeiro NaN vira o máximo e nunca é
        # substituído. Um único ponto degenerado sequestraria a escolha.
        finite = [index for index, value in enumerate(all_logp)
                  if math.isfinite(value)]
        if not finite:
            raise RuntimeError("no sample in the chain has a finite likelihood")
        best_index = max(finite, key=all_logp.__getitem__)
        best = all_samples[best_index]

        # O valor gravado pela cadeia é o autoritativo: foi ele que guiou a
        # aceitação. A reavaliação serve só para recuperar a grade de contagens
        # esperadas, e quando ela falha — worker que engasgou, parâmetro que o
        # motor recusa na fronteira — o certo é dizer que a grade não veio, e
        # não reportar -inf como "melhor verossimilhança" ao lado de um resumo
        # de posteriori perfeitamente válido.
        best_logp = all_logp[best_index]
        _, expected = pool.model(problem, best)
        model_available = expected is not None
    finally:
        pool.close()
    summaries = []
    autocorrelations = []
    for parameter_index, name in enumerate(problem.names):
        values = [sample[parameter_index] for sample in all_samples]
        by_chain = [[sample[parameter_index] for sample in trajectory]
                    for trajectory in result["trajectories"]]
        if is_cyclic(name):
            centre = circular_mean(values)
            values = unwrap_around(values, centre)
            by_chain = [unwrap_around(chain, centre) for chain in by_chain]
        diagnostic = autocorrelation_ess(by_chain)
        autocorrelations.append(diagnostic["autocorrelation"])
        summaries.append({"name": name, "median": percentile(values, 0.5),
                          "lower": percentile(values, 0.16), "upper": percentile(values, 0.84),
                          "best": best[parameter_index], "rhat": rhat(by_chain),
                          "ess": diagnostic["ess"],
                          "autocorrelationTime": diagnostic["tau"],
                          # sem isto o leitor nao distingue um ESS estimado de
                          # um ESS cuja soma a cadeia cortou. Nao certifica o
                          # ESS: so diz se ha passos suficientes por tau.
                          "essConfiavel": diagnostic["confiavel"],
                          "passosPorTau": diagnostic["passosPorTau"],
                          "atBound": "" if is_cyclic(name) else
                          at_bound(values, problem.bounds[parameter_index]),
                          "cyclic": is_cyclic(name)})
    posterior_pulses = result["posterior_pulses"]
    if not posterior_pulses:
        raise RuntimeError("no posterior pulse predictions were retained")
    pulse_columns = [[pulse[phase] for pulse in posterior_pulses]
                     for phase in range(problem.phase_bins)]
    predictive_bands = {
        "lower2Sigma": [percentile(values, 0.02275) for values in pulse_columns],
        "lower1Sigma": [percentile(values, 0.15865) for values in pulse_columns],
        "median": [percentile(values, 0.5) for values in pulse_columns],
        "upper1Sigma": [percentile(values, 0.84135) for values in pulse_columns],
        "upper2Sigma": [percentile(values, 0.97725) for values in pulse_columns],
        "sampleCount": len(posterior_pulses),
        "includesPoissonNoise": False,
    }
    stride = max(1, len(all_samples) // 1200)
    trace_stride = max(1, iterations // 250)
    trace_indices = list(range(0, iterations, trace_stride))
    trace_walker_stride = max(1, math.ceil(walkers / 24))
    trace_walkers = list(range(0, walkers, trace_walker_stride))[:24]
    diagnostics = {
        # do MAIOR, nao do primeiro: um parametro congelado devolve uma linha de
        # um elemento, e tomar o comprimento da linha 0 anunciava 101 lags para
        # linhas que tinham 1.
        "autocorrelationLags": list(range(max(len(a) for a in autocorrelations)
                                          if autocorrelations else 0)),
        "autocorrelation": autocorrelations,
        "traceIterations": [burn_in + index + 1 for index in trace_indices],
        "traceWalkerIndices": trace_walkers,
        "traces": [[result["trajectories"][walker][index]
                    for index in trace_indices] for walker in trace_walkers],
    }
    import hashlib
    exe = ROOT / "build" / "magnus_engine"
    try:
        motor_sha = hashlib.sha256(exe.read_bytes()).hexdigest()
    except OSError:
        motor_sha = None
    cmd = [str(x) for x in problem.worker_command()]
    print(json.dumps({"status": "ok",
                      "backend": "affine-invariant ensemble MCMC + persistent MAGNUS C++ engine",
                      "algorithm": "Goodman-Weare stretch move",
                      "superficie": superficie_efetiva(request),
                      "motor": {"caminho": str(exe), "sha256": motor_sha},
                      "comandoEfetivo": cmd,
                      "comandoEfetivoSha256":
                          hashlib.sha256(" ".join(cmd).encode()).hexdigest(),
                      "instrument": problem.instrument, "selectedEvents": problem.selected,
                      "exposure": problem.exposure,
                      "exposureSource": ("request" if request.get("exposure")
                                         else "metadata" if metadata.get("exposure_s")
                                         else "event time span"),
                      "absorption": {"model": "TBabs (wilm)" if problem.absorption_table
                                     else "none", "fitted": problem.fit_nh,
                                     "nh_1e22": problem.nh},
                      "diagnosticoDoMotor": problem.diagnostico.relatorio(),
                      "background": {"applied": any(problem.background),
                                     "expectedCounts": sum(
                                         rate * problem.energy_width * problem.exposure
                                         for rate in problem.background)},
                      "phaseBins": problem.phase_bins, "energyBins": problem.energy_bins,
                      "chains": walkers, "walkers": walkers, "dimensions": dimensions,
                      "iterations": iterations, "iterationsRequested": requested,
                      "walkersRecommended":
                          WALKERS_PER_DIMENSION_RECOMMENDED * len(problem.names),
                      "burnIn": burn_in,
                      "workers": workers, "runtimeSeconds": time.monotonic() - started,
                      "stretchScale": stretch_scale, "persistentWorkers": True,
                      "acceptance": result["acceptance"],
                      "parameterNames": problem.names, "summary": summaries,
                      "bestLogLikelihood": best_logp,
                      "modelGridAvailable": model_available,
                      "observed": problem.observed,
                      "expected": expected, "posteriorPredictive": predictive_bands,
                      "diagnostics": diagnostics,
                      "samples": all_samples[::stride],
                      "sampleLogLikelihood": all_logp[::stride], "metadata": metadata}))


if __name__ == "__main__":
    main()
