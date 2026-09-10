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
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "build" / "magnus_engine"
#: Os dados de instrumento vivem no PULSARIS e são usados SOMENTE PARA LEITURA —
#: a fronteira do projeto é essa: o que volta para lá é tabela, nunca escrita.
PULSARIS = Path("/home/rafael/Codes/PULSARIS")
#: Anisotropia da opacidade na fotosfera, derivada das tabelas de
#: Potekhin & Chabrier (2003) por scripts/build_magnetic_anisotropy.py.
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
_GM_SUN_C2_KM = 1.476625


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


def autocorrelation_ess(chain_values: list[list[float]], max_lag: int = 100) -> dict:
    """Estimate the within-chain ACF and autocorrelation effective sample size.

    The integrated autocorrelation time uses Geyer's initial-positive paired
    sequence.  A constant or too-short ensemble is reported as unavailable
    instead of receiving a misleadingly large ESS.
    """
    if not chain_values:
        return {"autocorrelation": [], "tau": None, "ess": None}
    length = min(map(len, chain_values))
    if length < 4:
        return {"autocorrelation": [1.0], "tau": None, "ess": None}
    chains = [values[-length:] for values in chain_values]
    lag_limit = min(max_lag, length - 1)
    correlations = []
    for values in chains:
        mean = statistics.fmean(values)
        centered = [value - mean for value in values]
        variance = statistics.fmean(value * value for value in centered)
        if variance <= 0.0:
            continue
        correlations.append([
            sum(centered[index] * centered[index + lag]
                for index in range(length - lag)) /
            ((length - lag) * variance)
            for lag in range(lag_limit + 1)
        ])
    if not correlations:
        return {"autocorrelation": [1.0], "tau": None, "ess": None}
    acf = [statistics.fmean(values[lag] for values in correlations)
           for lag in range(lag_limit + 1)]
    tau = 1.0
    lag = 1
    while lag < len(acf):
        pair = acf[lag]
        if lag + 1 < len(acf):
            pair += acf[lag + 1]
        if pair <= 0.0:
            break
        tau += 2.0 * pair
        lag += 2
    tau = max(1.0, tau)
    total = len(chains) * length
    return {"autocorrelation": acf, "tau": tau,
            "ess": min(float(total), total / tau)}


class FitProblem:
    def __init__(self, request: dict, events: list[tuple[float, float]],
                 metadata: dict[str, str] | None = None,
                 background: list[tuple[float, float]] | None = None):
        self.request = request
        self.metadata = metadata or {}
        self.period = float(request["period"])
        self.distance = float(request["distance"])
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
        # Modo de camadas: contínuo de corpo negro + feixe da atmosfera, em toda a
        # superfície (atmosfera fina sobre condensada, à la Hambaryan).
        self.layered_atmosphere = bool(request.get("layeredAtmosphere", False))
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
            if u >= 0.985 or not (0.8 <= mass <= 2.5):
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
        """Termo gaussiano informado somado ao log-posterior (0 se não houver)."""
        total = 0.0
        mp = self.gaussian_priors.get("mass")
        if mp:
            mu, sigma = float(mp[0]), float(mp[1])
            if self.fit_compactness:
                mass = values[0] * values[1] / (2.0 * _GM_SUN_C2_KM)
            else:
                mass = values[0]
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
                   "--time-bins", str(self.phase_bins), "--instrument", self.instrument,
                   "--instrument-label", str(p["label"]), "--time-resolution-us",
                   str(p.get("time_resolution_us", 0.0)), "--dead-time-us",
                   str(p.get("dead_time_us", 0.0))])
        if self.absorption_table is not None:
            command.extend(["--absorption", str(self.absorption_table),
                            "--nh", str(self.nh)])
        if self.atmosphere_active:
            command.extend(["--atmosphere", str(self.atmosphere or 1.0)])
            if self.fit_atmosphere:
                command.append("--fit-atmosphere")
            if self.magnetic_field > 0.0 and ANISOTROPY_TABLE.is_file():
                command.extend(["--magnetic-field", str(self.magnetic_field),
                                "--anisotropy-table", str(ANISOTROPY_TABLE)])
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
        if self.atmosphere_fraction < 1.0 or self.fit_atmosphere_fraction:
            command.extend(["--atmosphere-fraction", str(self.atmosphere_fraction)])
            if self.fit_atmosphere_fraction:
                command.append("--fit-atmosphere-fraction")
        if self.fit_magnetic_colatitude:
            command.append("--fit-magnetic-colatitude")
        if self.fit_magnetic_azimuth:
            command.append("--fit-magnetic-azimuth")
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
        for spot in self.unpack_spots(self.initial):
            command.extend(["--spot", ",".join(str(spot[key]) for key in
                                                ("theta", "phi", "radius", "temperature"))])
        if self.blackbody_spots:
            command.append("--blackbody-spots")
        if self.spot_overlay:
            command.append("--spot-overlay")
        if self.line_cyclotron:
            command.append("--line-cyclotron")
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
        if self.absorption_table is not None:
            # Sempre no fim da linha, ajustada ou fixa: é onde o motor a espera.
            fields.append(values[-1] if self.fit_nh else self.nh)
        return ",".join(format(value, ".17g") for value in fields)

    def spot_vector_length(self) -> int:
        """Quantas posições os pontos ocupam no vetor de parâmetros."""
        return 3 * self.spot_count + max(0, self.spot_count - 1)

    def score_document(self, document: dict, return_model: bool = False):
        if document.get("status") != "ok":
            return (-math.inf, None) if return_model else -math.inf
        grid = document["detected_count_rate"]
        complete_cycles = math.floor(self.exposure / self.period)
        partial = self.exposure - complete_cycles * self.period
        phase_bin_duration = self.period / self.phase_bins
        represented_time = [complete_cycles * phase_bin_duration + max(
            0.0, min(phase_bin_duration, partial - ip * phase_bin_duration))
            for ip in range(self.phase_bins)]
        expected = [max(1.0e-14,
                        (grid[ip][ie] + self.background[ie])
                        * self.energy_width * represented_time[ip])
                    for ip in range(self.phase_bins) for ie in range(self.energy_bins)]
        if self.fit_light_curve:
            return self._score_light_curve(grid, represented_time, return_model)
        log_likelihood = sum(n * math.log(mu) - mu for n, mu in zip(self.observed, expected))
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
        if not self.in_prior(values):
            if return_model or return_pulse:
                return -math.inf, None
            return -math.inf
        extra = self.extra_log_prior(values)
        try:
            if return_model or return_pulse:
                log_likelihood, expected = self.score_document(
                    worker.evaluate(self.worker_line(values)), True)
                log_likelihood += extra
                if return_model:
                    return log_likelihood, expected
                return log_likelihood, self.phase_pulse(expected) if expected else None
            return self.score_document(worker.evaluate(self.worker_line(values))) + extra
        except (BrokenPipeError, json.JSONDecodeError, OSError, RuntimeError):
            if return_model or return_pulse:
                return -math.inf, None
            return -math.inf


class EngineWorker:
    def __init__(self, command: list[str]):
        self.process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
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
                 resume_states: list[list[float]] | None = None) -> dict:
    """Goodman-Weare red/blue stretch move with parallel half-ensemble updates.

    Se `resume_states` vier (as posições finais de uma corrida anterior), o
    enxame nasce EXATAMENTE onde a corrida parou, em vez de ser sorteado da
    priori. Goodman-Weare é sem memória — só o estado atual dos caminhantes
    importa — então isso continua a mesma cadeia, sem desperdiçar o que já
    misturou. Use com burn_in=0 (já está equilibrado)."""
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
    initial_results = pool.evaluate(problem, states, with_pulse=True)
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
                                     with_pulse=True)
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
    if degenerate_axes > 0:
        raise RuntimeError(
            f"o enxame inicial gera só {rank} das {dimensions - frozen} "
            f"direções livres do espaço de parâmetros, e o passo de "
            f"esticamento nunca sai da envoltória afim onde nasce: "
            f"{degenerate_axes} parâmetro(s) ficariam congelados a corrida "
            f"inteira, com R-hat reportado como 1,00. Aumente o número de "
            f"caminhantes ou afrouxe a priori.")

    trajectories: list[list[list[float]]] = [[] for _ in range(walkers)]
    trajectory_logps: list[list[float]] = [[] for _ in range(walkers)]
    accepted = [0] * walkers
    posterior_pulses: list[list[float]] = []
    pulse_samples_seen = 0
    pulse_reservoir_size = 512
    total = burn_in + iterations
    halves = [list(range(0, walkers, 2)), list(range(1, walkers, 2))]
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
            proposed_results = pool.evaluate(problem, proposals, with_pulse=True)
            for walker, proposal, proposed_result, stretch in zip(
                    active, proposals, proposed_results, stretch_values):
                proposal_logp, proposal_pulse = proposed_result
                log_acceptance = ((dimensions - 1) * math.log(stretch) +
                                  proposal_logp - logps[walker])
                if (math.isfinite(proposal_logp) and
                        math.log(max(rng.random(), 1e-300)) < log_acceptance):
                    states[walker] = proposal
                    logps[walker] = proposal_logp
                    pulses[walker] = proposal_pulse
                    accepted[walker] += 1
        if step_index >= burn_in:
            for walker in range(walkers):
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
    return {"trajectories": trajectories, "trajectory_logps": trajectory_logps,
            "acceptance": [count / max(1, total) for count in accepted],
            "posterior_pulses": posterior_pulses}


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
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    metadata, events = read_events(args.events)
    background = read_background(args.background) if args.background else None
    problem = FitProblem(request, events, metadata, background)
    dimensions = len(problem.names)
    requested_walkers = int(request.get("walkers", request.get("chains", 24)))
    walkers = max(2 * dimensions, min(128, requested_walkers))
    if walkers % 2:
        walkers += 1
    requested = int(request.get("iterations", 200))
    iterations = max(20, min(MAX_ITERATIONS, requested))
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
        result = run_ensemble(problem, pool, walkers, iterations, burn_in,
                              seed, stretch_scale,
                              checkpoint_path=request.get("checkpointPath"),
                              checkpoint_every=int(request.get("checkpointEvery", 50)),
                              resume_states=resume_states)
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
        "autocorrelationLags": list(range(len(autocorrelations[0]))),
        "autocorrelation": autocorrelations,
        "traceIterations": [burn_in + index + 1 for index in trace_indices],
        "traceWalkerIndices": trace_walkers,
        "traces": [[result["trajectories"][walker][index]
                    for index in trace_indices] for walker in trace_walkers],
    }
    print(json.dumps({"status": "ok",
                      "backend": "affine-invariant ensemble MCMC + persistent PULSARIS C++",
                      "algorithm": "Goodman-Weare stretch move",
                      "instrument": problem.instrument, "selectedEvents": problem.selected,
                      "exposure": problem.exposure,
                      "exposureSource": ("request" if request.get("exposure")
                                         else "metadata" if metadata.get("exposure_s")
                                         else "event time span"),
                      "absorption": {"model": "TBabs (wilm)" if problem.absorption_table
                                     else "none", "fitted": problem.fit_nh,
                                     "nh_1e22": problem.nh},
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
