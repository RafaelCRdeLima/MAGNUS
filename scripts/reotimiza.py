#!/usr/bin/env python3
"""Reotimiza um vetor com os 17 parametros livres sob a quadratura corrigida.

Pequena reotimizacao da lei termica ANTIGA, prevista no item 2 do plano de
recuperacao: so depois que a quadratura ficou estavel e que faz sentido perguntar
se os maximos sobrevivem a integracao corrigida. Isso NAO e evidencia bayesiana;
comparar maximos e diagnostico de ajuste, e os pesos dos modos continuam
indeterminados ate uma nova amostragem.

Decisoes registradas:

- A faixa de inclinacao e a UNIAO das duas bacias (88 a 100 graus). A divisao em
  bacias era do amostrador: para procurar maximo ela e uma parede artificial, e
  como `extra_log_prior` nao depende da faixa, alargar nao muda o lnL de nenhum
  ponto interior. Com isso a busca pode atravessar a fronteira i = 93,3.
- `phaseOffset` e `lineEnergy` ficam nos valores do pedido (2 congelados, 17
  livres). A ordem do vetor e validada contra `prob.names`, sem excecao: um
  vetor com nomes fora de ordem produziria um numero plausivel e errado.
- A busca corre em coordenadas normalizadas pelos limites do prior, porque os
  parametros vao de 0,02 (baseKT) a 180 (magAzim) e o simplex e sensivel a
  escala. O simplex inicial e explicito: o padrao do scipy perturba x*1,05, que
  para o modo colado (a1 ~ 2e-8) daria um simplex degenerado.
- Fora do prior o custo nao e infinito, e sim 1e6 mais a violacao de massa, que
  cresce monotonamente: o simplex precisa de um gradiente para voltar.
- DIFERENCA OBSERVADA ENTRE RESOLUCOES, em convergencia_quadratura.json: a 120,
  240 e 400 bandas o lnL de um mesmo vetor varia cerca de 0,25, e as diferencas
  entre vetores variam 0,11 dentro de B e 0,45 entre A e B. Isso e uma diferenca
  MEDIDA entre tres malhas, nao um limite demonstrado do erro da quadratura:
  nenhuma extrapolacao com bracketing foi feita e o residuo do limbo nao esta
  corrigido. `fatol` fica em 0,05 em vez de 0,01 porque apertar mais gasta
  avaliacoes de 35 s perseguindo digitos que a quadratura nao sustenta.
- CADA OTIMO ENCONTRADO TEM DE SER REAVALIADO a 400 bandas, por imagem e por
  bin, com scripts/valida_otimo.py. Se o otimo cair em outra regiao do espaco de
  parametros, os testes de convergencia feitos nos tres vetores antigos NAO o
  cobrem, e a variacao de 0,25 medida neles nao se transfere.

Uso:
  MAGNUS_INSTRUMENT_DIR=.../instrumento python3 scripts/reotimiza.py \
      --vetor docs/incidente_resolucao_2026-10-04/modosB.json --rotulo colado \
      --saida exploracoes/reotimizacao/B_colado.json
"""
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mcmc_fit as m  # noqa: E402

PENALIDADE = 1.0e6
INCLINACAO_UNIAO = (88.0, 100.0)


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def monta_pedido(pedido: Path, bandas: int, refino: int, uniao: bool,
                 teto_de_massa: float | None = None) -> dict:
    base = json.loads(pedido.read_text())
    if uniao:
        base["inclinationRange"] = list(INCLINACAO_UNIAO)
    base["sphereBands"] = int(bandas)
    base["boundaryRefine"] = int(refino)
    if teto_de_massa is not None:
        base["massCap"] = float(teto_de_massa)
    return base


def valida_ordem(nomes_arquivo, nomes_prob) -> None:
    """Obrigatoria: um vetor fora de ordem devolve um lnL plausivel e errado."""
    if list(nomes_arquivo) != list(nomes_prob):
        faltam = [n for n in nomes_prob if n not in nomes_arquivo]
        sobram = [n for n in nomes_arquivo if n not in nomes_prob]
        trocados = [(i, a, b) for i, (a, b) in
                    enumerate(zip(nomes_arquivo, nomes_prob)) if a != b]
        raise SystemExit(
            "ordem dos nomes do vetor nao bate com prob.names.\n"
            "  no arquivo: %s\n  no problema: %s\n"
            "  ausentes no arquivo: %s\n  nao reconhecidos: %s\n"
            "  primeiras posicoes divergentes: %s"
            % (nomes_arquivo, nomes_prob, faltam, sobram, trocados[:5]))


def _estatuto_do_ponto(rodadas, limiar):
    """Se a busca estabilizou no criterio de orcamento, ou apenas parou."""
    if not rodadas:
        return "sem rodadas: o ponto e a propria partida"
    ultimo = rodadas[-1].get("ganhoDaRodada")
    if ultimo is None:
        return "ganho da ultima rodada nao registrado"
    if ultimo > limiar:
        return ("MELHOR PONTO ENCONTRADO, NAO estabilizado: a ultima rodada "
                "ainda rendeu %.4f em ln L, acima do limiar de orcamento de "
                "%.4f. Abrir mais rodadas provavelmente melhoraria" % (ultimo, limiar))
    return ("estabilizado no criterio de ORCAMENTO: a ultima rodada rendeu "
            "%.4f, abaixo do limiar de %.4f. Isso e parada por ganho "
            "decrescente, nao prova de convergencia: a sonda do ponto final e a "
            "revalidacao contra a referencia fina acompanham a parada"
            % (ultimo, limiar))


def _commit_se_houver():
    """O commit do HEAD, ou um marcador de por que ele nao esta aqui.

    NAO chame `git` sem guarda neste ponto. Os nos de computo deste cluster nao
    tem git no PATH, e esta chamada, que e so de REGISTRO, derrubou com
    FileNotFoundError os jobs 47531, 47532 e 47533 na gravacao FINAL, depois de
    7 a 8,5 horas de calculo cada. Os parciais por rodada sobreviveram, mas a
    sonda do ponto final foi perdida nas tres. Registro nao pode derrubar
    resultado, e por isso aqui devolve-se SEMPRE uma string: quem le a saida
    precisa saber que o commit nao foi registrado, e por que.
    """
    import shutil
    if shutil.which("git") is None:
        return "git ausente no PATH"
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or "git nao devolveu HEAD"
    except Exception as erro:                      # noqa: BLE001
        return "git falhou: %s" % type(erro).__name__


class Custo:
    """-log posterior em coordenadas normalizadas, com penalidade de massa."""

    def __init__(self, prob, trabalhador, referencia: float = 0.0):
        self.prob = prob
        self.w = trabalhador
        # OBJETIVO CENTRADO. Subtrair lnP(ref) nao muda o otimo, mas muda o
        # critério de parada RELATIVO do Powell, que no scipy instalado e
        #     bnd = ftol*(|f_antes| + |f_depois|) + 1e-20
        #     para se 2*(f_antes - f_depois) <= bnd
        # Com |f| ~ 1,7289e6 e ftol = 1e-6 isso para quando a melhora de uma
        # iteracao externa cai abaixo de 1,729 em lnL. Foi por isso que Powell
        # parou com ganho 0,865 no primeiro teste, e NAO por estagnacao em vale
        # estreito, como eu havia escrito. Centrado, o limiar fica MUITO menor,
        # da ordem de ftol*|ganho|, mas NAO desaparece: a parada interna do
        # Powell continua podendo encerrar uma rodada, e por isso o motivo da
        # parada e registrado em tres categorias separadas. Nelder-Mead nao era
        # afetado, porque xatol e fatol do scipy sao ABSOLUTOS.
        self.referencia = float(referencia)
        # BANDA DE MASSA NO OBJETIVO. Com restricao nao linear explicita o
        # objetivo tem de ser avaliavel FORA da banda, senao a penalidade de 1e6
        # reaparece justamente na fronteira e o modelo quadratico do COBYQA
        # encontra um degrau onde deveria ver superficie suave. Passar
        # NonlinearConstraint nao remove a penalidade por si: quem rejeitava era
        # o in_prior. Com isto em False a banda sai do objetivo e vai para a
        # restricao, enquanto os limites INDISPENSAVEIS seguem protegidos: a
        # caixa dos limites e u < 0,985, que o motor precisa para a tabela de
        # raios. Conferido neste pedido: as unicas portas ativas do in_prior sao
        # a caixa, a banda de massa e u < 0,985, porque fit_beaming e False e
        # spot_count e 0, entao break_symmetry nao age.
        self.banda_de_massa_no_objetivo = True
        self.livres = [i for i, n in enumerate(prob.names) if n not in prob.frozen]
        self.lo = np.array([prob.bounds[i][0] for i in self.livres], float)
        self.hi = np.array([prob.bounds[i][1] for i in self.livres], float)
        self.base = np.array([prob.frozen.get(n, 0.0) for n in prob.names], float)
        self.avaliacoes = 0
        self.melhor = (math.inf, None)

    def para_x(self, vetor) -> np.ndarray:
        v = np.asarray(vetor, float)[self.livres]
        return (v - self.lo) / (self.hi - self.lo)

    def para_vetor(self, x) -> list:
        v = self.base.copy()
        v[self.livres] = self.lo + np.clip(np.asarray(x, float), 0.0, 1.0) * (self.hi - self.lo)
        return [float(t) for t in v]

    def massa(self, x) -> float:
        v = self.para_vetor(x)
        return v[0] * v[1] / (2.0 * m._GM_SUN_C2_KM)

    def restricoes(self, x):
        """M_max - M >= 0, M - M_min >= 0, 0.985 - u >= 0, em coordenadas x.

        Dada EXPLICITAMENTE ao COBYQA, no lugar de representar a fronteira pela
        penalidade de 1e6. Os tres pontos de partida estao a 99,4 a 99,6 por
        cento do teto, entao a fronteira nao e um detalhe do contorno: ela passa
        pelo proprio otimo.
        """
        v = self.para_vetor(x)
        massa = v[0] * v[1] / (2.0 * m._GM_SUN_C2_KM)
        return [self.prob.mass_cap - massa,
                massa - max(0.8, self.prob.mass_floor),
                0.985 - v[0]]

    def violacao(self, vetor) -> float:
        u, raio = vetor[0], vetor[1]
        massa = u * raio / (2.0 * m._GM_SUN_C2_KM)
        p = 0.0
        if u >= 0.985:
            p += (u - 0.985) + 1.0
        piso = max(0.8, self.prob.mass_floor)
        if massa < piso:
            p += piso - massa
        if massa > self.prob.mass_cap:
            p += massa - self.prob.mass_cap
        return p

    def viavel(self, vetor) -> bool:
        if self.banda_de_massa_no_objetivo:
            return self.prob.in_prior(vetor)
        if any(not lo <= v <= hi for v, (lo, hi) in zip(vetor, self.prob.bounds)):
            return False
        return not (self.prob.fit_compactness and vetor[0] >= 0.985)

    def __call__(self, x) -> float:
        vetor = self.para_vetor(x)
        self.avaliacoes += 1
        if not self.viavel(vetor):
            return PENALIDADE * (1.0 + self.violacao(vetor))
        posterior = self.prob.evaluate(self.w, vetor)
        if not math.isfinite(posterior):
            return PENALIDADE * (1.0 + self.violacao(vetor))
        custo = -(posterior - self.referencia)
        if custo < self.melhor[0]:
            self.melhor = (custo, vetor)
        return custo


def calibra_passos(custo, x0: np.ndarray, alvo: float = 1.0,
                   tentativas: int = 5, passo0: float = 0.01):
    """Passo por coordenada que piora o objetivo em cerca de `alvo` em lnL.

    Uma FRACAO FIXA da largura do prior nao serve, e medir mostrou o tamanho do
    erro. No vetor B aberto, com passo de 0,02 da largura, a piora por
    coordenada ia de 1,3 em poleTilt2b a 12532 em distance, e os vertices de
    compactness e radius caiam direto na PENALIDADE, custo 1e6: os tres pontos
    de partida estao a 99,4 a 99,6 por cento do teto de massa de 2,5 M_sol, com
    folga em u de so 0,4 a 0,6 por cento da largura do prior, e o passo pedia
    cinco vezes mais que isso. Um vertice em 1e6 domina o simplex e Nelder-Mead
    gasta o orcamento inteiro contraindo.

    Aqui o passo e escolhido pelo EFEITO e nao pelo prior. Perto de um otimo a
    piora vai como o quadrado do passo, entao escalar por sqrt(alvo/piora)
    converge em duas ou tres avaliacoes por coordenada. Se os dois sentidos
    forem inviaveis, o passo encurta; se so um for, usa-se o viavel, o que trata
    o caso de borda sem precisar conhecer a restricao de massa explicitamente.

    Custo: cerca de 2 a 3 avaliacoes por coordenada livre, ou 20 a 30 min a 120
    bandas, contra as 19,6 h da otimizacao. Feito NA MESMA resolucao da
    otimizacao, de proposito: calibrar numa malha mais grosseira seria barato,
    mas a decisao de 04/10 foi nao trazer nada de resolucao inferior.
    """
    c0 = custo(x0)
    passos, registro = [], []
    for i in range(x0.size):
        passo, achado = passo0, None
        for _ in range(tentativas):
            viavel = None
            for sinal in (1.0, -1.0):
                x = x0.copy()
                x[i] = min(1.0, max(0.0, x0[i] + sinal * passo))
                if x[i] == x0[i]:
                    continue
                c = custo(x)
                if c < PENALIDADE:
                    viavel = (sinal, c - c0)
                    break
            if viavel is None:
                passo *= 0.25
                continue
            sinal, piora = viavel
            achado = (passo, sinal, piora)
            if 0.3 * alvo <= piora <= 3.0 * alvo:
                break
            passo = float(min(0.5, max(1.0e-7,
                                       passo * math.sqrt(alvo / max(piora, 1.0e-12)))))
        if achado is None:
            achado = (passo0, 1.0, float("nan"))
        passos.append(achado[0] * achado[1])
        registro.append({"passo": achado[0], "sinal": achado[1],
                         "piora_em_lnL": achado[2]})
    return np.asarray(passos), registro


def simplex_inicial(x0: np.ndarray, passos) -> np.ndarray:
    """Simplex explicito, um passo CALIBRADO e com sinal por coordenada."""
    n = x0.size
    passos = (np.full(n, float(passos)) if np.isscalar(passos)
              else np.asarray(passos, float))
    s = np.empty((n + 1, n))
    s[0] = x0
    for i in range(n):
        s[i + 1] = x0
        s[i + 1, i] = min(1.0, max(0.0, x0[i] + passos[i]))
    return s


def sonda_direcoes(custo, x, passos, fatores=(1.0, 0.25)):
    """Direcoes viaveis testadas NO PONTO FINAL, inclusive ao longo do teto.

    McKinnon construiu funcoes, algumas estritamente convexas, em que o simplex
    de Nelder-Mead converge para um ponto NAO estacionario. Reinicios ajudam na
    pratica e nao constituem certificado, entao o ponto final e sondado a mao.

    Inclui uma sonda SOBRE o teto de massa, que e onde os tres pontos de partida
    vivem. M constante significa uR constante, e a direcao tangente (u, -R)
    preserva isso apenas em PRIMEIRA ORDEM: para passo finito a massa cai em
    t^2. A sonda usa u' = u e^t e R' = R e^-t, que mantem uR exato em qualquer
    t. Sem essa sonda, uma melhora que exija trocar compacidade por raio
    mantendo a massa fica invisivel, porque mover u sozinho sai do prior.

    Registra ausencia de melhora. NAO prova maximo, nem local nem global.
    """
    c0 = custo(x)
    achados = []
    nomes = [custo.prob.names[i] for i in custo.livres]
    for fator in fatores:
        for i, nome in enumerate(nomes):
            for sinal in (1.0, -1.0):
                y = x.copy()
                y[i] = min(1.0, max(0.0, x[i] + sinal * fator * passos[i]))
                if y[i] == x[i]:
                    continue
                c = custo(y)
                if c < c0 - 1.0e-9:
                    achados.append({"direcao": nome, "sinal": sinal,
                                    "fator": fator, "melhora": float(c0 - c)})
    # SOBRE o teto de massa, com mapa EXPONENCIAL. A direcao (du, dR) ~ (u, -R)
    # preserva uR so em PRIMEIRA ORDEM: para passo finito,
    # (u + ut)(R - Rt) = uR(1 - t^2), ou seja a massa cai em t^2 e a sonda sai
    # da fronteira em vez de correr sobre ela. Com u' = u e^t e R' = R e^-t o
    # produto uR fica EXATO em qualquer t, porque e^t e^-t = 1. Os demais
    # limites sao conferidos a cada ponto, e um ponto que saia da caixa e
    # descartado em vez de recortado, que recortar quebraria a conservacao.
    iu_n = custo.prob.names.index("compactness")
    ir_n = custo.prob.names.index("radius")
    iu = custo.livres.index(iu_n)
    ir = custo.livres.index(ir_n)
    v = custo.para_vetor(x)
    lo_u, hi_u = custo.prob.bounds[iu_n]
    lo_r, hi_r = custo.prob.bounds[ir_n]
    escala_t = float(np.linalg.norm(passos)) / np.sqrt(len(passos))
    ao_longo = []
    for fator in fatores:
        for sinal in (1.0, -1.0):
            # t escolhido para dar um deslocamento em u comparavel ao passo
            # calibrado daquela coordenada
            t = sinal * fator * max(passos[iu] * (hi_u - lo_u) / max(v[0], 1e-12),
                                    1.0e-6 * escala_t)
            u_novo, r_novo = v[0] * math.exp(t), v[1] * math.exp(-t)
            dentro = (lo_u <= u_novo <= hi_u and lo_r <= r_novo <= hi_r)
            registro = {"sinal": sinal, "fator": fator, "t": t,
                        "u": u_novo, "R": r_novo,
                        "massa": u_novo * r_novo / (2.0 * m._GM_SUN_C2_KM),
                        "dentro_da_caixa": dentro}
            if not dentro:
                registro["variacao"] = None
                ao_longo.append(registro)
                continue
            y = x.copy()
            y[iu] = (u_novo - lo_u) / (hi_u - lo_u)
            y[ir] = (r_novo - lo_r) / (hi_r - lo_r)
            c = custo(y)
            registro["variacao"] = float(c0 - c)
            ao_longo.append(registro)
            if c < c0 - 1.0e-9:
                achados.append({"direcao": "sobre_o_teto_de_massa",
                                "sinal": sinal, "fator": fator,
                                "melhora": float(c0 - c)})
    return {"melhoras_encontradas": achados,
            "sobre_o_teto": ao_longo,
            "mapa_do_teto": "u' = u e^t, R' = R e^-t, que preserva uR exatamente",
            "direcoes_testadas": 2 * len(fatores) * len(nomes) + 2 * len(fatores),
            "nota": ("ausencia de melhora nestas direcoes NAO e prova de maximo: "
                     "McKinnon mostra convergencia a ponto nao estacionario mesmo "
                     "em funcao estritamente convexa")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vetor", required=True, help="json com nomes e rotulos")
    ap.add_argument("--rotulo", required=True, help="colado, aberto, ou um nome do json")
    ap.add_argument("--pedido", default="exploracoes/ajuste_geometria_2026-10-01/A_i91/pedido.json")
    ap.add_argument("--eventos", default="dados/eventos.csv")
    ap.add_argument("--bandas", type=int, default=120,
                    help="120 por omissao: a 26 bandas a diferenca entre os "
                         "vetores vale 8,90 contra 10,06 a 10,17 de 120 para cima")
    ap.add_argument("--refino", type=int, default=8)
    #: MALHA E TABELA, repassadas ao motor. Sem elas a reotimizacao rodaria na
    #: malha de PRODUCAO com a tabela padrao de 1800 nos, isto e FORA da
    #: configuracao aprovada: medido, a malha de producao fica a 0,2190 da
    #: referencia e a tabela em 1800 move as diferencas em ate 0,1986. O
    #: lancador da frente C passa psi0/480/115200.
    ap.add_argument("--malha", default=None, choices=[None, "psi0"])
    ap.add_argument("--tabela", type=int, default=None)
    #: M, a quadratura de fase do ALVO. Sem isto a reotimizacao acharia o maximo
    #: de uma FUNCAO DIFERENTE da que os pilotos amostram: medido, M = 1 erra
    #: 0,63 nas diferencas de ln L contra M = 15, e M = 5 erra 0,0908. Eu havia
    #: escrito no lancador que M nao entrava na reotimizacao, e estava errado.
    ap.add_argument("--media-de-fase", type=int, default=1,
                    dest="media_de_fase")
    ap.add_argument("--avaliacoes", type=int, nargs="+", default=[500, 500, 500, 500],
                    help="orcamento de cada rodada de simplex. Quatro rodadas por "
                         "omissao: o PLATO entre elas e a evidencia de estabilidade "
                         "numerica do maximo, e vale mais que o numero final")
    ap.add_argument("--passo", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.125],
                    help="fator sobre o passo CALIBRADO em cada rodada; o da "
                         "primeira rodada define a escala, 1.0 = o calibrado")
    ap.add_argument("--metodo", choices=("powell", "nelder-mead", "cobyqa"),
                    default="nelder-mead",
                    help="nelder-mead por omissao. Medido a 26 bandas com o mesmo "
                         "orcamento, Powell DECLAROU SUCESSO nas duas rodadas e "
                         "parou 0,46 abaixo do ponto que o simplex achou, usando "
                         "708 avaliacoes contra 1039: a busca em linha estagna "
                         "num vale estreito e chama isso de convergencia. Aceitar "
                         "aquele `terminated successfully` seria reportar "
                         "convergencia que nao existe.")
    ap.add_argument("--teto-de-massa", type=float, default=None,
                    dest="teto_de_massa",
                    help="sobrescreve massCap. Para a sensibilidade ao teto: os "
                         "maximos vivem a 99,4 a 99,6 por cento de 2,5 e a cauda "
                         "da posteriori de A toca 2,50000, entao o limite trunca o "
                         "suporte e a influencia dele sobre z so se mede variando-o")
    ap.add_argument("--parada-absoluta", type=float, default=0.01,
                    dest="parada_absoluta",
                    help="para de abrir rodadas quando o ganho de uma fica abaixo "
                         "deste valor em lnL. Parada EXTERNA e absoluta, porque "
                         "com o objetivo centrado o ftol relativo do Powell deixa "
                         "de operar e o fatol do simplex e por simplex")
    ap.add_argument("--alvo-calibracao", type=float, default=1.0,
                    dest="alvo_calibracao",
                    help="piora em lnL que cada vertice do simplex deve ter")
    ap.add_argument("--perturba", type=float, default=0.0,
                    help="desvio da partida, fracao da largura do prior")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--bacia", action="store_true",
                    help="mantem a faixa de inclinacao da bacia do pedido")
    ap.add_argument("--por-observacao", action="store_true",
                    dest="por_observacao",
                    help="otimiza o ALVO POR OBSERVACAO: soma das treze ln L, "
                         "fundo marginalizado b(o,e), offsets condicionados e "
                         "resolucao efetiva zero, numa chamada do motor por "
                         "avaliacao. Sem isto a reotimizacao roda no alvo "
                         "COADICIONADO, que NAO e o alvo cientifico escolhido: "
                         "otimizar num alvo e amostrar em outro acha o maximo de "
                         "uma funcao diferente")
    ap.add_argument("--threads", type=int, default=1,
                    help="threads do laco de fases do motor, com "
                         "--por-observacao. Exige um binario com OpenMP, que se "
                         "escolhe por MAGNUS_ENGINE=build/magnus_engine_omp. A "
                         "previsao e identica bit a bit, medido na estacao em "
                         "docs/paraleliza_fases.json; no no ainda nao foi "
                         "medido, e por isso o padrao e 1")
    ap.add_argument("--fundo-fixo", action="store_true", dest="fundo_fixo",
                    help="com --por-observacao, usa fundo FIXO em vez de "
                         "marginalizado. So para controle")
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()

    if len(a.passo) != len(a.avaliacoes):
        raise SystemExit("--passo e --avaliacoes precisam ter o mesmo numero de rodadas")

    pedido = Path(a.pedido)
    base = monta_pedido(pedido, a.bandas, a.refino, not a.bacia, a.teto_de_massa)
    if a.media_de_fase != 1:
        base["phaseAverage"] = int(a.media_de_fase)
    meta, eventos = m.read_events(Path(a.eventos))
    if a.por_observacao:
        import alvo_por_observacao as APO
        prob = APO.AlvoPorObservacao(
            bandas=a.bandas, refino=a.refino, media_de_fase=a.media_de_fase,
            marginal=not a.fundo_fixo, malha=a.malha, tabela=a.tabela,
            teto_de_massa=a.teto_de_massa, uniao=not a.bacia,
            eventos=a.eventos, threads=a.threads)
        print("ALVO POR OBSERVACAO: %s" % json.dumps(prob.descreve(),
                                                     ensure_ascii=False),
              flush=True)
    else:
        prob = m.FitProblem(base, eventos, meta, None)

    dados = json.loads(Path(a.vetor).read_text())
    valida_ordem(dados["nomes"], prob.names)
    if a.rotulo not in dados:
        raise SystemExit("rotulo %r ausente; disponiveis: %s"
                         % (a.rotulo, [k for k in dados if k != "nomes"]))
    partida = list(dados[a.rotulo]["vetor"])
    if len(partida) != len(prob.names):
        raise SystemExit("vetor tem %d coordenadas, o problema tem %d"
                         % (len(partida), len(prob.names)))
    # congelados nos valores do pedido: o residuo de float32 do checkpoint
    # (0.30000001192092896) fica abaixo da tolerancia de 1e-6, mas o canonico e
    # o que o amostrador impoe.
    for nome, valor in prob.frozen.items():
        partida[prob.names.index(nome)] = float(valor)

    # PROJECAO DA PARTIDA. Com um teto menor que a massa do vetor de partida, o
    # ponto inicial fica FORA do prior e o otimizador comecaria na penalidade de
    # 1e6, onde o simplex nao tem informacao. A projecao escala u mantendo R,
    # porque M = uR/(2GM/c^2) e o que o teto limita, e fica registrada: o ponto
    # de partida da varredura de teto NAO e o mesmo das outras partidas.
    projecao = None
    massa_partida = partida[0] * partida[1] / (2.0 * m._GM_SUN_C2_KM)
    if massa_partida > prob.mass_cap:
        u_novo = partida[0] * prob.mass_cap / massa_partida
        lo_u, hi_u = prob.bounds[0]
        u_novo = min(hi_u, max(lo_u, u_novo * (1.0 - 1.0e-9)))
        projecao = {"massa_antes": massa_partida, "u_antes": partida[0],
                    "u_depois": u_novo, "teto": prob.mass_cap,
                    "regra": "u escalado por teto/massa, R mantido"}
        print("partida projetada para dentro do teto %.2f: u de %.6f para %.6f, "
              "massa de %.4f para %.4f"
              % (prob.mass_cap, partida[0], u_novo, massa_partida,
                 u_novo * partida[1] / (2.0 * m._GM_SUN_C2_KM)), flush=True)
        partida[0] = u_novo

    comando = list(prob.worker_command())
    if not a.por_observacao:
        if a.malha == "psi0":
            comando += ["--malha-psi0", "--psi-bandas", str(a.bandas)]
        if a.tabela:
            comando += ["--ray-table-size", str(a.tabela)]
    print("configuracao do motor: malha %s, bandas %d, tabela %s, M = %d "
          "(%d fases pedidas ao motor)"
          % (a.malha or "producao", a.bandas, a.tabela or "padrao 1800",
             a.media_de_fase, prob.phase_bins * prob.media_de_fase), flush=True)
    trabalhador = m.EngineWorker(comando)
    custo = Custo(prob, trabalhador)
    if a.metodo == "cobyqa":
        # a banda de massa sai do objetivo e vai para a restricao explicita.
        # Antes disso, conferir que nenhuma OUTRA porta do in_prior esta ativa,
        # senao tira-se silenciosamente um vinculo junto com a banda.
        if prob.fit_beaming:
            raise SystemExit("fit_beaming ativo: in_prior tem um vinculo de "
                             "beaming que sairia junto com a banda de massa")
        if prob.break_symmetry and prob.spot_count == 1:
            raise SystemExit("break_symmetry ativo com um ponto: a ordem "
                             "inclinacao <-> colatitude sairia junto com a banda")
        custo.banda_de_massa_no_objetivo = False
        print("cobyqa: banda de massa [%.2f, %.2f] passada como restricao; o "
              "objetivo fica avaliavel fora dela, com a caixa e u < 0,985 "
              "mantidos" % (max(0.8, prob.mass_floor), prob.mass_cap), flush=True)

    if a.perturba > 0.0:
        rng = np.random.default_rng(a.semente)
        x0 = custo.para_x(partida)
        for _ in range(500):
            tentativa = np.clip(x0 + rng.normal(0.0, a.perturba, x0.size), 0.0, 1.0)
            if prob.in_prior(custo.para_vetor(tentativa)):
                x0 = tentativa
                break
        else:
            raise SystemExit("nao achei partida perturbada dentro do prior")
        partida = custo.para_vetor(x0)

    x = custo.para_x(partida)
    diretas = [0]

    def avalia_direto(v):
        diretas[0] += 1
        return prob.evaluate(trabalhador, v)

    posterior0 = avalia_direto(partida)
    custo.referencia = float(posterior0)
    extra0 = prob.extra_log_prior(partida)
    print("partida %s/%s: ln posterior = %.4f (lnL = %.4f, prior = %.4f)"
          % (Path(a.vetor).stem, a.rotulo, posterior0, posterior0 - extra0, extra0),
          flush=True)

    t0 = time.time()
    marca = time.time()
    passos_cal, registro_cal = calibra_passos(custo, x, alvo=a.alvo_calibracao)
    n_calibracao = custo.avaliacoes
    print("calibracao: %d avaliacoes, %.1f min. piora por coordenada entre "
          "%.2f e %.2f em lnL"
          % (custo.avaliacoes - 1, (time.time() - marca) / 60.0,
             min(d["piora_em_lnL"] for d in registro_cal),
             max(d["piora_em_lnL"] for d in registro_cal)), flush=True)
    rodadas = []
    anterior = posterior0

    def grava(parcial: bool, vetor_atual, posterior_atual, sonda_atual=None):
        """Grava o artefato a cada rodada, nao so no fim.

        Em 05/10/2026 isto faltava e custou uma decisao: com a campanha na
        rodada 2 de 4, o revisor pediu pausa antes dos reinicios restantes, e
        matar os processos destruiria os vetores, porque o artefato so existia
        ao final. Agora cada rodada deixa o melhor ponto em disco, e um arquivo
        PARAR ao lado da saida encerra depois da rodada corrente.
        """
        extra_a = prob.extra_log_prior(vetor_atual)
        massa_a = vetor_atual[0] * vetor_atual[1] / (2.0 * m._GM_SUN_C2_KM)
        doc = {
            "parcial": parcial,
            "data": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metodo": a.metodo,
            "tetoDeMassa": prob.mass_cap,
            "partidaProjetada": projecao,
            "paradaAbsoluta": a.parada_absoluta,
            "objetivoCentradoEm": custo.referencia,
            "pedidoEfetivo": base,
            "origem": {"vetor": str(a.vetor), "rotulo": a.rotulo,
                       "pedidoSha256": hashlib.sha256(
                           json.dumps(base, sort_keys=True).encode()).hexdigest(),
                       "pedido": str(pedido), "perturba": a.perturba,
                       "semente": a.semente,
                       "faixaInclinacao": base["inclinationRange"],
                       "uniaoDasBacias": not a.bacia},
            "quadratura": {"bandas": a.bandas, "refino": a.refino,
                           "superficie": m.superficie_efetiva(base)},
            "alvo": (prob.descreve() if a.por_observacao
                     else {"alvo": "coadicionado", "npz":
                           base.get("preparedDataNpz")}),
            "nomes": prob.names, "congelados": prob.frozen,
            "livres": [prob.names[i] for i in custo.livres],
            "calibracao": {"alvo_em_lnL": a.alvo_calibracao,
                           "por_coordenada": {prob.names[i]: registro_cal[k]
                                              for k, i in enumerate(custo.livres)}},
            "partida": {"vetor": partida, "lnPosterior": posterior0,
                        "lnL": posterior0 - extra0, "logPrior": extra0},
            "otimo": {"vetor": vetor_atual, "lnPosterior": posterior_atual,
                      "lnL": posterior_atual - extra_a, "logPrior": extra_a},
            "ganho": posterior_atual - posterior0,
        #: ESTATUTO DO PONTO, escrito pelo proprio arquivo. Chamar de "otimo" o
        #: ponto final de um orcamento esgotado e abuso de linguagem: se a
        #: ultima rodada ainda rendeu mais que o limiar, a busca NAO estabilizou
        #: e o que se tem e o MELHOR PONTO ENCONTRADO. O limiar e regra de
        #: orcamento e nao prova de convergencia, e quem le a saida precisa
        #: saber em qual dos dois casos esta.
        "estatutoDoPonto": _estatuto_do_ponto(rodadas, a.parada_absoluta),
            "derivados": {"massa": massa_a, "compactness": vetor_atual[0],
                          "MsobreR_Msol_por_km": massa_a / vetor_atual[1],
                          "GMsobreRc2": vetor_atual[0] / 2.0,
                          "a1": vetor_atual[prob.names.index("peaking")],
                          "a2": vetor_atual[prob.names.index("peaking2")],
                          "z": 1.0 / math.sqrt(1.0 - vetor_atual[0]) - 1.0},
            "pontoFinalDentroDaBandaDeMassa": bool(
                max(0.8, prob.mass_floor) <= massa_a <= prob.mass_cap),
            "bandaDeMassaNoObjetivo": custo.banda_de_massa_no_objetivo,
            "sondaDoPontoFinal": sonda_atual,
            "rodadas": rodadas,
            "avaliacoesDoMotor": {"ate_agora": custo.avaliacoes + diretas[0]},
            "segundos": time.time() - t0,
        }
        destino = Path(a.saida)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        return doc

    parar = Path(a.saida).with_name("PARAR")
    for indice, (orcamento, passo) in enumerate(zip(a.avaliacoes, a.passo), 1):
        antes = custo.avaliacoes
        marca = time.time()
        escala = passos_cal * (passo / a.passo[0])
        if a.metodo == "cobyqa":
            # a fronteira de massa entra como RESTRICAO, nao como penalidade de
            # 1e6: COBYQA e sem derivadas, com modelo quadratico e regiao de
            # confianca, e aceita restricao nao linear explicita.
            from scipy.optimize import NonlinearConstraint
            r = minimize(custo, x, method="cobyqa",
                         bounds=[(0.0, 1.0)] * x.size,
                         constraints=[NonlinearConstraint(custo.restricoes,
                                                          0.0, np.inf)],
                         options={"maxfev": orcamento,
                                  "initial_tr_radius": float(np.mean(np.abs(escala))),
                                  "final_tr_radius": 1.0e-6})
        elif a.metodo == "powell":
            # Powell faz busca em LINHA ao longo de um conjunto de direcoes, e o
            # passo calibrado e exatamente a escala inicial de cada direcao.
            # Nelder-Mead em 17 dimensoes esgotou 1000 avaliacoes sem convergir
            # no teste a 26 bandas, ganhando 1,33; com 35 s por avaliacao isso
            # entregaria ponto melhorado e nao maximo.
            direcoes = np.diag(escala)
            r = minimize(custo, x, method="Powell",
                         bounds=[(0.0, 1.0)] * x.size,
                         options={"direc": direcoes, "maxfev": orcamento,
                                  "xtol": 1.0e-4, "ftol": 1.0e-6, "disp": False})
            # com o objetivo centrado o limiar deste ftol cai para a ordem de
            # ftol*|ganho|, muito menor, mas ele CONTINUA OPERANDO e pode
            # encerrar a rodada: o motivo da parada entra no artefato.
        else:
            r = minimize(custo, x, method="Nelder-Mead",
                         bounds=[(0.0, 1.0)] * x.size,
                         options={"initial_simplex": simplex_inicial(x, escala),
                                  "maxfev": orcamento, "adaptive": True,
                                  "xatol": 1.0e-4, "fatol": 5.0e-2, "disp": False})
        x = np.clip(r.x, 0.0, 1.0)
        vetor = custo.para_vetor(x)
        posterior = avalia_direto(vetor)
        msg = str(r.message)
        # TRES categorias distintas, e nao um "convergiu" so. Tolerancia
        # satisfeita nao e maximo, orcamento esgotado nao e convergencia, e a
        # parada externa e minha decisao e nao do metodo.
        if "function evaluations" in msg or "iterations" in msg.lower():
            motivo = "orcamento_esgotado"
        else:
            motivo = "parada_interna_do_metodo"
        rodadas.append({"rodada": indice, "passo": passo, "orcamento": orcamento,
                        "avaliacoes": custo.avaliacoes - antes,
                        "segundos": time.time() - marca,
                        "lnPosterior": posterior,
                        "metodo": a.metodo,
                        "motivoDaParada": motivo,
                        "convergiu": bool(r.success), "mensagem": msg})
        print("rodada %d: ln posterior = %.4f apos %d avaliacoes (%.1f min) [%s]"
              % (indice, posterior, custo.avaliacoes - antes,
                 (time.time() - marca) / 60.0, r.message), flush=True)
        # PARADA EXTERNA E ABSOLUTA. Com o objetivo centrado, nenhum dos metodos
        # tem mais um criterio de parada na escala certa: o do Powell e relativo
        # e virou inoperante, e o do Nelder-Mead e absoluto mas no simplex, nao
        # na rodada. Quem decide e esta comparacao, em lnL, com limiar declarado
        # e bem abaixo da variacao de 0,25 entre malhas.
        ganho_da_rodada = posterior - anterior
        anterior = posterior
        rodadas[-1]["ganhoDaRodada"] = float(ganho_da_rodada)
        grava(True, vetor, posterior)
        if parar.exists():
            print("   arquivo %s presente: encerro depois desta rodada, com o "
                  "artefato parcial gravado" % parar, flush=True)
            rodadas[-1]["motivoDaParada"] = "arquivo_PARAR"
            break
        if ganho_da_rodada < a.parada_absoluta:
            print("   ganho da rodada %+.4f abaixo do limiar %.4f: paro aqui, "
                  "sem gastar as rodadas restantes"
                  % (ganho_da_rodada, a.parada_absoluta), flush=True)
            rodadas[-1]["motivoDaParada"] = "parada_externa_absoluta"
            rodadas[-1]["ganhoDaRodada"] = float(ganho_da_rodada)
            break

    vetor = custo.para_vetor(x)
    posterior = avalia_direto(vetor)
    n_antes_da_sonda = custo.avaliacoes
    if (custo.melhor[1] is not None
            and -custo.melhor[0] + custo.referencia > posterior):
        # o simplex pode terminar acima do melhor ponto que visitou
        vetor, posterior = custo.melhor[1], -custo.melhor[0] + custo.referencia
        print("o melhor ponto visitado supera o fim do simplex: uso o visitado",
              flush=True)
    marca = time.time()
    sonda = sonda_direcoes(custo, custo.para_x(vetor), passos_cal)
    print("sonda do ponto final: %d direcoes viaveis testadas em %.1f min, "
          "%d com melhora" % (sonda["direcoes_testadas"],
                              (time.time() - marca) / 60.0,
                              len(sonda["melhoras_encontradas"])), flush=True)
    for d in sonda["melhoras_encontradas"][:5]:
        print("   melhora %+.4f em %s, sinal %+.0f, fator %.3g"
              % (d["melhora"], d["direcao"], d["sinal"], d["fator"]), flush=True)
    extra = prob.extra_log_prior(vetor)
    trabalhador.close()

    massa = vetor[0] * vetor[1] / (2.0 * m._GM_SUN_C2_KM)
    piso_real, teto_real = max(0.8, prob.mass_floor), prob.mass_cap
    dentro_da_banda = piso_real <= massa <= teto_real
    if not dentro_da_banda:
        print("ATENCAO: o ponto final tem M = %.4f, FORA da banda [%.2f, %.2f]. "
              "Com a banda como restricao, o metodo pode terminar violando-a: "
              "este ponto nao e admissivel sob o prior declarado."
              % (massa, piso_real, teto_real), flush=True)
    saida = {
        "data": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pedidoEfetivo": base,
        "metodo": a.metodo,
        "tetoDeMassa": prob.mass_cap,
        "partidaProjetada": projecao,
        "paradaAbsoluta": a.parada_absoluta,
        "objetivoCentradoEm": custo.referencia,
        "origem": {"vetor": str(a.vetor), "rotulo": a.rotulo,
                   "pedidoSha256": hashlib.sha256(
                       json.dumps(base, sort_keys=True).encode()).hexdigest(),
                   "pedido": str(pedido), "perturba": a.perturba,
                   "semente": a.semente,
                   "faixaInclinacao": base["inclinationRange"],
                   "uniaoDasBacias": not a.bacia},
        "quadratura": {"bandas": a.bandas, "refino": a.refino,
                       "superficie": m.superficie_efetiva(base)},
        "alvo": (prob.descreve() if a.por_observacao
                 else {"alvo": "coadicionado", "npz":
                   base.get("preparedDataNpz")}),
        "nomes": prob.names,
        "congelados": prob.frozen,
        "livres": [prob.names[i] for i in custo.livres],
        "partida": {"vetor": partida, "lnPosterior": posterior0,
                    "lnL": posterior0 - extra0, "logPrior": extra0},
        "otimo": {"vetor": vetor, "lnPosterior": posterior,
                  "lnL": posterior - extra, "logPrior": extra},
        "ganho": posterior - posterior0,
        #: Mesmo estatuto do documento parcial: "melhor ponto encontrado" quando
        #: a ultima rodada ainda rendeu mais que o limiar de orcamento.
        "estatutoDoPonto": _estatuto_do_ponto(rodadas, a.parada_absoluta),
        "derivados": {"massa": massa, "compactness": vetor[0],
                      "MsobreR_Msol_por_km": massa / vetor[1],
                      "GMsobreRc2": vetor[0] / 2.0,
                      "a1": vetor[prob.names.index("peaking")],
                      "a2": vetor[prob.names.index("peaking2")],
                      "z": 1.0 / math.sqrt(1.0 - vetor[0]) - 1.0},
        "calibracao": {"alvo_em_lnL": a.alvo_calibracao,
                       "por_coordenada": {prob.names[i]: registro_cal[k]
                                          for k, i in enumerate(custo.livres)}},
        "pontoFinalDentroDaBandaDeMassa": bool(dentro_da_banda),
        "bandaDeMassaNoObjetivo": custo.banda_de_massa_no_objetivo,
        "tetoDeMassaAtivo": {
            "massa_da_partida": partida[0] * partida[1] / (2.0 * m._GM_SUN_C2_KM),
            "teto": prob.mass_cap,
            "folga_em_u": ((prob.mass_cap - partida[0] * partida[1]
                            / (2.0 * m._GM_SUN_C2_KM)) * 2.0 * m._GM_SUN_C2_KM
                           / partida[1]),
            "nota": ("os tres pontos de partida estao a 99,4 a 99,6 por cento do teto, "
                     "entao o otimo em compactness e de BORDA DO PRIOR e nao dos dados; "
                     "a sensibilidade ao teto e item proprio do plano")},
        "sondaDoPontoFinal": sonda,
        "avaliacoesDoMotor": {
            "calibracao": n_calibracao,
            "rodadas": n_antes_da_sonda - n_calibracao,
            "sonda_do_ponto_final": custo.avaliacoes - n_antes_da_sonda,
            "diretas_de_conferencia": diretas[0],
            "total": custo.avaliacoes + diretas[0],
            "nota": ("conta TUDO que chamou o motor, inclusive calibracao, sonda e as "
                     "avaliacoes diretas de conferencia no fim de cada rodada. "
                     "Comparacao entre metodos tem de usar o total.")},
        "rodadas": rodadas,
        "avaliacoes": custo.avaliacoes,
        "segundos": time.time() - t0,
        "ambiente": {"scipy": __import__("scipy").__version__,
                     "python": platform.python_version(),
                     "motor": comando[0],
                     "motorSha256": sha256(Path(comando[0])),
                     "comando": comando,
                     "comandoSha256": hashlib.sha256(
                         " ".join(comando).encode()).hexdigest(),
                     "commit": _commit_se_houver()},
    }
    destino = Path(a.saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, indent=2))
    print("ln posterior: %.4f -> %.4f (ganho %+.4f) em %d avaliacoes do motor "
          "(%d calibracao, %d rodadas, %d sonda, %d diretas), %.1f min"
          % (posterior0, posterior, posterior - posterior0,
             custo.avaliacoes + diretas[0], n_calibracao,
             n_antes_da_sonda - n_calibracao,
             custo.avaliacoes - n_antes_da_sonda, diretas[0],
             saida["segundos"] / 60.0))
    print("motivo da parada de cada rodada: %s"
          % ", ".join("%d=%s" % (d["rodada"], d["motivoDaParada"]) for d in rodadas))
    print("a1 = %.6g, a2 = %.6g, u = %.6f, z = %.4f, M = %.3f, R = %.3f, i = %.3f"
          % (saida["derivados"]["a1"], saida["derivados"]["a2"], vetor[0],
             saida["derivados"]["z"], massa, vetor[1], vetor[2]))
    print("gravado em", destino)


if __name__ == "__main__":
    main()
