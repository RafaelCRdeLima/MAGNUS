#!/usr/bin/env python3
"""O alvo POR OBSERVACAO como objetivo otimizavel: soma das treze.

Existe porque a frente C, como estava, reotimizava no alvo COADICIONADO
(`combinado13.npz`, sem fundo marginalizado e sem offsets por observacao),
enquanto o alvo cientifico escolhido pelo revisor em 09/10/2026 e o de um fundo
latente b(o,e) por observacao com os offsets condicionados. Otimizar num alvo e
amostrar em outro acha o maximo de uma funcao diferente, que foi exatamente o
erro que o revisor apontou sobre M.

O QUE ESTA CLASSE E: um objeto com a interface que `reotimiza.Custo` consome
(`names`, `bounds`, `frozen`, `in_prior`, `evaluate`, ...), mas cujo `evaluate`
faz UMA chamada do motor com as treze observacoes como slots e devolve

    ln P = ln priori + soma_o ln L_o

O QUE ELA NAO E: nao tem dado proprio. Pedir-lhe `observed`, `background` ou
`exposure` e erro, e ela RECUSA em vez de devolver o do coadicionado, que seria
silenciosamente o dado errado.

APROXIMACOES DECLARADAS, todas medidas e nenhuma suposta:
  - periodo comum a montante na rota de reuso: desvio relativo 8,86e-05 entre os
    treze, move a previsao em 1,640e-08 (docs/por_observacao.json);
  - resolucao temporal EFETIVA ZERO nos slots. Em M = 5 isto nao e escolha de
    conveniencia: o operador hoje implementado nada faz nesse M, por guarda nas
    seis LARGEWINDOW e por janela 1 nas sete FULLWINDOW
    (docs/suavizacao_em_m.json). Passa-la explicitamente mantem a rota identica
    a validada se M mudar;
  - independencia entre bins de energia no fundo: os 39 canais partidos geram
    dois latentes onde deveria haver um compartilhado. Medido em
    docs/correlaciona_fundo.json;
  - offsets FIXOS, em resolucao inteira, sem propagar a incerteza sub-bin.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import mcmc_fit as m  # noqa: E402
import por_observacao as PO  # noqa: E402
import reotimiza as r  # noqa: E402

#: Atributos de DADO que nao podem ser delegados ao problema base: devolver o do
#: coadicionado seria o dado errado, em silencio.
PROIBIDOS = ("observed", "background", "exposure", "exposicao_por_fase",
             "fundo_pesos", "fundo_contagens", "fundo_deslocamento",
             "fundo_largura_do_canal", "fundo_canal_do_termo", "_fundo_k",
             "_fundo_nos", "_fundo_peso_pi")


class AlvoPorObservacao:
    def __init__(self, bandas=480, refino=8, media_de_fase=1, marginal=True,
                 malha=None, tabela=None, teto_de_massa=None, uniao=True,
                 eventos=None, threads=1, excluir=()):
        obss = {o["obsid"]: o for o in PO.prepara(verbose=False, regravar=False)}
        #: EXCLUIR observacoes, por identificador. Existe para a comparacao com
        #: e sem a 0402851001, que tem nivel de particulas cerca de vinte vezes
        #: o das outras doze mesmo contra as de mesma area e submodo
        #: (docs/diagnostico_flare.json). A exclusao e EXPLICITA e vai para o
        #: registro do alvo: um alvo com doze observacoes nao pode se passar por
        #: um alvo com treze.
        self.excluidas = sorted(str(x) for x in excluir)
        desconhecidas = [x for x in self.excluidas if x not in obss]
        if desconhecidas:
            raise ValueError("pedido para excluir observacao inexistente: %s"
                             % desconhecidas)
        for x in self.excluidas:
            obss.pop(x)
        if not obss:
            raise ValueError("todas as observacoes foram excluidas")
        self.ordem = sorted(obss)
        self.dado_conferido = self._confere_o_dado(obss)
        self.obss = obss
        self.bandas = int(bandas)
        self.refino = int(refino)
        self.malha = malha
        self.tabela = tabela
        self.marginal = bool(marginal)
        #: THREADS do laco de fases. 1 por omissao. Com mais de uma, o comando
        #: passa --threads ao motor, o que SO tem efeito num binario compilado
        #: com OpenMP (build/magnus_engine_omp, via MAGNUS_ENGINE). A identidade
        #: bit a bit da previsao foi medida na estacao, em
        #: docs/paraleliza_fases.json; no no ela ainda NAO foi medida, e e por
        #: isso que o padrao e 1.
        self.threads = max(1, int(threads))
        meta, ev = m.read_events(Path(eventos) if eventos
                                 else RAIZ / "dados/eventos.csv")
        # O PRIOR E O MODELO vem do pedido canonico. Ele e montado pela rota
        # POR OBSERVACAO da primeira, e nao por r.monta_pedido direto, por um
        # motivo concreto: o pedido canonico declara o instrumento
        # `xmm_epn_imaging_combinado13`, que nao existe no catalogo de perfis, e
        # quem o resolvia era justamente o `responseProfile` por observacao. A
        # uniao de inclinacao e o teto de massa sao aplicados em seguida, pelos
        # mesmos valores que `r.monta_pedido` usaria.
        base = PO.monta_pedido(obss[self.ordem[0]], bandas, refino)
        if uniao:
            base["inclinationRange"] = list(r.INCLINACAO_UNIAO)
        if teto_de_massa is not None:
            base["massCap"] = float(teto_de_massa)
        base["outputPrecision"] = 17
        if media_de_fase != 1:
            base["phaseAverage"] = int(media_de_fase)
        self._base = m.FitProblem(base, ev, meta, None)
        # Um problema por observacao, SO para pontuar: dado, exposicao e fundo
        # daquela observacao.
        self.probs = {}
        for obsid in self.ordem:
            p = PO.monta_pedido(obss[obsid], bandas, refino)
            p["outputPrecision"] = 17
            if marginal:
                p["marginalizeBackground"] = True
            if media_de_fase != 1:
                p["phaseAverage"] = int(media_de_fase)
            prob = m.FitProblem(p, ev, meta, None)
            if marginal:
                # Falha cedo e com mensagem, nao na milesima avaliacao.
                prob.prepara_fundo_marginal()
            self.probs[obsid] = prob
        self.avaliacoes = 0
        self.falhas = 0

    @staticmethod
    def _confere_o_dado(obss):
        """O npz de cada observacao e o mesmo que o registro declara.

        Existe porque escrita atomica resolve corrupcao PARCIAL e nao
        substituicao: `os.replace` troca o arquivo inteiro mesmo quando ele esta
        somente leitura, porque o que governa renomear e a permissao do
        DIRETORIO. Em 10/10/2026 tres npz foram corrompidos por escrita
        concorrente, e o diretorio passou a ser congelado; esta conferencia e a
        segunda linha, a que detecta troca silenciosa de entrada mesmo com o
        congelamento desfeito por acidente.

        A referencia e `sha256_npz` do proprio registro preparado, que por sua
        vez e conferido contra docs/manifesto_do_alvo_provisorio.json quando ele
        existe. Divergencia e ERRO, nao aviso: avaliar com dado diferente do
        declarado produz numero plausivel e errado.
        """
        manifesto = RAIZ / "docs/manifesto_do_alvo_provisorio.json"
        declarado = {}
        if manifesto.is_file():
            try:
                m = json.loads(manifesto.read_text())
                declarado = {o["obsid"]: o["sha256_npz"]
                             for o in m["1_dados_por_observacao"]["por_observacao"]}
            except (KeyError, ValueError):
                declarado = {}
        divergentes = []
        for obsid in sorted(obss):
            caminho = Path(obss[obsid]["npz"])
            atual = hashlib.sha256(caminho.read_bytes()).hexdigest()
            for fonte, esperado in (("registro", obss[obsid].get("sha256_npz")),
                                    ("manifesto", declarado.get(obsid))):
                if esperado and atual != esperado:
                    divergentes.append("%s: %s esperava %s, o arquivo tem %s"
                                       % (obsid, fonte, esperado[:12], atual[:12]))
        if divergentes:
            raise ValueError(
                "o dado nao e o declarado, em %d caso(s):\n  %s\nUm npz foi "
                "substituido depois de preparado. Nao avalio com dado "
                "divergente: o resultado seria plausivel e errado."
                % (len(divergentes), "\n  ".join(divergentes)))
        return {"observacoes": len(obss), "contra_o_manifesto": bool(declarado),
                "divergentes": 0}

    # ---- delegacao explicita do que e MODELO e PRIOR -----------------------
    def __getattr__(self, nome):
        if nome.startswith("_") or nome in PROIBIDOS:
            raise AttributeError(
                "%r nao existe no alvo por observacao: ele nao tem dado "
                "proprio, e delegar ao problema base devolveria o do "
                "COADICIONADO. Use self.probs[obsid].%s" % (nome, nome))
        return getattr(self._base, nome)

    def worker_command(self):
        """O comando do motor: emissao comum mais os treze slots."""
        comando = list(self._base.worker_command())
        if self.malha == "psi0":
            comando += ["--malha-psi0", "--psi-bandas", str(self.bandas)]
        if self.tabela:
            comando += ["--ray-table-size", str(int(self.tabela))]
        if self.threads > 1:
            # Passado SO quando maior que 1, para que o caso de uma thread
            # produza o MESMO comando de antes.
            comando += ["--threads", str(self.threads)]
        for obsid in self.ordem:
            o = self.obss[obsid]
            comando += [
                "--obs-id", obsid,
                "--obs-profile", o["perfil_csv"],
                "--obs-response", o["perfil_rmfbin"],
                # RESOLUCAO EFETIVA ZERO, declarada acima.
                "--obs-time-resolution-us", "0.0",
                "--obs-dead-time-us", "0.0",
                "--obs-period", repr(float(o["periodo"])),
                "--obs-offset", repr(PO.offset_em_ciclos(int(o["desloc_v3_bins"])))]
        return comando

    def resolucao_declarada(self):
        """O que o manifesto PRESERVA, ainda que o operador efetivo seja zero."""
        return {obsid: float(self.obss[obsid]["resolucao_us"])
                for obsid in self.ordem}

    # ---- o objetivo --------------------------------------------------------
    def termos(self, worker, valores):
        """ln L de cada observacao, UMA chamada do motor."""
        doc = worker.evaluate(self._base.worker_line(valores))
        if doc.get("status") != "ok":
            raise m.FalhaDoMotor("motor devolveu status %r: %s"
                                 % (doc.get("status"),
                                    str(doc.get("message"))[:200]))
        if "observations" not in doc:
            raise m.FalhaDoMotor(
                "o motor nao devolveu 'observations': a rota de reuso nao foi "
                "acionada, logo isto NAO e o alvo por observacao")
        grades = {o["id"]: o["detected_count_rate"] for o in doc["observations"]}
        if set(grades) != set(self.ordem):
            raise m.FalhaDoMotor("slots devolvidos %s nao batem com os pedidos %s"
                                 % (sorted(grades), self.ordem))
        return {obsid: float(self.probs[obsid].score_document(
                    {"status": "ok", "detected_count_rate": grades[obsid]}))
                for obsid in self.ordem}

    def evaluate(self, worker, valores, return_model=False, return_pulse=False):
        """ln priori + soma dos treze. `-inf` SO fora da priori."""
        if return_model or return_pulse:
            raise NotImplementedError(
                "return_model/return_pulse nao existem nesta rota: ha treze "
                "modelos, um por observacao. Use termos() e os probs")
        self.avaliacoes += 1
        if not self._base.in_prior(valores):
            return -math.inf
        extra = self._base.extra_log_prior(valores)
        for tentativa in (1, 2):
            try:
                termos = self.termos(worker, valores)
                break
            except (m.FalhaDoMotor, BrokenPipeError, OSError, RuntimeError,
                    ValueError) as erro:
                self.falhas += 1
                if tentativa == 2:
                    # NAO devolve -inf: falha do motor nao e recusa do vetor, e
                    # mascara-la como verossimilhanca baixa corromperia o otimo.
                    raise
                # Worker NOVO com o MESMO comando: distingue processo corrompido
                # de vetor recusado, como faz FitProblem.evaluate.
                worker.reiniciar()
                self.ultimo_erro = str(erro)[:300]
        total = extra + sum(termos.values())
        return total if math.isfinite(total) else -math.inf

    def perfis_por_observacao(self, worker, valores):
        """Os TREZE perfis de pulso, um por observacao, com procedencia.

        Decisao do revisor em 11/10/2026, sobre o que registrar como resultado
        cientifico: os treze perfis POR OBSERVACAO, com suas respostas,
        exposicoes e offsets, e uma visualizacao combinada apenas como
        complemento. Isso encerra a pendencia de desenho que bloqueava o
        amostrador: nao se escolhe UM perfil entre treze, registram-se os treze.

        Cada perfil vem do mesmo caminho que pontua: a grade do slot, promediada
        para as 32 fases observadas quando M > 1, e somada em energia com a
        largura do bin e o tempo representado daquela observacao. O que acompanha
        cada um e o que o torna interpretavel: a resposta usada, a exposicao, o
        offset aplicado e a convencao de fase.
        """
        doc = worker.evaluate(self._base.worker_line(valores))
        if doc.get("status") != "ok" or "observations" not in doc:
            raise m.FalhaDoMotor("sem 'observations' para extrair os perfis")
        grades = {o["id"]: o["detected_count_rate"] for o in doc["observations"]}
        saida = {}
        for obsid in self.ordem:
            prob = self.probs[obsid]
            ln, esperado = prob.score_document(
                {"status": "ok", "detected_count_rate": grades[obsid]},
                return_model=True)
            esp = np.asarray(esperado, float).reshape(prob.phase_bins, -1)
            o = self.obss[obsid]
            saida[obsid] = {
                "perfil_contagens_por_fase": esp.sum(axis=1).tolist(),
                "ln_L": float(ln),
                "resposta": o["perfil_id"],
                "sha256_da_resposta": o.get("sha256_perfil_rmfbin"),
                "exposicao_s": float(o["exposicao"]),
                "offset_em_ciclos": PO.offset_em_ciclos(int(o["desloc_v3_bins"])),
                "resolucao_declarada_us": float(o["resolucao_us"]),
                "submodo": o["submodo"],
                "convencao_de_fase": o.get("convencao_de_fase"),
            }
        return saida

    def descreve(self):
        return {
            "alvo": "por observacao, soma de treze ln L",
            "fundo": "marginalizado b(o,e)" if self.marginal else "fixo",
            "observacoes": list(self.ordem),
            "quantas": len(self.ordem),
            "excluidas": list(self.excluidas),
            "bandas": self.bandas, "refino": self.refino,
            "malha": self.malha or "producao",
            "tabela": self.tabela or "padrao 1800",
            "media_de_fase": int(getattr(self._base, "media_de_fase", 1)),
            "fases_pedidas_ao_motor": int(self._base.phase_bins
                                          * getattr(self._base, "media_de_fase", 1)),
            "threads_do_laco_de_fases": self.threads,
            "dado_conferido": self.dado_conferido,
            "resolucao_efetiva_us": 0.0,
            "perfil_do_comando_base": self.obss[self.ordem[0]]["perfil_id"],
            "por_que_o_perfil_base_nao_contamina": (
                "a resposta aplicada e a do SLOT, uma por observacao; o perfil "
                "do comando base existe porque o motor exige um e foi medido "
                "inerte na rota de reuso, identidade em docs/por_observacao.json"),
            "resolucao_declarada_us": self.resolucao_declarada(),
            "offsets_em_ciclos": {o: PO.offset_em_ciclos(
                int(self.obss[o]["desloc_v3_bins"])) for o in self.ordem},
            "aproximacoes_pendentes": [
                "independencia energetica do fundo nos 39 canais partidos, "
                "medida em docs/correlaciona_fundo.json",
                "incerteza sub-bin dos offsets NAO propagada",
                "periodo comum a montante na rota de reuso",
                "resposta temporal sub-bin ausente: o operador efetivo e zero",
            ],
        }
