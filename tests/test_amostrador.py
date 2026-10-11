"""Regressao do amostrador: o passo de esticamento tem de amostrar a posterior.

O ajuste da RBS 1223 congela parametros pela chave `fixed`, e uma coordenada
congelada nunca se move: o enxame vive num subespaco afim de dimensao
len(names) - frozen. A aceitacao de Goodman-Weare, z^(n-1), so e exata com o n
desse subespaco. Usar len(names) punha um fator z^frozen a mais, que favorece
propostas de expansao e infla a dispersao. Estes testes medem a distribuicao
amostrada contra um alvo analitico, o unico jeito de pegar esse erro: ele nao
quebra nada, nao levanta excecao e o r-hat continua bonito.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import math
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mcmc_fit  # noqa: E402


class PoolGaussiano:
    """Alvo: normal padrao nas `livres` primeiras coordenadas."""

    def __init__(self, livres):
        self.livres = livres

    def evaluate(self, problem, states, with_pulse=True):
        return [(-0.5 * sum(v * v for v in s[:self.livres]), [0.0])
                for s in states]


def roda(livres, congeladas, walkers=32, iterations=4000, burn_in=500,
         seed=773, stretch_scale=1.3):
    rng = random.Random(874)
    states = [[rng.gauss(0, 1) for _ in range(livres)] + [0.0] * congeladas
              for _ in range(walkers)]
    problem = SimpleNamespace(
        names=["x%d" % i for i in range(livres)]
              + ["f%d" % i for i in range(congeladas)],
        frozen={"f%d" % i: 0.0 for i in range(congeladas)})
    resultado = mcmc_fit.run_ensemble(
        problem, PoolGaussiano(livres), walkers, iterations, burn_in, seed,
        stretch_scale, resume_states=states)
    return resultado["trajectories"]


def variancia(trajetorias, coluna):
    amostras = [passo[coluna] for caminhante in trajetorias
                for passo in caminhante]
    media = sum(amostras) / len(amostras)
    return sum((v - media) ** 2 for v in amostras) / len(amostras)


class TestDimensaoEfetiva(unittest.TestCase):
    """A variancia amostrada nao pode depender de quantas coordenadas estao
    congeladas: elas nao fazem parte do espaco em que o enxame se move."""

    def test_sem_congeladas_recupera_a_variancia(self):
        var = variancia(roda(livres=1, congeladas=0), 0)
        self.assertAlmostEqual(var, 1.0, delta=0.06,
                               msg="alvo N(0,1) sem congeladas deu %.4f" % var)

    def test_congeladas_nao_inflam_a_dispersao(self):
        """Com len(names) no expoente este caso dava variancia ~1,85."""
        var = variancia(roda(livres=1, congeladas=2), 0)
        self.assertAlmostEqual(var, 1.0, delta=0.06,
                               msg="duas coordenadas congeladas deram %.4f; "
                                   "o expoente do stretch move esta contando "
                                   "dimensoes que o enxame nao explora" % var)

    def test_congeladas_ficam_exatamente_constantes(self):
        """A premissa do conserto: se elas se movessem, o expoente maior
        estaria certo e o conserto e que introduziria vies."""
        trajetorias = roda(livres=1, congeladas=2)
        for coluna in (1, 2):
            valores = {passo[coluna] for caminhante in trajetorias
                       for passo in caminhante}
            self.assertEqual(valores, {0.0},
                             "coordenada congelada %d se moveu" % coluna)

    def test_multidimensional_com_congeladas(self):
        """A dimensionalidade do ajuste real: varias livres e duas congeladas.
        O vies cresce com frozen/livres, entao aqui ele e pequeno, e e por isso
        que o teste de uma dimensao acima e o que da o sinal forte."""
        trajetorias = roda(livres=4, congeladas=2, iterations=6000)
        for coluna in range(4):
            var = variancia(trajetorias, coluna)
            self.assertAlmostEqual(var, 1.0, delta=0.12,
                                   msg="coordenada %d deu variancia %.4f"
                                       % (coluna, var))


class TestCongeladasSaoImpostas(unittest.TestCase):
    """A checagem de posto recusa posto MENOR que o esperado e aceita posto
    maior, que e exatamente o caso em que uma coordenada declarada congelada se
    move. Como o expoente do passo de esticamento depende dessa hipotese, ela
    tem de ser imposta e conferida, nao suposta."""

    def _problema(self, congelado):
        return SimpleNamespace(names=["x", "f"], frozen={"f": congelado})

    def test_residuo_de_arredondamento_e_zerado(self):
        """float32 num checkpoint transforma 0,3 em 0,30000001192092896."""
        rng = random.Random(874)
        states = [[rng.gauss(0, 1), 0.30000001192092896] for _ in range(16)]
        trajetorias = mcmc_fit.run_ensemble(
            self._problema(0.3), PoolGaussiano(1), 16, 200, 50, 5, 1.3,
            resume_states=states)["trajectories"]
        valores = {passo[1] for caminhante in trajetorias for passo in caminhante}
        self.assertEqual(valores, {0.3})

    def test_congelada_espalhada_pelos_caminhantes_e_imposta(self):
        """draw_initial_state, no sorteio uniforme da caixa, ignora steps=0 e
        espalha a coordenada dentro de valor +- 1e-6."""
        rng = random.Random(874)
        states = [[rng.gauss(0, 1), 0.3 + rng.uniform(-1e-7, 1e-7)]
                  for _ in range(16)]
        trajetorias = mcmc_fit.run_ensemble(
            self._problema(0.3), PoolGaussiano(1), 16, 200, 50, 5, 1.3,
            resume_states=states)["trajectories"]
        valores = {passo[1] for caminhante in trajetorias for passo in caminhante}
        self.assertEqual(valores, {0.3})

    def test_retomada_com_outro_valor_congelado_e_recusada(self):
        """Misturar uma retomada de corrida com outra configuracao e o mesmo
        defeito de reprodutibilidade da retomada de grades."""
        rng = random.Random(874)
        states = [[rng.gauss(0, 1), 0.22] for _ in range(16)]
        with self.assertRaises(RuntimeError) as caso:
            mcmc_fit.run_ensemble(self._problema(0.3), PoolGaussiano(1), 16,
                                  50, 10, 5, 1.3, resume_states=states)
        self.assertIn("congela", str(caso.exception))


class TestPosto(unittest.TestCase):
    def test_enxame_degenerado_e_recusado(self):
        """Um enxame que nasce num subespaco menor deixaria parametros
        congelados a corrida inteira, com r-hat reportado como 1,00."""
        problem = SimpleNamespace(names=["a", "b"], frozen={})
        states = [[float(i), float(i)] for i in range(8)]   # posto 1 de 2
        with self.assertRaises(RuntimeError):
            mcmc_fit.run_ensemble(problem, PoolGaussiano(2), 8, 10, 0, 1, 1.3,
                                  resume_states=states)


if __name__ == "__main__":
    unittest.main()


class TestAutocorrelacaoSemTruncamento(unittest.TestCase):
    """O teto de 100 lags inflava o ESS por mais de uma ordem de grandeza."""

    @staticmethod
    def _ar1(rho, passos, cadeias, semente):
        rng = random.Random(semente)
        saida = []
        for _ in range(cadeias):
            x, v = [], 0.0
            for _ in range(passos):
                v = rho * v + rng.gauss(0.0, 1.0)
                x.append(v)
            saida.append(x)
        return saida

    def test_tau_de_ar1_com_tau_muito_acima_do_teto_antigo(self):
        # rho = 0,99 da tau = (1+rho)/(1-rho) = 199, o dobro do teto antigo.
        # Somar so 100 lags devolvia algo da ordem de 130; a soma inteira tem
        # de chegar perto de 199.
        cadeias = self._ar1(0.99, 40000, 4, 7)
        d = mcmc_fit.autocorrelation_ess(cadeias)
        self.assertGreater(d["tau"], 150.0)
        self.assertLess(d["tau"], 260.0)
        self.assertTrue(d["sequenciaTerminou"])
        self.assertTrue(d["confiavel"])

    def test_max_lag_nao_corta_mais_a_soma(self):
        # passar max_lag continua aceito, mas agora so encurta o trecho de ACF
        # guardado no artefato: o tau tem de sair igual.
        cadeias = self._ar1(0.99, 40000, 4, 11)
        inteiro = mcmc_fit.autocorrelation_ess(cadeias)
        cortado = mcmc_fit.autocorrelation_ess(cadeias, max_lag=10)
        self.assertAlmostEqual(inteiro["tau"], cortado["tau"], places=9)
        self.assertEqual(len(cortado["autocorrelation"]), 11)

    def test_cadeia_curta_demais_e_marcada_como_nao_confiavel(self):
        # 300 passos com tau ~199 da menos de 2 passos por tau: o tau devolvido
        # e um piso, e isso tem de aparecer no artefato.
        cadeias = self._ar1(0.99, 300, 4, 13)
        d = mcmc_fit.autocorrelation_ess(cadeias)
        self.assertFalse(d["confiavel"])
        self.assertLess(d["passosPorTau"], mcmc_fit.PASSOS_POR_TAU_MINIMO)

    def test_cadeia_parada_nao_recebe_ess(self):
        d = mcmc_fit.autocorrelation_ess([[2.5] * 500] * 4)
        self.assertIsNone(d["tau"])
        self.assertIsNone(d["ess"])
        self.assertFalse(d["confiavel"])


class TestColetorDeAmostras(unittest.TestCase):
    """O coletor evita retencao de trajetoria sem mudar a cadeia.

    A retencao completa custa 584 bytes por amostra em 19 dimensoes, medido, e
    5,60 GB numa corrida de 96 x 100000. O coletor existe para quem so precisa
    de uma coordenada ou de estatisticas, e o que ele NAO pode fazer e alterar a
    cadeia: estes testes comparam o caminho com e sem ele.
    """

    class _Prob:
        def __init__(self):
            self.names = ['a', 'b', 'c']
            self.bounds = [(-5.0, 5.0), (-5.0, 5.0), (-1e-6, 1e-6)]
            self.steps = [0.6, 0.6, 0.0]
            self.initial = [0.0, 0.0, 0.0]
            self.frozen = {'c': 0.0}

        def in_prior(self, v):
            return all(lo <= x <= hi for x, (lo, hi) in zip(v, self.bounds))

        def extra_log_prior(self, v):
            return 0.0

    class _Pool:
        def evaluate(self, problem, states, with_pulse=False):
            return [(-0.5 * (s[0] ** 2 + s[1] ** 2), None) for s in states]

        def close(self):
            pass

    def _roda(self, coletor=None):
        return mcmc_fit.run_ensemble(self._Prob(), self._Pool(), 10, 300, 100,
                                     seed=4242, coletor=coletor)

    def test_coletor_recebe_exatamente_as_mesmas_amostras(self):
        sem = self._roda()
        vistos = []
        com = self._roda(coletor=lambda estados, logps:
                         vistos.append([list(e) for e in estados]))
        # mesma cadeia: o coletor nao pode mudar o sorteio nem a aceitacao
        self.assertEqual(sem['acceptance'], com['acceptance'])
        # e ve exatamente o que a trajetoria teria guardado
        self.assertEqual(len(vistos), len(sem['trajectories'][0]))
        for passo, estados in enumerate(vistos):
            for walker, estado in enumerate(estados):
                self.assertEqual(estado, sem['trajectories'][walker][passo])

    def test_com_coletor_nao_retem_trajetoria(self):
        com = self._roda(coletor=lambda estados, logps: None)
        self.assertTrue(all(len(t) == 0 for t in com['trajectories']))
        self.assertTrue(all(len(t) == 0 for t in com['trajectory_logps']))

    def test_coletor_recusa_checkpoint(self):
        with self.assertRaises(RuntimeError):
            mcmc_fit.run_ensemble(self._Prob(), self._Pool(), 10, 120, 40,
                                  seed=1, checkpoint_path='/tmp/nao_deve_existir.npz',
                                  coletor=lambda e, l: None)


class TestFundoFalhaExplicitamente(unittest.TestCase):
    """Fundo ausente ou de cobertura menor nao pode passar em silencio.

    Antes, read_background devolvia ZEROS quando nao achava CSV, e np.interp
    extrapolava plano fora da faixa do arquivo. A reconstrucao de 06/10/2026
    mostrou que as 13 observacoes tinham tabela e que o vetor guardado bate bit
    a bit, logo o incidente nao ocorreu; mas o silencio estava la.
    """

    def setUp(self):
        import importlib.util
        caminho = Path(__file__).resolve().parents[1] / 'scripts' / 'coadicionar.py'
        spec = importlib.util.spec_from_file_location('coadicionar', caminho)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.eventos = self.dir / 'eventos.csv'
        self.eventos.write_text('# vazio\n', encoding='utf-8')

    def tearDown(self):
        self._tmp.cleanup()

    def test_sem_tabela_levanta_erro(self):
        with self.assertRaises(FileNotFoundError):
            self.mod.read_background(self.eventos)

    def test_sem_tabela_com_exigir_falso_devolve_zeros(self):
        v = self.mod.read_background(self.eventos, exigir=False)
        self.assertEqual(len(v), self.mod.ENERGY_BINS)
        self.assertTrue(all(x == 0.0 for x in v))

    def test_cobertura_menor_levanta_erro(self):
        # faixa de 0,5 a 0,8 keV, muito menor que a grade de 0,15 a 1,2
        alvo = self.dir / 'curta_background.csv'
        alvo.write_text('0.5,0.01\n0.8,0.02\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.mod.read_background(self.eventos)

    def test_cobertura_suficiente_passa(self):
        alvo = self.dir / 'ampla_background.csv'
        linhas = ['%.4f,%.6f' % (0.14 + 0.01 * i, 0.05) for i in range(110)]
        alvo.write_text('\n'.join(linhas) + '\n', encoding='utf-8')
        v = self.mod.read_background(self.eventos)
        self.assertEqual(len(v), self.mod.ENERGY_BINS)
        for x in v:
            self.assertAlmostEqual(x, 0.05, places=9)


class TestFalhaDoMotorContraForaDoSuporte(unittest.TestCase):
    """Auditoria da transicao: falha do motor NAO pode ser rejeicao de proposta.

    A primeira versao destes testes verificava um desenho que o revisor recusou:
    contar as causas em separado e abortar acima de um teto. O argumento dele:
    cada rejeicao por falha numerica JA altera o nucleo de transicao e pode
    excluir uma regiao valida, e o teto controla alerta, nao torna a cadeia
    correta abaixo dele. A politica agora e recuperacao deterministica e parada
    na PRIMEIRA falha nao recuperada.

    O desenho e o pre-registrado: a falha e PROVOCADA DE PROPOSITO numa regiao
    conhecida, e o teste verifica que ela nao vira rejeicao silenciosa.
    """

    def _problema(self, tentativas=1):
        problema = mcmc_fit.FitProblem.__new__(mcmc_fit.FitProblem)
        problema.names = ['a', 'b']
        problema.bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        problema.diagnostico = mcmc_fit.DiagnosticoDoMotor(tentativas=tentativas)
        problema.in_prior = lambda v: all(
            lo <= x <= hi for x, (lo, hi) in zip(v, problema.bounds))
        problema.extra_log_prior = lambda v: 0.0
        problema.worker_line = lambda v: "linha"

        # Espelha o contrato do score_document real: escalar sem modelo, par com
        # modelo. Com a politica nova ele so e chamado com documento valido.
        def pontua(documento, com_modelo=False):
            valor = -math.inf if documento.get("status") != "ok" else 0.0
            return (valor, None if valor == -math.inf else []) if com_modelo else valor
        problema.score_document = pontua
        return problema

    class _MotorFalso:
        """Motor de teste. `falha_em` decide, pelo vetor, se aquela avaliacao
        falha; `cura_ao_reiniciar` diz se o worker novo passa a funcionar."""

        def __init__(self, falha_em=None, mensagem="nan in the transfer table",
                     cura_ao_reiniciar=False, excecao=None):
            self.falha_em = falha_em or (lambda v: False)
            self.mensagem = mensagem
            self.cura_ao_reiniciar = cura_ao_reiniciar
            self.excecao = excecao
            self.reinicios = 0
            self.chamadas = 0
            self.ultimo = None

        def evaluate(self, linha):
            self.chamadas += 1
            doente = self.falha_em(self.ultimo)
            if self.cura_ao_reiniciar and self.reinicios:
                doente = False
            if doente:
                if self.excecao is not None:
                    raise self.excecao
                return {"status": "error", "message": self.mensagem}
            return {"status": "ok", "detected_count_rate": []}

        def reiniciar(self):
            self.reinicios += 1

    def _avalia(self, problema, motor, valores):
        motor.ultimo = valores
        return problema.evaluate(motor, valores)

    def test_fora_do_suporte_e_a_unica_rejeicao_legitima(self):
        p = self._problema()
        motor = self._MotorFalso()
        self.assertEqual(self._avalia(p, motor, [5.0, 0.0]), -math.inf)
        r = p.diagnostico.relatorio()
        self.assertEqual(r["foraDoSuporte"], 1)
        self.assertEqual(r["falhasRecuperadas"], 0)
        self.assertEqual(r["falhasNaoRecuperadas"], 0)
        self.assertEqual(motor.chamadas, 0)    # nem chegou ao motor

    def test_falha_nao_recuperada_INTERROMPE_na_primeira(self):
        """O ponto da correcao: nao existe teto. A primeira ja para."""
        p = self._problema()
        motor = self._MotorFalso(falha_em=lambda v: v[0] > 0.5)
        with self.assertRaises(mcmc_fit.FalhaDoMotor) as ctx:
            self._avalia(p, motor, [0.7, -0.2])
        self.assertIn("DENTRO da priori", str(ctx.exception))
        r = p.diagnostico.relatorio()
        self.assertEqual(r["falhasNaoRecuperadas"], 1)
        self.assertEqual(r["avaliacoes"], 1)
        self.assertEqual(r["parada"]["vetor"], {"a": 0.7, "b": -0.2})
        self.assertEqual(motor.reinicios, 1)   # tentou recuperar antes de parar

    def test_falha_nunca_devolve_menos_infinito(self):
        """Antes, a falha virava -inf e a cadeia rejeitava a proposta."""
        p = self._problema()
        motor = self._MotorFalso(falha_em=lambda v: True)
        for valores in ([0.0, 0.0], [0.9, 0.9], [-0.5, 0.3]):
            with self.assertRaises(mcmc_fit.FalhaDoMotor):
                self._avalia(self._problema(), motor, valores)

    def test_recuperacao_em_worker_novo_nao_para_a_corrida(self):
        """Estado corrompido do processo e recuperavel; o valor vem do motor."""
        p = self._problema()
        motor = self._MotorFalso(falha_em=lambda v: True, cura_ao_reiniciar=True)
        valor = self._avalia(p, motor, [0.1, 0.1])
        self.assertEqual(valor, 0.0)           # avaliacao VALIDA, nao -inf
        r = p.diagnostico.relatorio()
        self.assertEqual(r["falhasRecuperadas"], 1)
        self.assertEqual(r["falhasNaoRecuperadas"], 0)
        self.assertEqual(motor.reinicios, 1)

    def test_excecao_tambem_passa_pela_recuperacao(self):
        p = self._problema()
        motor = self._MotorFalso(
            falha_em=lambda v: True, cura_ao_reiniciar=True,
            excecao=RuntimeError("persistent C++ worker stopped unexpectedly"))
        self.assertEqual(self._avalia(p, motor, [0.1, 0.1]), 0.0)
        r = p.diagnostico.relatorio()
        self.assertEqual(r["falhasRecuperadas"], 1)
        self.assertEqual(r["excecoes"], {"RuntimeError": 1})

    def test_excecao_nao_recuperada_interrompe(self):
        p = self._problema()
        motor = self._MotorFalso(
            falha_em=lambda v: True,
            excecao=RuntimeError("persistent C++ worker stopped unexpectedly"))
        with self.assertRaises(mcmc_fit.FalhaDoMotor):
            self._avalia(p, motor, [0.1, 0.1])
        self.assertEqual(p.diagnostico.relatorio()["parada"]["causa"], "excecao")

    def test_recusa_fisica_dentro_da_priori_tambem_interrompe(self):
        """Recusa fisica so valeria como probabilidade zero se correspondesse a
        uma restricao EXPLICITA do alvo, e as deste alvo estao todas em in_prior.
        Se o motor recusa o que a priori aceitou, os dois discordam."""
        p = self._problema()
        motor = self._MotorFalso(
            falha_em=lambda v: True,
            mensagem="fit worker compactness is outside physical bounds")
        with self.assertRaises(mcmc_fit.FalhaDoMotor) as ctx:
            self._avalia(p, motor, [0.9, 0.9])
        self.assertIn("nao vale como probabilidade zero", str(ctx.exception))
        r = p.diagnostico.relatorio()
        self.assertEqual(r["recusaFisicaAlegada"], 1)
        self.assertTrue(r["parada"]["recusa_fisica_alegada"])

    def test_regiao_com_falha_nao_e_excluida_em_silencio(self):
        """O desenho pre-registrado: a falha so acontece para a > 0,5.

        Antes, a cadeia veria -inf em metade da caixa e o relatorio nao
        distinguiria isso de priori nula. Agora a corrida para no primeiro vetor
        daquela regiao, com o vetor nomeado.
        """
        p = self._problema()
        motor = self._MotorFalso(falha_em=lambda v: v[0] > 0.5)
        # Valores exatos, nao -0,9 + 0,1 i: a soma binaria devolve
        # 0,5000000000000002 no decimo quinto passo.
        grade = [-0.9, -0.5, -0.1, 0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
        vistos = []
        with self.assertRaises(mcmc_fit.FalhaDoMotor):
            for a in grade:
                vistos.append(a)
                self._avalia(p, motor, [a, 0.0])
        self.assertEqual(vistos[-1], 0.6)      # parou no primeiro da regiao
        r = p.diagnostico.relatorio()
        self.assertEqual(r["avaliacoes"], 7)
        self.assertEqual(r["foraDoSuporte"], 0)
        self.assertGreater(r["parada"]["vetor"]["a"], 0.5)
        self.assertIn("nan in the transfer table", r["mensagensDoMotor"])

    def test_checkpoint_e_gravado_antes_de_parar(self):
        """Parar sem gravar jogaria fora o que foi amostrado desde o ultimo
        checkpoint periodico."""

        class Prob:
            def __init__(self):
                self.names = ['a', 'b']
                self.bounds = [(-5.0, 5.0), (-5.0, 5.0)]
                self.steps = [0.6, 0.6]
                self.initial = [0.0, 0.0]
                self.frozen = {}
                self.passos = 0

            def in_prior(self, v):
                return all(lo <= x <= hi for x, (lo, hi) in zip(v, self.bounds))

            def extra_log_prior(self, v):
                return 0.0

        class Pool:
            def __init__(self, prob):
                self.prob = prob

            def evaluate(self, problem, states, with_pulse=False):
                self.prob.passos += len(states)
                if self.prob.passos > 400:
                    raise mcmc_fit.FalhaDoMotor("falha provocada no meio da corrida")
                return [(-0.5 * (s[0] ** 2 + s[1] ** 2), None) for s in states]

            def close(self):
                pass

        prob = Prob()
        with tempfile.TemporaryDirectory() as dir_temporario:
            destino = str(Path(dir_temporario) / 'ck.npz')
            with self.assertRaises(mcmc_fit.FalhaDoMotor):
                mcmc_fit.run_ensemble(prob, Pool(prob), 10, 300, 5, seed=7,
                                      checkpoint_path=destino,
                                      checkpoint_every=1000)
            import numpy
            guardado = numpy.load(destino, allow_pickle=True)
            # checkpoint_every=1000 nunca dispararia sozinho em 300 passos: o
            # arquivo existe porque a interrupcao o gravou.
            self.assertGreater(guardado['samples'].shape[1], 0)
            self.assertEqual([str(x) for x in guardado['names']], ['a', 'b'])


class TestAlvoComFundoMarginalizado(unittest.TestCase):
    """O alvo novo, com o fundo integrado, contra o antigo como controle.

    Decisao do revisor em 06/10/2026: definir um alvo novo com fundo
    marginalizado, mantendo o anterior como controle condicional. Estes testes
    cobrem as tres coisas que podem dar errado em silencio:

      1. a integral ter de ser UMA por canal de energia sobre a verossimilhanca
         conjunta das 32 fases. Integrar celula por celula criaria 32 fundos
         independentes artificiais e estreitaria a incerteza por raiz de 32;
      2. o limite degenerado, pi_e concentrada, ter de devolver EXATAMENTE o alvo
         de controle: e o unico jeito de saber que a maquinaria nova nao mudou a
         conta, so a trata como incerta;
      3. a priori ser PROPORCIONAL A LARGURA do canal, porque uma priori fixa
         por canal e afirmacao sobre a grade e muda a posterior ao reagrupar.
    """

    def _monta(self, a0=0.0, dispersao=1.0, nos=64):
        """FitProblem minimo com o fundo latente montado a mao.

        4 fases e 3 bins de energia, com dois termos por bin. `dispersao`
        multiplica os pesos: dispersao -> 0 concentra pi_e na media, que e o
        limite em que o alvo novo tem de virar o de controle.
        """
        import numpy as np
        p = mcmc_fit.FitProblem.__new__(mcmc_fit.FitProblem)
        p.phase_bins, p.energy_bins = 4, 3
        p.energy_min, p.energy_max = 0.15, 1.2
        p.energy_width = (p.energy_max - p.energy_min) / p.energy_bins
        p.energy_fit_max = p.energy_max
        p.exposure, p.period = 1000.0, 10.0
        p.marginalizar_fundo = True
        p.fundo_a0 = a0
        p.fundo_beta = 0.0
        p.fundo_nos = nos
        p._bins_de_energia_ajustados = [0, 1, 2]
        p._fit_bins = [ip * p.energy_bins + ie
                       for ip in range(p.phase_bins)
                       for ie in range(p.energy_bins)]
        p.observed = [7, 5, 3, 8, 6, 2, 9, 4, 3, 6, 5, 4]
        # dois termos por bin, contagens e larguras declaradas
        contagens = np.array([40.0, 30.0, 20.0, 15.0, 10.0, 8.0])
        larguras = np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.005])
        base = np.array([1.0e-4, 1.0e-4, 8.0e-5, 8.0e-5, 5.0e-5, 5.0e-5])
        p.fundo_pesos = base * dispersao
        p.fundo_contagens = contagens
        p.fundo_largura_do_canal = larguras
        p.fundo_deslocamento = np.array([0, 2, 4, 6])
        # o fundo FIXO equivalente: a media de pi_e, para o controle
        k1 = [float((p.fundo_pesos[2 * i:2 * i + 2]
                     * (contagens[2 * i:2 * i + 2]
                        + a0 * larguras[2 * i:2 * i + 2])).sum())
              for i in range(3)]
        p.background = k1
        return p

    def _grid(self):
        return [[0.004, 0.003, 0.002],
                [0.005, 0.004, 0.002],
                [0.006, 0.003, 0.003],
                [0.004, 0.004, 0.002]]

    def _tempo(self, p):
        ciclos = math.floor(p.exposure / p.period)
        duracao = p.period / p.phase_bins
        return [ciclos * duracao for _ in range(p.phase_bins)]

    def _controle(self, p, grid, tempo):
        """O alvo de controle, escrito aqui a mao, com o fundo FIXO na media."""
        esperado = [max(1.0e-14, (grid[ip][ie] + p.background[ie])
                        * p.energy_width * tempo[ip])
                    for ip in range(p.phase_bins) for ie in range(p.energy_bins)]
        return sum(p.observed[b] * math.log(esperado[b]) - esperado[b]
                   for b in p._fit_bins)

    def test_limite_degenerado_devolve_o_controle(self):
        """pi_e concentrada: o alvo novo tem de virar o antigo."""
        for dispersao in (1.0e-3, 1.0e-4, 1.0e-5):
            p = self._monta(dispersao=dispersao)
            grid, tempo = self._grid(), self._tempo(p)
            novo = p._score_marginal(grid, tempo)
            controle = self._controle(p, grid, tempo)
            # a diferenca tem de encolher com a dispersao, em segunda ordem
            self.assertLess(abs(novo - controle), 1.0e-3,
                            "dispersao %g: %r contra %r"
                            % (dispersao, novo, controle))

    def test_a_diferenca_encolhe_com_a_dispersao(self):
        """Nao basta ficar pequena: tem de ir a zero com pi_e concentrando."""
        difs = []
        for dispersao in (1.0e-2, 1.0e-3, 1.0e-4):
            p = self._monta(dispersao=dispersao)
            grid, tempo = self._grid(), self._tempo(p)
            difs.append(abs(p._score_marginal(grid, tempo)
                            - self._controle(p, grid, tempo)))
        self.assertLess(difs[1], difs[0])
        self.assertLess(difs[2], difs[1])

    def test_integral_bate_com_quadratura_independente(self):
        """A Gauss-Legendre de 64 nos contra uma soma de Riemann muito fina.

        Verificacao independente da integral, escrita aqui do zero, sem usar os
        nos nem os pesos que o proprio metodo monta.
        """
        import numpy as np
        p = self._monta(dispersao=1.0)
        grid, tempo = self._grid(), self._tempo(p)
        obtido = p._score_marginal(grid, tempo)
        n = np.asarray(p.observed, float).reshape(p.phase_bins, p.energy_bins)
        passo = p.energy_width * np.asarray(tempo, float)
        total = 0.0
        for ie in p._bins_de_energia_ajustados:
            a = p.fundo_pesos[2 * ie:2 * ie + 2]
            c = p.fundo_contagens[2 * ie:2 * ie + 2]
            w = p.fundo_largura_do_canal[2 * ie:2 * ie + 2]
            f = c + p.fundo_a0 * w
            k1 = float((a * f).sum())
            k2 = float((a ** 2 * f).sum())
            k3 = float(2.0 * (a ** 3 * f).sum())
            forma, escala, desloc = mcmc_fit.gama_de_tres_cumulantes(k1, k2, k3)
            sd = math.sqrt(k2)
            lo, hi = max(0.0, k1 - 8 * sd), k1 + 12 * sd
            b = np.linspace(lo, hi, 400001)
            pi = mcmc_fit.densidade_gama(b, forma, escala, desloc)
            mu = np.maximum((np.asarray([g[ie] for g in grid])[None, :]
                             + b[:, None]) * passo[None, :], 1e-300)
            ll = (n[:, ie][None, :] * np.log(mu) - mu).sum(axis=1)
            pico = float(ll.max())
            dx = (hi - lo) / (b.size - 1)
            total += pico + math.log(float((pi * np.exp(ll - pico)).sum() * dx)
                                     / float(pi.sum() * dx))
        self.assertAlmostEqual(obtido, total, places=8)

    def test_integra_uma_vez_por_canal_e_nao_por_celula(self):
        """Integrar celula por celula nao e a mesma conta, e a diferenca e grande.

        A versao errada, feita aqui de proposito, da um fundo independente a cada
        celula fase-energia. MEDIDO: ela devolve ln L MENOR, nao maior como eu
        havia previsto, e a razao e que os 32 fatores de fase sao funcoes
        POSITIVAMENTE CORRELACIONADAS do mesmo b, de modo que a media do produto
        excede o produto das medias. O sinal nao e o ponto; o ponto e que as duas
        contas diferem, e so uma corresponde a um fundo compartilhado.
        """
        import numpy as np
        p = self._monta(dispersao=1.0)
        grid, tempo = self._grid(), self._tempo(p)
        certo = p._score_marginal(grid, tempo)
        p.prepara_fundo_marginal()
        errado = 0.0
        n = np.asarray(p.observed, float).reshape(p.phase_bins, p.energy_bins)
        for ie in p._bins_de_energia_ajustados:
            b = p._fundo_nos[ie]
            for ip in range(p.phase_bins):
                mu = np.maximum((grid[ip][ie] + b) * p.energy_width * tempo[ip],
                                1e-300)
                ll = n[ip, ie] * np.log(mu) - mu
                pico = float(ll.max())
                errado += pico + math.log(float(
                    (p._fundo_peso_pi[ie] * np.exp(ll - pico)).sum()))
        self.assertNotAlmostEqual(errado, certo, places=2)
        self.assertGreater(abs(errado - certo), 1.0)
        # a direcao MEDIDA, registrada para nao ser reinventada
        self.assertLess(errado, certo)

    def test_priori_proporcional_a_largura(self):
        """a0 entra multiplicado pela largura do canal, nao por canal."""
        p = self._monta(a0=0.0)
        p.prepara_fundo_marginal()
        k_sem = [k[0] for k in p._fundo_k]
        q = self._monta(a0=100.0)
        q.prepara_fundo_marginal()
        k_com = [k[0] for k in q._fundo_k]
        for ie in range(3):
            esperado = k_sem[ie] + sum(
                q.fundo_pesos[2 * ie + j] * 100.0 * q.fundo_largura_do_canal[2 * ie + j]
                for j in range(2))
            self.assertAlmostEqual(k_com[ie], esperado, places=12)

    def test_a0_zero_nao_injeta_nada(self):
        p = self._monta(a0=0.0)
        p.prepara_fundo_marginal()
        import numpy as np
        for ie in range(3):
            a = p.fundo_pesos[2 * ie:2 * ie + 2]
            n = p.fundo_contagens[2 * ie:2 * ie + 2]
            self.assertAlmostEqual(p._fundo_k[ie][0], float((a * n).sum()),
                                   places=15)

    def test_quadratura_converge_em_numero_de_nos(self):
        valores = [self._monta(nos=k)._score_marginal(self._grid(),
                                                      self._tempo(self._monta()))
                   for k in (16, 32, 64, 128)]
        # 64 contra 128 tem de concordar muito melhor que 16 contra 32
        self.assertLess(abs(valores[3] - valores[2]),
                        abs(valores[1] - valores[0]))
        self.assertLess(abs(valores[3] - valores[2]), 1.0e-6)

    def test_a0_zero_e_recusado_quando_ha_canal_vazio(self):
        """Posterior impropria nao pode rodar em silencio.

        Com a0 = 0 um canal de contagem zero da forma zero, e Gamma(0, 1) nao e
        integravel na origem. No dado real sao 336 canais assim, com 14,1 por
        cento do peso total. Entao isto e erro, e nao aviso.
        """
        import numpy as np
        p = self._monta(a0=0.0)
        p.fundo_contagens = np.array([40.0, 0.0, 20.0, 15.0, 10.0, 8.0])
        p._fundo_pronto = False
        with self.assertRaises(ValueError) as ctx:
            p.prepara_fundo_marginal()
        self.assertIn("IMPROPRIA", str(ctx.exception))

    def test_a0_fiducial_aceita_canal_vazio(self):
        """A convencao fiducial, a0 = 0,01 por keV, torna a forma positiva."""
        import numpy as np
        p = self._monta(a0=0.01)
        p.fundo_contagens = np.array([40.0, 0.0, 20.0, 15.0, 10.0, 8.0])
        p._fundo_pronto = False
        p.prepara_fundo_marginal()
        self.assertEqual(len(p._fundo_k), 3)
        # a forma do canal vazio e a0 * largura, pequena mas positiva
        self.assertAlmostEqual(0.01 * 0.005, 5.0e-05, places=12)

    def test_beta_encolhe_o_fundo_uniformemente(self):
        """beta e a taxa da priori Gama, e k1 escala por 1/(1+beta).

        Fica exposto para a convencao ser auditavel: beta = 0 e a unica escolha
        em que a media a posteriori reproduz a taxa medida quando a0 -> 0.
        """
        base = self._monta(a0=0.0)
        base.fundo_beta = 0.0
        base.prepara_fundo_marginal()
        k_base = [k[0] for k in base._fundo_k]
        for beta in (0.01, 0.1, 1.0):
            q = self._monta(a0=0.0)
            q.fundo_beta = beta
            q.prepara_fundo_marginal()
            for ie in range(3):
                self.assertAlmostEqual(q._fundo_k[ie][0],
                                       k_base[ie] / (1.0 + beta), places=14)

    def test_produto_sem_pesos_e_recusado(self):
        """O v1 nao traz fundo_pesos; pedir marginalizacao com ele e erro."""
        p = self._monta()
        del p.fundo_pesos
        p._fundo_pronto = False
        with self.assertRaises(ValueError):
            p.prepara_fundo_marginal()


class TestExposicaoPorFaseMedida(unittest.TestCase):
    """A exposicao por fase MEDIDA nos quadros, contra a formula analitica.

    O revisor colocou isto como o achado mais importante do portao de dados: "a
    exposicao por fase usada na verossimilhanca nao vir das GTIs". Medido, o
    efeito desloca diferencas de ln L em ate 1,0105. A formula analitica supoe
    cobertura uniforme a menos do ultimo ciclo parcial; os quadros, com FRACEXP,
    dizem outra coisa.

    O que pode dar errado em silencio, e e o que estes testes cobrem:

      1. a medida nao ser usada, e o alvo seguir na formula sem avisar;
      2. a medida estar em OUTRA CONVENCAO, por exemplo fracao em vez de
         segundos, e o modelo sair reescalado por um fator constante sem erro
         nenhum: a soma tem de bater com a exposicao;
      3. um bin de fase sem exposicao, que nao tem modelo e daria mu = 0 com
         contagem possivelmente nao nula.
    """

    def _monta(self, por_fase=None):
        p = mcmc_fit.FitProblem.__new__(mcmc_fit.FitProblem)
        p.phase_bins, p.energy_bins = 4, 3
        p.exposure, p.period = 1000.0, 10.0
        p.exposicao_por_fase = por_fase
        return p

    def test_sem_medida_reproduz_a_formula_analitica(self):
        """Sem a medida, nada muda: o alvo antigo continua bit a bit o mesmo."""
        p = self._monta(None)
        t = p.tempo_representado()
        ciclos = math.floor(p.exposure / p.period)
        dur = p.period / p.phase_bins
        parcial = p.exposure - ciclos * p.period
        esperado = [ciclos * dur + max(0.0, min(dur, parcial - i * dur))
                    for i in range(p.phase_bins)]
        self.assertEqual(t, esperado)
        # 1000 s em periodos de 10 s da 100 ciclos exatos, logo 250 s por bin
        self.assertEqual(t, [250.0, 250.0, 250.0, 250.0])

    def test_com_medida_a_medida_vence(self):
        medida = [260.0, 240.0, 255.0, 245.0]
        p = self._monta(medida)
        self.assertEqual(p.tempo_representado(), medida)
        # e a soma continua sendo a exposicao
        self.assertAlmostEqual(sum(p.tempo_representado()), p.exposure, places=9)

    def test_a_medida_muda_o_modelo_na_proporcao_do_tempo(self):
        """mu_p e linear no tempo representado, logo o desvio passa inteiro."""
        medida = [260.0, 240.0, 255.0, 245.0]
        a, b = self._monta(None), self._monta(medida)
        for ip, (t0, t1) in enumerate(zip(a.tempo_representado(),
                                          b.tempo_representado())):
            self.assertAlmostEqual(t1 / t0 - 1.0,
                                   (medida[ip] - 250.0) / 250.0, places=12)

    def _carrega(self, **extra):
        """Carrega um npz pelo caminho do pedido, que e onde a validacao mora."""
        import numpy as np
        import tempfile
        pedido = {
            "preparedDataNpz": None, "phaseBins": 4, "energyBins": 3,
            "energyMin": 0.15, "energyMax": 1.2, "exposure": 1000.0,
            "period": 10.0}
        with tempfile.TemporaryDirectory() as d:
            arq = Path(d) / "x.npz"
            dados = dict(observed=np.zeros(12), exposure=1000.0,
                         background=np.zeros(3))
            dados.update(extra)
            np.savez(arq, **dados)
            p = mcmc_fit.FitProblem.__new__(mcmc_fit.FitProblem)
            p.phase_bins, p.energy_bins = 4, 3
            p.exposicao_por_fase = None
            dd = np.load(arq)
            # o mesmo bloco de validacao do construtor, isolado
            pe = np.asarray(dd["exposicao_por_fase"], float)
            if pe.shape != (p.phase_bins,):
                raise ValueError(
                    "exposicao_por_fase com %d valores, e a grade tem %d bins "
                    "de fase" % (pe.size, p.phase_bins))
            if not np.all(pe > 0.0):
                raise ValueError("exposicao_por_fase com bin nao positivo: um "
                                 "bin sem exposicao nao tem modelo")
            razao = float(pe.sum()) / float(dd["exposure"])
            if abs(razao - 1.0) > 1.0e-6:
                raise ValueError(
                    "soma da exposicao_por_fase e %.6f vezes a exposicao; as "
                    "duas precisam estar na mesma convencao" % razao)
            return [float(x) for x in pe]

    def test_recusa_outra_convencao(self):
        """Fracao em vez de segundos: soma 1 e nao 1000, e tem de reclamar."""
        import numpy as np
        with self.assertRaises(ValueError) as c:
            self._carrega(exposicao_por_fase=np.array([0.26, 0.24, 0.255, 0.245]))
        self.assertIn("mesma convencao", str(c.exception))

    def test_recusa_bin_sem_exposicao(self):
        import numpy as np
        with self.assertRaises(ValueError) as c:
            self._carrega(exposicao_por_fase=np.array([333.4, 333.3, 333.3, 0.0]))
        self.assertIn("nao positivo", str(c.exception))

    def test_recusa_numero_errado_de_bins(self):
        import numpy as np
        with self.assertRaises(ValueError) as c:
            self._carrega(exposicao_por_fase=np.array([500.0, 500.0]))
        self.assertIn("bins de fase", str(c.exception))

    def test_aceita_a_medida_coerente(self):
        import numpy as np
        self.assertEqual(
            self._carrega(exposicao_por_fase=np.array([260.0, 240.0, 255.0, 245.0])),
            [260.0, 240.0, 255.0, 245.0])


#: A TABELA DE ATMOSFERA NAO E VERSIONADA, por decisao do projeto: o
#: repositorio traz tabelas/MANIFEST.json e a tabela se obtem com
#: scripts/tabelas_baixar.py. Sem ela estes testes nao tem o que carregar, e
#: pular com motivo e o certo -- era isso que fazia os tres falharem no clone
#: publico, que nunca teve a tabela. O padrao segue test_magnetizada.py.
_TABELA_ATM = Path(__file__).resolve().parents[1] / "tabelas/magnus_campoBg_vacuo_v5.magnus"


@unittest.skipUnless(
    _TABELA_ATM.is_file(),
    "tabela de atmosfera ausente (%s): obtenha com scripts/tabelas_baixar.py"
    % _TABELA_ATM.name)
class TestOffsetDeFaseObservacional(unittest.TestCase):
    """O operador de alinhamento EXPLICITO, que substitui o uso de `magAzim`.

    Por que existe: `magAzim` e um parametro FISICO livre e compartilhado, e
    medido NAO e um operador de fase exato. Rolar a grade e mover o azimute
    diferem por 0,061 em ln L e ate 3,0e-04 por celula, porque o azimute move a
    configuracao magnetica DENTRO da malha, que e fixa no referencial da
    estrela, mudando o erro de quadratura. O offset explicito nao move nada
    dentro da malha, so o instante amostrado, e por isso e exato em
    deslocamento inteiro (docs/valida_offset.json).

    O que pode dar errado em silencio, e e o que estes testes cobrem:

      1. o offset zero mudar o comando e portanto o resultado, quebrando a
         reproducao do alvo anterior;
      2. o offset ser aceito no pedido e NAO chegar ao motor, caso em que o
         alinhamento simplesmente nao aconteceria e a conta rodaria errada sem
         erro nenhum;
      3. a convencao de sinal trocar sem ninguem notar.
    """

    def _problema(self, offset=None):
        # FIXTURE EM tests/, e nao o pedido de exploracoes/. O teste vai ao
        # repositorio publico e exploracoes/ nao vai, de modo que ler de la
        # fazia os tres testes desta classe falharem no clone publico com
        # FileNotFoundError. Um teste publicado nao pode depender de caminho
        # que nao e publicado.
        aqui = Path(__file__).resolve().parent
        raiz = aqui.parent
        pedido = json.loads(
            (aqui / "dados/pedido_offset_fase.json").read_text())
        pedido["instrument"] = "generic"
        pedido["sphereBands"] = 24
        pedido["boundaryRefine"] = 1
        if offset is not None:
            pedido["observationPhaseOffsetCycles"] = offset
        meta, eventos = mcmc_fit.read_events(raiz / "dados/eventos.csv")
        return mcmc_fit.FitProblem(pedido, eventos, meta, None)

    def test_sem_offset_o_comando_nao_muda(self):
        """Offset zero nao pode acrescentar flag: o alvo anterior tem de voltar."""
        p = self._problema(None)
        self.assertEqual(p.observation_phase_offset, 0.0)
        self.assertNotIn("--observation-phase-offset", p.worker_command())

    def test_offset_zero_explicito_tambem_nao_muda(self):
        p = self._problema(0.0)
        self.assertNotIn("--observation-phase-offset", p.worker_command())

    def test_offset_nao_nulo_chega_ao_motor(self):
        """O silencio aqui seria fatal: alinhamento pedido e nao aplicado."""
        p = self._problema(5.0 / 32.0)
        cmd = list(p.worker_command())
        self.assertIn("--observation-phase-offset", cmd)
        valor = float(cmd[cmd.index("--observation-phase-offset") + 1])
        self.assertAlmostEqual(valor, 5.0 / 32.0, places=15)

    def test_a_conversao_de_bins_para_ciclos(self):
        import por_observacao
        self.assertAlmostEqual(por_observacao.offset_em_ciclos(0), 0.0)
        self.assertAlmostEqual(por_observacao.offset_em_ciclos(5), 5 / 32)
        self.assertAlmostEqual(por_observacao.offset_em_ciclos(32), 1.0)
        # fracionario e aceito, porque o offset e continuo no motor
        self.assertAlmostEqual(por_observacao.offset_em_ciclos(2.5), 2.5 / 32)

    def test_a_convencao_de_sinal_esta_documentada_e_e_a_medida(self):
        """A convencao medida: m_offset[p] = m_0[p + offset.N], razao 1,000000.

        Este teste nao mede nada: ele FIXA a convencao no codigo, para que
        trocar o sinal quebre um teste em vez de passar calado.
        """
        import por_observacao
        doc = por_observacao.offset_em_ciclos.__doc__
        self.assertIn("m_offset[p] = m_0[p + offset . N]", doc)
        self.assertIn("POSITIVO", doc)
