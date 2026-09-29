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
import math
import random
import sys
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
