"""O portão do leitor de tabela — estágio 0 do plano de desenvolvimento.

Não há teste de unidade que pegue física errada; há, e são estes, os que pegam
LEITOR errado. Um índice trocado na interpolação de cinco eixos não levanta
exceção nenhuma: só devolve o feixe de outra temperatura, e isso apareceria
seis meses depois como "a atmosfera não fecha o portão do estágio 2".

Três portões, e cada um mede uma coisa que os outros não medem:

* uma tabela de zeros tem de reproduzir o motor SEM atmosfera, dígito a dígito
  — porque lg w = 0 é o corpo negro isotrópico exato, e é isso que faz o modelo
  aninhar;
* uma tabela com o feixe (1 + a mu) tem de reproduzir a `--beaming a` do próprio
  motor — a mesma forma angular por dois caminhos independentes, o que exercita
  o eixo de mu e a interpolação;
* o theta_B de cada ponto tem de sair do dipolo e ser DIFERENTE entre pontos em
  colatitudes diferentes, que é a razão de o eixo existir.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "build" / "magnus_engine"
sys.path.insert(0, str(ROOT / "scripts"))

import tabela_intensidade as ti                                    # noqa: E402

#: Dois pontos em colatitudes bem separadas: é a diferença entre eles que o eixo
#: theta_B existe para carregar.
SPOTS = ["--spot", "30,0,10,2.5", "--spot", "120,200,12,2.5"]
GRID = ["--spectral-grid", "--energy-bins", "24", "--time-bins", "16", "--rings", "5"]


def engine(*arguments: str) -> dict:
    finished = subprocess.run([str(ENGINE), *arguments], capture_output=True, text=True)
    if finished.returncode != 0:
        raise AssertionError(f"motor falhou: {finished.stderr.strip()}")
    return json.loads(finished.stdout)


def relative_difference(first: dict, second: dict) -> float:
    a = np.asarray(first["photon_flux"], dtype=float)
    b = np.asarray(second["photon_flux"], dtype=float)
    scale = np.maximum(np.abs(a), np.abs(b))
    interesting = scale > scale.max() * 1.0e-12
    return float(np.max(np.abs(a - b)[interesting] / scale[interesting]))


class TestFormatoDeIntensidade(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not ENGINE.exists():
            subprocess.run(["make", "engine"], cwd=ROOT, check=True, capture_output=True)
        cls.blackbody = ti.blackbody_table(ROOT / "build" / "gabarito_corpo_negro.magnus")
        cls.beam = ti.linear_beam_table(ROOT / "build" / "gabarito_feixe_linear.magnus", 1.5)

    def test_ida_e_volta_do_formato(self) -> None:
        table = ti.read(self.blackbody)
        self.assertEqual(table["log_w"].shape,
                         (len(ti.GATE_LOG_T), len(ti.GATE_LOG_G), len(ti.GATE_THETA_B),
                          len(ti.GATE_MU), len(ti.GATE_LOG_E)))
        self.assertTrue(np.all(table["log_w"] == 0.0))
        self.assertAlmostEqual(float(table["mu"][-1]), 1.0, places=6)

    def test_zeros_reproduzem_o_corpo_negro(self) -> None:
        """lg w = 0 é o corpo negro isotrópico EXATO, não uma aproximação dele."""
        sem = engine(*GRID, *SPOTS)
        com = engine(*GRID, *SPOTS, "--atmosphere-table", str(self.blackbody))
        self.assertEqual(sem["model"], "Schwarzschild+Doppler+blackbody+isotropic")
        self.assertTrue(com["model"].endswith("+magnus_intensity_table"))
        difference = relative_difference(sem, com)
        self.assertLess(difference, 1.0e-9, f"diferença relativa máxima {difference:.2e}")

    def test_o_eixo_de_mu_reproduz_o_feixe_linear(self) -> None:
        """A mesma forma angular, pela fórmula em C++ e pela tabela interpolada.

        O resíduo é de interpolação e nada mais: 65 nós em mu, interpolação
        linear do logaritmo de uma função suave. Fica em 1e-4, e o portão do
        plano pedia 1e-3.
        """
        formula = engine(*GRID, *SPOTS, "--beaming", "1.5")
        tabela = engine(*GRID, *SPOTS, "--atmosphere-table", str(self.beam))
        difference = relative_difference(formula, tabela)
        self.assertLess(difference, 1.0e-3, f"diferença relativa máxima {difference:.2e}")

    def test_theta_b_vem_do_dipolo(self) -> None:
        """Colatitude magnética 30 e 120 graus dão 16,10 e 40,89 — não o mesmo.

        Os valores saem de cos(theta_B) = 2 cos(t) / sqrt(1 + 3 cos^2 t), que é o
        dipolo e não escolha nossa. Se este teste passar a devolver dois números
        iguais, o eixo theta_B parou de fazer trabalho nenhum.
        """
        result = engine(*GRID, *SPOTS, "--atmosphere-table", str(self.blackbody))
        angles = result["atmosphere"]["theta_b_deg_by_spot"]
        self.assertEqual(len(angles), 2)
        self.assertAlmostEqual(angles[0], 16.102114, places=4)
        self.assertAlmostEqual(angles[1], 40.893395, places=4)

    def test_eixo_inclinado_muda_os_angulos(self) -> None:
        """Inclinar o eixo magnético tem de mexer nos dois ângulos."""
        aligned = engine(*GRID, *SPOTS, "--atmosphere-table", str(self.blackbody))
        tilted = engine(*GRID, *SPOTS, "--atmosphere-table", str(self.blackbody),
                        "--magnetic-colatitude", "30")
        self.assertNotAlmostEqual(aligned["atmosphere"]["theta_b_deg_by_spot"][0],
                                  tilted["atmosphere"]["theta_b_deg_by_spot"][0], places=3)
        # O primeiro ponto está em colatitude 30 e azimute 0: com o eixo levado
        # para lá, ele fica NO polo magnético, e theta_B tem de ser zero.
        self.assertAlmostEqual(tilted["atmosphere"]["theta_b_deg_by_spot"][0], 0.0, places=6)

    def test_tabela_truncada_e_recusada(self) -> None:
        """Um arquivo curto tem de dar erro, não zeros silenciosos."""
        truncated = ROOT / "build" / "tabela_truncada.magnus"
        truncated.write_bytes(self.blackbody.read_bytes()[:-400])
        finished = subprocess.run(
            [str(ENGINE), *GRID, "--atmosphere-table", str(truncated)],
            capture_output=True, text=True)
        self.assertNotEqual(finished.returncode, 0)
        self.assertIn("ends early", finished.stderr)

    def test_arquivo_alheio_e_recusado(self) -> None:
        alien = ROOT / "build" / "nao_e_tabela.magnus"
        alien.write_bytes(b"NSMAXG01" + bytes(64))
        finished = subprocess.run(
            [str(ENGINE), *GRID, "--atmosphere-table", str(alien)],
            capture_output=True, text=True)
        self.assertNotEqual(finished.returncode, 0)
        self.assertIn("not a MAGNUS intensity table", finished.stderr)


if __name__ == "__main__":
    unittest.main()
