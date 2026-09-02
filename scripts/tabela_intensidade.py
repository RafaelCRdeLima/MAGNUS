"""O formato de intercâmbio do MAGNUS: I(E, mu, theta_B), e quem o escreve.

**Por que o formato vem antes da física.** No plano original a tabela só era
escrita no estágio 4, depois de tudo pronto. Aqui ela vem no estágio 0, e a
razão é que assim cada estágio seguinte entrega uma tabela na mesma escala, que
vai direto ao ajuste — o portão final vira leitura contínua, e a verossimilhança
de cada estágio se compara com a do anterior sem tradução no meio.

**O que fica gravado é lg w, com w = I / B_E(T_ef)** — a razão para a
intensidade de corpo negro ISOTRÓPICA de mesma temperatura efetiva. A escolha
não é de conveniência:

1. w é de ordem 1 e suave, ao passo que I varia dez ordens de grandeza ao longo
   da grade de temperatura;
2. a escala T^4 sai analiticamente, e o motor já sabe fazer corpo negro;
3. **w = 1 devolve o corpo negro isotrópico exato.** Um corpo negro isotrópico
   a T_ef emite exatamente sigma T_ef^4, então o modelo aninha — e uma tabela de
   zeros é o portão do leitor: ela tem de reproduzir o motor sem atmosfera
   nenhuma, dígito a dígito.

E o motor interpola linearmente em lg w, ou seja LOGARITMICAMENTE na
intensidade. Não é detalhe: interpolar intensidade linearmente em mu perto de
mu -> 0 borra justamente a região rasante, que é onde lápis e leque se
distinguem. O formato já resolve isso ao guardar o logaritmo.

**O eixo que o X-PSI não tem.** O formato de cinco colunas da comunidade é
(logT, logg, mu, logE, intensidade), feito para atmosferas não magnéticas. Aqui
entra theta_B, o ângulo entre B e a normal da superfície. Num dipolo ele vale
tan(theta_B) = tan(theta_s)/2 e portanto MUDA de ponto para ponto: colatitude
magnética de 30 e 120 graus dão 16,1 e 40,9. Sem o eixo, dois pontos quentes em
colatitudes diferentes usariam o mesmo feixe, o que é falso por dezenas de
graus. A exportação para a comunidade sai depois em cinco colunas, na fatia
theta_B = 0, para continuar intercambiável.

Formato binário, little-endian, no mesmo estilo dos perfis de instrumento e da
tabela NSMAXG que o motor já lê:

    char[8]  "MAGNUSI1"
    u32 n_T   f32 lg_T[n_T]          (K)
    u32 n_g   f32 lg_g[n_g]          (cm/s^2)
    u32 n_b   f32 theta_B[n_b]       (graus, 0 a 90)
    u32 n_mu  f32 mu[n_mu]           (0 a 1)
    u32 n_E   f32 lg_E[n_E]          (keV)
    f32 lg_w[n_T * n_g * n_b * n_mu * n_E]

com o índice ((((iT*n_g + ig)*n_b + ib)*n_mu + imu)*n_E + iE) — T por fora, E
por dentro.

Executado direto, escreve as duas tabelas do portão do estágio 0 em build/.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC = b"MAGNUSI1"
ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, log_t, log_g, theta_b_deg, mu, log_e, log_w) -> Path:
    """Grava uma tabela. `log_w` tem forma (n_T, n_g, n_b, n_mu, n_E)."""
    axes = [np.asarray(a, dtype=np.float32) for a in (log_t, log_g, theta_b_deg, mu, log_e)]
    values = np.asarray(log_w, dtype=np.float32)
    expected = tuple(len(a) for a in axes)
    if values.shape != expected:
        raise ValueError(f"lg w tem forma {values.shape}, esperada {expected}")
    for name, axis in zip(("lg T", "lg g", "theta_B", "mu", "lg E"), axes):
        if len(axis) > 1 and not np.all(np.diff(axis) > 0):
            raise ValueError(f"o eixo {name} não é estritamente crescente")
    if not np.all(np.isfinite(values)):
        raise ValueError("lg w tem valor não finito; o motor não tem como interpolar isso")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(MAGIC)
        for axis in axes:
            handle.write(struct.pack("<I", len(axis)))
            handle.write(axis.astype("<f4").tobytes())
        handle.write(values.astype("<f4").ravel(order="C").tobytes())
    return path


def read(path: Path) -> dict:
    """Lê de volta o que `write` gravou — usado pelos testes e pela conferência."""
    with path.open("rb") as handle:
        if handle.read(8) != MAGIC:
            raise ValueError(f"não é uma tabela de intensidade do MAGNUS: {path}")
        axes = []
        for _ in range(5):
            count = struct.unpack("<I", handle.read(4))[0]
            axes.append(np.frombuffer(handle.read(4 * count), dtype="<f4").copy())
        total = int(np.prod([len(a) for a in axes]))
        values = np.frombuffer(handle.read(4 * total), dtype="<f4").copy()
        if values.size != total:
            raise ValueError(f"tabela truncada: {values.size} valores de {total}")
    names = ("log_t", "log_g", "theta_b_deg", "mu", "log_e")
    table = dict(zip(names, axes))
    table["log_w"] = values.reshape([len(a) for a in axes])
    return table


# --------------------------------------------------------------------------- #
# As duas tabelas do portão


#: Grade mínima que cobre a configuração base do motor com folga. Os eixos de T,
#: g e theta_B ficam curtos de propósito: estas tabelas testam o LEITOR, não a
#: física, e um eixo de dois pontos já exercita a interpolação em cada um.
GATE_LOG_T = [5.5, 6.0, 6.5, 6.9]
GATE_LOG_G = [13.6, 14.4, 15.0]
GATE_THETA_B = [0.0, 45.0, 90.0]
GATE_LOG_E = [-1.5, 0.0, 1.5]
#: 65 nós em mu. O feixe linear é suave, mas a interpolação é do LOGARITMO dele,
#: e o erro de interpolação vai com o quadrado do passo — 65 nós põem o resíduo
#: em 1e-4, abaixo do que o portão exige e acima do que valeria a pena refinar.
GATE_MU = np.linspace(0.0, 1.0, 65)


def blackbody_table(path: Path) -> Path:
    """lg w = 0 em toda parte: o corpo negro isotrópico, exato."""
    shape = (len(GATE_LOG_T), len(GATE_LOG_G), len(GATE_THETA_B), len(GATE_MU), len(GATE_LOG_E))
    return write(path, GATE_LOG_T, GATE_LOG_G, GATE_THETA_B, GATE_MU, GATE_LOG_E,
                 np.zeros(shape))


def linear_beam_table(path: Path, a: float) -> Path:
    """O feixe (1 + a mu) / (1 + 2a/3), que o motor já sabe fazer sozinho.

    Serve de portão do eixo de mu: a mesma forma angular, por dois caminhos
    independentes — a fórmula em C++ e a interpolação da tabela — tem de dar o
    mesmo número. A normalização é a mesma do motor, `int shape(mu) mu dmu` =
    1/2, então o feixe muda a forma e não mexe no fluxo.
    """
    shape_mu = (1.0 + a * GATE_MU) / (1.0 + 2.0 * a / 3.0)
    log_w = np.log10(shape_mu)[None, None, None, :, None] * np.ones(
        (len(GATE_LOG_T), len(GATE_LOG_G), len(GATE_THETA_B), 1, len(GATE_LOG_E)))
    return write(path, GATE_LOG_T, GATE_LOG_G, GATE_THETA_B, GATE_MU, GATE_LOG_E, log_w)


if __name__ == "__main__":
    build = ROOT / "build"
    print(blackbody_table(build / "gabarito_corpo_negro.magnus"))
    print(linear_beam_table(build / "gabarito_feixe_linear.magnus", 1.5))
