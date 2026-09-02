"""O aferidor: transforma os gabaritos em teste automático.

**Por que isto existe.** O estágio 0 do plano diz que não há teste de unidade
que pegue física errada — só comparação com quem já fez certo. Então a primeira
coisa a construir não é física, é a régua: um programa que lê os 26 arquivos
`.in` e diz, por arquivo e por temperatura, se o gabarito é o que dizemos que
ele é. Sem isso, todo alvo de "5%" ou "20%" dos estágios seguintes é medido
contra um número que ninguém conferiu.

**Auditoria 1 — sigma T^4 nos espectros.** Classificação, primeiro: a página do
modelo NSMAXG diz que há DOIS conjuntos, "um com um único B e T_ef de superfície
[...] e um construído com B e T_ef variando sobre a superfície segundo o modelo
de dipolo". Os quatro arquivos marcados `Thm` são o segundo conjunto — e são
exatamente os quatro que reprovam o teste abaixo, porque um modelo de superfície
inteira com distribuição de temperatura não tem por que emitir sigma T^4 do
rótulo. Eles NÃO são gabarito de atmosfera local, e ficam declarados aqui para
que ninguém os meça de novo contra a régua errada.

 Uma atmosfera em equilíbrio
radiativo emite exatamente sigma T_ef^4. Se a tabela conserva isso, a razão
C = integral(F dE) / T_ef^4 é a mesma constante em todas as temperaturas e em
todos os arquivos — ela é só a escala de unidade do fluxo tabelado. Onde C
varia, ou a banda tabelada perde fluxo pelas bordas, ou o rótulo "T" daquele
arquivo não é T_ef.

A correção de banda usa a fração de um corpo negro que cai dentro da banda.
É uma aproximação declarada e conservadora POR BAIXO: a atmosfera magnetizada
é mais dura que o corpo negro, então ela perde MAIS fluxo acima do corte do que
o corpo negro perde, e a correção sub-corrige. Por isso o veredito estrito só
olha as temperaturas onde a fração passa de 99,5% — ali a correção é pequena e
o que sobra é sinal, não borda.

**Auditoria 2 — a contabilidade das frações do Potekhin.** O estágio 2 depende
de saber onde a ionização é completa, e isso sai das colunas de fração. A página
do Ioffe define x(H) como "átomos com níveis não destruídos", x(H0) como o
subconjunto no estado fundamental, x(H2) como o NÚMERO DE MOLÉCULAS dividido por
N e x(pert.) como prótons em aglomerados e átomos fortemente perturbados. A
pergunta que decide o uso é se a molécula conta uma vez ou duas, e ela se
resolve por medição: no canto frio e denso a matéria tem de estar toda
recombinada. Com peso 2 a fração ionizada ali vai a zero; com peso 1 sobra uma
ionização teimosa de 10 a 14%, que é exatamente o segundo próton de cada
molécula. Logo

    x_ionizado = 1 - x(H) - 2 x(H2) - x(pert.)

**Auditoria 3 — o piso das opacidades do Potekhin.** As colunas 13 e 14 das
tabelas do Potekhin são as médias de Rosseland ao longo e perpendicular a B.
Em densidade baixa o livre-livre desaparece e sobra só espalhamento por
elétron, que em campo forte é suprimido por ~(E/E_Be)^2 no modo extraordinário.
Com os dois modos carregando o fluxo em paralelo,

    1/K = <(1/2)(1/kappa_O + 1/kappa_X)>_Rosseland

o piso previsto é K ~ 2 kappa_T / (1 + 0,1266 (E_Be/kT)^2). Bater esse piso
contra o que está tabelado testa duas coisas de uma vez: que as colunas são
mesmo opacidades de Rosseland em cm^2/g, e que a DEFINIÇÃO com peso 1/2 e média
harmônica é a que o Potekhin usou. É a primeira coisa que o módulo de opacidade
do estágio 2 vai ter de reproduzir, e ela sai de graça, hoje.

Uso:  python3 scripts/auditoria_gabaritos.py
Saída não-zero se algum arquivo que deveria passar não passa.
"""

from __future__ import annotations

import gzip
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NSMAXG = ROOT / "atmosphere_data" / "nsmaxg_ho"
POTEKHIN = ROOT / "atmosphere_data" / "potekhin_magnetic_h"

KEV_PER_KELVIN = 8.617333262e-8
#: Espalhamento de Thomson por grama de hidrogênio, sigma_T / m_p.
THOMSON_CM2_G = 0.39774
#: Energia de cíclotron do elétron: hbar e B / (m_e c), em keV por 1e12 G.
E_BE_KEV_PER_B12 = 11.577
#: <E^-2> com o peso de Rosseland vale (pi^2/3)/(4 pi^4/15) / (kT)^2.
ROSSELAND_INV_SQUARE = (math.pi ** 2 / 3.0) / (4.0 * math.pi ** 4 / 15.0)


# --------------------------------------------------------------------------- #
# Leitura


def read_in(path: Path) -> dict:
    """Um arquivo .in: seis linhas de cabeçalho e um espectro por (T, g).

    A ordem dos espectros é TEMPERATURA por fora, GRAVIDADE por dentro —
    espectro[it * ng + ig]. Verificado no nsx_HB0000ThB00, que tem 15
    gravidades: com esta ordem o pico do espectro anda com a temperatura
    (0,120 / 0,380 / 1,148 keV em lg T = 5,5 / 6,0 / 6,5) e com a ordem trocada
    fica parado. Um erro aqui não levanta exceção nenhuma, só desloca o
    espectro — por isso está escrito.
    """
    with gzip.open(path, "rt", encoding="latin-1") as handle:
        rows = [line.split() for line in handle]
    log_t = [float(v) for v in rows[1]]
    log_g = [float(v) for v in rows[3]]
    energy = [float(v) for v in rows[5]]
    spectra = [[float(v) for v in rows[6 + i]] for i in range(len(log_t) * len(log_g))]
    return {"log_t": log_t, "log_g": log_g, "energy": energy, "spectra": spectra}


def trapezoid(x: list[float], y: list[float]) -> float:
    return sum(0.5 * (y[k] + y[k + 1]) * (x[k + 1] - x[k]) for k in range(len(x) - 1))


def planck_shape(energy_kev: float, log_t: float) -> float:
    """Fluxo de energia de Planck a menos de constante: E^3 / (exp(E/kT) - 1)."""
    x = energy_kev / (KEV_PER_KELVIN * 10.0 ** log_t)
    return 0.0 if x > 700.0 else energy_kev ** 3 / math.expm1(x)


def band_fraction(energy: list[float], log_t: float) -> float:
    """Fração de um corpo negro de mesma T que cai dentro da banda tabelada."""
    wide = [math.exp(math.log(1.0e-5) + k * (math.log(1.0e3) - math.log(1.0e-5)) / 4000)
            for k in range(4001)]
    inside = trapezoid(energy, [planck_shape(v, log_t) for v in energy])
    whole = trapezoid(wide, [planck_shape(v, log_t) for v in wide])
    return inside / whole if whole > 0.0 else 0.0


# --------------------------------------------------------------------------- #
# Auditoria 1


#: Constante comum a que 24 dos 26 arquivos obedecem, em unidades do fluxo
#: tabelado. Equivale a sigma T^4 / integral = 2,42e17.
COMMON_CONSTANT = 2.344e-22
#: Os modelos de superfície inteira, com B e T_ef variando segundo o dipolo.
#: Não conservam sigma T^4 do rótulo e não devem: o rótulo não é a T_ef de um
#: pedaço plano-paralelo. Servem no fim, como portão de ponta a ponta do
#: conjunto atmosfera + distribuição de temperatura + integração angular, e é
#: por isso que estão listados em vez de apagados.
DIPOLE_SURFACE_MODELS = {
    "nsmaxg_HB1226Thm00g1420.in.gz", "nsmaxg_HB1226Thm90g1420.in.gz",
    "nsmaxg_HB1300Thm00g1420.in.gz", "nsmaxg_HB1300Thm90g1420.in.gz",
}
#: Fração mínima da banda para uma temperatura entrar no veredito estrito.
SAFE_FRACTION = 0.995
#: Deriva tolerada dentro da janela segura. 2,5% não é escolha de conforto: é o
#: patamar em que os arquivos de Z médio ficam (C, O, Ne dão 1,5 a 2,0%, com as
#: bordas fotoelétricas deles empurrando fluxo para fora da banda), e abaixo do
#: qual todos os de hidrogênio com marca ThB ficam com folga — o pior deles dá
#: 0,71%. Quem não passa, então, não passa por causa da banda.
SAFE_SPREAD_PERCENT = 2.5


def audit_spectra() -> list[str]:
    print("=" * 96)
    print("AUDITORIA 1 — sigma T^4 nos 26 gabaritos de espectro")
    print("=" * 96)
    print(f"{'arquivo':32s} {'nT':>3} {'ng':>3} {'nE':>4} "
          f"{'C mediano':>11} {'cru%':>7} {'corr%':>7} {'seguro%':>8} {'nT seg':>7}  veredito")
    failures: list[str] = []
    for path in sorted(NSMAXG.glob("*.in.gz")):
        table = read_in(path)
        energy, log_g = table["energy"], table["log_g"]
        raw, corrected, safe = [], [], []
        for index, log_t in enumerate(table["log_t"]):
            spectrum = table["spectra"][index * len(log_g)]      # a primeira gravidade
            constant = trapezoid(energy, spectrum) / (10.0 ** log_t) ** 4
            fraction = band_fraction(energy, log_t)
            raw.append(constant)
            corrected.append(constant / fraction)
            if fraction >= SAFE_FRACTION:
                safe.append(constant / fraction)

        def spread(values: list[float]) -> float:
            if len(values) < 2:
                return float("nan")
            middle = sorted(values)[len(values) // 2]
            return (max(values) - min(values)) / middle * 100.0

        median = sorted(raw)[len(raw) // 2]
        offset = abs(median - COMMON_CONSTANT) / COMMON_CONSTANT
        # Dois vereditos diferentes. A escala fora do comum é o defeito grave:
        # significa que o rótulo "T" daquele arquivo não é a T_ef de um
        # espectro que conserva sigma T^4, e nenhum corte de banda produz isso,
        # porque corte só tira fluxo das bordas.
        verdict = "ok"
        if path.name in DIPOLE_SURFACE_MODELS:
            print(f"{path.name[:32]:32s} {len(table['log_t']):3d} {len(log_g):3d} {len(energy):4d} "
                  f"{median:11.4e} {spread(raw):7.2f} {spread(corrected):7.2f} "
                  f"{spread(safe):8.2f} {len(safe):7d}  superficie dipolar (nao e' gabarito local)")
            continue
        if offset > 0.05:
            verdict = "<<< ESCALA FORA (nao usar como gabarito)"
            failures.append(path.name)
        elif not math.isnan(spread(safe)) and spread(safe) > SAFE_SPREAD_PERCENT:
            verdict = "<<< deriva na janela segura"
            failures.append(path.name)
        elif len(safe) < 3:
            verdict = "janela segura curta demais"
        print(f"{path.name[:32]:32s} {len(table['log_t']):3d} {len(log_g):3d} {len(energy):4d} "
              f"{median:11.4e} {spread(raw):7.2f} {spread(corrected):7.2f} "
              f"{spread(safe):8.2f} {len(safe):7d}  {verdict}")
    return failures


# --------------------------------------------------------------------------- #
# Auditoria 2


def read_potekhin(path: Path) -> dict[float, list[list[float]]]:
    """Blocos (lg R, lg P, ..., lg K_long, lg K_transv) indexados por lg T."""
    blocks: dict[float, list[list[float]]] = {}
    current: list | None = None
    with gzip.open(path, "rt", encoding="latin-1") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) == 2:
                try:
                    current = blocks.setdefault(float(fields[0]), [])
                except ValueError:
                    continue
            elif len(fields) >= 14 and current is not None:
                try:
                    current.append([float(v) for v in fields])
                except ValueError:
                    continue
    return blocks


def scattering_floor(log_t: float, log_b: float) -> float:
    """lg do piso de opacidade: só espalhamento, com o modo X suprimido.

    Os dois modos conduzem o fluxo em paralelo, então as opacidades inversas se
    somam com peso 1/2 cada. Com kappa_O = kappa_T e kappa_X = kappa_T (E/E_Be)^2,

        1/K = (1/2)(1/kappa_T) + (1/2)(E_Be^2/kappa_T) <E^-2>

    e <E^-2> = 0,1266 / (kT)^2 com o peso de Rosseland.
    """
    kt_kev = KEV_PER_KELVIN * 10.0 ** log_t
    e_be = E_BE_KEV_PER_B12 * 10.0 ** (log_b - 12.0)
    suppression = ROSSELAND_INV_SQUARE * (e_be / kt_kev) ** 2
    return math.log10(2.0 * THOMSON_CM2_G / (1.0 + suppression))


def audit_ionisation_bookkeeping() -> list[str]:
    """Decide, por medicao, se a molecula conta uma vez ou duas na soma."""
    print()
    print("=" * 96)
    print("AUDITORIA 2 — a contabilidade das fracoes do Potekhin")
    print("=" * 96)
    print("No canto frio e denso a materia esta toda recombinada, entao a fracao ionizada tem")
    print("de ir a zero. E' isso que decide o peso de x(H2), e nao a leitura do cabecalho.")
    print()
    problems: list[str] = []
    for name in ("hmag13_5.dat.gz", "hmag13_0.dat.gz", "hmag12_0.dat.gz"):
        path = POTEKHIN / name
        if not path.exists():
            continue
        rows = [row for block in read_potekhin(path).values() for row in block]
        # As cinco linhas de maior fracao molecular: e' onde as duas convencoes
        # mais se afastam, e portanto onde a medida decide.
        molecular = sorted(rows, key=lambda r: -r[10])[:5]
        one = [1.0 - (r[8] + r[10] + r[11]) for r in molecular]
        two = [1.0 - (r[8] + 2.0 * r[10] + r[11]) for r in molecular]
        print(f"{name:16s} x(H2) ate {max(r[10] for r in molecular):.3f} | "
              f"ionizado com peso 1: {min(one):+.4f} a {max(one):+.4f} | "
              f"com peso 2: {min(two):+.4f} a {max(two):+.4f}")
        # Peso 2 pode passar de 1 por arredondamento da tabela, que tem tres a
        # quatro digitos; abaixo de -0,002 ja nao seria arredondamento.
        worst = min(1.0 - (r[8] + 2.0 * r[10] + r[11]) for r in rows)
        if worst < -0.002:
            problems.append(f"{name}: fracao ionizada {worst:.4f} com peso 2")
    print()
    print("Veredito: x_ionizado = 1 - x(H) - 2 x(H2) - x(pert.). Com peso 1 sobra ionizacao")
    print("de 10 a 14% justamente onde ha molecula, que e' o segundo proton de cada uma.")
    return problems


def audit_opacity_floor() -> None:
    print()
    print("=" * 96)
    print("AUDITORIA 3 — o piso das opacidades do Potekhin e a definicao da media")
    print("=" * 96)
    print("Em densidade baixa sobra so espalhamento. Se o piso tabelado bate com a formula,")
    print("as colunas sao mesmo Rosseland em cm^2/g E a media harmonica com peso 1/2 por modo")
    print("e a definicao certa — que e a armadilha do portao do estagio 2.")
    print()
    print(f"{'arquivo':16s} {'lgB':>6} {'lgT':>6} {'lgR min':>8} "
          f"{'lgK tab':>8} {'lgK prev':>9} {'razao':>7}")
    for name in ("hmag13_5.dat.gz", "hmag13_0.dat.gz", "hmag12_0.dat.gz"):
        path = POTEKHIN / name
        if not path.exists():
            continue
        log_b = float(name[4:8].replace("_", "."))
        blocks = read_potekhin(path)
        for log_t in sorted(blocks)[:3]:
            rows = blocks[log_t]
            row = min(rows, key=lambda r: r[0])                  # a densidade menor
            tabulated, predicted = row[12], scattering_floor(log_t, log_b)
            print(f"{name:16s} {log_b:6.2f} {log_t:6.2f} {row[0]:8.2f} "
                  f"{tabulated:8.3f} {predicted:9.3f} {10.0 ** (tabulated - predicted):7.2f}")


if __name__ == "__main__":
    failed = audit_spectra()
    failed += audit_ionisation_bookkeeping()
    audit_opacity_floor()
    print()
    if failed:
        print(f"{len(failed)} arquivo(s) fora do padrao: {', '.join(failed)}")
        sys.exit(1)
    print("todos os gabaritos passam")
