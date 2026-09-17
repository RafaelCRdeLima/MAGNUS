"""Validacao externa: perfil T(tau) do MAGNUS contra van Adelsberg e Lai (2006).

O artigo (MNRAS 373, 1495) publica, na Tabela 1 e na Eq. 48, ajustes analiticos
do perfil de temperatura de atmosferas de H e He COM polarizacao do vaco e
conversao PARCIAL de modos, o mesmo caso que o MAGNUS calcula com
`vacuum=True, conversion="partial"`. Os modelos de 4e13 e 7e13 G cercam o campo
polar do RBS 1223, o que torna essa comparacao a prova de funcionamento mais
direta que existe para o ramo do vaco.

Hipoteses que o artigo adota e que este script reproduz: campo ao longo da
normal (theta_B = 0), H totalmente ionizado, gas ideal sem degenerescencia,
M = 1,4 M_sol e R = 10 km, logo g = 2,4e14 cm/s^2. A profundidade optica da
Eq. 48 e a de Thomson, tau = 0,4 y, com y em g/cm^2.

Uso:
    python3 scripts/valida_vanadelsberg2006.py            # os dois modelos vizinhos do RBS 1223
    python3 scripts/valida_vanadelsberg2006.py --todos    # os nove modelos de H
    python3 scripts/valida_vanadelsberg2006.py --modelo 7e13_1e6 --iteracoes 400
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

G_VAL06 = 2.4e14          # cm/s^2, do M = 1,4 M_sol e R = 10 km do artigo
KAPPA_T = 0.4             # cm^2/g, a opacidade de Thomson que define o tau deles

#: Tabela 1 de van Adelsberg e Lai (2006). Para cada modelo:
#: (B em G, T_eff em K, composicao, tau_mid, a1, a2, [a3..a6], [b3..b6]).
#: A Eq. 48 usa os a_i acima de tau_mid e os b_i abaixo, com a1 e a2 comuns.
TABELA1 = {
    "1e13_5e6":  (1.0e13, 5.0e6, "H",  27.1,  0.793,   0.122,
                  [-0.502, 0.548, -0.205, 0.0266],
                  [0.00445, 0.0108, 0.00211, 0.0000574]),
    "4e13_1e6":  (4.0e13, 1.0e6, "H",  4.27, -0.0599,  0.192,
                  [0.0225, 0.0115, -0.0072, 0.00116],
                  [0.109, 0.0828, 0.0256, 0.00286]),
    "4e13_5e6":  (4.0e13, 5.0e6, "H",  11.9,  0.623,  -0.0425,
                  [0.0991, 0.0412, -0.026, 0.0036],
                  [-0.0851, -0.00392, 0.0034, 0.000418]),
    "7e13_1e6":  (7.0e13, 1.0e6, "H", 0.888, -0.0455, -0.158,
                  [0.221, -0.0469, 0.00231, 0.000307],
                  [-0.329, -0.118, -0.00387, 0.00304]),
    "7e13_5e6":  (7.0e13, 5.0e6, "H",  21.6,  0.789,   0.123,
                  [-0.650, 0.726, -0.274, 0.0354],
                  [-0.0105, -0.00406, -0.00242, -0.000411]),
    "1e14_1e6":  (1.0e14, 1.0e6, "H", 0.683,  0.00828, 0.0614,
                  [-0.304, 0.266, -0.0719, 0.00652],
                  [-0.313, -0.374, -0.162, -0.0234]),
    "1e14_1e6_He": (1.0e14, 1.0e6, "He", 0.749, -0.0935, -0.154,
                    [0.262, -0.0904, 0.0167, -0.00124],
                    [-0.197, 0.0197, 0.0428, 0.0083]),
    "1e14_5e6":  (1.0e14, 5.0e6, "H",  30.6,  0.799,   0.115,
                  [-0.537, 0.617, -0.241, 0.0326],
                  [-0.00603, 0.00409, 0.000351, -0.0000978]),
    "5e14_1e6":  (5.0e14, 1.0e6, "H",  32.9,  0.0939,  0.0181,
                  [-0.0153, -0.0413, 0.0376, -0.00578],
                  [0.0504, 0.0462, 0.00991, 0.000676]),
    "5e14_5e6":  (5.0e14, 5.0e6, "H",  63.2,  0.761,   0.00198,
                  [0.267, -0.356, 0.179, -0.0282],
                  [-0.118, -0.0336, -0.00428, -0.00022]),
    "5e14_5e6_He": (5.0e14, 5.0e6, "He", 23.5, 0.707,  0.0467,
                    [0.342, -0.417, 0.174, -0.0234],
                    [-0.109, -0.040, -0.0069, -0.000481]),
}

#: Os dois que cercam o campo polar do RBS 1223 (6e13 a 8,6e13 G) na temperatura
#: certa (T_eff ~ 0,7 MK observado, 1 MK tabelado).
PADRAO = ["4e13_1e6", "7e13_1e6"]

FAIXA_TAU = (1.0e-3, 2.0e4)   # validade declarada do ajuste


def perfil_publicado(tau: np.ndarray, chave: str) -> np.ndarray:
    """log10 T_6 pela Eq. 48 de van Adelsberg e Lai (2006)."""
    _b, _t, _c, tau_mid, a1, a2, a_altos, b_baixos = TABELA1[chave]
    x = np.log10(tau)
    dx = x - np.log10(tau_mid)
    fundo = a1 + a2 * dx
    poli_a = sum(c * dx ** (i + 2) for i, c in enumerate(a_altos))
    poli_b = sum(c * dx ** (i + 2) for i, c in enumerate(b_baixos))
    return np.where(tau > tau_mid, fundo + poli_a, fundo + poli_b)


def roda_magnus(chave: str, iteracoes: int, mu_nodes: int) -> dict:
    from atmosfera import magnetizada
    campo, t_eff, comp, *_ = TABELA1[chave]
    if comp != "H":
        raise SystemExit(f"{chave}: o MAGNUS so tem hidrogenio; modelo de {comp} fora do teste")
    t0 = time.time()
    sol = magnetizada.solve(log_t_eff=np.log10(t_eff), log_g=np.log10(G_VAL06),
                            field_g=campo, theta_b=0.0, vacuum=True,
                            conversion="partial", iterations=iteracoes,
                            mu_nodes=mu_nodes)
    sol["segundos"] = time.time() - t0
    return sol


def compara(chave: str, sol: dict) -> dict:
    y = np.asarray(sol["columns"], float)
    t = np.asarray(sol["temperature"], float)
    bom = y > 0.0
    tau = KAPPA_T * y[bom]
    dentro = (tau >= FAIXA_TAU[0]) & (tau <= FAIXA_TAU[1])
    tau = tau[dentro]
    nosso = np.log10(t[bom][dentro] / 1.0e6)
    deles = perfil_publicado(tau, chave)
    dif = nosso - deles
    # diferenca relativa em T, que e o que interessa: 10^dif - 1
    rel = 10.0 ** dif - 1.0
    return {
        "modelo": chave, "pontos": int(tau.size),
        "tau": tau.tolist(), "log10T6_magnus": nosso.tolist(),
        "log10T6_val06": deles.tolist(),
        "dif_relativa_mediana": float(np.median(np.abs(rel))),
        "dif_relativa_maxima": float(np.max(np.abs(rel))),
        "dif_relativa_rms": float(np.sqrt(np.mean(rel ** 2))),
        "erro_fluxo": float(sol["flux_error"]), "iteracoes": int(sol["iterations"]),
        "segundos": float(sol.get("segundos", 0.0)),
    }


K_B_CGS = 1.380649e-16
M_H_CGS = 1.67262192369e-24


def diagnostico_ressonancia(r: dict) -> list[dict]:
    """Onde cai a ressonancia de vaco, em tau, e qual o desvio da T ali.

    Usa a hidrostatica do proprio perfil (P = g y, gas ideal de H ionizado) para
    converter tau em densidade, e compara com rho_V = 0,96 E^2 B_14^2 g/cm^3.
    """
    campo = TABELA1[r["modelo"]][0]
    tau = np.array(r["tau"])
    t = 1.0e6 * 10.0 ** np.array(r["log10T6_magnus"])
    rho = G_VAL06 * (tau / KAPPA_T) * M_H_CGS / (2.0 * K_B_CGS * t)
    rel = 10.0 ** (np.array(r["log10T6_magnus"]) - np.array(r["log10T6_val06"])) - 1.0
    saida = []
    for energia in (0.15, 0.26, 0.5, 1.0):
        rho_v = 0.96 * energia ** 2 * (campo / 1.0e14) ** 2
        i = int(np.argmin(np.abs(np.log(rho / rho_v))))
        saida.append({"energia_keV": energia, "rho_V": float(rho_v),
                      "tau": float(tau[i]), "dif_relativa": float(rel[i])})
    return saida


def figura(resultados: list[dict], destino: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(resultados)
    fig, eixos = plt.subplots(2, n, figsize=(5.2 * n, 6.4), sharex="col",
                              gridspec_kw={"height_ratios": [2.2, 1.0]}, squeeze=False)
    for j, r in enumerate(resultados):
        tau = np.array(r["tau"]); cima, baixo = eixos[0][j], eixos[1][j]
        cima.plot(tau, 10.0 ** np.array(r["log10T6_val06"]), lw=2.4, color="0.55",
                  label="van Adelsberg e Lai (2006), Eq. 48")
        cima.plot(tau, 10.0 ** np.array(r["log10T6_magnus"]), lw=1.3, color="crimson",
                  label="MAGNUS, vaco com conversao parcial")
        cima.set_xscale("log"); cima.set_yscale("log")
        cima.set_ylabel(r"$T$ (MK)"); cima.legend(fontsize=8, frameon=False)
        campo, t_eff = TABELA1[r["modelo"]][0], TABELA1[r["modelo"]][1]
        cima.set_title(f"B = {campo:.0e} G, "
                       f"$T_{{\\rm ef}}$ = {t_eff:.0e} K, H\n"
                       f"mediana {100*r['dif_relativa_mediana']:.1f} %, "
                       f"maxima {100*r['dif_relativa_maxima']:.1f} %", fontsize=9)
        rel = 10.0 ** (np.array(r["log10T6_magnus"]) - np.array(r["log10T6_val06"])) - 1.0
        baixo.axhline(0.0, color="0.7", lw=0.8)
        baixo.plot(tau, 100.0 * rel, lw=1.2, color="crimson")
        baixo.set_xscale("log"); baixo.set_xlabel(r"$\tau$ (Thomson)")
        baixo.set_ylabel("diferenca (%)")
        for d in r.get("ressonancia", []):
            for eixo in (cima, baixo):
                eixo.axvline(d["tau"], color="steelblue", lw=0.8, ls=":")
            cima.annotate(f"{d['energia_keV']:.2f} keV", (d["tau"], 0.97),
                          xycoords=("data", "axes fraction"), fontsize=6,
                          color="steelblue", rotation=90, va="top", ha="right")
    fig.tight_layout()
    fig.savefig(destino, dpi=140)
    fig.savefig(destino.with_suffix(".pdf"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modelo", action="append", choices=sorted(TABELA1),
                   help="modelo da Tabela 1 (repetivel); padrao: os vizinhos do RBS 1223")
    p.add_argument("--todos", action="store_true", help="todos os modelos de hidrogenio")
    p.add_argument("--iteracoes", type=int, default=300)
    p.add_argument("--mu", type=int, default=6, help="nos de Gauss-Legendre em mu")
    p.add_argument("--saida", default="exploracoes/validacao_val06")
    p.add_argument("--apenas-analise", action="store_true",
                   help="refaz diagnostico e figura a partir do resultado.json ja gravado")
    args = p.parse_args()

    if args.todos:
        chaves = [k for k, v in TABELA1.items() if v[2] == "H"]
    else:
        chaves = args.modelo or PADRAO

    destino = RAIZ / args.saida
    destino.mkdir(parents=True, exist_ok=True)
    if args.apenas_analise:
        resultados = json.loads((destino / "resultado.json").read_text())
        for r in resultados:
            r["ressonancia"] = diagnostico_ressonancia(r)
            print(f"[{r['modelo']}] mediana {100*r['dif_relativa_mediana']:.2f} %, "
                  f"maxima {100*r['dif_relativa_maxima']:.2f} %")
            for d in r["ressonancia"]:
                print(f"[{r['modelo']}] ressonancia de {d['energia_keV']:.2f} keV em "
                      f"tau = {d['tau']:.1e}, desvio de T ali {100*d['dif_relativa']:+.0f} %")
        (destino / "resultado.json").write_text(json.dumps(resultados, indent=1))
        figura(resultados, destino / "fig_perfil.png")
        print("gravado em", destino)
        return
    resultados = []
    for chave in chaves:
        print(f"[{chave}] rodando o MAGNUS ...", flush=True)
        sol = roda_magnus(chave, args.iteracoes, args.mu)
        r = compara(chave, sol)
        r["ressonancia"] = diagnostico_ressonancia(r)
        resultados.append(r)
        print(f"[{chave}] erro de fluxo {r['erro_fluxo']:.2e} em {r['iteracoes']} it, "
              f"{r['segundos']:.0f} s\n"
              f"[{chave}] T contra o ajuste publicado: mediana "
              f"{100*r['dif_relativa_mediana']:.2f} %, rms "
              f"{100*r['dif_relativa_rms']:.2f} %, maxima "
              f"{100*r['dif_relativa_maxima']:.2f} %", flush=True)
        for d in r["ressonancia"]:
            print(f"[{chave}] ressonancia de {d['energia_keV']:.2f} keV em "
                  f"tau = {d['tau']:.1e}, desvio de T ali {100*d['dif_relativa']:+.0f} %",
                  flush=True)
    (destino / "resultado.json").write_text(json.dumps(resultados, indent=1))
    figura(resultados, destino / "fig_perfil.png")
    print("gravado em", destino)


if __name__ == "__main__":
    main()
