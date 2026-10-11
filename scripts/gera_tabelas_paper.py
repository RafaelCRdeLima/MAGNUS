#!/usr/bin/env python3
"""Gera as tabelas e os numeros do artigo DIRETO dos checkpoints.

Quase todos os erros encontrados nas auditorias de 29/09 e 01/10 foram numeros
transcritos a mao que ficaram para tras: eixos de tabela, r-hat, profundidade da
linha, correlacoes, margens de EoS. Este script fecha essa porta: a tabela da
posterior, a de convergencia e um JSON com cada numero citado em linha saem dos
artefatos, com a convencao declarada no cabecalho do arquivo gerado.

Convencoes, todas explicitas na saida:
  janela    segunda metade da cadeia
  intervalo percentis 16 e 84 em torno da mediana
  exclusao  caminhante cujo lnL medio na janela fique mais de 100 abaixo da
            mediana dos caminhantes, listado por numero na saida

Uso:
  python3 scripts/gera_tabelas_paper.py \
      --fiducial exploracoes/.../B_i95 --outra exploracoes/.../A_i91 \
      --saida ~/Codes/paper-xmm/gerado
"""
import argparse
import json
from pathlib import Path

import numpy as np

GM_SUN_C2_KM = 1.4766250385          # identico ao do motor
E_CP_KEV_POR_GAUSS = 6.3048787654e-15  # idem
ATRASO_MAXIMO = 100.0


def carrega(d: Path, janela=0.5):
    ck = np.load(d / "checkpoint.npz", allow_pickle=True)
    nomes = [str(v) for v in ck["names"]]
    s = ck["samples"]; lp = ck["logps"].astype(np.float64)
    corte = int(s.shape[1] * (1.0 - janela))
    jan = lp[:, corte:]
    atraso = np.median(jan) - jan.mean(axis=1)
    perdidos = np.where(atraso > ATRASO_MAXIMO)[0]
    bons = np.setdiff1d(np.arange(s.shape[0]), perdidos)
    V = s[bons, corte:].reshape(-1, s.shape[2]).astype(np.float64)
    return dict(nomes=nomes, V=V, perdidos=[int(x) for x in perdidos],
                atrasos=[float(atraso[x]) for x in perdidos],
                corte=corte, passos=int(s.shape[1]), walkers=int(s.shape[0]),
                lnL_max=float(lp.max()), samples=s, logps=lp)


def q(x):
    a, b, c = np.percentile(x, [16, 50, 84])
    return float(b), float(c - b), float(b - a)


def derivados(d):
    nm, V = d["nomes"], d["V"]
    u = V[:, nm.index("compactness")]
    R = V[:, nm.index("radius")]
    z = 1.0 / np.sqrt(1.0 - u) - 1.0
    M = u * R / (2.0 * GM_SUN_C2_KM)
    lgB = V[:, nm.index("logMagneticField")]
    Ecp = E_CP_KEV_POR_GAUSS * 10.0 ** lgB
    col = {n: V[:, i] for i, n in enumerate(nm)}
    col.update(z=z, mass=M, MsobreR=M / R, Ecp=Ecp, Eobs=Ecp / (1.0 + z))
    return col


def rhat(x):
    m, n = x.shape
    W = x.var(axis=1, ddof=1).mean(); B = n * x.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * W + B / n) / W)) if W > 0 else float("nan")


def tabela_convergencia(fid, out):
    linhas = []
    for N in (10000, 20000, 40000, 80000, 140000):
        vals = []
        for d in (fid, out):
            if N > d["passos"]:
                vals.append(None); continue
            keep = np.setdiff1d(np.arange(d["walkers"]), d["perdidos"])
            sub = d["samples"][keep, N // 2:N].astype(np.float64)
            r = {k: rhat(sub[:, :, i]) for i, k in enumerate(d["nomes"])
                 if sub[:, :, i].std() > 0}
            vals.append(max(r.values()))
        if vals[0] is not None:
            linhas.append((N, vals[0], vals[1]))
    return linhas


def fmt(v, dv_mais, dv_menos, casas=3, grau=False):
    g = "^{\\circ}" if grau else ""
    if abs(dv_mais - dv_menos) < 0.05 * max(dv_mais, 1e-12):
        return "$%.*f%s\\pm%.*f$" % (casas, v, g, casas, (dv_mais + dv_menos) / 2)
    return "$%.*f%s{}^{+%.*f}_{-%.*f}$" % (casas, v, g, casas, dv_mais, casas, dv_menos)


LINHAS = [
    ("Compactness $M/R\\,[M_\\odot\\,{\\rm km}^{-1}]$", "MsobreR", 4, "measured"),
    ("Surface redshift $z$", "z", 3, "measured"),
    ("$u=2GM/Rc^{2}$", "compactness", 3, "measured"),
    ("Polar field $\\lg B_{p}\\,[{\\rm G}]$", "logMagneticField", 3, "measured"),
    ("Absorbing column $N_{\\rm H}\\,[10^{22}]$", "nh", 4, "free"),
    ("Distance $d$\\,[kpc]", "distance", 3, "at prior floor"),
    ("Mass $M\\,[M_\\odot]$", "mass", 2, "implied, at ceiling"),
    ("Radius $R$\\,[km]", "radius", 2, "implied"),
    (None, None, None, "Atmosphere and surface"),
    ("Effective thickness $f$", "atmFraction", 3, ""),
    ("Pole temperature $kT_{p1}$\\,[keV]", "baseKT", 3, ""),
    ("Pole temperature $kT_{p2}$\\,[keV]", "baseKT2", 3, ""),
    ("Heat concentration $a_{1}$", "peaking", 2, ""),
    ("Heat concentration $a_{2}$", "peaking2", 2, ""),
    ("Magnetic obliquity $\\Theta_{B}$", "magColat", 1, True, ""),
    ("Magnetic azimuth $\\varphi_{B}$", "magAzim", 1, True, ""),
    ("Inclination $i$", "inclination", 1, True, "basin restricted"),
    ("Pole-2 offset $\\beta$", "poleTilt2", 1, True, ""),
    ("Pole-2 azimuth $\\gamma$", "poleTilt2b", 1, True, "$\\gamma>0$ imposed"),
    (None, None, None, "Absorption feature (redshifted cyclotron)"),
    ("Rest energy $E_{c,p}(B_{p})$\\,[keV]", "Ecp", 3, "from $B_{p}$"),
    ("Observed energy $E_{0}$\\,[keV]", "Eobs", 3, "at the poles"),
    ("Line width $\\sigma$\\,[keV]", "lineWidth", 3, ""),
    ("Line depth $\\tau$", "lineDepth", 2, ""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiducial", required=True)
    ap.add_argument("--outra", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--janela", type=float, default=0.5)
    a = ap.parse_args()
    saida = Path(a.saida); saida.mkdir(parents=True, exist_ok=True)

    fid = carrega(Path(a.fiducial), a.janela)
    out = carrega(Path(a.outra), a.janela)
    cf, co = derivados(fid), derivados(out)

    cab = ("%% GERADO por scripts/gera_tabelas_paper.py; nao editar a mao.\n"
           "%% fiducial: %s (%d x %d)\n%% outra:    %s (%d x %d)\n"
           "%% janela: segunda metade | intervalo: percentis 16 e 84\n"
           "%% excluidos no fiducial: %s | na outra: %s\n"
           % (a.fiducial, fid["walkers"], fid["passos"],
              a.outra, out["walkers"], out["passos"],
              fid["perdidos"] or "nenhum", out["perdidos"] or "nenhum"))

    numeros = {"_convencao": "segunda metade; percentis 16/84; excluidos listados",
               "fiducial": {"dir": a.fiducial, "walkers": fid["walkers"],
                            "passos": fid["passos"], "excluidos": fid["perdidos"],
                            "atraso_dos_excluidos": fid["atrasos"],
                            "lnL_max": fid["lnL_max"]},
               "outra": {"dir": a.outra, "passos": out["passos"],
                         "excluidos": out["perdidos"], "lnL_max": out["lnL_max"]}}

    corpo = [r"\begin{tabular}{lll}", r"\hline\hline",
             r"Parameter & Value & Note \\", r"\hline",
             r"\multicolumn{3}{l}{\textit{Global (feature tied to cyclotron)}}\\"]
    for item in LINHAS:
        rotulo, chave, casas = item[0], item[1], item[2]
        grau = len(item) == 5 and item[3] is True
        nota = item[-1]
        if chave is None:
            corpo.append("\\multicolumn{3}{l}{\\textit{%s}}\\\\" % nota)
            continue
        v, p, mi = q(cf[chave])
        numeros.setdefault("fiducial_parametros", {})[chave] = \
            dict(mediana=v, mais=p, menos=mi)
        corpo.append("%s & %s & %s \\\\" % (rotulo, fmt(v, p, mi, casas, grau), nota))

    zf = q(cf["z"])[0]; zo = q(co["z"])[0]
    mro = q(co["MsobreR"])[0]
    numeros["outra_parametros"] = {"z": zo, "MsobreR": mro}
    numeros["separacao_z"] = abs(zo - zf)
    numeros["delta_lnL_entre_bacias"] = fid["lnL_max"] - out["lnL_max"]

    PLACEHOLDER_RODAPE = len(corpo)
    corpo += [
        r"\hline",
        r"\multicolumn{3}{l}{\textit{Fit quality:} $\chi^{2}\simeq2.1$ per bin; "
        r"$\hat R\le%.2f$ (App.~\ref{app:convergence})}\\" % 0.0,
        r"\multicolumn{3}{l}{\textit{Other basin} ($i<93.3^{\circ}$): "
        r"$M/R=%.4f$, $z=%.3f$, $\Delta\ln\mathcal{L}=%+.1f$}\\"
        % (mro, zo, out["lnL_max"] - fid["lnL_max"]),
        r"\hline", r"\end{tabular}"]

    conv = tabela_convergencia(fid, out)
    numeros["convergencia"] = [dict(N=n, fiducial=r1, outra=r2) for n, r1, r2 in conv]
    numeros["rhat_fiducial"] = conv[-1][1]
    numeros["rhat_outra"] = conv[-1][2]

    corpo[PLACEHOLDER_RODAPE + 1] = (
        r"\multicolumn{3}{l}{\textit{Fit quality:} $\chi^{2}\simeq2.1$ per bin; "
        r"$\hat R\le%.2f$ (App.~\ref{app:convergence})}\\" % conv[-1][1])
    (saida / "tabela_posterior.tex").write_text(cab + "\n".join(corpo) + "\n")
    linhas_conv = ["$%s$ & $%.3f$ & $%.3f$ \\\\"
                   % (("10^{4}" if n == 10000 else "%g\\times10^{4}" % (n / 1e4)), r1, r2)
                   for n, r1, r2 in conv]
    conv_tex = ([r"\begin{tabular}{rcc}", r"\hline\hline",
                 r"$N$ & $\hat R$ (fiducial basin) & $\hat R$ (other basin) \\",
                 r"\hline"] + linhas_conv + [r"\hline", r"\end{tabular}"])
    (saida / "tabela_convergencia.tex").write_text(cab + "\n".join(conv_tex) + "\n")
    (saida / "numeros.json").write_text(json.dumps(numeros, indent=2) + "\n")

    print("gravado em", saida)
    print("  z fiducial = %.4f (+%.4f -%.4f) | outra = %.4f | separacao %.4f"
          % (*q(cf["z"]), zo, abs(zo - zf)))
    print("  r-hat: fiducial %.3f, outra %.3f" % (conv[-1][1], conv[-1][2]))
    print("  excluidos: fiducial %s, outra %s" % (fid["perdidos"], out["perdidos"]))


if __name__ == "__main__":
    main()
