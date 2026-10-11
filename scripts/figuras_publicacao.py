"""Figuras de publicação a partir de um ajuste do MAGNUS.

Estilo de periódico: em inglês, SEM grade, ticks para dentro nos quatro lados,
fonte serif, saída vetorial (PDF) + PNG. Amostra a posteriori pelo motor e
desenha as bandas 1σ/2σ do pulso e do espectro sobre os dados.

Uso:
    python3 scripts/figuras_publicacao.py <dir_do_ajuste> [n_amostras]

O diretório precisa ter resultado.json e pedido.json. As figuras saem nele.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcmc_fit as m                                                  # noqa: E402

# ---- estilo de publicação --------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 13,
    "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.minor.size": 3, "ytick.minor.size": 3,
    "axes.grid": False,
    "legend.frameon": False,
    "figure.dpi": 120,
})

DATA = "#111111"          # data points
MED = "#1a3d6e"           # posterior median
S1 = "#7fa8d0"            # 1 sigma
S2 = "#cfe0ee"            # 2 sigma


def sample_models(result_dir: Path, n: int):
    d = json.loads((result_dir / "resultado.json").read_text())
    req = json.loads((result_dir / "pedido.json").read_text())
    S = np.asarray(d["samples"], float)
    L = np.asarray(d["sampleLogLikelihood"], float)
    fin = S[np.isfinite(L)]
    # dados: do preparedDataNpz se houver, senão dos eventos
    ev = req.get("_events") or _guess_events(req)
    meta, events = m.read_events(Path(ev))
    bg = None
    problem = m.FitProblem(req, events, meta, m.read_background(Path(req["_background"]))
                           if req.get("_background") else None)
    nph, nen = problem.phase_bins, problem.energy_bins
    emin = req.get("energyMin", 0.15); emax = req.get("energyMax", 1.2)
    # Recorte da banda dura: plota (e soma o pulso) apenas nos bins de energia
    # cujo CENTRO fica abaixo de energyFitMax, coerente com a mascara do ajuste.
    efm = float(req.get("energyFitMax", emax))
    width = (emax - emin) / nen
    centers = emin + (np.arange(nen) + 0.5) * width
    emask = centers < efm
    rng = np.random.default_rng(11)
    draw = fin[rng.choice(len(fin), size=min(n, len(fin)), replace=False)]
    w = m.EngineWorker(problem.worker_command())
    LC, SP = [], []
    for v in draw:
        _, exp = problem.evaluate(w, list(v), return_model=True)
        if exp is None:
            continue
        g = np.array(exp, float).reshape(nph, nen)
        LC.append(g[:, emask].sum(1)); SP.append(g[:, emask].sum(0))
    w.close()
    obs = np.array(problem.observed, float).reshape(nph, nen)
    return (np.array(LC), np.array(SP), obs[:, emask].sum(1), obs[:, emask].sum(0),
            nph, centers[emask])


def _guess_events(req):
    import os
    ev = req.get("events") or os.environ.get("MAGNUS_EVENTS")
    if not ev:
        raise SystemExit("arquivo de eventos nao informado: use --events, a chave "
                         "'events' do pedido ou a variavel MAGNUS_EVENTS")
    return ev


def _finish(ax):
    for s in ax.spines.values():
        s.set_linewidth(1.0)


def lightcurve(LC, lo, nph, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ph = (np.arange(nph) + 0.5) / nph
    ph2 = np.concatenate([ph, ph + 1]); dbl = lambda x: np.concatenate([x, x])
    P = np.percentile(LC, [2.5, 16, 50, 84, 97.5], axis=0)
    ax.fill_between(ph2, dbl(P[0]), dbl(P[4]), color=S2,
                    label=r"$2\sigma$, expected counts")
    ax.fill_between(ph2, dbl(P[1]), dbl(P[3]), color=S1, label=r"$1\sigma$")
    ax.plot(ph2, dbl(P[2]), color=MED, lw=1.8, label="Posterior median")
    ax.errorbar(ph2, dbl(lo), yerr=np.sqrt(dbl(lo)), fmt="o", ms=4, color=DATA,
                elinewidth=0.9, capsize=0, label="RBS 1223", zorder=5)
    ax.set_xlabel("Rotational phase")
    ax.set_ylabel("Counts")
    ax.set_xlim(0, 2)
    # A legenda vai ACIMA dos eixos, numa linha so. Dentro do quadro ela ficava
    # no canto inferior direito, exatamente onde caem os pontos do ultimo minimo
    # (fases 1,9 a 2,0): os marcadores encostavam nas entradas e o olho lia os
    # dados como parte da legenda.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=4,
              fontsize=9.5, frameon=False, borderpad=0.0, columnspacing=1.6,
              handlelength=1.6, handletextpad=0.6)
    _finish(ax)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200)


def spectrum(SP, so, energy, out):
    """Espectro somado em fase, com painel de residuos.

    O painel de baixo entrou a pedido de arbitragem: concordancia visual em escala
    logaritmica nao resolve uma discrepancia estatistica, e o ajuste tem chi2 de
    2,1 por bin. O residuo e padronizado por Poisson, (obs - modelo)/sqrt(modelo),
    e as faixas marcam 1 e 2 sigma.
    """
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3.0, 1.0], hspace=0.06))
    Q = np.percentile(SP, [2.5, 16, 50, 84, 97.5], axis=0)
    ax.fill_between(energy, Q[0], Q[4], color=S2,
                    label=r"$2\sigma$, expected counts")
    ax.fill_between(energy, Q[1], Q[3], color=S1, label=r"$1\sigma$")
    ax.plot(energy, Q[2], color=MED, lw=1.8, label="Posterior median")
    ax.errorbar(energy, so, yerr=np.sqrt(np.maximum(so, 1)), fmt="o", ms=4,
                color=DATA, elinewidth=0.9, capsize=0, label="RBS 1223", zorder=5)
    ax.set_yscale("log")
    ax.set_ylabel("Counts (phase-summed)")
    ax.set_xlim(energy[0], energy[-1])
    ax.legend(loc="upper right", fontsize=10)
    _finish(ax)

    res = (so - Q[2]) / np.sqrt(np.maximum(Q[2], 1.0))
    axr.axhspan(-2, 2, color=S2, zorder=0)
    axr.axhspan(-1, 1, color=S1, zorder=0)
    axr.axhline(0, color=MED, lw=1.0, zorder=2)
    axr.plot(energy, res, "o", ms=3.5, color=DATA, zorder=5)
    axr.set_ylabel(r"$(N-M)/\sqrt{M}$", fontsize=9)
    axr.set_xlabel("Energy (keV)")
    lim = max(4.0, 1.15*np.abs(res).max())
    axr.set_ylim(-lim, lim)
    chi2 = float(np.mean(res**2))
    axr.annotate(r"$\chi^{2}/{\rm bin}=%.1f$ (phase-summed)" % chi2,
                 (0.03, 0.88), xycoords="axes fraction", fontsize=8, color="0.25", va="top")
    _finish(axr)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200)


def diagnosticos(d, LC, SP, lo, so, nph):
    """Numeros que o texto cita, com o ESTIMADOR identificado.

    A auditoria de 29/09 apontou que o artigo misturava os dois: a fracao
    pulsada de 16,5 por cento e o residuo de -5,3 sigma sao da amostra de
    MAXIMA verossimilhanca, enquanto as figuras mostram a mediana das previsoes,
    que da 16,7 e -5,0. Aqui os dois saem lado a lado.
    """
    import json
    def pf(c):
        return float((c.max() - c.min()) / (c.max() + c.min()))
    def pior(c, obs):
        r = (obs - c) / np.sqrt(np.maximum(c, 1e-9))
        return float(r[np.argmax(abs(r))])
    mediana = np.median(LC, axis=0)
    saida = {"fracao_pulsada": {"curva_mediana": pf(mediana)},
             "residuo_extremo_sigma": {"curva_mediana": pior(mediana, lo)}}

    # a amostra de maxima verossimilhanca, avaliada de novo
    try:
        res = json.loads((d / "resultado.json").read_text())
        req = json.loads((d / "pedido.json").read_text())
        S = np.asarray(res["samples"], float)
        L = np.asarray(res["sampleLogLikelihood"], float)
        v = S[int(np.nanargmax(L))]
        ev = req.get("_events") or _guess_events(req)
        meta, events = m.read_events(Path(ev))
        prob = m.FitProblem(req, events, meta,
                            m.read_background(Path(req["_background"]))
                            if req.get("_background") else None)
        w = m.EngineWorker(prob.worker_command())
        _, esp = prob.evaluate(w, list(v), return_model=True)
        w.close()
        g = np.array(esp, float).reshape(prob.phase_bins, prob.energy_bins)
        emin = req.get("energyMin", 0.15); emax = req.get("energyMax", 1.2)
        nen = prob.energy_bins
        c = emin + (np.arange(nen) + 0.5) * (emax - emin) / nen
        mk = c < float(req.get("energyFitMax", emax))
        cur = g[:, mk].sum(1)
        saida["fracao_pulsada"]["amostra_de_maximo"] = pf(cur)
        saida["residuo_extremo_sigma"]["amostra_de_maximo"] = pior(cur, lo)
    except Exception as erro:                       # nao impedir as figuras
        saida["amostra_de_maximo"] = "indisponivel: %s" % erro

    saida["_nota"] = ("as bandas das figuras sao da contagem ESPERADA sobre a "
                      "posterior; nao incluem o sorteio de Poisson de uma nova "
                      "realizacao")
    (d / "numeros_figuras.json").write_text(json.dumps(saida, indent=2) + "\n")
    return saida


def main():
    d = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 140
    # Cache das amostras. Ajustar estilo de figura nao pode custar outra varredura
    # da posterior pelo motor (sao n execucoes do C++, minutos). Reusa se o cache
    # tiver pelo menos tantas amostras quanto se pediu e for mais novo que o
    # resultado; --refazer ignora o cache.
    refazer = "--refazer" in sys.argv
    cache = d / "amostras_figuras.npz"
    usar = (cache.exists() and not refazer
            and cache.stat().st_mtime >= (d / "resultado.json").stat().st_mtime)
    if usar:
        z = np.load(cache)
        if int(z["n"]) >= n:
            LC, SP, lo, so = z["LC"], z["SP"], z["lo"], z["so"]
            nph, energy = int(z["nph"]), z["energy"]
            print(f"reusando {cache.name} ({int(z['n'])} amostras)")
        else:
            usar = False
    if not usar:
        LC, SP, lo, so, nph, energy = sample_models(d, n)
        np.savez(cache, LC=LC, SP=SP, lo=lo, so=so, nph=nph, energy=energy, n=n)
    n_ = diagnosticos(d, LC, SP, lo, so, nph)
    fp = n_["fracao_pulsada"]
    print("fracao pulsada: curva mediana %.4f | amostra de maximo %s"
          % (fp["curva_mediana"], ("%.4f" % fp["amostra_de_maximo"])
             if "amostra_de_maximo" in fp else "indisponivel"))
    lightcurve(LC, lo, nph, d / "fig_lightcurve")
    spectrum(SP, so, energy, d / "fig_spectrum")
    print(f"figuras de publicação salvas em {d}/ (fig_lightcurve, fig_spectrum; .pdf + .png)")


if __name__ == "__main__":
    main()
