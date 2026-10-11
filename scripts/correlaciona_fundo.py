#!/usr/bin/env python3
"""A correlacao que o estimador por sobreposicao cria, e quanto ela custa.

O revisor, 10/10/2026: "a diagonalidade demonstrada no estimador antigo nao se
transfere ao novo estimador por sobreposicao. Um canal dividido entre dois bins
cria dependencia; suas contagens nao devem gerar duas medicoes independentes.
Precisamos preservar o latente compartilhado ou demonstrar que ignorar essa
correlacao cabe no orcamento, comparando a verossimilhanca nos casos
representativos."

O QUE EXATAMENTE ESTA ERRADO HOJE. O alvo monta, para cada bin de energia e,

    b_e = soma_{j em e} a_j lambda_j ,   lambda_j ~ Gamma(N_j + a0 w_j, 1)

e integra cada bin SEPARADAMENTE. Os componentes j sao pares (canal, bin), e um
canal partido entre dois bins aparece DUAS vezes, com pesos diferentes mas com a
MESMA contagem N. O alvo sorteia para ele dois lambda independentes. O certo e
um lambda por CANAL, entrando com peso a_j em cada um dos dois bins.

CONSEQUENCIA, pela algebra e nao por suposicao: a marginal de cada b_e nao muda,
porque Var(b_e) = soma_j a_j^2 (N_j + a0 w_j) e a mesma dos dois lados. O que o
alvo perde e so a COVARIANCIA entre bins vizinhos, que ele poe em zero:

    cov(b_e, b_{e+1}) = a_j^{(e)} a_j^{(e+1)} (N_j + a0 w_j)  somada nos partidos

Logo o efeito nao e um vies no fundo esperado: e uma dependencia ignorada na
verossimilhanca conjunta dos bins.

COMO SE MEDE AQUI. Duas integrais com a MESMA familia de base, as Gammas
exatas, estimadas por Monte Carlo com VARIAVEIS COMUNS:

  independente: ln E[exp(soma_e ll_e)] com um lambda por COMPONENTE
  compartilhado: ln E[exp(soma_e ll_e)] com um lambda por CANAL

Os sorteios dos componentes nao partidos sao os MESMOS nos dois lados, e so os
partidos diferem: isso correlaciona numerador e denominador e faz a razao, que e
a diferenca de ln L, ter erro muito menor que cada lado. O erro de Monte Carlo e
reportado por reamostragem e nao suposto.

CONFERENCIA DO ESTIMADOR, em separado: o lado independente tem valor por
quadratura dentro do proprio alvo, e e comparado com ele. Essa diferenca mede a
Gamma deslocada casada em tres cumulantes mais a quadratura, NAO a correlacao.

O que decide nao e o valor absoluto, e o CONTRASTE entre vetores: e o contraste
que move conclusao.

Uso: python3 scripts/correlaciona_fundo.py [--amostras 200000] [--bandas 120]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import mcmc_fit as m  # noqa: E402
import por_observacao as PO  # noqa: E402

FASES = PO.FASES


def grades_conjuntas(obss, ordem, bandas, refino, vetor, meta, ev):
    """UMA chamada do motor, as treze como slots, resolucao efetiva zero."""
    base = PO.monta_pedido(obss[ordem[0]], bandas, refino)
    base["outputPrecision"] = 17
    prob = m.FitProblem(base, ev, meta, None)
    comando = list(prob.worker_command())
    for obsid in ordem:
        o = obss[obsid]
        comando += ["--obs-id", obsid,
                    "--obs-profile", o["perfil_csv"],
                    "--obs-response", o["perfil_rmfbin"],
                    "--obs-time-resolution-us", "0.0",
                    "--obs-dead-time-us", "0.0",
                    "--obs-period", repr(float(o["periodo"])),
                    "--obs-offset", repr(PO.offset_em_ciclos(int(o["desloc_v3_bins"])))]
    caminho, rotulo = [x[1:] for x in PO.VETORES if x[0] == vetor][0]
    v = PO.le_vetor(prob, caminho, rotulo)
    trab = m.EngineWorker(comando)
    t0 = time.time()
    doc = trab.evaluate(prob.worker_line(v))
    dt = time.time() - t0
    trab.close()
    if doc.get("status") != "ok" or "observations" not in doc:
        raise SystemExit("motor falhou ou rota de reuso nao acionada: %s"
                         % str(doc)[:300])
    return {o["id"]: np.asarray(o["detected_count_rate"], float)
            for o in doc["observations"]}, dt


def estrutura_do_fundo(prob):
    """Os componentes, seus bins, seus canais e suas formas."""
    desl = np.asarray(prob.fundo_deslocamento, int)
    a = np.asarray(prob.fundo_pesos, float)
    n = np.asarray(prob.fundo_contagens, float)
    w = np.asarray(prob.fundo_largura_do_canal, float)
    canal = np.asarray(prob.fundo_canal_do_termo, int)
    forma = n + prob.fundo_a0 * w
    bin_do_termo = np.zeros(a.size, int)
    for ie in range(prob.energy_bins):
        bin_do_termo[desl[ie]:desl[ie + 1]] = ie
    # canais partidos: os que aparecem em mais de um bin
    unicos, inverso, contas = np.unique(canal, return_inverse=True,
                                        return_counts=True)
    partidos = unicos[contas > 1]
    peso_partido = float(a[np.isin(canal, partidos)].sum())
    return {"desl": desl, "a": a, "n": n, "w": w, "canal": canal,
            "forma": forma, "bin": bin_do_termo, "inverso": inverso,
            "n_canais": unicos.size, "n_componentes": a.size,
            "partidos": partidos, "peso_relativo_dos_partidos":
            peso_partido / float(a.sum()) if a.sum() else 0.0}


def ll_por_bin(S, n_obs, passo, bins_ok, b):
    """soma_e ll_e(b_e) para um lote de b: b tem forma (K, energias)."""
    # S (fases, energias), n_obs (fases, energias), passo (fases,), b (K, E)
    So = S[:, bins_ok]                      # (fases, E')
    no = n_obs[:, bins_ok]                  # (fases, E')
    bo = b[:, bins_ok]                      # (K, E')
    mu = (So[None, :, :] + bo[:, None, :]) * passo[None, :, None]
    np.maximum(mu, 1.0e-300, out=mu)
    return (no[None, :, :] * np.log(mu) - mu).sum(axis=(1, 2))   # (K,)


def monta_b(est, lam_por_componente, energy_bins):
    """b_e = soma_{j em e} a_j lambda_j, para um lote de lambdas (K, J)."""
    K = lam_por_componente.shape[0]
    contrib = lam_por_componente * est["a"][None, :]
    b = np.zeros((K, energy_bins))
    np.add.at(b.T, est["bin"], contrib.T)
    return b


def mc_de_uma_obs(prob, S, est, amostras, bloco, rng):
    """Os dois lados, com variaveis comuns, em blocos para nao estourar memoria."""
    E = prob.energy_bins
    bins_ok = np.asarray(prob._bins_de_energia_ajustados, int)
    n_obs = np.asarray(prob.observed, float).reshape(prob.phase_bins, E)
    t = np.asarray(prob.tempo_representado(), float)
    passo = prob.energy_width * t
    forma = est["forma"]
    J = forma.size
    inv = est["inverso"]              # componente -> indice do canal unico
    C = est["n_canais"]
    # forma por CANAL: identica nos componentes do mesmo canal, por construcao
    forma_canal = np.zeros(C)
    forma_canal[inv] = forma
    conferencia = float(np.abs(forma - forma_canal[inv]).max())
    lls_i, lls_c = [], []
    feitas = 0
    while feitas < amostras:
        K = min(bloco, amostras - feitas)
        # lambda por CANAL, o lado certo
        lam_c = rng.standard_gamma(forma_canal, size=(K, C))
        lam_comp_comp = lam_c[:, inv]                 # compartilhado
        # lado independente: MESMOS sorteios, menos nos componentes partidos,
        # que recebem sorteio proprio. Para o canal com p componentes, o
        # primeiro componente herda lam_c e os outros sao novos.
        lam_comp_ind = lam_comp_comp.copy()
        primeiro = np.zeros(C, int)
        vistos = np.zeros(C, bool)
        extra = []
        for j in range(J):
            c = inv[j]
            if not vistos[c]:
                vistos[c] = True
                primeiro[c] = j
            else:
                extra.append(j)
        if extra:
            extra = np.asarray(extra, int)
            lam_comp_ind[:, extra] = rng.standard_gamma(
                forma[extra], size=(K, extra.size))
        b_c = monta_b(est, lam_comp_comp, E)
        b_i = monta_b(est, lam_comp_ind, E)
        lls_c.append(ll_por_bin(S, n_obs, passo, bins_ok, b_c))
        lls_i.append(ll_por_bin(S, n_obs, passo, bins_ok, b_i))
        feitas += K
    return (np.concatenate(lls_i), np.concatenate(lls_c), conferencia)


def log_media_exp(x, pico=None):
    p = float(x.max()) if pico is None else pico
    return p + math.log(float(np.exp(x - p).mean()))


def ess(x, pico):
    p = np.exp(x - pico)
    s = p.sum()
    return float(s * s / (p * p).sum()) if s > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amostras", type=int, default=200000)
    ap.add_argument("--bloco", type=int, default=2000)
    ap.add_argument("--bandas", type=int, default=120)
    ap.add_argument("--refino", type=int, default=8)
    ap.add_argument("--vetores", nargs="+",
                    default=["A", "B colado", "B aberto"])
    ap.add_argument("--semente", type=int, default=20261010)
    ap.add_argument("--saida", default=str(RAIZ / "docs/correlaciona_fundo.json"))
    a = ap.parse_args()

    obss = {o["obsid"]: o for o in PO.prepara()}
    ordem = sorted(obss)
    meta, ev = m.read_events(RAIZ / "dados/eventos.csv")
    probs = {}
    for obsid in ordem:
        base = PO.monta_pedido(obss[obsid], a.bandas, a.refino)
        base["outputPrecision"] = 17
        base["marginalizeBackground"] = True
        p = m.FitProblem(base, ev, meta, None)
        p.prepara_fundo_marginal()
        probs[obsid] = p

    rel = {"o_que_e": "a correlacao entre bins que o estimador por sobreposicao "
                      "cria, e seu custo em ln L",
           "data": "2026-10-10",
           "pedido_do_revisor": (
             "um canal dividido entre dois bins cria dependencia; suas contagens "
             "nao devem gerar duas medicoes independentes. Preservar o latente "
             "compartilhado ou demonstrar que ignorar essa correlacao cabe no "
             "orcamento"),
           "o_que_a_algebra_ja_diz": (
             "a marginal de cada b_e e IDENTICA nos dois lados, porque "
             "Var(b_e) = soma a_j^2 (N_j + a0 w_j) nao depende de os lambda "
             "serem compartilhados; o que o alvo poe em zero e a covariancia "
             "entre bins vizinhos"),
           "estimador": ("Monte Carlo com Gammas exatas e VARIAVEIS COMUNS: so "
                         "os componentes partidos recebem sorteio diferente"),
           "configuracao": {"bandas": a.bandas, "refino": a.refino,
                            "amostras": a.amostras, "semente": a.semente,
                            "resolucao_efetiva_us": 0.0},
           "estrutura": {}, "por_vetor": {}}

    ests = {}
    for obsid in ordem:
        est = estrutura_do_fundo(probs[obsid])
        ests[obsid] = est
        rel["estrutura"][obsid] = {
            "canais": int(est["n_canais"]), "componentes": int(est["n_componentes"]),
            "canais_partidos": int(est["partidos"].size),
            "peso_relativo_dos_partidos": est["peso_relativo_dos_partidos"]}
    e0 = rel["estrutura"][ordem[0]]
    print("estrutura (%s): %d canais em %d componentes, %d partidos, "
          "%.2f por cento do peso" % (ordem[0], e0["canais"], e0["componentes"],
                                      e0["canais_partidos"],
                                      100 * e0["peso_relativo_dos_partidos"]))

    rng = np.random.default_rng(a.semente)
    for vetor in a.vetores:
        print("\n=== vetor %s ===" % vetor, flush=True)
        grades, dt = grades_conjuntas(obss, ordem, a.bandas, a.refino, vetor,
                                      meta, ev)
        soma_i = soma_c = 0.0
        soma_quad = 0.0
        linhas = []
        for obsid in ordem:
            p, est = probs[obsid], ests[obsid]
            S = grades[obsid]
            lli, llc, conf = mc_de_uma_obs(p, S, est, a.amostras, a.bloco, rng)
            pico = float(max(lli.max(), llc.max()))
            vi, vc = log_media_exp(lli, pico), log_media_exp(llc, pico)
            # o lado independente tem valor por QUADRATURA dentro do alvo
            quad = float(p.score_document({"status": "ok",
                                           "detected_count_rate": S.tolist()}))
            # erro de Monte Carlo da DIFERENCA, por reamostragem em blocos
            nb = 20
            corte = (lli.size // nb) * nb
            di = lli[:corte].reshape(nb, -1)
            dc = llc[:corte].reshape(nb, -1)
            difs = [log_media_exp(dc[k], pico) - log_media_exp(di[k], pico)
                    for k in range(nb)]
            erro = float(np.std(difs, ddof=1) / math.sqrt(nb))
            linha = {"obsid": obsid,
                     "ln_independente_mc": vi, "ln_compartilhado_mc": vc,
                     "diferenca": vc - vi, "erro_de_monte_carlo": erro,
                     "ln_independente_quadratura": quad,
                     "mc_contra_quadratura": vi - quad,
                     "ess_independente": ess(lli, pico),
                     "ess_compartilhado": ess(llc, pico),
                     "conferencia_forma_por_canal": conf}
            linhas.append(linha)
            soma_i += vi
            soma_c += vc
            soma_quad += quad
            print("  %-12s d(ln L) = %+.5f +- %.5f   mc-quad %+.4f  ESS %6.0f/%6.0f"
                  % (obsid, vc - vi, erro, vi - quad,
                     linha["ess_independente"], linha["ess_compartilhado"]),
                  flush=True)
        rel["por_vetor"][vetor] = {
            "soma_independente_mc": soma_i, "soma_compartilhado_mc": soma_c,
            "diferenca_da_soma": soma_c - soma_i,
            "soma_independente_quadratura": soma_quad,
            "segundos_do_motor": dt, "por_observacao": linhas}
        print("  SOMA das treze: independente %.4f  compartilhado %.4f  "
              "diferenca %+.5f" % (soma_i, soma_c, soma_c - soma_i))

    # o que decide: o CONTRASTE entre vetores, dos dois lados
    vs = [v for v in a.vetores if v in rel["por_vetor"]]
    if len(vs) > 1:
        base = vs[0]
        rel["contrastes"] = {}
        print("\n=== contrastes contra %s, que e o que move conclusao ===" % base)
        for v in vs[1:]:
            ci = (rel["por_vetor"][v]["soma_independente_mc"]
                  - rel["por_vetor"][base]["soma_independente_mc"])
            cc = (rel["por_vetor"][v]["soma_compartilhado_mc"]
                  - rel["por_vetor"][base]["soma_compartilhado_mc"])
            rel["contrastes"]["%s - %s" % (v, base)] = {
                "independente": ci, "compartilhado": cc, "mudanca": cc - ci}
            print("  %-22s independente %+.4f   compartilhado %+.4f   "
                  "muda %+.5f" % (v + " - " + base, ci, cc, cc - ci))
    Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print("\ngravado em %s" % a.saida)


if __name__ == "__main__":
    main()
