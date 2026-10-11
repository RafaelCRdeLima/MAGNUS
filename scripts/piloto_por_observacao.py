#!/usr/bin/env python3
"""Piloto do alvo por observacao: amostras efetivas por hora, por par (T, W).

O revisor, 11/10/2026: "pilotos MCMC podem ser preparados sem exigir que todos
os maximos sejam conhecidos exatamente", e antes disso: a escolha de threads e
workers para a producao vem de AMOSTRAS EFETIVAS POR HORA, nao de iteracoes por
hora.

O QUE ESTE PILOTO MEDE, e so isso:
  o ESS por hora de z e das funcoes cientificas relevantes, massa, raio e M/R,
    que e o que escolhe o par (T, W): a producao existe para medir essas
    quantidades, e nao os parametros de amostragem;
  o PIOR parametro como diagnostico COMPLEMENTAR, e junto dele a CORRELACAO
    dele com z. A ressalva do revisor em 11/10/2026: um parametro lento pode
    afetar z indiretamente, por correlacao ou por transicao entre regioes.
    Correlacao LINEAR pequena nesta janela NAO garante que ele seja
    irrelevante: ele pode controlar transicoes entre regioes que uma cadeia
    curta nao visita, e nesse caso a correlacao dentro de um modo nao diz nada
    sobre o efeito de mudar de modo;
  e a estabilidade da estimativa, entre janelas de Sokal e entre metades da
    cadeia.

O QUE ELE NAO FAZ: nao produz intervalos, nao decide maximo, nao substitui a
producao.

SOBRE O LIMIAR DE 50 TAU, corrigido em 11/10/2026. Eu havia escrito que abaixo
dele o ESS por hora seria um LIMITE SUPERIOR. Nao e: abaixo de 50 tau a
estimativa fica potencialmente INSTAVEL e OTIMISTA, sem garantia unilateral. O
estimador pode errar nos dois sentidos, e nada assegura que o valor verdadeiro
esteja abaixo do medido. O relatorio passou a dizer isso, e a trazer dois
indicios de instabilidade em vez de um rotulo: o tau com duas constantes de
janela, c = 5 e c = 10, e o tau das duas metades da cadeia.

E a validacao do estimador tambem foi rotulada: o teste contra AR(1) com tau
teorico 9 valida AQUELE regime, exponencial e estacionario, e nao todos. Cadeia
com cauda longa, multimodalidade ou deriva nao esta coberta por ele.

O pulso posterior fica DESLIGADO: no alvo por observacao ha treze previsoes e
portanto treze perfis, e escolher qual registrar e decisao cientifica. Um piloto
de desempenho nao precisa das bandas de credibilidade do pulso.

Uso: python3 scripts/piloto_por_observacao.py --pares 2:32 4:16 --iteracoes 60
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import mcmc_fit as m  # noqa: E402
import por_observacao as PO  # noqa: E402
import alvo_por_observacao as APO  # noqa: E402
import revalida_ponto as RP  # noqa: E402


def tau_integrado(cadeia, c=5.0):
    """Tempo de autocorrelacao integrado, com janela automatica a Sokal.

    `cadeia` tem forma (passos, caminhantes). A funcao de autocorrelacao e
    estimada pela media das cadeias, via FFT, e a janela e o menor M com
    M >= c * tau(M). Devolve tambem a janela, porque um tau com janela igual ao
    comprimento da cadeia nao e estimativa, e saturacao.
    """
    x = np.asarray(cadeia, float)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    if n < 4:
        return float("nan"), 0
    f = np.zeros(n)
    for k in range(x.shape[1]):
        y = x[:, k] - x[:, k].mean()
        N = 1
        while N < 2 * n:
            N *= 2
        F = np.fft.fft(y, n=N)
        acf = np.fft.ifft(F * np.conjugate(F))[:n].real
        if acf[0] > 0:
            f += acf / acf[0]
    f /= x.shape[1]
    taus = 2.0 * np.cumsum(f) - 1.0
    janela = n - 1
    for M in range(1, n):
        if M >= c * taus[M]:
            janela = M
            break
    return float(taus[janela]), int(janela)


def cientificas(traj, nomes):
    """As quantidades que a producao existe para medir, das trajetorias.

    z, a massa, o raio e M/R. Sao FUNCOES dos parametros amostrados, e e o ESS
    delas que decide o par (threads, workers): um parametro de amostragem que
    mistura devagar sem afetar z nao e motivo para trocar de arranjo.
    """
    iu, ir = nomes.index("compactness"), nomes.index("radius")
    u = traj[:, :, iu]
    R = traj[:, :, ir]
    massa = u * R / (2.0 * m._GM_SUN_C2_KM)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = 1.0 / np.sqrt(np.maximum(1.0 - u, 1.0e-12)) - 1.0
    return {"z": z, "massa": massa, "raio": R, "MsobreR": massa / np.maximum(R, 1e-12)}


def deriva(cad):
    """Deriva da cadeia: a media se move entre o inicio e o fim?

    Comparacao a Geweke, entre o primeiro decimo e o ultimo meio, em unidades do
    erro padrao de cada trecho corrigido pelo tau. Nao e teste de hipotese com
    nivel declarado: e indicio. |escore| grande diz que a cadeia ainda esta
    andando, e nesse caso o tau e o ESS descrevem uma distribuicao que a cadeia
    nao esta amostrando, e sim atravessando.
    """
    x = np.asarray(cad, float)
    n = x.shape[0]
    if n < 20:
        return {"escore": None, "motivo": "cadeia curta demais para a comparacao"}
    a_ = x[: max(2, n // 10)]
    b_ = x[n // 2:]
    ta, _ = tau_integrado(a_)
    tb, _ = tau_integrado(b_)
    def _var(y, t):
        if not np.isfinite(t) or t <= 0:
            t = 1.0
        n_ef = max(y.size / t, 1.0)
        return float(y.var(ddof=1) / n_ef) if y.size > 1 else float("inf")
    va, vb = _var(a_.ravel(), ta), _var(b_.ravel(), tb)
    denom = (va + vb) ** 0.5
    esc = float((a_.mean() - b_.mean()) / denom) if denom > 0 else None
    return {"escore": esc, "tau_inicio": ta, "tau_fim": tb,
            "leitura": ("|escore| acima de cerca de 2 e indicio de deriva, nao "
                        "prova. Com deriva, tau e ESS descrevem uma "
                        "distribuicao que a cadeia atravessa e nao amostra")}


def estabilidade(cad):
    """Dois indicios de instabilidade do tau, em vez de um rotulo.

    O tau com duas constantes de janela, c = 5 e c = 10, e o tau das duas
    metades da cadeia. Discordancia grande entre eles diz que a estimativa nao
    esta estavel, sem supor em que direcao ela erra.
    """
    t5, j5 = tau_integrado(cad, c=5.0)
    t10, j10 = tau_integrado(cad, c=10.0)
    meio = cad.shape[0] // 2
    ta, _ = tau_integrado(cad[:meio]) if meio >= 4 else (float("nan"), 0)
    tb, _ = tau_integrado(cad[meio:]) if meio >= 4 else (float("nan"), 0)
    razoes = [x for x in (t10 / t5 if t5 else None,
                          tb / ta if ta else None) if x and np.isfinite(x)]
    return {"tau_c5": t5, "janela_c5": j5, "tau_c10": t10, "janela_c10": j10,
            "tau_primeira_metade": ta, "tau_segunda_metade": tb,
            "pior_razao": max(razoes + [1.0 / r for r in razoes]) if razoes else None}


def um_par(T, W, iteracoes, ponto, cfg, semente, excluir=(), walkers=None):
    """Um piloto com T threads por avaliacao e W trabalhadores simultaneos.

    CAMINHANTES E TRABALHADORES SAO COISAS DIFERENTES, e a primeira versao disto
    usava W para os dois. O numero de trabalhadores e imposto pelos NUCLEOS; o
    de caminhantes, pela DIMENSAO e pela implementacao.

    A CONTAGEM, conferida e nao suposta, porque eu a errei duas vezes. O vetor
    tem 19 parametros, mas `phaseOffset` e `lineEnergy` estao congelados: a
    dimensao LIVRE e 17. E a regra nao e "dois por dimensao", que daria 34:
    `required_walkers` implementa 2*(2d+1), ou 4d+2, que da 70.

    ESTATUTO DESSE 4d+2, restringido em 11/10/2026: ele e a REGRA DESTE PROJETO,
    vinda da constante WALKERS_PER_DIMENSION_MINIMUM = 2 escolhida aqui, e NAO
    um minimo universal do stretch move. O requisito do metodo e mais fraco:
    cada metade precisa alcancar o espaco afim, o que pede d+1 por metade, e os
    35 que a regra entrega deixam folga de 17 sobre isso. Conferir o rank das
    duas metades e guarda CONSERVADORA util, nao um teorema.

    Conferido empiricamente num enxame como o que este script cria: rank afim 17
    de 17 no enxame todo e em cada metade. Uma regra verbal nao bastaria.
    """
    alvo = APO.AlvoPorObservacao(refino=8, marginal=True, threads=T,
                                 excluir=excluir, **cfg)
    if list(alvo.names) != list(ponto["nomes"]):
        raise SystemExit("ordem dos nomes nao bate com o problema")
    livres = sum(1 for n in alvo.names if n not in alvo.frozen)
    minimo = m.required_walkers(livres)
    n_walk = int(walkers or minimo)
    if n_walk < minimo:
        raise SystemExit(
            "%d caminhantes para %d dimensoes livres: o minimo implementado e "
            "%d, que e 2*(2d+1) e nao 2d, porque cada metade do ensemble "
            "red/blue tem de gerar o espaco. Com menos, o tau nao significa "
            "nada" % (n_walk, livres, minimo))
    # NAO basta a regra: confere o rank afim das duas metades do enxame real.
    def _confere_rank(estados_):
        pares, impares = estados_[0::2], estados_[1::2]
        for nome_, metade in (("par", pares), ("impar", impares)):
            r = m.affine_rank(metade)
            if r < livres:
                raise SystemExit(
                    "a metade %s do enxame tem rank afim %d para %d dimensoes "
                    "livres: a proposta stretch nao alcanca o espaco todo"
                    % (nome_, r, livres))
        return {"enxame": m.affine_rank(estados_),
                "metade_par": m.affine_rank(pares),
                "metade_impar": m.affine_rank(impares),
                "dimensoes_livres": livres}
    pool = m.WorkerPool(alvo, W)
    rng = __import__("random").Random(semente)
    # O enxame nasce em torno do melhor ponto, com o passo do proprio problema:
    # um piloto que nasce da priori gastaria o orcamento so para chegar perto.
    estados = []
    for _ in range(n_walk):
        v = list(ponto["vetor"])
        for _ in range(200):
            cand = [x + rng.gauss(0, 0.05 * p) for x, p in zip(v, alvo.steps)]
            if alvo.in_prior(cand):
                v = cand
                break
        estados.append(v)
    rank = _confere_rank(estados)
    t0 = time.time()
    r = m.run_ensemble(alvo, pool, n_walk, iteracoes, 0, semente,
                       resume_states=estados, com_pulso=False)
    dt = time.time() - t0
    for w in pool.workers:
        w.close()
    traj = np.asarray(r["trajectories"], float)      # (caminhantes, passos, dim)
    livres = [i for i, n in enumerate(alvo.names) if n not in alvo.frozen]
    taus, janelas = {}, {}
    for i in livres:
        cad = traj[:, :, i].T                        # (passos, caminhantes)
        t, j = tau_integrado(cad)
        taus[alvo.names[i]] = t
        janelas[alvo.names[i]] = j
    finitos = {k: v for k, v in taus.items() if np.isfinite(v) and v > 0}
    pior = max(finitos, key=lambda k: finitos[k]) if finitos else None
    n_total = traj.shape[0] * traj.shape[1]
    ess = n_total / finitos[pior] if pior else float("nan")
    passos = traj.shape[1]
    # AS CIENTIFICAS, que e o que decide o par
    cie = cientificas(traj, list(alvo.names))
    por_cientifica = {}
    for nome, arr in cie.items():
        cad = arr.T                                  # (passos, caminhantes)
        t, j = tau_integrado(cad)
        est = estabilidade(cad)
        der = deriva(cad)
        e = n_total / t if (t and np.isfinite(t) and t > 0) else float("nan")
        por_cientifica[nome] = {
            "tau": t, "janela": j, "ess": e,
            "ess_por_hora": e * 3600.0 / dt if dt else None,
            "passos_sobre_tau": passos / t if (t and np.isfinite(t) and t > 0) else None,
            "estabilidade": est,
            "deriva": der,
        }
    z = por_cientifica.get("z", {})
    poucos_tau = bool(z.get("passos_sobre_tau") and z["passos_sobre_tau"] < 50)
    # A RESSALVA: o parametro lento pode agir em z por correlacao.
    acoplamento = None
    if pior is not None:
        ip = list(alvo.names).index(pior)
        cp = traj[:, :, ip].ravel()
        cz = cie["z"].ravel()
        if cp.std() > 0 and cz.std() > 0:
            r = float(np.corrcoef(cp, cz)[0, 1])
            acoplamento = {
                "parametro": pior, "correlacao_com_z": r,
                "leitura": ("|r| alto significa que exploracao incompleta deste "
                            "parametro e exploracao incompleta de z, mesmo que o "
                            "tau de z pareca bom. |r| baixo nesta janela nao "
                            "exclui acoplamento por transicao entre regioes, que "
                            "uma cadeia curta nao visita"),
                "merece_atencao": bool(abs(r) > 0.3 or (z.get("passos_sobre_tau") or 99) < 50),
            }
    return {
        "threads": T, "workers": W, "caminhantes": n_walk,
        "minimo_de_caminhantes": minimo,
        "nucleos_usados": T * W, "iteracoes_pedidas": iteracoes,
        "rank_afim_conferido": rank,
        "passos_registrados": int(passos), "amostras_totais": int(n_total),
        "segundos": dt, "segundos_por_iteracao": dt / max(passos, 1),
        "aceitacao": float(np.mean(r.get("acceptance", r.get("aceitacao", [np.nan])))
                           if isinstance(r.get("acceptance", r.get("aceitacao")), (list, tuple))
                           else float("nan")),
        "por_cientifica": por_cientifica,
        "ess_por_hora_de_z": z.get("ess_por_hora"),
        "diagnostico_complementar": {
            "tau_por_parametro": taus, "janela_por_parametro": janelas,
            "pior_parametro": pior, "pior_tau": finitos.get(pior),
            "ess_pelo_pior": ess,
            "ess_por_hora_pelo_pior": ess * 3600.0 / dt if dt else None,
            "para_que_serve": ("indicar parametro que mistura devagar. Um "
                               "parametro lento que nao afeta z nao e motivo "
                               "para trocar de par, mas ver o acoplamento"),
            "acoplamento_com_z": acoplamento,
        },
        "passos_sobre_tau_de_z": z.get("passos_sobre_tau"),
        "ressalva": (None if not poucos_tau else
                     "passos < 50 tau em z: a estimativa fica potencialmente "
                     "INSTAVEL e OTIMISTA, sem garantia unilateral. Nao e um "
                     "limite superior: o estimador pode errar nos dois "
                     "sentidos. Confira a estabilidade entre janelas e entre "
                     "metades, e repita com outra semente"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ponto", required=True, help="saida de reotimizacao")
    ap.add_argument("--pares", nargs="+", default=["2:32", "4:16"])
    ap.add_argument("--iteracoes", type=int, default=60)
    ap.add_argument("--caminhantes", type=int, default=None,
                    help="padrao: o minimo imposto pela dimensao, 2 por "
                         "parametro livre. NAO e o numero de trabalhadores")
    ap.add_argument("--sementes", type=int, nargs="+", default=[20261011, 7703],
                    help="PILOTOS INDEPENDENTES, um por semente, com o mesmo "
                         "alvo, a mesma duracao e o mesmo descarte. A dispersao "
                         "do ESS por hora entre eles e a medida de estabilidade "
                         "que decide, junto do valor")
    ap.add_argument("--configuracao", default="grosso", choices=["grosso", "fino"])
    ap.add_argument("--sem-1001", action="store_true", dest="sem_1001")
    ap.add_argument("--saida", default=str(RAIZ / "docs/piloto_por_observacao.json"))
    a = ap.parse_args()
    cfg = RP.GROSSO if a.configuracao == "grosso" else RP.FINO
    ponto = RP.le_ponto(a.ponto)
    excluir = ("0402851001",) if a.sem_1001 else ()
    rel = {"o_que_e": "piloto do alvo por observacao: amostras efetivas por hora",
           "data": time.strftime("%Y-%m-%d"),
           "metrica": ("ESS pelo PIOR parametro dividido pelo tempo de parede. "
                       "Iteracoes por hora nao serve: cadeias que misturam "
                       "devagar entregam menos amostra por iteracao"),
           "ponto_de_partida": {"arquivo": a.ponto, "rotulo": ponto["rotulo"],
                                "estatuto": ("melhor ponto encontrado; o piloto "
                                             "NAO exige que o maximo seja "
                                             "conhecido exatamente")},
           "configuracao": a.configuracao, "excluidas": list(excluir),
           "pulso_posterior": "DESLIGADO, ver o cabecalho do script",
           "pares": []}
    rel["desenho_dos_pilotos"] = {
        "alvo": "o mesmo em todos: %s observacoes%s" % (
            13 - len(excluir), " (sem %s)" % ", ".join(excluir) if excluir else ""),
        "duracao": "%d iteracoes em todos" % a.iteracoes,
        "descarte": "ZERO: burn_in = 0, porque o enxame nasce em torno do melhor "
                    "ponto encontrado e o piloto mede mistura local, nao "
                    "aquecimento desde a priori",
        "sementes": list(a.sementes),
        "por_que_independentes": ("comparar pilotos de semente diferente e o que "
                                  "separa estabilidade de sorte numa cadeia "
                                  "curta"),
    }
    for par in a.pares:
        T, W = (int(x) for x in par.split(":"))
        for semente in a.sementes:
            print("=== T=%d W=%d, %d iteracoes, semente %d ==="
                  % (T, W, a.iteracoes, semente), flush=True)
            r = um_par(T, W, a.iteracoes, ponto, cfg, semente, excluir,
                       walkers=a.caminhantes)
            r["semente"] = semente
            rel["pares"].append(r)
        print("  %.1f s, %d passos, pior tau %s em %s"
              % (r["segundos"], r["passos_registrados"],
                 ("%.2f" % r["pior_tau"]) if r["pior_tau"] else "n/d",
                 r["pior_parametro"]), flush=True)
        for nome in ("z", "massa", "raio", "MsobreR"):
            c = r["por_cientifica"].get(nome, {})
            if c:
                print("  %-9s tau %7.2f  ESS %8.1f  ESS/h %9.1f  N/tau %6.1f  "
                      "pior razao de janela %s"
                      % (nome, c["tau"], c["ess"], c["ess_por_hora"] or 0.0,
                         c["passos_sobre_tau"] or 0.0,
                         ("%.2f" % c["estabilidade"]["pior_razao"])
                         if c["estabilidade"]["pior_razao"] else "n/d"), flush=True)
        if r["ressalva"]:
            print("  RESSALVA: %s" % r["ressalva"][:110], flush=True)
    # AGREGA POR PAR, entre as sementes: valor E estabilidade, nao so o maximo.
    por_par = {}
    for p in rel["pares"]:
        if p["ess_por_hora_de_z"]:
            por_par.setdefault((p["threads"], p["workers"]), []).append(p)
    rel["resumo_por_par"] = []
    print("\n=== resumo por par: valor E estabilidade entre pilotos ===")
    for (T, W), ps in sorted(por_par.items()):
        v = np.array([x["ess_por_hora_de_z"] for x in ps], float)
        disp = float(v.max() / v.min()) if v.min() > 0 and v.size > 1 else None
        res = {"threads": T, "workers": W, "pilotos": len(ps),
               "ess_por_hora_de_z": {"media": float(v.mean()),
                                     "minimo": float(v.min()),
                                     "maximo": float(v.max())},
               "dispersao_entre_pilotos": disp,
               "alguma_ressalva": [x["ressalva"] for x in ps if x["ressalva"]],
               "acoplamento": [x["diagnostico_complementar"].get("acoplamento_com_z")
                               for x in ps]}
        rel["resumo_por_par"].append(res)
        print("  T=%-3d W=%-3d  ESS/h de z: media %9.1f  faixa %9.1f a %9.1f  "
              "dispersao %s" % (T, W, v.mean(), v.min(), v.max(),
                                ("%.2fx" % disp) if disp else "n/d"), flush=True)
    # VEREDITO que pode ser INCONCLUSIVO, e isso nao e falha do piloto.
    todas = rel["pares"]
    derivou = [p for p in todas
               if (p["por_cientifica"].get("z", {}).get("deriva", {}).get("escore")
                   or 0) and abs(p["por_cientifica"]["z"]["deriva"]["escore"]) > 2]
    instavel = [p for p in todas
                if (p["por_cientifica"].get("z", {}).get("estabilidade", {})
                    .get("pior_razao") or 1) > 1.5]
    curtas = [p for p in todas if p["ressalva"]]
    rel["veredito"] = {
        "pares_com_indicio_de_deriva_em_z": len(derivou),
        "pares_com_tau_instavel_em_z": len(instavel),
        "pares_com_passos_abaixo_de_50_tau": len(curtas),
        "conclusivo": not (derivou or instavel or curtas),
        "se_inconclusivo": (
            "cadeias curtas podem dar apenas diagnostico inconclusivo, e nesse "
            "caso NAO se escolhe a configuracao pelo maior ESS estimado: "
            "aumenta-se a duracao do piloto ou se aceita decidir por custo "
            "medido, declarando que a mistura nao foi caracterizada"),
    }
    rel["o_que_reportar_ao_fechar"] = [
        "estabilidade entre replicacoes independentes",
        "deriva de z e das funcoes cientificas",
        "ESS por hora dessas grandezas",
        "sensibilidade a janela de autocorrelacao",
    ]
    rel["como_escolher"] = (
        "pelo ESS por hora das grandezas CIENTIFICAS e pela ESTABILIDADE entre "
        "pilotos independentes, nao pelo maior numero estimado. Um par com media "
        "alta e dispersao grande entre sementes nao esta medido, esta "
        "flutuando. A escolha nao e automatica e nao e feita por este script")
    Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False,
                                        default=float) + "\n")
    print("\ngravado em %s" % a.saida)


if __name__ == "__main__":
    main()
