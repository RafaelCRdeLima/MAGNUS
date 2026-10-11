#!/usr/bin/env python3
"""Coadição contra previsões por observação, em DIFERENÇAS de ln L.

O revisor, em 07/10/2026: "A condição de equivalência da coadição está correta
se os pesos forem conhecidos e independentes dos parâmetros, com o mesmo
tratamento de fundo. Para comparar as formulações, use diferenças de ln L entre
vetores ou remova explicitamente a constante de agregação; valores absolutos não
bastam. As avaliações dos três vetores serão um diagnóstico inicial, não uma
validação da equivalência em toda a posterior."

POR QUE DIFERENÇAS. A coadição soma contagens de 13 observações numa grade; a
forma por observação soma 13 verossimilhanças. Mesmo que as duas descrevessem a
mesma física, os ln L absolutos diferem por uma constante de agregação,

    soma_o n_o ln w_o - (soma_o n_o) ln(soma_o w_o),

que não depende dos parâmetros. Logo comparar valores absolutos não mede nada, e
comparar DIFERENÇAS entre vetores remove a constante sem precisar calculá-la.

O QUE ESTÁ IGUAL E O QUE NÃO ESTÁ nas duas formulações aqui:

  IGUAL: as contagens (nenhuma é deslocada nem interpolada, as 13 somam 329972
  nas duas), a exposição por fase medida nos quadros, o vetor de parâmetros, as
  120 bandas, o refino 8.

  DIFERENTE, e é o que se mede: na coadição há UMA resposta e UMA resolução
  temporal para as 13; por observação há 13 respostas, duas resoluções e 13
  períodos, e o alinhamento entra como `magAzim + 11,25 . s_o` em vez de um
  `roll` das contagens.

  DIFERENTE, e DECLARADO como limitação: o alvo aqui é o de CONTROLE, com fundo
  fixo, e não o v2 com fundo marginalizado, porque o v2 precisa de `fundo_pesos`
  por observação e o produto por observação ainda não os traz. A coadição usa o
  fundo por área ponderado pela exposição; cada observação usa o seu próprio. A
  soma ponderada dos 13 é o fundo coadicionado, mas a verossimilhança não é
  linear no fundo, então sobra diferença, e ela entra na medida.

O QUE SE REPORTA, pedido pelo revisor em 07/10/2026: "reporte diferencas entre
vetores, discrepancias por fase/energia e tempo efetivo. Isso permitira decidir
se a aproximacao combinada e suficiente no conjunto testado."

E A REFERENCIA E A SOMA DAS PREVISOES INDIVIDUAIS, nao a saida da resposta
combinada: "A referencia e a soma das previsoes individuais, com cada exposicao,
ARF e RMF." No quadro coadicionado isso e

    referencia(p, e) = soma_o roll(mu_o, s_o)(p, e)

porque o bin q da observacao o aparece no bin q + s_o do coadicionado. Compara-se
contra `mu_coad(p, e)` da resposta combinada, celula a celula, e em seguida
marginalizado por fase e por energia.

FONTE E FUNDO SEPARADOS. As duas partes nao erram pelo mesmo motivo: a fonte erra
pela resposta e pela suavizacao, e o fundo erra porque o fundo coadicionado e uma
media ponderada pelas exposicoes TOTAIS, enquanto a soma individual pondera pela
exposicao POR FASE de cada observacao. Reportar so a soma esconderia qual das
duas domina.

Uso: python3 scripts/compara_formulacoes.py [--bandas 120]
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
import reotimiza as r  # noqa: E402
import por_observacao as PO  # noqa: E402

COMBINADO = PO.V3
#: A resposta combinada da RECONSTRUCAO, construida por
#: scripts/resposta_ponderada.py com procedencia completa. NAO uso o
#: xmm_epn_imaging_combinado13: ele e de 01/10, com procedencia nula, construido
#: das reducoes que eu mesmo invalidei; roda-lo contra as contagens do v3
#: misturaria dados da reconstrucao com uma resposta descartada.
ID_COMBINADO = "xmm_epn_coadd13_recon5906a57"


def perfil_combinado():
    perfis = json.loads(PO.MANIFESTO.read_text())["profiles"]
    p = [x for x in perfis if x["id"] == ID_COMBINADO]
    if not p:
        raise SystemExit(
            "o manifesto local nao tem %s; construa com\n"
            "  python3 scripts/resposta_ponderada.py --origem .../produtos "
            "--combinado %s --id %s" % (ID_COMBINADO, COMBINADO, ID_COMBINADO))
    p = p[0]
    return {"id": p["id"], "label": p["label"],
            "profile_file": str(PO.caminho_absoluto(p["profile_file"])),
            "response_file": str(PO.caminho_absoluto(p["response_file"])),
            "time_resolution_us": float(p["time_resolution_us"]),
            "dead_time_us": float(p.get("dead_time_us", 0.0)),
            "redistribution": "full_sparse_ogip_rmf"}


def partes(prob, doc):
    """mu da FONTE e mu do FUNDO, em contagens, separados.

    mu_fonte(p, e) = grade(p, e) . largura . T(p)
    mu_fundo(p, e) = B(e)        . largura . T(p)

    Separados porque as duas partes nao erram pelo mesmo motivo, e somadas elas
    escondem qual domina.
    """
    g = np.asarray(doc["detected_count_rate"], float)
    T = np.asarray(prob.tempo_representado(), float)[:, None]
    B = np.asarray(prob.background, float)[None, :]
    return g * prob.energy_width * T, np.broadcast_to(
        B, g.shape) * prob.energy_width * T, T[:, 0]


def avalia_coadicao(vetores, bandas, refino):
    """Um FitProblem na grade coadicionada, os vetores avaliados nele."""
    base = r.monta_pedido(PO.PEDIDO, bandas, refino, False)
    base["preparedDataNpz"] = str(COMBINADO)
    base["responseProfile"] = perfil_combinado()
    meta, eventos = m.read_events(RAIZ / "dados/eventos.csv")
    prob = m.FitProblem(base, eventos, meta, None)
    trab = m.EngineWorker(list(prob.worker_command()))
    saida, grades = {}, {}
    for nome, caminho, rotulo in vetores:
        v = PO.le_vetor(prob, caminho, rotulo)
        t0 = time.time()
        doc = trab.evaluate(prob.worker_line(v))
        dt = time.time() - t0
        if doc.get("status") != "ok":
            raise SystemExit("motor falhou na coadicao: %s" % doc)
        extra = float(prob.extra_log_prior(v))
        fonte, fundo, T = partes(prob, doc)
        grades[nome] = {"fonte": fonte, "fundo": fundo, "tempo": T}
        saida[nome] = {"ln_verossimilhanca": float(prob.score_document(doc)),
                       "segundos": dt, "prior_extra": extra}
        saida[nome]["ln_posterior"] = saida[nome]["ln_verossimilhanca"] + extra
        print("  coadicao %-10s ln posterior %14.4f  (%.1f s)"
              % (nome, saida[nome]["ln_posterior"], dt), flush=True)
    trab.close()
    return saida, grades, prob.names


def avalia_por_observacao(vetores, obss, bandas, refino):
    """13 FitProblems, um por observacao, com o offset no azimute."""
    saida = {nome: {"por_obs": {}, "ln_posterior": 0.0, "segundos": 0.0}
             for nome, _, _ in vetores}
    grades = {}
    for obsid in sorted(obss):
        obs = obss[obsid]
        prob, trab = PO.problema_de(obs, bandas, refino)
        ia = PO.indice_do_azimute(prob)
        s_o = 0 if obsid == PO.REFERENCIA else int(obs["desloc_v3_bins"])
        for nome, caminho, rotulo in vetores:
            v = PO.le_vetor(prob, caminho, rotulo)
            azim0 = v[ia]
            v[ia] = PO.azimute_com_offset(azim0, s_o)
            t0 = time.time()
            doc = trab.evaluate(prob.worker_line(v))
            dt = time.time() - t0
            if doc.get("status") != "ok":
                raise SystemExit("motor falhou em %s: %s" % (obsid, doc))
            extra = float(prob.extra_log_prior(v))
            lnl = float(prob.score_document(doc))
            fonte, fundo, T = partes(prob, doc)
            # TRANSPORTADO PARA O QUADRO COADICIONADO: o bin q desta observacao
            # aparece no bin q + s_o do coadicionado, logo roll(., s_o)
            grades.setdefault(nome, {})[obsid] = {
                "fonte": np.roll(fonte, s_o, axis=0),
                "fundo": np.roll(fundo, s_o, axis=0),
                "tempo": np.roll(T, s_o)}
            saida[nome]["por_obs"][obsid] = {
                "ln_posterior": lnl + extra, "ln_verossimilhanca": lnl,
                "s_o": s_o, "azimute": v[ia], "azimute_sem_offset": azim0,
                "segundos": dt, "prior_extra": extra}
            saida[nome]["segundos"] += dt
            print("  %-10s %-12s s=%2d azim %8.3f  ln posterior %13.4f  (%.1f s)"
                  % (nome, obsid, s_o, v[ia], lnl + extra, dt), flush=True)
        trab.close()
    # A SOMA. A priori entra UMA vez, nao 13: ela e sobre os parametros, nao
    # sobre os dados, e somar 13 copias multiplicaria a priori por 13.
    for nome in saida:
        por = saida[nome]["por_obs"]
        qualquer = next(iter(por.values()))
        ln_l = sum(d["ln_posterior"] - d["prior_extra"] for d in por.values())
        saida[nome]["ln_verossimilhanca"] = ln_l
        saida[nome]["ln_posterior"] = ln_l + qualquer["prior_extra"]
        saida[nome]["prior_extra"] = qualquer["prior_extra"]
    return saida, grades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandas", type=int, default=120)
    ap.add_argument("--refino", type=int, default=8)
    ap.add_argument("--saida", default=str(RAIZ / "docs/compara_formulacoes.json"))
    a = ap.parse_args()
    vetores = list(PO.VETORES)
    obss = {o["obsid"]: o for o in PO.prepara()}
    print("=== formulacao 1: coadicao, %s ===" % COMBINADO.name, flush=True)
    coad, gco, nomes = avalia_coadicao(vetores, a.bandas, a.refino)
    print("=== formulacao 2: 13 previsoes, offset no azimute ===", flush=True)
    poro, gpo = avalia_por_observacao(vetores, obss, a.bandas, a.refino)

    # ---- O OPERADOR COMBINADO CONTRA A SOMA DAS PREVISOES INDIVIDUAIS ----
    #
    # O revisor: "O espalhamento a 1 keV nao certifica a combinacao: confira o
    # operador completo em energia. A referencia e a soma das previsoes
    # individuais, com cada exposicao, ARF e RMF."
    print()
    print("=== o operador combinado contra a SOMA das previsoes individuais ===")
    operador = {}
    for nome, _, _ in vetores:
        soma_f = sum(g["fonte"] for g in gpo[nome].values())
        soma_b = sum(g["fundo"] for g in gpo[nome].values())
        soma_T = sum(g["tempo"] for g in gpo[nome].values())
        comb_f, comb_b = gco[nome]["fonte"], gco[nome]["fundo"]
        comb_T = gco[nome]["tempo"]

        def resumo(a, b, rotulo):
            """a = combinado, b = referencia. Relativo, e em sigmas de Poisson."""
            d = a - b
            rel = d / np.maximum(np.abs(b), 1e-300)
            sig = d / np.maximum(np.sqrt(np.maximum(b, 0.0)), 1e-300)
            ip, ie = np.unravel_index(int(np.argmax(np.abs(rel))), d.shape)
            return {"rotulo": rotulo,
                    "total_combinado": float(a.sum()),
                    "total_referencia": float(b.sum()),
                    "diferenca_do_total_relativa": float(a.sum() / b.sum() - 1.0),
                    "pior_celula_relativa": float(rel[ip, ie]),
                    "pior_celula": {"fase": int(ip), "energia": int(ie)},
                    "mediana_absoluta_relativa": float(np.median(np.abs(rel))),
                    "pior_celula_em_sigmas": float(np.max(np.abs(sig))),
                    "por_fase_relativa_max": float(np.max(np.abs(
                        a.sum(1) / np.maximum(b.sum(1), 1e-300) - 1.0))),
                    "por_energia_relativa_max": float(np.max(np.abs(
                        a.sum(0) / np.maximum(b.sum(0), 1e-300) - 1.0))),
                    "por_fase_relativa": (a.sum(1) / np.maximum(b.sum(1), 1e-300)
                                          - 1.0).tolist(),
                    "por_energia_relativa": (a.sum(0) / np.maximum(b.sum(0), 1e-300)
                                             - 1.0).tolist()}

        r_f = resumo(comb_f, soma_f, "fonte")
        r_b = resumo(comb_b, soma_b, "fundo")
        r_t = resumo(comb_f + comb_b, soma_f + soma_b, "fonte+fundo")
        tempo = {"combinado_total": float(comb_T.sum()),
                 "soma_individual_total": float(soma_T.sum()),
                 "razao_do_total": float(comb_T.sum() / soma_T.sum()),
                 "por_fase_relativa_max": float(np.max(np.abs(
                     comb_T / np.maximum(soma_T, 1e-300) - 1.0))),
                 "por_fase_relativa": (comb_T / np.maximum(soma_T, 1e-300)
                                       - 1.0).tolist()}
        operador[nome] = {"fonte": r_f, "fundo": r_b, "total": r_t,
                          "tempo_efetivo": tempo}
        print("  %s" % nome)
        for rr in (r_f, r_b, r_t):
            print("    %-12s total %+.3e  pior celula %+.3e (fase %2d, en %2d)  "
                  "mediana |rel| %.3e  pior %.2f sigma"
                  % (rr["rotulo"], rr["diferenca_do_total_relativa"],
                     rr["pior_celula_relativa"], rr["pior_celula"]["fase"],
                     rr["pior_celula"]["energia"],
                     rr["mediana_absoluta_relativa"],
                     rr["pior_celula_em_sigmas"]))
            print("    %-12s por fase max %+.3e | por energia max %+.3e"
                  % ("", rr["por_fase_relativa_max"],
                     rr["por_energia_relativa_max"]))
        print("    tempo efetivo: total %.1f contra %.1f s, razao %.9f, "
              "por fase max %+.3e"
              % (tempo["combinado_total"], tempo["soma_individual_total"],
                 tempo["razao_do_total"], tempo["por_fase_relativa_max"]))

    ref = vetores[0][0]
    print()
    print("=== DIFERENCAS de ln L contra %s, que e o que se compara ===" % ref)
    print("%-12s %16s %16s %14s" % ("vetor", "coadicao", "por observacao",
                                    "discordancia"))
    difs = []
    for nome, _, _ in vetores:
        dc = coad[nome]["ln_posterior"] - coad[ref]["ln_posterior"]
        dp = poro[nome]["ln_posterior"] - poro[ref]["ln_posterior"]
        difs.append({"vetor": nome, "diferenca_coadicao": dc,
                     "diferenca_por_observacao": dp, "discordancia": dp - dc})
        print("%-12s %16.4f %16.4f %14.4f" % (nome, dc, dp, dp - dc))
    pior = max(abs(d["discordancia"]) for d in difs)
    print()
    print("pior discordancia entre as formulacoes: %.4f em ln L" % pior)
    print("constante de agregacao, nao comparavel: coadicao %.1f, soma por "
          "observacao %.1f, diferenca %.1f"
          % (coad[ref]["ln_posterior"], poro[ref]["ln_posterior"],
             poro[ref]["ln_posterior"] - coad[ref]["ln_posterior"]))
    print("custo: coadicao %.1f s para 3 vetores, por observacao %.1f s"
          % (sum(coad[n]["segundos"] for n in coad),
             sum(poro[n]["segundos"] for n in poro)))

    saida = {
      "o_que_e": ("coadicao contra previsoes por observacao, comparadas em "
                  "DIFERENCAS de ln L entre vetores"),
      "por_que_diferencas": (
        "os ln L absolutos diferem por uma constante de agregacao que nao "
        "depende dos parametros; diferencas a removem sem calcula-la"),
      "bandas": a.bandas, "refino": a.refino,
      "alvo": ("CONTROLE, com fundo fixo. O v2 com fundo marginalizado precisa "
               "de fundo_pesos por observacao, que o produto por observacao "
               "ainda nao traz"),
      "o_que_esta_igual": ["as contagens, nenhuma deslocada nem interpolada",
                           "a exposicao por fase medida nos quadros",
                           "o vetor, as bandas e o refino"],
      "resposta_da_coadicao": ID_COMBINADO,
      "por_que_nao_o_combinado13_antigo": (
        "o xmm_epn_imaging_combinado13 e de 01/10, com procedencia nula, "
        "construido das reducoes que eu mesmo invalidei; usa-lo contra as "
        "contagens do v3 misturaria a reconstrucao com uma resposta descartada"),
      "a_resolucao_da_coadicao_e_convencao": (
        "66715,5 us, ponderada pela exposicao entre 73400 e 47700. Nao existe "
        "uma resolucao correta para a mistura, porque ela teria de ser a "
        "combinacao dos operadores de suavizacao; o custo medido dessa escolha e "
        "residuo de forma de 3,34e-02"),
      "o_que_difere": ["13 respostas contra 1", "2 resolucoes contra 1",
                       "13 periodos contra 1",
                       "alinhamento por magAzim + 11,25.s_o contra roll das "
                       "contagens",
                       "fundo proprio de cada observacao contra o ponderado"],
      "coadicao": coad, "por_observacao": poro,
      "o_operador_combinado_contra_a_soma_individual": operador,
      "a_referencia": ("a SOMA das previsoes individuais, cada uma com a sua "
                       "exposicao, o seu ARF e o seu RMF, transportada para o "
                       "quadro coadicionado por roll(., s_o)"),
      "caminhos_resolvidos": dict(PO.RESOLVIDOS),
      "diferencas": difs,
      "pior_discordancia": pior,
      "constante_de_agregacao_nao_comparavel": (
        poro[ref]["ln_posterior"] - coad[ref]["ln_posterior"]),
      "o_que_isto_NAO_valida": (
        "a equivalencia em toda a posterior. Sao tres vetores, um diagnostico "
        "inicial, como o revisor declarou"),
      "o_roll_e_so_para_deslocamento_INTEIRO": (
        "transportar as previsoes por roll(., s_o) e adequado para ESTE "
        "diagnostico, cujos deslocamentos sao inteiros porque e assim que a "
        "coadicao os mede. Ele NAO valida offsets sub-bin nem a incerteza deles: "
        "um deslocamento fracionario exigiria o operador continuo, que no modelo "
        "e magAzim, e a propagacao da incerteza do offset, que aqui nao e feita. "
        "A medida sub-bin que existe, de -0,280 +- 0,243 bins em "
        "docs/offset_relativo.json, e uma ESTIMATIVA do alinhamento, nao uma "
        "validacao do transporte em sub-bin"),
      "a_media_do_fundo_NAO_demonstra_o_alvo_MARGINALIZADO": {
        "o_ponto": (
          "esta comparacao mede a previsao MEDIA do fundo. Equivalencia da media "
          "nao demonstra equivalencia da verossimilhanca marginalizada, porque "
          "nela entram tambem as INCERTEZAS individuais de cada canal"),
        "e_ha_uma_diferenca_de_MODELO_e_nao_so_de_normalizacao": (
          "no alvo v2 coadicionado existe UM fundo latente por canal de energia, "
          "compartilhado pelas 32 fases, com pi_e construida da soma ponderada "
          "dos termos das 13 observacoes. Na forma por observacao o natural sao "
          "TREZE fundos latentes por canal, um por observacao, cada um com a sua "
          "priori vinda das contagens de fundo daquela observacao. Integrar 13 "
          "latentes independentes nao e o mesmo que integrar 1 compartilhado, "
          "nem a menos de constante: sao modelos estatisticos diferentes, e qual "
          "deles se quer e uma DECISAO a declarar, nao um detalhe de "
          "implementacao"),
        "qual_e_mais_defensavel": (
          "o de 13 latentes, porque o fundo e medido em cada observacao "
          "separadamente e nao ha razao para impor que seja o mesmo valor em "
          "epocas separadas por anos. Mas isso nao e meu para decidir"),
        "estado": (
          "nao implementado: o produto por observacao nao traz fundo_pesos nem "
          "fundo_contagens, e esta comparacao usa o alvo de CONTROLE com fundo "
          "fixo nas duas formulacoes")},
      "para_que_serve_se_passar": (
        "economizar avaliacoes NO DOMINIO TESTADO, com revalidacao das regioes "
        "novas. Nao se busca equivalencia perfeita"),
    }
    # AS GRADES GUARDADAS. Sem elas, decompor os 0,1367 entre o colapso do fundo
    # e o de resposta/resolucao exigiria repetir as 42 chamadas, 25 minutos. Com
    # elas, a decomposicao e aritmetica: basta misturar fonte combinada com fundo
    # somado e vice-versa. Aprendi isso pagando: a primeira corrida nao as
    # guardou.
    npz = Path(a.saida).with_suffix(".grades.npz")
    guarda = {}
    for nome in gco:
        chave = nome.replace(" ", "_")
        guarda["coad_fonte_" + chave] = gco[nome]["fonte"]
        guarda["coad_fundo_" + chave] = gco[nome]["fundo"]
        guarda["coad_tempo_" + chave] = gco[nome]["tempo"]
        for obsid, g in gpo[nome].items():
            guarda["po_fonte_%s_%s" % (chave, obsid)] = g["fonte"]
            guarda["po_fundo_%s_%s" % (chave, obsid)] = g["fundo"]
            guarda["po_tempo_%s_%s" % (chave, obsid)] = g["tempo"]
    np.savez_compressed(npz, **guarda)
    saida["grades_guardadas"] = str(npz)
    print("grades guardadas em %s" % npz)
    Path(a.saida).write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n")
    print("gravado em %s" % a.saida)


if __name__ == "__main__":
    main()
