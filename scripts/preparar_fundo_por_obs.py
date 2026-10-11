#!/usr/bin/env python3
"""Fundos INDIVIDUAIS: um latente por observação e canal, da reconstrução.

DECISÃO DO REVISOR, 07/10/2026: "Escolho o alvo por observação como referência
científica, com um fundo latente por observação e canal, condicionado à medição
de fundo correspondente. Não é necessário acrescentar essas dimensões ao MCMC:
mantenha a marginalização, preservando o compartilhamento entre as fases de cada
observação."

E a correção de descrição dele: as contagens da região de fundo entram como
VEROSSIMILHANÇA DA MEDIÇÃO, não como priori. A distribuição condicionada a elas
é que entra na marginalização da região da fonte, com cada medição usada uma
única vez e os hiperparâmetros declarados (`a0 = 0,01/keV`, `β = 0`, na família
proporcional à largura do canal).

POR QUE NÃO PRECISA MEXER NA VEROSSIMILHANÇA. A maquinaria de marginalização do
adaptador opera POR PROBLEMA: cada observação, com o seu npz, integra os seus 40
latentes sobre as SUAS 32 fases. Somar os ln L das treze é exatamente o produto
das marginais, porque os latentes são independentes ENTRE observações. O que
falta é só o dado, e é o que este script produz.

O QUE ELE MUDA NO PESO. No produto coadicionado v2 cada termo leva o fator
`exp_o/T` da coadição. Aqui NÃO: o latente é o fundo DAQUELA observação, e o
peso é só o do estimador por área.

Uso: python3 scripts/preparar_fundo_por_obs.py [--conferir]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
PULSARIS = Path(os.environ.get("MAGNUS_PULSARIS", Path.home() / "Codes/PULSARIS"))
sys.path.insert(0, str(PULSARIS / "scripts"))
import build_instrument_profiles as B  # noqa: E402
sys.path.insert(0, str(RAIZ / "scripts"))
import por_observacao as PO  # noqa: E402

RECON = PO.RECON
VERSAO = "por_observacao-1.0.0"
E_MIN, E_MAX, N_E, N_P = 0.15, 1.2, 40, 32
BORDAS = np.linspace(E_MIN, E_MAX, N_E + 1)
ESTIMADOR = ("rebin ponderado pela area de sobreposicao canal-bin, normalizado "
             "pela largura coberta, SEM o peso de coadicao")


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as fh:
        for p in iter(lambda: fh.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()


def espectros(ev):
    """Os espectros e a rmf daquela observacao, na reconstrucao.

    Os CSVs levam o prefixo LONGO e ficam em pulsaris/; os espectros e a
    resposta ficam um nivel acima com o prefixo CURTO. Descobri isso olhando o
    diretorio, nao supondo, e e a mesma armadilha que o preparar_v3 documenta.
    """
    longo = ev.name.replace("_events.csv", "")
    curto = "_".join(longo.split("_")[-2:]) if longo.count("_") >= 2 else longo
    base = ev.parent.parent
    for pref in (curto, longo):
        f = {"fundo": base / ("%s_bkg_spec.fits" % pref),
             "fonte": base / ("%s_src_spec.fits" % pref),
             "rmf": base / ("%s_src.rmf" % pref)}
        if all(v.is_file() for v in f.values()):
            return f
    raise SystemExit("sem espectros para %s em %s" % (longo, base))


def termos_da_observacao(ev):
    """Os termos (peso, contagem, canal, largura) por bin de energia.

    b_{o,k} = soma_j a_j . N_j, com
        a_j = escala . ov_j / (w_j . exposicao . coberto_k)
    onde ov_j e a sobreposicao do canal j com o bin k, w_j a largura do canal e
    coberto_k a largura do bin coberta por canais.
    """
    f = espectros(ev)
    cab_b, dados_b = B.extension(f["fundo"], "SPECTRUM")
    conta = {int(r["CHANNEL"]): float(r["COUNTS"])
             for r in B.table_rows(cab_b, dados_b)}
    cab_s, _ = B.extension(f["fonte"], "SPECTRUM")
    escala = float(cab_s["BACKSCAL"]) / float(cab_b["BACKSCAL"])
    exposicao = float(cab_b["EXPOSURE"])
    cabe, dadose = B.extension(f["rmf"], "EBOUNDS")
    eb = list(B.table_rows(cabe, dadose))
    cl = np.array([int(r["CHANNEL"]) for r in eb])
    elo = np.array([float(r["E_MIN"]) for r in eb])
    ehi = np.array([float(r["E_MAX"]) for r in eb])
    cen = 0.5 * (elo + ehi)
    sel = (cen >= E_MIN) & (cen <= E_MAX) & (ehi > elo)
    cl, elo, ehi = cl[sel], elo[sel], ehi[sel]
    N = np.array([conta.get(int(c), 0.0) for c in cl])

    # O DESLOCAMENTO COMECA EM ZERO. A convencao do alvo e N+1 LIMITES, usados
    # como desl[ie]:desl[ie+1], e nao N finais: o preparar_fundo_v2.py inicia a
    # lista com [0] e eu iniciei vazia, gravando 40 limites para 40 bins. O alvo
    # recusou com "fundo_deslocamento traz 40 limites para 40 bins de energia",
    # que e a mensagem certa e me poupou um resultado errado.
    pesos, contagens, canais, larguras = [], [], [], []
    deslocamento = [0]
    fundo = np.zeros(N_E)
    cobertura = np.zeros(N_E)
    for k in range(N_E):
        a, b = BORDAS[k], BORDAS[k + 1]
        ov = np.clip(np.minimum(ehi, b) - np.maximum(elo, a), 0.0, None)
        coberto = float(ov.sum())
        cobertura[k] = coberto / (b - a)
        if coberto > 0.0:
            quais = np.nonzero(ov)[0]
            a_j = escala * ov[quais] / ((ehi[quais] - elo[quais])
                                        * exposicao * coberto)
            pesos.extend(a_j.tolist())
            contagens.extend(N[quais].tolist())
            canais.extend(cl[quais].tolist())
            larguras.extend((ehi[quais] - elo[quais]).tolist())
            fundo[k] = float((a_j * N[quais]).sum())
        deslocamento.append(len(pesos))
    proc = {"espectro_de_fundo": str(f["fundo"]),
            "sha256_do_espectro_de_fundo": sha256(f["fundo"]),
            "espectro_da_fonte": str(f["fonte"]),
            "sha256_do_espectro_da_fonte": sha256(f["fonte"]),
            "rmf": str(f["rmf"]), "sha256_da_rmf": sha256(f["rmf"]),
            "backscal_fonte": float(cab_s["BACKSCAL"]),
            "backscal_fundo": float(cab_b["BACKSCAL"]),
            "escala_de_area": escala, "exposicao_do_espectro_s": exposicao,
            "canais_na_banda": int(cl.size),
            "contagens_brutas_na_banda": float(N.sum())}
    return (np.array(pesos), np.array(contagens), np.array(canais, int),
            np.array(larguras), np.array(deslocamento, int), fundo,
            cobertura, proc)


def canais_compartilhados(canais, deslocamento, pesos, contagens):
    """Quantos canais entram em MAIS DE UM bin, e quanto peso carregam.

    POR QUE ISTO IMPORTA. O revisor exigiu que cada medicao seja usada uma unica
    vez. No estimador por area, um canal que ATRAVESSA a fronteira entre dois
    bins contribui para os dois, com as sobreposicoes somando a largura dele.
    Isso nao duplica o dado, porque a soma dos pesos respeita a largura, mas
    torna os latentes de bins adjacentes CORRELACIONADOS, enquanto a
    marginalizacao os trata como independentes. A minha medida anterior de
    covariancia diagonal era da rota de INTERPOLACAO, nao desta.
    """
    bin_do_termo = np.zeros(canais.size, int)
    ini = 0
    for k, fim in enumerate(deslocamento):
        bin_do_termo[ini:fim] = k
        ini = fim
    por_canal = {}
    for i, c in enumerate(canais):
        por_canal.setdefault(int(c), []).append(i)
    compart = {c: idx for c, idx in por_canal.items() if len(idx) > 1}
    peso_total = float((pesos * np.maximum(contagens, 0.0)).sum())
    peso_compart = float(sum(
        (pesos[i] * max(contagens[i], 0.0)) for idx in compart.values()
        for i in idx))
    return {
      "canais_na_banda": len(por_canal),
      "canais_em_mais_de_um_bin": len(compart),
      "fracao_dos_canais": len(compart) / max(1, len(por_canal)),
      "peso_dos_compartilhados_na_soma": (peso_compart / peso_total
                                          if peso_total else 0.0),
      "pares_de_bins_que_compartilham": sorted(
          {tuple(sorted({int(bin_do_termo[i]) for i in idx}))
           for idx in compart.values()})[:50],
      "o_que_isto_implica": (
        "os latentes de bins que compartilham canal sao CORRELACIONADOS, e a "
        "marginalizacao atual os integra como independentes. A correlacao entra "
        "pela fracao de peso acima; se ela for pequena, o erro e dessa ordem, "
        "mas isso precisa ser declarado e nao suposto"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(PO.DESTINO))
    a = ap.parse_args()
    indice = json.loads((RECON / "indice.json").read_text())
    print("=== termos de fundo por observacao, da reconstrucao ===")
    print("%-12s %8s %8s %10s %12s %10s" % ("obsid", "canais", "termos",
                                            "cobertura", "fundo somado",
                                            "contra npz"))
    relato = {"o_que_e": "um fundo latente por observacao e canal",
              "versao": VERSAO, "estimador": ESTIMADOR,
              "as_contagens_de_fundo_sao": (
                "VEROSSIMILHANCA DA MEDICAO, nao priori. A distribuicao "
                "condicionada a elas entra na marginalizacao da fonte"),
              "hiperparametros_declarados": {"a0_por_keV": 0.01, "beta": 0.0,
                                             "familia": "proporcional a largura do canal"},
              "sem_o_peso_de_coadicao": (
                "no v2 coadicionado cada termo leva exp_o/T; aqui nao, porque o "
                "latente e o fundo DAQUELA observacao"),
              "observacoes": []}
    for s in indice["sets"]:
        obsid = str(s["obsid"])
        ev = RECON / s["files"]["events"]
        (pesos, contagens, canais, larguras, desl, fundo, cob,
         proc) = termos_da_observacao(ev)
        npz = Path(a.saida) / ("obs_%s.npz" % obsid)
        atual = dict(np.load(npz, allow_pickle=True))
        antigo = np.asarray(atual["background"], float)
        dif = float(np.max(np.abs((fundo - antigo)
                                  / np.maximum(np.abs(antigo), 1e-300))))
        atual.update(fundo_pesos=pesos, fundo_contagens=contagens,
                     fundo_canal_do_termo=canais,
                     fundo_largura_do_canal=larguras,
                     fundo_deslocamento=desl,
                     versao=np.array(VERSAO))
        np.savez(npz, **atual)
        comp = canais_compartilhados(canais, desl, pesos, contagens)
        relato["observacoes"].append({
            "obsid": obsid, "npz": str(npz), "termos": int(pesos.size),
            "cobertura_minima": float(cob.min()),
            "fundo_somado": float(fundo.sum()),
            "diferenca_relativa_max_contra_o_npz": dif,
            "canais_compartilhados": comp, "procedencia": proc})
        print("%-12s %8d %8d %10.6f %12.6f %10.2e"
              % (obsid, proc["canais_na_banda"], pesos.size, cob.min(),
                 fundo.sum(), dif), flush=True)
    piores = max(o["diferenca_relativa_max_contra_o_npz"]
                 for o in relato["observacoes"])
    comp0 = relato["observacoes"][0]["canais_compartilhados"]
    print()
    print("pior diferenca contra o vetor `background` que ja estava no npz: %.3e"
          % piores)
    print("  (os dois sao calculados por rotas independentes: aqui pelos TERMOS, "
          "e no preparar_v3 pelo estimador por area direto)")
    print()
    print("=== canais compartilhados entre bins, na primeira observacao ===")
    print("  canais na banda: %d" % comp0["canais_na_banda"])
    print("  em mais de um bin: %d (%.2f por cento dos canais)"
          % (comp0["canais_em_mais_de_um_bin"],
             100 * comp0["fracao_dos_canais"]))
    print("  peso deles na soma do fundo: %.4f por cento"
          % (100 * comp0["peso_dos_compartilhados_na_soma"]))
    print("  implicacao: os latentes desses bins sao CORRELACIONADOS e a "
          "marginalizacao os integra como independentes")
    relato["pior_diferenca_contra_o_npz"] = piores
    Path(RAIZ / "docs/fundo_por_observacao.json").write_text(
        json.dumps(relato, indent=2, ensure_ascii=False) + "\n")
    print()
    print("gravado em docs/fundo_por_observacao.json")


if __name__ == "__main__":
    main()
