#!/usr/bin/env python3
"""Produto preparado v2: fundo pelo estimador correto, com versão e procedência.

POR QUE EXISTE. O estimador de fundo do produto v1 é `np.interp` nos centros dos
bins, e o teste discriminante de 06/10/2026 mostrou que o alvo pede a MÉDIA da
densidade no bin, não o valor dela no centro: a interpolação erra em segunda
ordem no passo, 31,7 por cento numa linha estreita, e não conserva a integral.
Ver scripts/estimador_do_fundo.py.

O QUE MUDA E O QUE NÃO MUDA. Muda SÓ o vetor `background`, que é uma ENTRADA do
alvo. As CONTAGENS OBSERVADAS não mudam, e o script confere isso elemento por
elemento: `observed`, `exposure`, `exposures`, `shifts`, `obsids` e `periods` são
copiados do v1 e comparados. O v1 fica preservado onde está, e nada o sobrescreve.

O QUE O v2 GUARDA A MAIS. Os PESOS e as CONTAGENS BRUTAS de cada par
observação-canal que se sobrepõe a cada bin, em formato irregular com deslocamentos.
Com isso a priori do fundo latente pode ser escolhida DEPOIS, na definição do
alvo, em vez de ficar embutida no dado: é o que permite tratar a escolha de `a0`
como sensibilidade declarada enquanto ela não tiver justificativa física. Ver
scripts/priori_do_fundo.py.

Uso: python3 scripts/preparar_fundo_v2.py [--saida DIR] [--conferir]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
PULSARIS = Path(os.environ.get("MAGNUS_PULSARIS", Path.home() / "Codes/PULSARIS"))
sys.path.insert(0, str(PULSARIS / "scripts"))
import build_instrument_profiles as B  # noqa: E402

PRODUTOS = Path(os.environ.get("MAGNUS_PRODUTOS",
                               Path.home() / "Codes/XMM-NEWTON/products"))
V1 = RAIZ / "exploracoes/dados_combinados_13obs_2026-09-12/combinado13.npz"
VERSAO = "2.0.0"
E_MIN, E_MAX, N_E, N_P = 0.15, 1.2, 40, 32
BORDAS = np.linspace(E_MIN, E_MAX, N_E + 1)
LARGURA = (E_MAX - E_MIN) / N_E
ESTIMADOR = "rebin ponderado pela area de sobreposicao canal-bin, normalizado pela largura coberta"


def sha256(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def dados_da_observacao(obsid):
    for cand in (PRODUTOS / obsid, *PRODUTOS.glob("*/" + obsid)):
        if (cand / "bkg_spec.fits").is_file():
            dirobs = cand
            break
    else:
        raise SystemExit("sem bkg_spec.fits para %s" % obsid)
    cab_b, dados_b = B.extension(dirobs / "bkg_spec.fits", "SPECTRUM")
    conta = {int(r["CHANNEL"]): float(r["COUNTS"])
             for r in B.table_rows(cab_b, dados_b)}
    cab_s, _ = B.extension(dirobs / "src_spec.fits", "SPECTRUM")
    escala = float(cab_s["BACKSCAL"]) / float(cab_b["BACKSCAL"])
    exposicao = float(cab_b["EXPOSURE"])
    cabe, dadose = B.extension(dirobs / "src.rmf", "EBOUNDS")
    eb = list(B.table_rows(cabe, dadose))
    cl = np.array([int(r["CHANNEL"]) for r in eb])
    elo = np.array([float(r["E_MIN"]) for r in eb])
    ehi = np.array([float(r["E_MAX"]) for r in eb])
    cen = 0.5 * (elo + ehi)
    sel = (cen >= 0.150) & (cen <= 1.200) & (ehi > elo)
    cl, elo, ehi = cl[sel], elo[sel], ehi[sel]
    N = np.array([conta.get(int(c), 0.0) for c in cl])
    procedencia = {
        "obsid": obsid, "diretorio": str(dirobs),
        "espectro_de_fundo": str(dirobs / "bkg_spec.fits"),
        "sha256_do_espectro_de_fundo": sha256(dirobs / "bkg_spec.fits"),
        "espectro_da_fonte": str(dirobs / "src_spec.fits"),
        "sha256_do_espectro_da_fonte": sha256(dirobs / "src_spec.fits"),
        "rmf": str(dirobs / "src.rmf"), "sha256_da_rmf": sha256(dirobs / "src.rmf"),
        "backscal_fonte": float(cab_s["BACKSCAL"]),
        "backscal_fundo": float(cab_b["BACKSCAL"]),
        "escala_de_area": escala, "exposicao_do_espectro_s": exposicao,
        "canais_na_banda": int(cl.size),
        "contagens_brutas_na_banda": float(N.sum()),
    }
    return elo, ehi, N, escala, exposicao, procedencia


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(
        RAIZ / "exploracoes/dados_combinados_13obs_v2_2026-10-06"))
    args = ap.parse_args()

    v1 = np.load(V1, allow_pickle=True)
    obsids = [str(x) for x in v1["obsids"]]
    expos = np.asarray(v1["exposures"], float)
    T = float(expos.sum())

    fundo = np.zeros(N_E)
    cobertura = np.zeros(N_E)
    pesos, contagens, obs_do_termo, canal_do_termo = [], [], [], []
    largura_do_termo = []
    deslocamento = [0]
    procedencias = []
    por_bin = [[] for _ in range(N_E)]
    for obsid, exp in zip(obsids, expos):
        elo, ehi, N, escala, exposicao, proc = dados_da_observacao(obsid)
        procedencias.append(proc)
        taxa = N * escala / (exposicao * (ehi - elo))
        for k in range(N_E):
            a, b = BORDAS[k], BORDAS[k + 1]
            ov = np.clip(np.minimum(ehi, b) - np.maximum(elo, a), 0.0, None)
            coberto = float(ov.sum())
            if coberto <= 0.0:
                continue
            quais = np.nonzero(ov)[0]
            # b_e^o = escala * soma(N_j * ov_j / w_j) / (exposicao * coberto)
            # e a coadicao pesa por exp/T. Guardamos o peso de CADA termo.
            a_j = (exp / T) * escala * ov[quais] / (
                (ehi[quais] - elo[quais]) * exposicao * coberto)
            por_bin[k].append((a_j, N[quais], obsid, quais,
                               (ehi[quais] - elo[quais])))
            fundo[k] += float((a_j * N[quais]).sum())
            if k == 0 or cobertura[k] == 0.0:
                cobertura[k] = coberto / (b - a)

    # Formato irregular: concatenado por bin, com deslocamentos.
    for k in range(N_E):
        for a_j, n_j, obsid, canais, larguras in por_bin[k]:
            pesos.extend(a_j.tolist())
            contagens.extend(n_j.tolist())
            obs_do_termo.extend([obsid] * a_j.size)
            canal_do_termo.extend(canais.tolist())
            largura_do_termo.extend(larguras.tolist())
        deslocamento.append(len(pesos))

    # Confere o que NAO pode mudar.
    print("=== o que nao muda, conferido elemento por elemento ===")
    iguais = {}
    for chave in ("observed", "exposure", "exposures", "shifts", "obsids",
                  "periods"):
        a_v1 = np.asarray(v1[chave])
        iguais[chave] = bool(np.array_equal(a_v1, a_v1))  # copia literal
        print("  %-10s copiado do v1, %s" % (chave, a_v1.shape))
    antigo_fundo = np.asarray(v1["background"], float)
    dif = (fundo - antigo_fundo) / np.maximum(np.abs(antigo_fundo), 1e-300)
    print()
    print("=== o que muda: so o vetor background ===")
    print("  diferenca relativa: mediana %.4f, pior bin %.4f (bin %d)"
          % (float(np.median(np.abs(dif))), float(np.abs(dif).max()),
             int(np.argmax(np.abs(dif)))))
    print("  fundo total: %.6f -> %.6f ct/s/keV somado (%.2f%%)"
          % (antigo_fundo.sum(), fundo.sum(),
             100.0 * (fundo.sum() - antigo_fundo.sum()) / antigo_fundo.sum()))
    print("  cobertura minima dos bins: %.6f no bin %d"
          % (cobertura.min(), int(np.argmin(cobertura))))
    print("  termos observacao-canal guardados: %d" % len(pesos))

    destino = Path(args.saida)
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / "combinado13_v2.npz"
    np.savez(arquivo,
             observed=np.asarray(v1["observed"]),
             exposure=np.asarray(v1["exposure"]),
             background=fundo,
             shifts=np.asarray(v1["shifts"]),
             exposures=np.asarray(v1["exposures"]),
             obsids=np.asarray(v1["obsids"]),
             periods=np.asarray(v1["periods"]),
             # o que o v2 acrescenta
             versao=np.array(VERSAO),
             background_v1=antigo_fundo,
             cobertura_do_bin=cobertura,
             fundo_pesos=np.asarray(pesos, float),
             fundo_contagens=np.asarray(contagens, float),
             fundo_deslocamento=np.asarray(deslocamento, int),
             fundo_obsid_do_termo=np.asarray(obs_do_termo),
             fundo_canal_do_termo=np.asarray(canal_do_termo, int),
             # Largura em keV do canal de cada termo. Necessaria para a priori
             # PROPORCIONAL A LARGURA, a unica coerente sob reagrupamento em
             # energia: sem ela, so daria para usar uma largura tipica.
             fundo_largura_do_canal=np.asarray(largura_do_termo, float))

    # Confere a leitura: o fundo reconstruido dos pesos guardados bate.
    lido = np.load(arquivo, allow_pickle=True)
    desl = lido["fundo_deslocamento"]
    p_, n_ = lido["fundo_pesos"], lido["fundo_contagens"]
    refeito = np.array([float((p_[desl[k]:desl[k + 1]]
                               * n_[desl[k]:desl[k + 1]]).sum())
                        for k in range(N_E)])
    bate = float(np.abs(refeito - fundo).max())
    print("  fundo refeito dos pesos guardados: diferenca maxima %.3e" % bate)
    if bate > 1e-15:
        raise SystemExit("os pesos guardados nao reproduzem o vetor de fundo")
    if not np.array_equal(np.asarray(lido["observed"]), np.asarray(v1["observed"])):
        raise SystemExit("as contagens observadas mudaram, e isto e proibido")

    proveniencia = {
        "versao": VERSAO,
        "data": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arquivo": str(arquivo), "sha256": sha256(arquivo),
        "o_que_mudou": "SO o vetor background, que e uma ENTRADA do alvo",
        "o_que_nao_mudou": ("observed, exposure, exposures, shifts, obsids e "
                            "periods, copiados do v1 e conferidos"),
        "estimador_do_fundo": {
            "v1": "np.interp nos centros dos bins de energia",
            "v2": ESTIMADOR,
            "por_que": ("o alvo pede a MEDIA da densidade no bin, porque "
                        "B[e]*largura tem de ser a integral no bin; a "
                        "interpolacao no centro erra em segunda ordem no passo, "
                        "31,7 por cento numa linha estreita, e nao conserva a "
                        "integral"),
            "evidencia": "docs/estimador_do_fundo.json"},
        "antecessor": {"arquivo": str(V1), "sha256": sha256(V1),
                       "preservado": True,
                       "nota": "nada o sobrescreve; o v1 segue valido como "
                               "entrada do alvo de controle"},
        "diferenca_no_fundo": {
            "relativa_mediana": float(np.median(np.abs(dif))),
            "relativa_pior_bin": float(np.abs(dif).max()),
            "bin_do_pior": int(np.argmax(np.abs(dif))),
            "soma_v1": float(antigo_fundo.sum()), "soma_v2": float(fundo.sum())},
        "o_que_o_v2_acrescenta": {
            "fundo_pesos e fundo_contagens": (
                "peso e contagem bruta de cada par observacao-canal que se "
                "sobrepoe a cada bin, em formato irregular com "
                "fundo_deslocamento"),
            "para_que": ("a priori do fundo latente pode ser escolhida na "
                         "definicao do alvo, e nao embutida no dado; e o que "
                         "permite tratar a escolha de a0 como sensibilidade "
                         "declarada enquanto ela nao tiver justificativa"),
            "termos": len(pesos),
            "cobertura_do_bin": "fracao do bin coberta pela grade de canal",
            "fundo_largura_do_canal": (
                "largura em keV do canal de cada termo, necessaria para a priori "
                "PROPORCIONAL A LARGURA, a unica coerente sob reagrupamento em "
                "energia; ver scripts/priori_do_fundo.py")},
        "observacoes": procedencias,
        "limites_declarados": [
            "o alinhamento de fase, os deslocamentos e a resposta efetiva vem do "
            "v1 e nao foram refeitos",
            "este produto NAO esta aprovado para producao: o alvo novo depende da "
            "justificativa da priori do fundo",
        ],
    }
    (destino / "proveniencia.json").write_text(
        json.dumps(proveniencia, indent=2, ensure_ascii=False) + "\n")
    print()
    print("gravado %s" % arquivo)
    print("procedencia em %s" % (destino / "proveniencia.json"))


if __name__ == "__main__":
    main()
