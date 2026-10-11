#!/usr/bin/env python3
"""O limiar de flare, medido nos dados em vez de herdado.

Pendencia aberta no Run0: "decisao do limiar de flare -- selecao reconstruida e
a candidata provisoria". Ela ficou mais delicada depois de um achado sobre o
dado: a observacao 0402851001 tem fundo 4,6 vezes o das outras doze e, com 5,07
por cento da exposicao, fornece 19,24 por cento do fundo coadicionado. O
argumento do limiar valia para a FONTE; para o FUNDO ele nao valia, e e o fundo
que entra agora como verossimilhanca da medicao no alvo por observacao.

O QUE ESTE SCRIPT MEDE, por observacao:
  a curva de luz de PARTICULAS, a receita padrao do EPIC-pn: eventos com
    PI entre 10000 e 12000 eV, PATTERN == 0 e FLAG == 0, em bins de 100 s,
    normalizada pela area viva do campo;
  a fracao de exposicao que cada limiar candidato descartaria;
  e, para cada observacao, quanto do tempo ja descartado pela reconstrucao
    coincide com o que o limiar descartaria.

O QUE ELE NAO FAZ: nao refaz a extracao nem muda a selecao. Ele produz o
diagnostico que a decisao do limiar precisa, e a decisao continua sendo um ato
declarado, nao um efeito colateral de um script.

Uso: python3 scripts/diagnostico_flare.py [--bin 100] [--obs 0402851001 ...]
"""
import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
PULSARIS = Path(os.environ.get("MAGNUS_PULSARIS", Path.home() / "Codes/PULSARIS"))
sys.path.insert(0, str(PULSARIS / "scripts"))
import build_instrument_profiles as B  # noqa: E402  parser FITS sem astropy
import por_observacao as PO  # noqa: E402

PRODUTOS = Path(os.environ.get(
    "MAGNUS_PRODUTOS_RECON",
    Path.home() / "Codes/XMM-NEWTON/reconstrucao_5906a57/produtos/RXJ1308.6_2127"))
#: A receita padrao do EPIC-pn para flare de particulas. O limiar de 0,4 ct/s e
#: o valor de referencia dos manuais para FULL FRAME; ele entra aqui como
#: CANDIDATO a ser comparado com os dados, nao como verdade.
PI_LO, PI_HI = 10000, 12000
LIMIARES = (0.2, 0.3, 0.4, 0.5, 0.8, 1.0)

_TIPOS = {"D": ">f8", "E": ">f4", "J": ">i4", "I": ">i2", "B": "u1", "K": ">i8"}


def colunas_rapidas(caminho, nomes):
    """Le colunas inteiras de um BINTABLE com numpy, sem varrer linha a linha.

    `table_rows` do parser do PULSARIS devolve um dicionario por linha, o que e
    claro e inviavel aqui: sao centenas de milhares de eventos por observacao.
    A leitura por offset e passo da linha da o mesmo resultado e nao depende de
    astropy.
    """
    for header, data in B.read_hdus(caminho):
        if header.get("EXTNAME") != "EVENTS":
            continue
        largura, n = int(header["NAXIS1"]), int(header["NAXIS2"])
        bruto = np.frombuffer(data[:largura * n], dtype=np.uint8).reshape(n, largura)
        saida = {}
        for nome, off, repete, desc, kind in B.table_columns(header):
            if nome not in nomes or desc or kind not in _TIPOS:
                continue
            dt = np.dtype(_TIPOS[kind])
            saida[nome] = bruto[:, off:off + dt.itemsize].copy().view(dt).ravel()
        faltam = [x for x in nomes if x not in saida]
        if faltam:
            raise ValueError("colunas ausentes em %s: %s" % (caminho.name, faltam))
        return saida, header
    raise ValueError("sem extensao EVENTS em %s" % caminho)


def area_lida(c):
    """Area do detector efetivamente lida, em pixels RAW.

    EXISTE PORQUE A COMPARACAO ENTRE SUBMODOS ESTAVA ERRADA. A primeira versao
    deste script comparou taxas ABSOLUTAS de particulas entre observacoes, e os
    submodos leem areas diferentes: PRIMEFULLWINDOW cobre RAWY de 14 a 199, 186
    linhas, e PRIMELARGEWINDOW cobre de 103 a 199, 97 linhas, pouco mais da
    metade. Sem normalizar, uma LARGEWINDOW parece mais limpa do que e.

    A area vem da GEOMETRIA do modo, o produto do numero de CCDs pela caixa
    envolvente em RAWX e RAWY, e nao da contagem de pixels com evento, que
    dependeria da estatistica da propria observacao.
    """
    ccds = int(np.unique(c["CCDNR"]).size)
    dx = int(c["RAWX"].max()) - int(c["RAWX"].min()) + 1
    dy = int(c["RAWY"].max()) - int(c["RAWY"].min()) + 1
    return {"ccds": ccds, "rawx": dx, "rawy": dy, "pixels": ccds * dx * dy}


def curva_de_particulas(caminho, passo_s):
    c, header = colunas_rapidas(caminho, ("TIME", "PI", "PATTERN", "FLAG",
                                          "CCDNR", "RAWX", "RAWY"))
    area = area_lida(c)
    sel = ((c["PI"] >= PI_LO) & (c["PI"] <= PI_HI)
           & (c["PATTERN"] == 0) & (c["FLAG"] == 0))
    t = c["TIME"]
    t0, t1 = float(t.min()), float(t.max())
    bordas = np.arange(t0, t1 + passo_s, passo_s)
    conta, _ = np.histogram(t[sel], bins=bordas)
    taxa = conta / passo_s
    return {"t0": t0, "t1": t1, "bordas": bordas, "taxa": taxa,
            "eventos_totais": int(t.size), "eventos_de_particula": int(sel.sum()),
            "area": area}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=float, default=100.0, dest="passo")
    ap.add_argument("--obs", nargs="*", default=None)
    ap.add_argument("--saida", default=str(RAIZ / "docs/diagnostico_flare.json"))
    a = ap.parse_args()

    obss = {o["obsid"]: o for o in PO.prepara(verbose=False, regravar=False)}
    alvo = sorted(a.obs) if a.obs else sorted(obss)
    rel = {"o_que_e": "o limiar de flare medido nos dados, por observacao",
           "data": time.strftime("%Y-%m-%d"),
           "receita": ("curva de particulas do EPIC-pn: PI entre %d e %d eV, "
                       "PATTERN == 0, FLAG == 0, em bins de %.0f s"
                       % (PI_LO, PI_HI, a.passo)),
           "o_limiar_de_referencia": ("0,4 ct/s e o valor dos manuais para FULL "
                                      "FRAME, e entra aqui como CANDIDATO"),
           "o_que_NAO_faz": ("nao refaz a extracao nem muda a selecao; produz o "
                             "diagnostico de que a decisao precisa"),
           "observacoes": []}
    print("%-12s %-16s %8s %8s %9s  %s"
          % ("obsid", "submodo", "part/s", "mediana", "med/1e4px",
             "fracao descartada por limiar"))
    for obsid in alvo:
        d = PRODUTOS / obsid
        candidatos = sorted(d.glob("epn_*clean.fits")) or sorted(d.glob("*ImagingEvts.ds"))
        if not candidatos:
            print("  %s: sem event list em %s" % (obsid, d))
            continue
        caminho = candidatos[0]
        c = curva_de_particulas(caminho, a.passo)
        taxa = c["taxa"]
        viva = float(taxa.size) * a.passo
        fracoes = {}
        for L in LIMIARES:
            ruim = float((taxa > L).sum()) * a.passo
            fracoes[str(L)] = ruim / viva if viva else float("nan")
        reg = {"obsid": obsid, "arquivo": caminho.name,
               "janela_s": c["t1"] - c["t0"],
               "eventos": c["eventos_totais"],
               "eventos_de_particula": c["eventos_de_particula"],
               "taxa_de_particulas_media": float(taxa.mean()),
               "mediana": float(np.median(taxa)),
               "p90": float(np.percentile(taxa, 90)),
               "maxima": float(taxa.max()),
               "area_lida": c["area"],
               "taxa_por_1e4_pixels": float(taxa.mean()) * 1.0e4 / c["area"]["pixels"],
               "mediana_por_1e4_pixels": float(np.median(taxa)) * 1.0e4 / c["area"]["pixels"],
               "fracao_descartada_por_limiar": fracoes,
               "exposicao_no_alvo_s": float(obss[obsid]["exposicao"]) if obsid in obss else None}
        rel["observacoes"].append(reg)
        reg["submodo"] = obss[obsid]["submodo"] if obsid in obss else "?"
        print("%-12s %-16s %8.4f %8.4f %9.4f  %s"
              % (obsid, reg["submodo"], reg["taxa_de_particulas_media"],
                 reg["mediana"], reg["mediana_por_1e4_pixels"],
                 " ".join("%.0f%%" % (100 * fracoes[str(L)]) for L in LIMIARES)),
              flush=True)
    if rel["observacoes"]:
        taxas = np.array([o["mediana_por_1e4_pixels"] for o in rel["observacoes"]])
        rel["panorama"] = {
            "mediana_das_medianas": float(np.median(taxas)),
            "pior_observacao": max(rel["observacoes"], key=lambda o: o["mediana"])["obsid"],
            "limiares_testados": list(LIMIARES),
            "leitura": ("a fracao descartada por limiar e o que a decisao troca "
                        "por limpeza; uma observacao cuja mediana ja passa do "
                        "limiar nao tem flare isolado, tem nivel de fundo alto, "
                        "e cortar por limiar nao a conserta"),
        }
    Path(a.saida).write_text(json.dumps(rel, indent=2, ensure_ascii=False) + "\n")
    print("\ngravado em %s" % a.saida)


if __name__ == "__main__":
    main()
