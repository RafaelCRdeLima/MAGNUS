#!/usr/bin/env python3
"""Reconstroi o vetor de fundo dos produtos preservados e compara com o NPZ.

Fecha a lacuna que o rastreamento da receita nao fechava: saber que a escala e
aplicada UMA VEZ nao diz QUAIS fundos entraram, porque
coadicionar.py::read_background devolve ZEROS EM SILENCIO quando nao acha o CSV.

Refaz exatamente o que o coadicionar faz: np.interp nos 40 centros de energia e
media das observacoes PONDERADA PELA EXPOSICAO. Registra, por observacao, o
arquivo, o hash, a faixa de energia coberta, a escala declarada no cabecalho e a
exposicao.
"""
import hashlib, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRODUTOS = Path.home() / 'Codes/XMM-NEWTON/products'
NPZ = ROOT / 'exploracoes/dados_combinados_13obs_2026-09-12/combinado13.npz'
E_MIN, E_MAX, N_E = 0.15, 1.2, 40
ECEN = E_MIN + (np.arange(N_E) + 0.5) * (E_MAX - E_MIN) / N_E

def acha_csv(obsid):
    p = list(PRODUTOS.glob('%s/pulsaris/*_background.csv' % obsid))
    if not p:
        p = list(PRODUTOS.glob('*/%s/pulsaris/*_background.csv' % obsid))
    return p[0] if p else None

def main():
    d = np.load(NPZ, allow_pickle=True)
    obsids = [str(x) for x in d['obsids']]
    expos = np.array(d['exposures'], dtype=float)
    guardado = np.array(d['background'], dtype=float)

    registro, curvas, pesos, faltando = [], [], [], []
    print('%-12s %10s %-46s %18s' % ('obsid', 'exposicao', 'arquivo', 'faixa do CSV (keV)'))
    for obsid, exp in zip(obsids, expos):
        f = acha_csv(obsid)
        if f is None:
            faltando.append(obsid)
            print('%-12s %10.0f %-46s %18s' % (obsid, exp, 'AUSENTE', '-'))
            continue
        bruto = f.read_text(encoding='utf-8')
        linhas = [l for l in bruto.splitlines() if not l.startswith('#')]
        dados = np.array([[float(x) for x in l.split(',')] for l in linhas if l.strip()])
        cab = [l for l in bruto.splitlines() if l.startswith('#')]
        escala = next((l for l in cab if 'scal' in l.lower() or 'escal' in l.lower()), '')
        curva = np.interp(ECEN, dados[:, 0], dados[:, 1])
        curvas.append(curva); pesos.append(exp)
        registro.append({'obsid': obsid, 'arquivo': str(f),
                         'sha256': hashlib.sha256(f.read_bytes()).hexdigest()[:16],
                         'exposicao': float(exp),
                         'faixa_keV': [float(dados[:, 0].min()), float(dados[:, 0].max())],
                         'linhas': int(dados.shape[0]),
                         'cabecalho_escala': escala.strip(),
                         'extrapola_abaixo': bool(dados[:, 0].min() > ECEN.min()),
                         'extrapola_acima': bool(dados[:, 0].max() < ECEN.max())})
        print('%-12s %10.0f %-46s %7.4f a %7.4f'
              % (obsid, exp, f.name[:46], dados[:, 0].min(), dados[:, 0].max()))

    if faltando:
        print()
        print('AUSENTES: %s' % ', '.join(faltando))
    recon = np.average(np.array(curvas), axis=0, weights=np.array(pesos))
    print()
    print('=== reconstruido contra o guardado no NPZ ===')
    dif = recon - guardado
    rel = np.abs(dif) / np.maximum(np.abs(guardado), 1e-300)
    print('observacoes usadas: %d de %d' % (len(curvas), len(obsids)))
    print('soma reconstruida %.6f | soma guardada %.6f | razao %.8f'
          % (recon.sum(), guardado.sum(), recon.sum()/guardado.sum()))
    print('maior diferenca relativa por bin: %.3e | mediana %.3e' % (rel.max(), np.median(rel)))
    # NAO e identidade binaria: o CSV guarda a taxa com %.9g e a reconstrucao
    # reaplica a receita, entao o que se afirma e CONCORDANCIA NUMERICA dentro
    # de 1e-12 relativo. O ajuste de termo foi pedido pelo revisor em 06/10/2026.
    print('concorda dentro de 1e-12 relativo: %s'
          % np.allclose(recon, guardado, rtol=1e-12, atol=0))
    if rel.max() > 1e-9:
        pior = int(np.argmax(rel))
        print('pior bin: %d, E = %.4f keV, reconstruido %.8g contra guardado %.8g'
              % (pior, ECEN[pior], recon[pior], guardado[pior]))
    extrap = [r['obsid'] for r in registro if r['extrapola_abaixo'] or r['extrapola_acima']]
    print()
    print('observacoes em que np.interp EXTRAPOLA plano: %s'
          % (', '.join(extrap) if extrap else 'nenhuma'))
    saida = {'reconstruido': recon.tolist(), 'guardado': guardado.tolist(),
             'razao_das_somas': float(recon.sum()/guardado.sum()),
             'maior_diferenca_relativa': float(rel.max()),
             'concorda_em_1e-12_relativo':
                 bool(np.allclose(recon, guardado, rtol=1e-12, atol=0)),
             'nota': 'concordancia numerica, nao identidade binaria',
             'observacoes': registro, 'ausentes': faltando,
             'extrapolam': extrap}
    (ROOT / 'docs/fundo_reconstruido.json').write_text(
        json.dumps(saida, indent=2, ensure_ascii=False) + "\n")
    print('gravado em docs/fundo_reconstruido.json')

if __name__ == '__main__':
    main()
