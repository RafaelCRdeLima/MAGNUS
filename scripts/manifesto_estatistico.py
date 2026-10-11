#!/usr/bin/env python3
"""Manifesto estatistico: gerado do CODIGO, nao escrito a mao.

Entrega de PE.1 e PE.2. Para cada grandeza: papel fisico, unidade, status
(amostrada, fixa, derivada, perfilada ou integrada analiticamente), prior,
suporte, dependencias e tratamento da incerteza.

A distincao que o portao exige, e que vale repetir aqui: o MODELO GLOBAL
DECLARADO tem inclinacao em [0, 180] graus, porque `inclinationRange` so entrou
em 25/09/2026 e os pedidos anteriores usam a faixa cheia. Os ajustes A e B sao
CONDICIONADOS por essa chave e amostram 6,67 por cento daquele suporte sob prior
uniforme em i, ou 10,43 por cento sob prior isotropico proporcional a sin(i). O
pedido efetivo de cada corrida define o alvo DELA, e nenhuma das duas e a
posterior do modelo global.

Uso: MAGNUS_INSTRUMENT_DIR=... python3 scripts/manifesto_estatistico.py <pedido.json>
"""
import json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mcmc_fit as m  # noqa: E402

PAPEL = {
 'compactness': ('u = 2GM/Rc^2, compacidade', 'adimensional'),
 'mass':        ('massa gravitacional', 'M_sol'),
 'radius':      ('raio de coordenada', 'km'),
 'inclination': ('angulo entre o eixo de rotacao e a linha de visada', 'graus'),
 'phaseOffset': ('fase do polo 1 na grade de tempo', 'ciclos'),
 'lineEnergy':  ('energia de repouso da feicao de absorcao', 'keV'),
 'lineWidth':   ('largura da feicao', 'keV'),
 'lineDepth':   ('profundidade otica da feicao no centro', 'adimensional'),
 'logMagneticField': ('lg do campo polar proprio', 'lg(G)'),
 'baseKT':      ('kT do polo 1 na lei termica', 'keV'),
 'peaking':     ('a1, concentracao do lobo 1', 'adimensional'),
 'baseKT2':     ('kT do polo 2', 'keV'),
 'peaking2':    ('a2, concentracao do lobo 2', 'adimensional'),
 'poleTilt2':   ('beta, desvio do polo 2 no plano magneto-rotacional', 'graus'),
 'poleTilt2b':  ('gamma, azimute do desvio do polo 2 em torno do eixo magnetico', 'graus'),
 'atmFraction': ('espessura efetiva da atmosfera sobre o corpo negro', 'adimensional'),
 'magColat':    ('colatitude do eixo magnetico', 'graus'),
 'magAzim':     ('azimute do eixo magnetico', 'graus'),
 'distance':    ('distancia', 'kpc'),
 'nh':          ('coluna de hidrogenio', '1e22 cm^-2'),
}

def main():
    pedido = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / 'exploracoes/ajuste_geometria_2026-10-01/A_i91/pedido.json'
    req = json.loads(pedido.read_text())
    meta, eventos = m.read_events(ROOT / 'dados/eventos.csv')
    prob = m.FitProblem(req, eventos, meta, None)
    K = 2.0 * m._GM_SUN_C2_KM

    linhas = []
    for i, nome in enumerate(prob.names):
        papel, unidade = PAPEL.get(nome, ('nao catalogado', '?'))
        lo, hi = prob.bounds[i]
        if nome in prob.frozen:
            status = 'FIXA'
            prior = 'nenhum: valor imposto'
            incerteza = 'NAO integrada'
        else:
            status = 'amostrada'
            prior = 'uniforme em [%.6g, %.6g]' % (lo, hi)
            incerteza = 'integrada pelas amostras'
        linhas.append({'nome': nome, 'papel': papel, 'unidade': unidade,
                       'status': status, 'prior': prior, 'suporte': [lo, hi],
                       'incerteza': incerteza})

    derivadas = [
      {'nome': 'massa', 'papel': 'M = u R / (2 GM_sol/c^2)', 'unidade': 'M_sol',
       'status': 'DERIVADA das amostras de u e R', 'prior': 'induzido, nao uniforme',
       'suporte': [max(0.8, prob.mass_floor), prob.mass_cap],
       'incerteza': 'integrada, por ser funcao de coordenadas amostradas'},
      {'nome': 'z', 'papel': 'z = 1/sqrt(1-u) - 1', 'unidade': 'adimensional',
       'status': 'DERIVADA de u', 'prior': 'induzido; a densidade cai como (1+z)^-3 '
                 'pelo jacobiano, e a banda de massa reduz mais',
       'suporte': [1/math.sqrt(1-prob.bounds[0][0])-1, 1/math.sqrt(1-prob.bounds[0][1])-1],
       'incerteza': 'integrada'},
      {'nome': 'energia da linha', 'papel': 'com lineCyclotron, E_cp = %g * B local'
                 % 6.3048787654e-15, 'unidade': 'keV',
       'status': 'DERIVADA do campo local, NAO do valor fixado de lineEnergy',
       'prior': 'induzido por logMagneticField e pela lei do dipolo',
       'suporte': ['depende de B e da posicao na superficie'],
       'incerteza': 'integrada via logMagneticField'},
    ]
    # TRES CATEGORIAS, porque elas NAO pedem o mesmo tratamento. Misturar tudo
    # numa lista de "entradas fixas" esconde que uma precisa de modelo de
    # incerteza, outra de validacao numerica e a terceira e hipotese de modelo.
    entradas_observacionais = [
      {'nome': 'fundo', 'origem': req.get('preparedDataNpz', 'nao declarado'),
       'papel': 'taxa de contagem nao vinda da fonte, somada bin a bin',
       'status': 'FIXO, previsao sem erro',
       'o_que_falta': ('erro, area de extracao, exposicao da regiao de fundo e metodo de '
                       'estimativa. So o valor central esta no npz.'),
       'tratamento_devido': ('modelo conjunto de contagens de fonte e regiao de fundo, com escala '
                             'de area e exposicao, OU marginalizacao analitica demonstrada. '
                             'Enquanto nao existir, o alvo e EXPLICITAMENTE CONDICIONAL ao fundo.'),
       'sensibilidade_medida': ('escala de 0,8 a 1,2 move as diferencas de ln L entre os tres '
                                'vetores fixos por 0,12. Isso e sensibilidade a um erro ESTIPULADO '
                                'de escala, nos tres vetores e nessa faixa, e NAO estabelece que o '
                                'fundo seja desprezivel nem aprova a aproximacao.')},
      {'nome': 'exposicao', 'valor': prob.exposure, 'unidade': 's',
       'papel': 'tempo vivo somado das 13 observacoes', 'status': 'FIXA',
       'o_que_falta': 'erro de tempo vivo por observacao',
       'tratamento_devido': 'propagar o erro de exposicao, ou declarar condicional'},
      {'nome': 'resposta do instrumento', 'valor': prob.instrument,
       'papel': 'area efetiva e redistribuicao em energia', 'status': 'FIXA',
       'o_que_falta': 'incerteza de calibracao',
       'tratamento_devido': 'auditoria da resposta e, se relevante, parametro de calibracao'},
      {'nome': 'periodo', 'valor': req.get('period'), 'unidade': 's',
       'papel': 'periodo de rotacao usado para dobrar em fase', 'status': 'FIXO',
       'o_que_falta': 'erro do periodo e da deriva entre as 13 epocas',
       'tratamento_devido': 'propagar, porque erro de periodo borra a curva de luz'},
    ]
    escolhas_de_modelo = [
      {'nome': 'temperatureMinFrac', 'valor': req.get('temperatureMinFrac'),
       'papel': 'piso de temperatura da lei termica, como fracao de baseKT',
       'status': 'HIPOTESE DE MODELO fixada', 'o_que_falta': 'justificativa fisica ou amostragem',
       'tratamento_devido': 'amostrar, ou declarar como hipotese e medir a sensibilidade'},
      {'nome': 'lei termica de Perez-Azorin', 'papel': 'lobe(c,a,T^4) por polo, com corte '
       'hemisferico', 'status': 'HIPOTESE DE MODELO',
       'o_que_falta': 'o limite a -> 0 e um degrau, e o plano de 04/10 propoe substituir por lei '
                      'continua', 'tratamento_devido': 'comparar familias termicas especificadas'},
      {'nome': 'lineCyclotron', 'valor': req.get('lineCyclotron'),
       'papel': 'amarra a energia da linha ao campo local em vez de deixa-la livre',
       'status': 'HIPOTESE DE MODELO', 'o_que_falta': 'nada: e escolha declarada',
       'tratamento_devido': 'comparar com linha livre, que e outro modelo'},
      {'nome': 'teto e piso de massa', 'valor': [max(0.8, prob.mass_floor), prob.mass_cap],
       'papel': 'recorte do suporte em (u, R)', 'status': 'ESCOLHA DE PRIOR',
       'o_que_falta': 'o teto esta ATIVO nos tres maximos, a 99,4 a 99,6 por cento',
       'tratamento_devido': 'variar o teto e medir o deslocamento, item do plano'},
    ]
    configuracao_numerica = [
      {'nome': 'maxImages', 'valor': prob.max_images,
       'papel': 'ordens de imagem incluidas no tracado', 'status': 'CONFIGURACAO NUMERICA',
       'tratamento_devido': ('VALIDACAO NUMERICA e nao modelo de incerteza: mostrar que aumentar a '
                             'ordem nao muda as previsoes alem da tolerancia, ou demonstrar '
                             'geometricamente quais ordens sao possiveis. Medido: a segunda imagem '
                             'carrega 2,32 por cento do fluxo.')},
      {'nome': 'sphereBands e boundaryRefine', 'valor': [req.get('sphereBands'), req.get('boundaryRefine')],
       'papel': 'resolucao da quadratura da superficie', 'status': 'CONFIGURACAO NUMERICA',
       'tratamento_devido': 'validacao contra o controle analitico, que P1 ainda reprova'},
      {'nome': 'alinhamento da malha', 'valor': 'offset 0, o padrao historico',
       'papel': 'origem azimutal da malha lat-lon', 'status': 'CONFIGURACAO NUMERICA',
       'tratamento_devido': ('medido: o offset 0 esta no EXTREMO da curva de erro, e trocar para '
                             '0,25 reduz o deficit por fator 92 a 240 bandas. Escolher alinhamento '
                             'pelo resultado seria vicio; o que falta e tolerancia validada.')},
    ]
    dados_usados = [
      {'nome': 'bins de energia e de fase', 'valor': [prob.energy_bins, prob.phase_bins],
       'papel': 'grade em que as contagens sao agregadas', 'status': 'DEFINE OS DADOS USADOS',
       'tratamento_devido': 'nao e incerteza: e escolha de agregacao, a declarar'},
      {'nome': 'energyFitMax', 'valor': prob.energy_fit_max, 'unidade': 'keV',
       'papel': 'corte superior de energia na verossimilhanca',
       'status': 'DEFINE OS DADOS USADOS',
       'tratamento_devido': ('exclui %d de %d bins, 20 por cento. Nao e incerteza: e selecao, e '
                             'os bins excluidos sao justamente os de maior fundo relativo, 4,7 a '
                             '11,9 por cento.'
                             % (prob.phase_bins*prob.energy_bins - len(prob._fit_bins),
                                prob.phase_bins*prob.energy_bins))},
    ]
    entradas_fixas = {'observacionais': entradas_observacionais,
                      'escolhas_de_modelo': escolhas_de_modelo,
                      'configuracao_numerica': configuracao_numerica,
                      'definem_os_dados': dados_usados}
    alvo = {
      'ln_verossimilhanca': ('soma sobre os bins selecionados de '
                             'n_b ln(mu_b) - mu_b, Poisson sem o termo ln(n_b!), '
                             'que e constante nos parametros'),
      'mu_b': 'taxa_modelo(fase, energia) + fundo_b, vezes largura do bin de energia '
              'vezes o tempo representado naquele bin de fase',
      'bins_incluidos': '%d de %d, por energyFitMax = %s keV'
                        % (len(prob._fit_bins), prob.phase_bins*prob.energy_bins,
                           prob.energy_fit_max),
      'termos_de_prior': ('uniforme na caixa, mais o recorte duro de massa em '
                          '[%.2f, %.2f] e u < 0,985. NAO ha fator jacobiano K/R no alvo, '
                          'o que e coerente com prior uniforme em (u, R)'
                          % (max(0.8, prob.mass_floor), prob.mass_cap)),
      'ln_posterior': 'ln L + extra_log_prior, com extra_log_prior = 0 aqui porque nao '
                      'ha prior gaussiano em massa neste pedido',
      'perfilado': ('nenhum. fitLightCurve esta %s, logo o alpha perfilado de '
                    '_score_light_curve nao entra' % ('ligado' if prob.fit_light_curve else 'desligado')),
      'integrado_analiticamente': 'nenhum',
    }
    doc = {'pedido': str(pedido), 'data': '2026-10-06',
           'modelo_global_declarado': {
             'inclinacao': 'suporte [0, 180] graus; inclinationRange so entrou em 25/09/2026',
             'alvo_desta_corrida': 'CONDICIONAL a inclinationRange = %s' % req.get('inclinationRange'),
             'fracao_do_suporte_amostrada': {'uniforme_em_i': 0.0666667, 'isotropico_sin_i': 0.1042738},
             'aviso': ('o pedido efetivo define o alvo DESTA corrida. Nem A nem B e a posterior '
                       'do modelo global, e misturar bem entre A e B nao valida uma posterior '
                       'global: falta o restante do suporte, ou uma reducao por simetria fisica '
                       'justificada e verificada')},
           'coordenadas': linhas, 'derivadas': derivadas,
           'entradas_fixas': entradas_fixas, 'alvo': alvo,
           'orcamento_operacional': {
             'bytes_por_amostra_medido': {'19_dims': 584, '6_dims': 302},
             'trajetorias_96x100000x19': '5,60 GB, medido por extrapolacao do custo por amostra',
             'nao_medido': 'propostas, reservatorio de pulsos, checkpoint, pos-processamento',
             'consequencia': 'a escala de producao nao cabe em 15 GB com os aplicativos abertos'},
           }
    saida = ROOT / 'docs/manifesto_estatistico.json'
    saida.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print('%-20s %-10s %-34s %s' % ('nome', 'status', 'suporte', 'incerteza'))
    for l in linhas:
        print('%-20s %-10s [%-14.6g, %-14.6g] %s'
              % (l['nome'], l['status'], l['suporte'][0], l['suporte'][1], l['incerteza']))
    print()
    print('derivadas: %s' % ', '.join(d['nome'] for d in derivadas))
    for cat, itens in entradas_fixas.items():
        print('%-22s %s' % (cat + ':', ', '.join(e['nome'] for e in itens)))
    print('perfilados: %s' % alvo['perfilado'])
    print('bins incluidos: %s' % alvo['bins_incluidos'])
    print('gravado em %s' % saida)

if __name__ == '__main__':
    main()
