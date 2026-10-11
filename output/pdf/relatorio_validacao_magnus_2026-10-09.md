MAGNUS / RBS 1223

Relatório consolidado de auditoria e validação

Da segunda opinião à preparação de uma nova produção científica

Corte de informação: 9 de outubro de 2026. Período principal: 29/09 a 09/10/2026. Documento elaborado para Rafael, o revisor e o auditor do projeto.

Objetivo: organizar as perguntas, os testes, as falhas encontradas e as decisões que efetivamente mudaram a inferência. Este relatório não substitui os artefatos reproduzíveis e não libera os intervalos antigos de redshift.

Situação no corte

Há uma candidata operacional de integração espacial e um alvo por observação definido. A integração temporal M=5 passou nos contrastes examinados com o eixo espacial fixo, mas consome quase todo o orçamento. A combinação espacial + temporal ainda está sendo verificada no cluster pelo job 47139, em node29, conforme o último estado informado. Nenhum resultado desse job foi presumido aqui.

As cadeias anteriores servem como diagnóstico e sementes. Preferências entre bacias e intervalos científicos de z permanecem suspensos até dados, operador e pilotos serem aprovados.

Como ler

Números antigos aparecem como histórico quando foram substituídos por diagnósticos posteriores. “Passou” significa que o teste cumpriu o critério no conjunto examinado; não significa prova universal. “Inconclusivo” significa que a referência ou o desenho não permitiu determinar a precisão.

Sumário e navegação

A narrativa organiza os testes por pergunta, não pela ordem de mensagens. As fontes indicadas ao fim de cada capítulo permitem retornar aos artefatos. O inventário final identifica os arquivos presentes no corte.

| Capítulo | Página |
|---|---|
| 01  Resumo executivo e mapa de evidências | 3 |
| 02  Pergunta científica e o que o modelo calcula | 4 |
| 03  Auditorias anteriores e limite térmico a₁→0 | 5 |
| 04  Correção estatística, marginais e mistura | 6 |
| 05  Restrições, otimização, memória e falhas | 7 |
| 06  Fundo: procedência, estimador e integração | 8 |
| 07  XREDUX e reconstrução dos dados | 9 |
| 08  Fase: exposição, alinhamento e rotação global | 10 |
| 09  Offset explícito, respostas e reuso | 11 |
| 10  Normalização: da malha à vizinhança antipodal | 12 |
| 11  Referência espacial: tabela de raios e ψ₀ | 13 |
| 12  Integração temporal e orçamento conjunto | 14 |
| 13  Cluster, custo e reprodutibilidade operacional | 15 |
| 14  Correções de interpretação que devem sobreviver ao relatório | 16 |
| 15  O que falta e o que pode entrar no paper | 17 |
| 16  Literatura e método de consolidação | 18 |
| 17  Inventário reproduzível de relatórios e artefatos | 19 |

Escopo: consolidação técnica das evidências disponíveis. O resultado pendente do job 47139 não foi antecipado; nenhum job ou teste científico foi iniciado para produzir este documento.

01  Resumo executivo e mapa de evidências

## Resumo executivo e mapa de evidências

O problema não foi um único defeito. A investigação separou quatro fontes: código de inferência, integração numérica do modelo físico, preparação observacional e definição do alvo estatístico. Corrigir uma delas não validou automaticamente as outras.

| Componente | Evidência atual | Limite |
|---|---|---|
| MCMC local | Gaussianas e suporte truncado recuperados em réplicas; dimensão livre corrigida | Não certifica exploração física/global |
| Fundo | 13 contribuições reconstruídas; estimador e incerteza rastreados | Tratamento final por observação requer validação integrada |
| Dados XMM | 13 observações reconstruídas com código congelado e semente SAS | Flare, centro e possível pile-up ainda requerem fechamento |
| Espaço | ψ₀ 480 + tabela 115200, passo observado até 0,0130 contra ψ₀ 960 | Três vetores e offsets testados |
| Fase | M9→M15: pior contraste 0,0046; M5→M15: 0,0908 | Espaço fixo em 480; não é orçamento conjunto |
| Produção | Teste conjunto lançado no cluster | Reotimização, pilotos e cadeias novas ainda pendentes |

Conclusão central

Foram reencontrados pontos de alto redshift, mas isso não confirma a medição científica antiga. Diferenças entre melhores pontos não são probabilidades de bacias. Parâmetros pouco informados são aceitáveis; erro numérico não controlado e aprisionamento de cadeia não são evidência de não identificação.

Critérios utilizados

O plano usa 0,1 em diferenças de ln L como orçamento operacional e 0,02 para estabilidade observada da referência. Essas tolerâncias são escolhas do projeto, não normas universais. Q_num e discrepâncias por bin complementam a análise; não determinam sozinhos Δln L.

Fontes principais: docs/plano_otimizado_producao_2026-10-07.md; docs/plano_tres_etapas_2026-10-08.md; docs/relatorio_rodada_coordenada_2026-10-07.md.

02  Pergunta científica e o que o modelo calcula

## Pergunta científica e o que o modelo calcula

RBS 1223 (RX J1308.6+2127) tem emissão térmica e período próximo de 10,31 s. MAGNUS combina geometria de dois polos, atmosfera magnetizada, absorção, traçado em Schwarzschild e resposta instrumental. A hipótese ciclotrônica vincula a energia de repouso ao campo local, mas campo, mapa térmico e geometria permanecem correlacionados; a identificação de z depende do modelo completo.

A compacidade é u=2GM/(Rc²), o redshift é z=(1−u)^(-1/2)−1 e a massa é derivada de u e R. O resultado inicial próximo de z=0,65 era acompanhado por limites de prior e mistura insuficiente. A comparação com trabalhos de emissão diferente não constitui, isoladamente, contradição ou confirmação.

Pipeline conceitual

Parâmetros físicos → emissão na superfície → transporte e integração espacial → integração em fase → resposta de cada observação → exposição efetiva por fase + fundo → verossimilhança → prior → amostragem e marginais.

Dados antigos e reconstruídos

| Produto | Contagens | Exposição | Papel |
|---|---|---|---|
| v1 | 346799 | 131978,1 s | Histórico coadicionado |
| v3 | 329972 | 125667,3 s | Reconstrução diagnóstica por observação |

A grade histórica tinha 32 fases × 40 canais, de 0,15 a 1,2 keV, com ajuste até 1,0 keV: 1024 de 1280 bins. A alteração de contagens e seleção entre produtos impede interpretar diferenças absolutas de ln L entre eles como preferência científica.

Convenções indispensáveis

Separar ln L de ln posterior; registrar suporte e prior em cada comparação. Constantes como ln(n!) podem ser omitidas para comparar parâmetros no mesmo alvo, mas agregação e marginalização podem usar constantes diferentes. Uma relação de densidades entre pontos não é o peso integrado de uma região.

Fontes principais: docs/segunda_opiniao_2026-10-04.md (histórico, com afirmações posteriormente corrigidas); docs/manifesto_estatistico.json; docs/normalizacao_do_alvo_v2.md.

03  Auditorias anteriores e limite térmico a₁→0

## Auditorias anteriores e limite térmico a₁→0

As auditorias iniciais identificaram uso da dimensão incorreta no passo de esticamento com parâmetros congelados, incoerência entre direção e intensidade do campo no segundo polo, clamp da atmosfera em ângulo campo-normal e resposta de uma observação aplicada ao conjunto. O briefing registrava Δln L≈225 para a geometria do campo no vetor examinado; os outros efeitos tinham escalas distintas. São diagnósticos históricos, não orçamento de erro da versão atual.

O erro conceitual no mapa térmico

a₁=0 não produz uma superfície inteira uniforme: o lobo térmico retorna zero para c≤0 e uniformiza apenas seu hemisfério. Valores negativos eram tratados como zero; ampliar o prior sem alterar o código não mudaria o modelo, e a expressão proposta poderia ter singularidades para −1<a<0.

O limite positivo estreito e o degrau em c=0 são difíceis para a malha lat-lon original. O equador magnético corta a malha obliquamente, e o resultado varia com a passagem das fronteiras pelos centros das células. Não se pode inferir resolução suficiente apenas contando células por camada.

| Caso preservado | Primeiro−último | Amplitude observada | Leitura |
|---|---|---|---|
| B colado | 11,70 | 24,79 | Não monotônico |
| B aberto | 1,04 | 2,65 | Não monotônico |
| A | 4,13 | 4,13 | Monotônico naquela sequência |

Tetos silenciosos

rings era limitado a 30 e as bandas a 60. A opção sphereBands separou discretização da esfera e spots, com teto ampliado e metadados de resolução efetiva. Compatibilidade sem a opção foi conferida nos três vetores. O stderr do motor deixou de ser descartado; a verificação inicial do erro fatal foi estrutural, não uma reprodução empírica daquela exceção.

Uma avaliação inicial com tratamento das fronteiras reduziu a preferência aparente de aproximadamente 27 para cerca de 10 em pontos antigos. Esse resultado não fixou a preferência convergida final: investigações posteriores encontraram erro adicional na normalização e nos raios.

Fontes principais: docs/parecer_segunda_opiniao_2026-10-04.md; docs/incidente_resolucao_2026-10-04/convergencia_quadratura.json; docs/verificacoes_do_revisor_2026-10-04.md.

04  Correção estatística, marginais e mistura

## Correção estatística, marginais e mistura

Nas configurações A/B examinadas eram armazenadas 19 coordenadas, com 17 livres e phaseOffset/lineEnergy fixadas; a energia física da linha ciclotrônica depende do campo. A marginalização dos parâmetros livres ocorre pelas amostras conjuntas. Massa e z devem ser calculados amostra por amostra, não por produto de quantis separados.

O fator de aceitação do stretch depende da dimensão livre. Os testes usaram o run_ensemble embarcado, com pool/problema de teste, e não uma reimplementação do algoritmo. Atualizações em grupos complementares e congelamento foram exercitados.

| Teste | Resultado | O que autoriza |
|---|---|---|
| Gaussiana correlacionada | 5 réplicas; 0/85 marginais fora; pior 2,30 erros MC | Correção local no alvo testado |
| Truncamento analítico | 5 réplicas; 0/85 fora; pior 3,02 erros MC | Suporte e momentos locais |
| Mistura 6D | Pesos finais próximos de 0,70; inicializações distintas | Recuperação naquela mistura |
| Mistura 19D | Média 0,6373→0,7113; amplitude 0,5413→0,0434 | Relaxação explica plausivelmente falha inicial |

A ocupação é um estimador válido de peso sob exploração adequada. Travessias não garantem precisão: o indicador de modo tem sua própria autocorrelação. Contagem de travessias/2 não substitui ESS. Na mistura 19D, a convergência ainda não estava completa no corte daquele teste; isso não exige repetir indefinidamente o sintético antes de validar o motor.

A cadeia antiga A

O estimador antigo truncava Geyer em 100 lags. A implementação FFT revelou τ≈1304–3266 e ESS≈1470–3681, contra τ≈179–193 e ESS≈52000 registrados no artefato. O pior caso tinha cerca de 15,3 passos por τ: a própria estimativa de τ é incerta. R-hat clássico entre caminhantes do mesmo ensemble não certifica convergência entre réplicas independentes.

Suporte e prior

A∪B cobria 6,67% do intervalo declarado [0°,180°] sob prior uniforme em inclinação; a fração sob prior isotrópico é diferente. Isso é massa prévia não explorada, não posterior perdida. Duas transformações de simetria falharam, mas não provaram inexistência de toda simetria. Prior uniforme em (u,R) não é uniforme em (M,R); K/R aparece ao transformar a densidade, não como correção automática do prior escolhido.

Fontes principais: docs/incidente_resolucao_2026-10-04/pre_registro_replicas.md; docs/incidente_resolucao_2026-10-04/mistura19.json; docs/amostrador_diagnostico.md.

05  Restrições, otimização, memória e falhas

## Restrições, otimização, memória e falhas

Os máximos encontrados estavam próximos do teto de 2,5 M☉. Isso limita o suporte e afeta a localização do máximo, mas não mede quanto do z vem do teto. Na cadeia A, a massa derivada corretamente apresentou mediana 2,42662, p84=2,48131, p99=2,49891; 8,90% acima de 2,49 e 0,92% acima de 2,499. A sensibilidade deve ser avaliada sob priors/cortes alternativos.

Otimização

Passos por fração do prior criavam vértices inviáveis e escalas de objetivo muito diferentes. A calibração pelo efeito e sondas viáveis melhoraram o desenho. A tolerância relativa de Powell no objetivo bruto ~1,7 milhão correspondia a ~1,729 em ln L; sua parada não demonstrava estagnação patológica. Centrar o objetivo corrigiu a interpretação. COBYQA foi testado com restrição não linear explícita de massa; sondas de massa constante usam u′=u exp(t), R′=R exp(−t).

Piloto registrado: Nelder-Mead +1,3328/1116 avaliações; Powell +0,8753/1116; COBYQA +1,2089/898. Ganhos por avaliação e estabilização são critérios distintos. Orçamento esgotado ou zero melhora em algumas direções não prova máximo global.

Memória e checkpoint

Trajetórias como listas Python custavam cerca de 584 bytes por amostra de 19 dimensões: ~5,60 GB na corrida de 100 mil passos e ~7,85 GB em 140 mil. Um coletor preservou sorteio, aceitação e trajetórias no teste comparativo, permitindo réplicas locais. O orçamento precisa incluir FFT e pós-processamento: armazenamento pequeno coexistiu com pico de 1090 MB antes da ACF em blocos, reduzido a cerca de 298 MB.

Falha não é suporte físico

Proposta fora do prior, recusa física e falha numérica antes eram indistinguíveis como −∞. A política final usa recuperação determinística validada e parada na primeira falha não recuperada, com checkpoint/diagnóstico. Permitir rejeições numéricas até 20 falhas ou 1% alterava a transição; essa primeira versão foi substituída.

Fontes principais: docs/verificacoes_do_revisor_2026-10-04.md; docs/incidente_resolucao_2026-10-04/fechamento_PE_2026-10-06b.md; docs/incidente_resolucao_2026-10-04/memoria_medida.py.

06  Fundo: procedência, estimador e integração

## Fundo: procedência, estimador e integração

A reconstrução rastreou contagens brutas, BACKSCAL de fonte/fundo, exposição e EBOUNDS até os produtos XMM. As 13 observações reproduziram o vetor guardado com diferença relativa máxima ~8×10⁻¹⁵: concordância numérica, não identidade binária. A comparação com CSV de %.9g atingiu ~4,9×10⁻⁹. A escala de área é aplicada uma vez; na observação de referência a razão era 0,09094.

Arquivo ausente antes retornava fundo zero e extrapolação plana via np.interp era silenciosa. Os caminhos passaram a falhar explicitamente. A propagação mostrou covariância diagonal nessa grade: os bins não compartilhavam nós, embora interpolação possa correlacionar grades diferentes.

O valor central também importa

Interpolação no centro descartava nós e diferia de integração por sobreposição de canais. Isso motivou o produto v2. Sensibilidade a escala ±20% moveu cada ln L ~1,4, mas contrastes apenas ~0,12 nos pontos fixos; não era incerteza medida nem avaliação de forma espectral.

| Alvo histórico | Colado−aberto | Colado−A | Aberto−A |
|---|---|---|---|
| Controle v1 | 10,0640 | 11,9129 | 1,8489 |
| Controle v2 | 10,5794 | 11,7970 | 1,2176 |
| Marginal, a0=0 | 10,5769 | 11,7946 | 1,2176 |
| Marginal, a0=100/keV | 10,6204 | 12,1835 | 1,5631 |

A priori por canal não podia depender arbitrariamente da quantidade de canais. A família com forma a0×largura explicita a medida em energia. β também precisava ser declarado. Para a0=0, componentes com contagem zero eram impróprios; 336/3224 termos estavam nessa condição. A massa pontual em zero era uma convenção implícita, não conclusão física.

Foi escolhido a0=0,01/keV, β=0 como convenção fiducial provisória, com sensibilidades declaradas. Posterior própria não torna a priori própria: evidências absolutas/fatores de Bayes entre modelos não são autorizados por essa escolha. O alvo individual integra um b(o,e) sobre todas as fases daquela observação uma vez. Contagens da região de fundo são medição, não priori; não devem ser contadas duas vezes.

A distribuição de uma soma ponderada de Gammas não é automaticamente Gamma. O protótipo examinou aproximação por cumulantes, quadratura independente e Monte Carlo. Esses testes precisam acompanhar a versão por observação e seu domínio; comparar apenas as médias não valida a marginal.

Fontes principais: docs/fundo_reconstruido.json; docs/incerteza_fundo.json; docs/alvo_estatistico_v2.md (especificação histórica); docs/alvo_por_observacao_decidido.md; docs/priori_do_fundo.json.

07  XREDUX e reconstrução dos dados

## XREDUX e reconstrução dos dados

A auditoria identificou estado obsoleto, retomada sem procedência, decimação com exposição integral, problemas de timeout, TAR, pile-up não medido classificado como limpo, texto LaTeX não vinculado a etapas, colisões e perda de parâmetros. Testes de código não demonstravam que esses riscos ocorreram nas 13 entradas; foi necessária auditoria dos produtos usados.

As primeiras reduções limpas montaram uma árvore em edição e foram invalidadas. Mesmo um commit citado não identifica código que muda durante a execução e importa módulos tardiamente. A reconstrução válida usa git archive do commit 5906a57, código somente leitura, manifesto de conteúdo e imagem identificada.

Reconstrução confirmada

Foram entregues 13 conjuntos, índice, comandos, regiões, respostas, GTIs, fundos e exposições por fase. Repetição com SAS_RAND_SEED fixada reproduziu eventos/espectros/período na observação de referência, descontando carimbos FITS. Sem semente, o SAS aleatoriza propriedades de eventos; duas execuções não definem um piso universal, mas medem variação observada.

| Achado | Evidência | Consequência |
|---|---|---|
| Origem temporal | Primeiro evento após cortes; até ~200 s | Convenção dependia da seleção |
| Resolução | 0163560101 declarava zero | Metadado corrigido |
| Flare | 0305900201 e 0402851001 perderam 52%/18% | Seleção científica precisa de justificativa |
| Pile-up | Não medido historicamente; 9 excessos de duplos ≥3σ | Diagnóstico anular autorizado |
| Centro | 5,6–10,3 segundos de arco do centróide | Recentrar e regenerar respostas |
| Procedência antiga | Imagem/código originais não recuperados | Não certificar retrospectivamente pelo código novo |

RMF/ARF e exposições concordaram onde seleções eram iguais; diferenças de região e limiar precisam ser distinguidas de randomização. Taxa na região da fonte semelhante em intervalos mantidos/removidos não limita sozinha o fundo: é preciso medir fundo por energia/tempo e sua incerteza.

0402851001 fornece ~19,24% do fundo coadicionado com ~5,07% da exposição. Isso prioriza diagnóstico; não justifica excluir automaticamente. O alvo individual evita espalhar esse fundo pelas outras épocas, mas não corrige contaminação residual.

Fontes principais: <XMM>/reconstrucao_5906a57/RELATORIO.md; docs/portao_de_dados.md; docs/frente_de_dados_2026-10-07.md.

08  Fase: exposição, alinhamento e rotação global

## Fase: exposição, alinhamento e rotação global

represented_time estimava cobertura quase uniforme do período/exposição total. EXPOSUnn fornece TIME e FRACEXP por quadro do CCD da fonte. Os quadros devem ser distribuídos pelo ciclo e convertidos à mesma referência baricêntrica dos eventos, com tempo vivo aplicado uma vez. Comparar amplitude pico a pico não substitui comparar vetores completos de exposição.

| Medida histórica | Resultado | Limite |
|---|---|---|
| Amplitude estimada | 7,8×10⁻⁵ | Receita quase uniforme |
| Amplitude por quadros | ~2,0×10⁻³ no diagnóstico | Depende do conjunto/convenção |
| Correção nos contrastes | Até ~1,00 em ln L | Três pontos fixos |
| Q_num conjunto | ~0,0528 após espalhar quadros | Não determina Δln L |

Efeitos por bin de até 0,12σ não garantiam desprezibilidade conjunta. Δln L era dominado por termo linear coerente. Correlação simples pequena entre exposição e resíduos não exclui uma soma ponderada relevante.

A inversão que não era preferência posterior

A reconstrução inicialmente mudou colado−A em −15,9585. Nas 11 observações não disputadas o contraste mudou −17,1738, mas isso não isolava causalmente flare. A diferença de perfil de até 15,1% era majoritariamente rotação global: removida translação sub-bin de 1,668 bins, o pior resíduo caiu a 2,55%, mediana 0,57%; harmônicos ficaram próximos.

| Contraste nas 11 | v1 | v3 reestimado | v3 transportado |
|---|---|---|---|
| Colado−aberto | 13,5960 | 3,6765 | 11,2472 |
| Colado−A | 16,8326 | −0,3412 | 13,5885 |
| Aberto−A | 3,2365 | −4,0178 | 2,3413 |

Transportar os números antigos de roll sem transportar origem/época/período era inconsistente. A fração ~81% descrevia aquele diagnóstico, não uma decomposição causal geral. Uma origem diferente é válida quando declarada consistentemente; o defeito era sua dependência da seleção. Os pontos físicos fixos não absorviam a rotação, portanto a inversão não demonstrava preferência por A.

A solução escolhida é preservar contagens não deslocadas e aplicar offsets ao modelo, com uma referência global fixa. Estimativas de correlação cruzada obtidas dos mesmos dados não devem ser usadas como priori independente. Períodos diferentes deixam deriva de fase e precisam ser registrados.

Fontes principais: docs/exposicao_por_fase.md; docs/isolamento_alinhamento_2026-10-07.json; docs/offset_relativo.json; docs/valida_exposunn.json.

09  Offset explícito, respostas e reuso

## Offset explícito, respostas e reuso

arrival_phase=2π[(p+1/2)/N+δ] implica mδ[p]=m0[p+δN]. Para δ=k/N a translação corresponde a np.roll(m0,−k). Offset zero, periodicidade, sinal e deslocamentos inteiros passaram nos casos testados; deslocamentos fracionários tiveram teste próprio. Concordância pontual não certifica integral sobre bin.

magAzim não reproduziu essa operação. Mesmo leitura de rotação rígida da geometria não basta para atribuir toda diferença à malha sem verificar equivalência física completa. A otimização não depende dessa atribuição: o offset explícito é o operador escolhido.

Reuso validado e aproximação de período

Uma chamada calcula emissão comum, aplica respostas/offsets e emite grades etiquetadas. Exposição e fundo permanecem no referencial observado. Deslocamentos inteiros permitem rolagem sem interpolar contagens. A diferença inicial ~1,640×10⁻⁸ não desapareceu com impressão em 17 dígitos; trocar somente período a reproduziu. O período entra a montante via Doppler, portanto seu reuso comum é aproximação declarada, não identidade universal.

No teste de período, efeito em ln L≈4,76×10⁻⁷, muito abaixo do orçamento; será revalidado em novos pontos. Ganhou-se ~4,97× em cinco observações. A impressão padrão preservada em 10 dígitos mantém compatibilidade, não prova igualdade interna.

Coadição não é alvo individual

A resposta combinada deve preservar a soma exposição×ARF×RMF, não apenas a área em 1 keV. Fonte e fundo foram comparados separadamente. Por bin: fonte mediana ~10⁻⁵, fundo mediana 3,14×10⁻⁴ e pior 2,98×10⁻³; tudo ≤0,01σ no teste. Ainda assim, contrastes de ln L diferiram até 0,1367 entre formulações. Médias próximas não certificam marginais iguais.

Os fundos individuais ponderam exposições de fase diferentes; seu agregado pode preservar o total e perder estrutura. O alvo científico adotado é individual. A emissão pode ser compartilhada, mas respostas, offsets, exposições e integrais de fundo permanecem individualizados.

O resolvedor registra base/caminho/hash e rejeita candidatos de conteúdo diferente. Cópias idênticas ou symlinks não são ambiguidade científica atual, mas precisam de identificação explícita.

Fontes principais: docs/valida_offset.json; docs/valida_reuso.json; docs/compara_formulacoes.json; docs/alvo_por_observacao_decidido.md.

10  Normalização: da malha à vizinhança antipodal

## Normalização: da malha à vizinhança antipodal

P1 foi observado numa esfera analítica, sem dados reais. Logo, problemas de redução não o explicam. Isotropia no plano da imagem verifica área/constantes por fatoração; um feixe normalizado acrescentou teste do mapeamento no controle, com invariância ~3,7×10⁻⁹. Não certifica toda física local.

Na superfície apareceu déficit ~7,731×10⁻⁴ e dependência de fase ~1,742×10⁻⁴. Próximo de ψ_max=π passa a existir contribuição antipodal e segunda imagem. O erro repartia-se entre imagens e podia cancelar no total. Refino de visibilidade era inerte na imagem direta porque o predicado não marcava células nesse regime; não era refutação física do limbo.

Experimentos que restringiram a causa

Média sobre offsets reduziu dispersão, mas não viés comum. Gauss-Legendre em μ passou na geometria e falhou no controle físico (pior déficit 4,75×10⁻³, modulação ~0,957%). Contagens azimutais múltiplas de 32 geraram degenerescências. Centroide em cosθ pouco alterou o viés; azimute global removeu alinhamento, mas não o erro dominante. Refino seletivo testado teve ganho pequeno a alto custo; isso não exclui toda adaptação espacial possível.

A localização colocou praticamente todo erro assinado nas duas últimas faixas de ψ₀, 171°–180°, no caso compacto. Frações com denominador quase zero abaixo do limiar eram instáveis. O clamp não foi acionado na malha examinada; esse resultado não foi automaticamente transferido à outra família.

Regularização geométrica

O fator 1/sinψ₀ do jacobiano cancela o sinψ₀ do elemento de área quando a contribuição completa é escrita em coordenadas centradas no observador. Em termos de b², a medida corresponde à área do plano da imagem. A singularidade era da representação, não uma divergência não integrável do fluxo naquele controle.

A malha centrada no observador reduziu o erro e tornou evidente a tabela de raios, antes mascarada. O ponto médio aritmético foi melhor na regra testada; isso não é exigência matemática de toda quadratura. O custo por fase precisa ser medido com atmosfera e respostas, não inferido de contagem de amostras incompatíveis.

Fontes principais: docs/incidente_resolucao_2026-10-04/P1_mecanismo_2026-10-06.md; docs/incidente_resolucao_2026-10-04/candidata_p1_2026-10-07.md; docs/incidente_resolucao_2026-10-04/plano_imagem_p1.json.

11  Referência espacial: tabela de raios e ψ₀

## Referência espacial: tabela de raios e ψ₀

Autoconvergência do motor antigo com Richardson não certificou o modelo real. Ausência de razão positiva e extrapolações discordantes tornaram o método inconclusivo, não medição de erro grande. Mesmo u=0,55 com ordens 1,65/1,09 não justificava chamar 0,7864 de erro certificado.

O toggle da segunda imagem alterou a sequência, mas suprimir uma contribuição muda cancelamentos: não demonstra duas causas independentes. Comparar as duas malhas é evidência de sensibilidade, não verdade externa. Q_num decrescente pode coexistir com contraste oscilante porque Δln L tem termo linear.

A sequência antes e depois da tabela fina

| Diagnóstico | Medida | Leitura |
|---|---|---|
| Tabela 1800 a 480 bandas | Contrastes movem até 0,1986 | Componente antes confundido com malha |
| ψ₀ 480→960, tabela grossa | Pior passo 0,1077 | Não certificava referência |
| ψ₀ 480→960, tabela 115200 | Pior passo 0,0130 | Estabilidade observada no domínio |
| Produção vs candidata fina | Discordância até 0,2190 | Configuração original não atende comparação |
| ψ₀ 240 vs referência | ~0,29 | Fora do orçamento operacional |

A tabela convergiu a partir de 28800 nos testes e foi mantida em 115200. Malha e tabela interagem: o contraste era quase comum na malha grossa, mas não na fina. Isso não prova interação malha–fase. Não se deve somar máximos de contrastes diferentes nem presumir separabilidade.

Escolha operacional: ψ₀ 480 + tabela 115200. Referência candidata espacial: ψ₀ 960 na mesma tabela. As três diferenças de pontos não são independentes: apenas duas são livres. O quarto nível não foi exigido neste estágio; nova região ou instabilidade reabre validação.

Por que ln L amplifica diferenças pequenas

Para μ′=μ+δ, Δln L=Σ[n log(1+δ/μ)−δ]. Em primeira ordem aparece Σ(n/μ−1)δ; Q_num=Σδ²/μ* mede uma soma quadrática diferente. Erros por bin pequenos não garantem contraste pequeno, e um máximo ou mediana relativa não resume a exigência da posterior.

Fontes principais: docs/cruza_tabela.json; docs/cruza_tabela_convergencia.json; docs/malha_tabela_fina_sequencia.json; docs/escolhe_producao.json.

12  Integração temporal e orçamento conjunto

## Integração temporal e orçamento conjunto

O motor historicamente amostra o centro do bin. Refinar fases reamostra também erro espacial dependente de fase, portanto primeiro foi fixada uma configuração espacial controlada. A regra instrumental discreta é inerte abaixo de um bin e ativa na grade fina: resolução foi zerada para isolar quadratura temporal. Isso não valida a resposta temporal física omitida nem é regra permanente.

M sub-amostras usa média das taxas finas no mesmo bin, não soma, antes de exposição. Peso uniforme testa perfil; exposição intra-bin variável exige tratamento coerente. Identidade de fases pontuais em dois níveis não demonstra estabilidade da integral.

| Contraste | M1 | M3 | M9 | M15 |
|---|---|---|---|---|
| A−B aberto | 5,0316 | 5,1097 | 5,1160 | 5,116087 |
| A−B colado | 3,8305 | 3,2791 | 3,1646 | 3,169169 |
| B aberto−colado | −1,2011 | −1,8306 | −1,9515 | −1,946918 |

M9→M15 teve pior passo ~0,00461, dentro da margem 0,02 no eixo espacial fixo. M3 ficou a ~0,1163 da referência no par B. M5 foi medido sequencialmente para evitar custos desnecessários.

| Contraste | M5 | M15 | M5−M15 |
|---|---|---|---|
| A−colado | 3,258802 | 3,169169 | +0,08963 |
| A−aberto | 5,114936 | 5,116087 | −0,00115 |
| Aberto−colado | −1,856134 | −1,946918 | +0,09078 |

M5 atende 0,1 no eixo temporal isolado, mas consome ~91% do orçamento. O passo espacial 0,0130 foi medido em outro corte do experimento. Soma ingênua ~0,1038 não é erro conjunto medido. Dois contrastes próximos do limite podem refletir uma discrepância concentrada em B colado; não são duas evidências independentes.

Último portão numérico no corte

Job 47139 compara (ψ₀480,M5) com (ψ₀960,M15), mesma tabela, no pior par, em node29. Se passar com margem suficiente, confirmar A. Se perto de 0,1, interpretar com a estabilidade limitada da referência; se exceder, testar M7 reaproveitando o canto fino. Nenhum resultado foi inventado para esse job.

Fontes principais: docs/temporal_psi0_A_480_115200_M15.json; docs/temporal_psi0_A_480_115200_M5.json; docs/temporal_psi0_B_aberto_480_115200_M5.json; último estado do job 47139 informado na conversa.

13  Cluster, custo e reprodutibilidade operacional

## Cluster, custo e reprodutibilidade operacional

As avaliações locais são validação; reotimização, pilotos e produção estão destinados ao sci-com. Custo no cluster foi medido em node28, AMD EPYC 9755, job 47091: mediana 32,98 s na configuração antiga e 174,49 s em ψ₀480/tabela115200, razão 5,29. Essa medição não incluiu M5/M9 e não mede throughput com muitos workers.

No nó não havia /usr/bin/time, portanto o artefato contém parede, não CPU/RSS. CPU/parede próximo de 1 no teste local indica pouca espera por execução, não ausência de contenção de memória. Corridas locais sob carga diferiram por fator ~7; custos assim não servem como benchmark isolado.

Requisitos encontrados no porte

GCC 11+ é necessário; o padrão do sistema era 9.4 e a compilação usou Spack gcc 11.4. Bootstrap dependente de caminho e reconstrução ausente no cluster exigiram correção/metadata cache. O cache deve ser identificado por conteúdo, não apenas caminhos reenraizados. Reproduzir um caso local verifica a transferência científica.

O epílogo relatado usa pkill por usuário: jobs simultâneos do mesmo usuário no mesmo nó correm risco. Enquanto 47139 estava em node29, recomenda-se excluir esse nó de novas submissões pertinentes. Não atribuir outros jobs a esta sessão sem evidência. A política operacional precisa ser compatível com concorrência antes da produção.

Orçamento ainda não é duração de cadeia

Estimativas de 5400 ou 43000 core-h foram provisórias/retiradas quando assumiam resolução ou escalabilidade não medidas. Iterações de ensemble e avaliações não são sinônimos: número de walkers e atualização por metades determinam chamadas e paralelismo. Planejar por custo medido, ESS/quantis e throughput, não por “100 mil” sem unidade.

Rodar em blocos com checkpoints e executável exclusivo. Código congelado em archive/worktree imutável; hashes de dados/respostas/flags. Não recompilar binário compartilhado enquanto jobs o usam. Memória total inclui coleta e FFT, além do RSS do motor.

Fontes principais: docs/tempo_scicom.json; docs/custo_isolado.json; docs/congelamento_2026-10-07.json; relato operacional do job 47139.

14  Correções de interpretação que devem sobreviver ao relatório

## Correções de interpretação que devem sobreviver ao relatório

| Afirmação abandonada | Formulação sustentada |
|---|---|
| “O resultado principal sobreviveu” | As execuções corrigidas reencontraram pontos de alto z; intervalos não certificados |
| “B é preferida por Δln L=27” | Preferência inicial sensível à discretização; pontos não são pesos de bacias |
| “a₁ de A está resolvido porque há células suficientes” | Precisão deve ser medida nas previsões e no alvo |
| “A ocupação não estima pesos” | É estimador sob exploração adequada; corrida inicial não relaxou |
| “A reconstrução prefere A” | Comparação de pontos fixos confundiu rotação global e convenção |
| “Erro por bin abaixo do ruído é irrelevante” | Termos coerentes acumulam; verificar alvo conjunto |
| “Tabela inerte” | Pode estar mascarada pelo erro de malha; testar cruzado |
| “Quantum decimal prova erro de impressão” | 17 dígitos manteve diferença; período explicava |
| “Três contrastes são independentes” | Só dois são livres para três vetores |
| “3 sub-amostras integra exatamente” | Quadratura mais fina com erro residual |
| “Falha até 1% pode virar rejeição” | Falha não recuperada interrompe; não altera suporte silenciosamente |
| “Refino seletivo está universalmente descartado” | Somente estratégia testada foi ineficiente; bins não são região espacial |

O padrão transversal foi atribuir um efeito ao último eixo variado sem controlar os demais. A resposta metodológica não é exigir perfeição: é comparação controlada, registro de convenções e distinção entre discrepância observada e erro certificado.

Fontes principais: docs/relatorio_rodada_coordenada_2026-10-07.md; campos CORRECAO dos artefatos JSON; decisões e retratações registradas na conversa.

15  O que falta e o que pode entrar no paper

## O que falta e o que pode entrar no paper

| Etapa | Próxima decisão | Libera |
|---|---|---|
| Precisão conjunta | Resultado 47139; confirmar A ou testar M7 | Configuração candidata para inferência |
| Dados e alvo | Recentramento/anel/flare; fundos e offsets na cadeia completa | Dados e alvo congelados |
| Reotimização | Três sementes, sondas e referência nos novos pontos | Sementes atualizadas |
| Pilotos | Réplicas, mistura, autocorrelação e throughput | Orçamento de produção |
| Produção | Cadeias novas por precisão de quantis | Resultados condicionais; global só com pesos |
| Robustez | Prior de massa, dados alternativos, resíduos | Alcance da interpretação física |

Frentes de dados e preparação do cluster podem ocorrer em paralelo à certificação numérica. Dependências científicas permanecem: não concatenar alvos antigos/novos; pilotos físicos exigem configuração definida. Novos pontos fora do domínio testado reabrem verificações locais, não toda a auditoria.

Critério de suficiência

Parâmetros podem permanecer pouco determinados e próximos do prior. O objetivo não é estreitar todos os intervalos. Sua incerteza precisa ser explorada e propagada para as funções científicas, especialmente z. Cadeia presa não prova posterior uniforme.

Meta inicial do plano: erro Monte Carlo dos quantis 16/50/84 ≤5% da largura p84−p16, concordância entre réplicas/janelas e ausência de deriva relevante. Cinquenta tempos de autocorrelação são heurística, não prova global. Resultados condicionais por região são publicáveis com alcance explícito; pesos globais exigem exploração ou integral validada.

Draft disponível agora

Pode-se escrever descrição dos dados reconstruídos, algoritmo, priors, testes analíticos, falhas e correções demonstradas, proveniência e arquitetura por observação. Não apresentar os intervalos antigos como resultados validados nem preferências de pontos como probabilidades. Dois testes sintéticos de recuperação autorizam teste interno, não cobertura nominal.

Não há deadline de entrega especificado no registro. O tempo até produção não pode ser prometido a partir de uma medição isolada; será dimensionado por configuração final e pilotos no cluster. Segurança de credenciais expostas é uma pendência separada: nenhuma senha ou token aparece neste relatório.

Fontes principais: docs/plano_otimizado_producao_2026-10-07.md; docs/plano_portoes_validacao_2026-10-05.md.

16  Literatura e método de consolidação

## Literatura e método de consolidação

A literatura abaixo foi consultada durante a revisão e informa escolhas de método. Ela não demonstra defeito específico no MAGNUS nem transfere precisão/custo entre atmosferas e implementações.

Bogdanov et al. (2021)

Verificação cruzada superfície→observador e plano da imagem; controles de modelos de pulsação.

https://arxiv.org/abs/2104.06928

Choudhury et al. (2024)

Comparação de códigos e geometrias difíceis; referência cara separada da resolução de inferência.

https://arxiv.org/abs/2406.07285

Zhang & Yu (2026), Monk-NS

Integração no plano da imagem e refino adaptativo; estratégia alternativa, não obrigação de migração.

https://arxiv.org/abs/2603.20870

Sherlock, Golightly & Henderson (2017)

Delayed acceptance com correção do alvo caro; usar depois de validar esse alvo.

https://arxiv.org/abs/1509.00172

Foreman-Mackey et al. (2013)

Ensemble afim; correção local e eficiência não equivalem a exploração global.

https://arxiv.org/abs/1202.3665

Cotter, Dashti & Stuart (2010)

Relação entre aproximação do modelo direto e posterior; necessidade de controlar erro do operador.

https://arxiv.org/abs/0909.2126

Fontes e limitações desta consolidação

Foram lidos relatórios estruturantes, examinados artefatos JSON e consolidados os números comunicados ao revisor. A execução histórica de cada teste não foi repetida para gerar este PDF. O inventário a seguir registra arquivos disponíveis e hashes no momento da elaboração; não é uma certificação de todos os seus conteúdos.

Artefatos podem conter vereditos antigos acompanhados de campos CORRECAO. A narrativa usa a interpretação corrigida; campos antigos não são automaticamente promovidos ao estado atual. Últimos estados de jobs e autorizações procedem da conversa quando ainda não há resultado consolidado no arquivo.

Os números de suítes (82, 86, 101, 113, 303 etc.) se referem a versões e projetos distintos. Não foram somados como “total de testes” e não substituem portões científicos. Teste unitário, réplica estatística, matriz numérica e redução real têm papéis diferentes.

17  Inventário reproduzível de relatórios e artefatos

## Inventário reproduzível de relatórios e artefatos

Caminhos relativos à raiz do MAGNUS, salvo os explicitamente externos. O inventário inclui Markdown e JSON em docs, sem credenciais e sem binários. Para rastreio completo, usar inventario_fontes_2026-10-09.json junto do PDF; cada entrada traz SHA-256 completo e tamanho.

docs/

| Arquivo | SHA-256 (prefixo) | Bytes |
|---|---|---|
| VERSAO_​NAO_​IDENTIFICADA_​reducoes_​2026-10-07.md | 4497d08ffad7 | 7791 |
| alvo_​estatistico_​v2.md | 3a5638304e1f | 11299 |
| alvo_​por_​observacao_​decidido.md | 352e5c0309b6 | 3322 |
| amostrador_​diagnostico.md | fe0230757e93 | 3223 |
| atmosfera_​fina_​ciclo_​limite.md | a40ca9a1e7c2 | 5478 |
| calibracao_​offset_​fase.json | 06a677f9422a | 2975 |
| compara_​formulacoes.json | 62d8655ef615 | 62591 |
| compara_​malhas.json | bc5948295e58 | 11088 |
| compara_​malhas_​fino.json | 90a0dbfd1fd0 | 13836 |
| compara_​reducao_​limpa.json | 38a1c6ded7df | 7838 |
| congelamento_​2026-10-07.json | 5777da120329 | 2620 |
| convergencia_​nas_​diferencas.json | 9c336decd470 | 2138 |
| convergencia_​producao_​1_​de_​2.json | 3458b376c1fc | 2990 |
| convergencia_​producao_​2_​de_​2.json | e10b80c510bb | 3321 |
| cruza_​tabela.json | 913f02cc8b52 | 5515 |
| cruza_​tabela_​convergencia.json | 1a39ecd25e2f | 7660 |
| custo_​isolado.json | 09f40a9d04a8 | 1899 |
| custo_​por_​fase.json | c9b6f49bd122 | 3831 |
| diagnostico_​pileup.json | 012239f8735b | 6006 |
| discretizacao_​conjunta_​A.json | 3cc9898126a5 | 3629 |
| discretizacao_​conjunta_​B_​aberto.json | 14c181458a15 | 3630 |
| discretizacao_​conjunta_​B_​colado.json | 9ed82a7edde1 | 3634 |
| discretizacao_​de_​fase.json | 005d97605065 | 6667 |
| discretizacao_​de_​fase_​A.json | 3a9dfa11367b | 3273 |
| discretizacao_​de_​fase_​B_​aberto.json | 55c0b4f54b24 | 3289 |
| discretizacao_​de_​fase_​B_​aberto_​M3.json | b3e8117622bc | 3285 |
| discretizacao_​de_​fase_​B_​colado.json | 005d97605065 | 6667 |
| distancia_​rbs1223.md | 1c7372ab1e9f | 3274 |
| erro_​da_​familia.json | 786e08f164b2 | 23031 |
| escolhe_​producao.json | 2d4ed3dff514 | 12965 |
| estimador_​do_​fundo.json | 9d8837fbd0dc | 14897 |
| exposicao_​por_​fase.json | 2895ada6fb87 | 38035 |
| exposicao_​por_​fase.md | 960117068455 | 7103 |
| fase_​consolidada.json | 7a8a2af94dc8 | 7674 |
| frente_​de_​dados_​2026-10-07.md | 62adea166a71 | 3287 |
| frente_​do_​integrador_​estatuto.json | c5475d790496 | 2272 |
| fundo_​por_​observacao.json | d0efbc72dac0 | 53245 |
| fundo_​reconstruido.json | 0417fc026182 | 8523 |
| grade_​ho_​vs_​magnus.md | 65fdff6092f1 | 4478 |
| incerteza_​fundo.json | de1f82d5f90b | 62496 |
| isolamento_​alinhamento_​2026-10-07.json | 0841c9140540 | 5965 |
| limite_​a0.json | 6e64f820fdf2 | 10841 |
| literatura_​numerica_​mcmc_​2026-10-05.md | 0e15eb908910 | 9652 |
| localiza_​malha.json | ab4dbbabd609 | 19464 |
| malha_​com_​tabela_​fina.json | 5706c58e2cb2 | 9830 |
| malha_​tabela_​fina_​sequencia.json | 09b3f1081bda | 13866 |
| manifesto_​estatistico.json | c89120f1b12e | 13808 |
| marginaliza_​fundo.json | d8e4c6c7ac1b | 9024 |
| normalizacao_​do_​alvo_​v2.md | ff5650ec5308 | 5359 |
| offset_​relativo.json | 0c1bdd05016d | 22284 |
| parecer_​segunda_​opiniao_​2026-10-04.md | cef7162e2519 | 8369 |
| pileup_​pre_​corte.json | d4886804e21a | 6850 |
| plano_​otimizado_​producao_​2026-10-07.md | c2cf504a751e | 11825 |
| plano_​portoes_​validacao_​2026-10-05.md | 2d7a17e3576d | 31380 |
| plano_​recuperacao_​inferencia_​2026-10-04.md | fba151a7c3d6 | 16412 |
| plano_​tres_​etapas_​2026-10-08.md | efe521b0a610 | 4036 |
| por_​observacao.json | 56a9e59a0c54 | 42936 |
| por_​observacao_​2026-10-07.md | 84a24f4b3407 | 8884 |
| portao_​de_​dados.json | c1d15842eee2 | 72032 |
| portao_​de_​dados.md | 246de93cd44b | 6795 |
| portao_​de_​dados_​2026-10-07.md | 2641d86da522 | 7612 |
| priori_​do_​fundo.json | ebb13689dc30 | 6732 |
| pulso_​reajuste.md | b335353bbf0c | 2899 |
| relatorio_​rodada_​coordenada_​2026-10-07.md | a3e8f056d4d5 | 15517 |
| reuso_​por_​observacao.json | 85d9f2117643 | 10669 |
| segunda_​opiniao_​2026-10-04.md | b44a2ca9b0b0 | 7087 |
| suavizacao_​inerte.json | 8349db7d3be3 | 1881 |
| tempo_​scicom.json | cc9e7b7245b9 | 2504 |
| temporal_​psi0_​A_​480_​115200.json | a6c2a8bf901f | 4931 |
| temporal_​psi0_​A_​480_​115200_​M15.json | 4b5a198dad11 | 3493 |
| temporal_​psi0_​A_​480_​115200_​M5.json | 650d8edca8dd | 3490 |
| temporal_​psi0_​B_​aberto_​480_​115200.json | c05f46404dd5 | 2316 |
| temporal_​psi0_​B_​aberto_​480_​115200_​M5.json | ad2e940bdcee | 3492 |
| temporal_​psi0_​B_​colado_​480_​115200.json | ee6867db2408 | 2315 |
| temporal_​psi0_​B_​colado_​480_​115200_​M5.json | f99cbd7c4fb5 | 3492 |
| tolerancia_​familia.json | 73257d3e71ff | 20650 |
| vacuo_​convergencia_​literatura.md | 458773953d71 | 27941 |
| valida_​exposunn.json | 950d7e511e23 | 15496 |
| valida_​malha_​psi0.json | 8d43d122ed98 | 7114 |
| valida_​offset.json | ced8c8e8f1d5 | 7086 |
| valida_​reuso.json | 368029bb4f40 | 7313 |
| validacoes_​para_​o_​artigo.md | b1aa2c0ae11c | 6040 |
| verificacoes_​do_​revisor_​2026-10-04.md | 31aaa5c7d798 | 6185 |

docs/incidente_resolucao_2026-10-04/

| Arquivo | SHA-256 (prefixo) | Bytes |
|---|---|---|
| P1_​mecanismo_​2026-10-06.md | ffaca7bbbf39 | 9140 |
| candidata_​p1.json | 77a58f0d3eed | 9191 |
| candidata_​p1_​2026-10-07.md | 158b990c7da6 | 8088 |
| candidatos_​p1.json | b0c775644277 | 71621 |
| convencao_​quadro.json | 8416e645ff36 | 7236 |
| convergencia_​quadratura.json | 9e518c23e2f5 | 9606 |
| corrida_​cluster_​46150.json | 9a653cb5ae66 | 2485 |
| determinismo_​do_​motor.json | a61f880566bd | 5400 |
| diagnostico_​alvo_​novo.json | d71a1f415ad7 | 3073 |
| fechamento_​PE_​2026-10-06.md | 243865bd6041 | 11813 |
| fechamento_​PE_​2026-10-06b.md | 0e42281052a8 | 11495 |
| fechamento_​tolerancia_​2026-10-06.md | 6a3b7fa0c925 | 7378 |
| limite_​a0_​2026-10-06.md | f48e9c04123e | 8565 |
| manifesto_​dados_​p1.json | 44cb4cc71d5c | 1900 |
| matriz_​gauss_​declarada.md | 68e3b22740b3 | 2718 |
| mecanismo_​p1.json | a3fa6a3babc5 | 3562 |
| mistura19.json | 665da72610ab | 14828 |
| modoA.json | cadb0045525d | 1140 |
| modosB.json | f7b99fead99d | 1147 |
| onde_​nasce_​o_​residuo.json | e3f6e8522975 | 11842 |
| plano_​imagem_​p1.json | 47559b84d093 | 13451 |
| por_​imagem_​p1.json | be5e6ba771fe | 5264 |
| portao_​P1.json | e2587de926ac | 15755 |
| portao_​PE.json | 2361abdc3a98 | 60830 |
| portao_​reproducao_​p1.md | 49edaef41556 | 3224 |
| pre_​registro_​candidatos_​p1.md | 23ddea3ba7df | 5894 |
| pre_​registro_​controles_​nao_​uniformes.md | 28c69ffabf05 | 4147 |
| pre_​registro_​plano_​imagem.md | f0c7c231074a | 4897 |
| pre_​registro_​replicas.md | d10230eb127c | 4545 |
| pre_​registro_​tolerancia_​familia.md | 14ff65710bd4 | 2759 |
| qnum_​exposicao.json | 7982ff6dbb30 | 12515 |
| registro.json | 2b8217ded57a | 2379 |
| resolucao_​nos_​casos_​dificeis.json | 9a7c46947e27 | 5613 |
| teste_​gauss.json | 8de8464c5ecf | 4643 |

docs/auditoria_2026-09-29/

| Arquivo | SHA-256 (prefixo) | Bytes |
|---|---|---|
| ALTERACOES.md | 77f5233cb920 | 12943 |
| RELATORIO.md | 1ee4deaf5075 | 20742 |
| RESPOSTA.md | 39c2b4440941 | 17930 |
| convergencia_​bacias.json | d67468640a17 | 2152 |
| dimensao.json | bd4e03a23c59 | 458 |
| dimensao_​17livres.json | 8aa67c629b6e | 1621 |
| evidencias.json | 3842df25973d | 23628 |
| modelo_​verificado.json | aef80466efbd | 977 |

docs/auditoria_2026-10-01/

| Arquivo | SHA-256 (prefixo) | Bytes |
|---|---|---|
| RELATORIO.md | 63e2576f6983 | 14301 |
| evidencias.json | d301c3e4c121 | 2166 |

externo

| Arquivo | SHA-256 (prefixo) | Bytes |
|---|---|---|
| RELATORIO.md | 64a05c723cb4 | 15218 |
