# Run0 — estado do projeto antes da produção

09/10/2026. Commit do congelamento: `e5586c6` (e seguintes). Executável validado:
`ψ₀` 480 bandas, tabela de raios 115200 nós.

Escrito para quem vai planejar e rodar a produção: diz o que está fechado, o que
não está, e o que a produção custa com o que está fechado.

---

## 1. O problema que estamos resolvendo

**O objeto.** RBS 1223 (RX J1308.6+2127), uma estrela de nêutrons isolada do
grupo dos XDINS. O alvo científico são **M, R e o desvio para o vermelho
gravitacional z**, com incerteza propagada — não um ajuste de melhor ponto.

**Os dados.** 13 observações XMM-Newton EPIC-pn, **329972 contagens em 125667,3
s**, resolvidas em 32 bins de fase × 40 bins de energia, de 0,15 a 1,0 keV.

**O modelo.** MAGNUS integra a superfície de uma estrela compacta com desvio
gravitacional e deflexão de luz, mapa térmico de dois polos (lei de
Perez-Azorin), campo dipolar com atmosfera magnetizada de hidrogênio (tabelas
NSMAXG), linha de absorção e absorção interestelar, e dobra o resultado pela
resposta do instrumento. São **19 parâmetros livres**.

**A dificuldade que organiza todo o trabalho recente.** As soluções candidatas
não estão bem separadas pelos dados. Os três vetores preservados — `B colado`,
`B aberto` (duas geometrias de campo) e `A` — estão separados por

| contraste | ln L |
|---|---|
| `B aberto` − `B colado` | ≈ **1,9** |
| `A` − `B colado` | ≈ **3,2** |

Diferenças dessa ordem **só decidem entre as hipóteses se o erro do modelo for
muito menor que elas**. E até esta semana vários efeitos eram da mesma ordem:

| efeito | em diferenças de ln L | natureza |
|---|---|---|
| exposição por fase não corrigida | **1,01** | erro do **operador observacional**, não do integrador |
| tabela de raios no padrão de 1800 nós | **0,20** | quadratura dos raios |
| discretização da malha | **0,23** | quadratura da superfície |
| quadratura de fase, ponto médio | ≈ **0,63** | quadratura temporal |

**Estes números não se somam e não formam uma fração única da evidência**: foram
medidos em **configurações diferentes**, e os eixos interagem — o peso da tabela
nas diferenças cresce conforme a malha refina, e a precisão espacial depende da
quadratura temporal. O que se pode dizer é que **o maior deles era da ordem de
metade da separação entre as duas geometrias de campo**, e que a escolha entre
elas não era separável do método.

Então o problema imediato não é astrofísico, é este: **tornar o erro numérico
pequeno o suficiente, e demonstrá-lo, para que a comparação entre as hipóteses e
o intervalo de z sejam decididos pelos dados e não pelo método.**

---

## 2. Resultados até o momento

### 2.1 Dados — fechados no alcance testado

| item | estado | métrica |
|---|---|---|
| reconstrução v3 das 13 observações | fechada | 329972 contagens, 125667,3 s, proveniência com sha256 e semente do SAS |
| exposição por fase, dos quadros (`EXPOSUnn`) | **implementada e no alvo** | soma bate com a exposição a 9 casas nas treze |
| fundo por observação, estimador por área | **implementado** | conferido contra rota independente a **5e-16** |
| 13 respostas e perfis registrados | fechado | manifesto com 22 entradas, base declarada em 21, zero ambiguidades |
| alinhamento por observação | validado em resolução inteira | regra `δ_o = s_o/32` ciclos |

**Dois achados sobre o dado e o instrumento:**

- a observação **`0402851001` tem fundo 4,6 vezes o das outras doze** e, com
  **5,07%** da exposição, fornece **19,24%** do fundo coadicionado; removê-la
  move o fundo em 17,55%. Isso justifica **diagnóstico, não exclusão
  automática**, e muda o argumento do limiar de *flare*, que valia para a fonte
  e não para o fundo;
- o motor **descarta em silêncio a resolução temporal do instrumento** quando
  ela é menor que um bin de fase. Os 73,4 ms reais do EPIC-pn valem 0,23 de um
  bin a 32 fases e são jogados fora inteiros. É um efeito instrumental real que
  o modelo não tem, e não estava declarado em lugar nenhum.

### 2.2 Alvo estatístico — decidido, parcialmente implementado

- **referência científica:** alvo **por observação**, com um fundo latente por
  observação e canal, condicionado à medição de fundo daquela observação. As
  contagens da região de fundo entram como **verossimilhança da medição**, não
  como priori, com `a0 = 0,01/keV` e `β = 0` declarados;
- **a coadição não é equivalente**, e isso está medido: um escalar por
  observação absorve a área efetiva, mas sobra mudança de forma de até
  **3,34e-02**, dominada pela diferença de **resposta entre submodos**
  PRIMEFULLWINDOW e PRIMELARGEWINDOW. Comparada em diferenças de ln L, a troca
  de formulação move até **0,1367**;
- **implementação:** rodada **ponta a ponta nas treze** em 10/10/2026. O que
  está medido, cada item com seu rótulo exato:
  - **fechamento da soma** (e não só associação): a mesma avaliação por duas
    rotas, treze chamadas isoladas contra uma chamada com treze slots, dá
    857464,7248638906 contra 857464,7248642457, diferença **3,55e-07**, com pior
    diferença de grade 1,979e-08. A rota conjunta custa **13,3 vezes menos**
    (36,5 s contra 485,2 s). O resíduo foi **atribuído por medição**, não por
    leitura de rótulo: repetindo as treze isoladas com o período da observação
    de referência, a diferença cai para **0,000e+00 exato** e o pior desvio de
    grade para 8,882e-15, de modo que ele é inteiramente a aproximação do
    período comum a montante;
  - **independência entre observações**: perturbar o fundo medido de
    `0157360101` por 1,10× move **só** o termo dela, +0,196677, e os outros doze
    ficam **idênticos bit a bit**;
  - **quadratura**: 64 nós contra 1024 muda 1,079e-08 no pior caso. Isto é
    **convergência da quadratura**, não referência independente do integrando;
  - **treze espectros de fundo distintos em treze**: isto verifica **associação**
    entre observações, não o fechamento acima, que está registrado em separado;
  - **correlação ignorada**: 39 dos 209 canais caem em dois bins e geram dois
    latentes onde deveria haver um compartilhado. Medido por Monte Carlo com
    Gammas exatas e variáveis comuns, 200 mil amostras: move o ln L absoluto em
    **+1,5e-03** e os contrastes em **menos de 2e-03**, dentro do próprio ruído
    de 8,1e-04. Cabe no orçamento desta exploração, e **continua sendo
    aproximação**, não item resolvido.

### 2.3 Infraestrutura de avaliação — fechada

| item | métrica |
|---|---|
| offset de fase **explícito** no motor | V1 a V5, **zero exato** nas cinco; razão de escala 1,000000 |
| rota multi-observação (uma chamada, 13 grades) | identidade até a aproximação declarada do período comum, **1,6e-08**; custo 4,97× em 5 observações contra 5 ideal |
| determinismo entre máquinas | `(480, M=5)` **bit a bit idêntico** no cluster e no local, arquiteturas e compiladores diferentes — evidência de portabilidade **nessas avaliações**, que dispensa repetir a validação lá, e não um atestado geral |
| política de falha do motor | para na primeira falha não recuperada, com checkpoint |

**Uma correção de fundo:** `magAzim` **não** é um operador de fase exato —
difere de uma rotação rígida por 0,015 a 0,061 em ln L e até 3,0e-04 por célula.
O offset explícito, que só muda o instante amostrado, é exato.

### 2.4 Precisão numérica — o bloqueio principal, agora com solução operacional

**O componente dominante era a tabela de raios, e era o mais barato de
corrigir.** No padrão do motor, **1800 nós**, ela move as diferenças de ln L em
até **0,1986** a 480 bandas — mais que o refino de malha de 480 para 960, que
movia 0,1077. Converge a partir de **28800** nós, e corrigi-la custa **3 s por
chamada**.

**A escolha da configuração espacial, contra a referência `ψ₀` 960 / 115200:**

| configuração | pior contraste | veredito |
|---|---|---|
| produção 120 bandas | 0,2190 | fora de 0,1 |
| produção 240 bandas | 3,7782 | patológica |
| produção 480 bandas | 0,2316 | fora de 0,1, **pior que 120** |
| **`ψ₀` 480 bandas + tabela 115200** | **0,0130** | **dentro** |

**Esse 0,0130 foi medido em M=1**, e portanto é específico desse corte
temporal. O componente espacial medido depois em M=15 vale **−0,0947**, sete
vezes maior — isto é, **a precisão espacial depende da quadratura temporal**, e
nenhuma das duas aprovações vale fora do par em que foi medida.

As três configurações **testadas** da malha antiga não atendem ao orçamento, e a
de 480 é pior que a de 120. Isso justifica escolher outra família **agora**; não
prova que nenhum refino futuro funcionaria.

**Eixo temporal, com o espacial fixo:**

| nível | pior contraste contra M=15 | veredito |
|---|---|---|
| M=1 (ponto médio atual) | ≈ 0,63 | longe |
| M=3 | 0,1163 | fora de 0,1 |
| **M=5** | **0,0908** | dentro, margem de 9% |
| M=9 | 0,0046 | **certificado como referência nos três vetores** |

**E o erro conjunto, medido no cluster**, canto grosso `(480, M=5)` contra canto
fino `(960, M=15)`: **0,0039**, com 96% de margem. **Mas ele se forma por
cancelamento:**

| componente | valor |
|---|---|
| temporal sozinho, a 480 bandas | **+0,0908** |
| espacial sozinho, em M=15 | **−0,0947** |
| soma | −0,0039 |

Os dois eixos têm sinais opostos e magnitudes quase iguais. **O cancelamento não
invalida o resultado** — a verossimilhança usa a previsão completa, e 0,0039 é o
que a previsão completa erra. Mas **0,0039 é a discrepância observada neste par,
não uma margem garantida de 96% para outros parâmetros**: se qualquer eixo se
mover, os componentes de ~0,09 reaparecem.

**Frente A FECHADA em 10/10/2026**, jobs 47139 e 47228. O erro conjunto
`(ψ₀480, M=5)` contra `(ψ₀960, M=15)` nos três contrastes:

| contraste | grosso | fino | discrepância |
|---|---|---|---|
| A − B colado | 3,258802 | 3,276952 | **−0,01815** |
| A − B aberto | 5,114936 | 5,129175 | −0,01424 |
| B aberto − B colado | −1,856134 | −1,852223 | −0,00391 |

Pior **0,0182**, dentro da tolerância de 0,1. A configuração está **aprovada
para reotimização e pilotos, com revalidação dos novos pontos**, e **não para
produção final**.

**O que isso não certifica:** estar abaixo de 0,02 aqui mede a concordância
**grosso–fino**, e não o erro da **referência fina** contra a integral contínua,
que permanece não medido.

**E a advertência sobre o cancelamento se confirmou:** os contrastes com `A`,
0,0182 e 0,0142, são **quatro vezes** o do par B, 0,0039 — o valor pequeno do par
B era em parte específico do cancelamento daquele par.

### 2.5 O que isto significa fisicamente

O maior efeito contaminante era **1,01**, da exposição por fase — que é erro do
**operador observacional** e não do integrador. Na configuração de hoje, a
discrepância medida entre o canto grosso e o canto fino é **0,0039**, e os
componentes isolados valem ~**0,09**. Contra a separação de **1,9** entre as duas
geometrias de campo, isso é **0,2%** e **5%** respectivamente.

**Três restrições sobre como ler isso**, e nenhuma é detalhe:

- os efeitos foram medidos em **configurações diferentes** e **não devem ser
  somados** nem apresentados como uma fração única da evidência;
- os números comparam **pontos preservados**, não hipóteses físicas já
  discriminadas, nem máximos reotimizados, nem pesos posteriores. O que se
  compara é ln L em três vetores fixos;
- **0,0039 é discrepância observada neste par**, não limite garantido.

Sobre a estrela — M, R, z, campo, temperatura — **nada novo foi encontrado**: o
trabalho desta semana foi todo de precisão do método.

---

## 3. O que falta para uma produção correta no cluster

### 3.1 O obstáculo é o custo, e ele é quantitativo

Custo por avaliação do **alvo completo das 13 observações**, medido no sci-com
(uma chamada do motor, rota de reuso):

| configuração | custo | fator |
|---|---|---|
| atual (120 bandas, tabela 1800, M=1) | **33 s** | 1 |
| aprovada espacial, M=1 | **174,5 s** | 5,3 |
| **aprovada + M=5** | **1620 s** | **49** |
| aprovada + M=9 | ~2916 s | 88 |

**Em 4 dias num nó de 128 núcleos (12288 core-horas), com 128 walkers:**

| configuração | avaliações | **passos** |
|---|---|---|
| atual | 1,34 M | **10473** |
| aprovada, M=1 | 254 k | **1981** |
| **aprovada, M=5** | 27 k | **213** |

**213 passos não é uma cadeia convergida em 19 dimensões.** Esse é o obstáculo
central: a configuração numericamente validada torna a produção ~49 vezes mais
cara, e um nó por 4 dias deixa de ser suficiente.

Para **3000 passos** com 128 walkers (384 mil avaliações):

| configuração | core-horas | num nó de 128 | na `ep-short` inteira (1920) |
|---|---|---|---|
| atual | 3520 | 1,1 dia | 0,1 dia |
| aprovada, M=1 | 18613 | 6,1 dias | 0,4 dia |
| **aprovada, M=5** | **172800** | 56 dias | **3,8 dias** |

Isto é: com a configuração validada, uma cadeia de 3000 passos em 4 dias exige
**a partição `ep-short` inteira** — seis nós, 1920 núcleos. Não é impossível,
mas não é um job, e colide com o epílogo que mata por usuário no nó.

### 3.2 As quatro decisões que precedem a submissão

1. **M=5 ou M=9?** M=5 passa por cancelamento frágil, a 1620 s; M=9 é a
   referência certificada, a 2916 s. A diferença é um fator 1,8 no orçamento
   inteiro. **Decisão do Rafael e do árbitro, não minha.**
2. **O orçamento de precisão vale para o quê?** O erro de 0,09 importa para
   comparar as hipóteses discretas (A contra B), cuja separação é 1,9 a 3,2. Para
   o **intervalo de z dentro de um modo** ele pode importar muito menos, e isso é
   **testável de forma barata**: avaliar o mesmo conjunto de pontos de uma cadeia
   curta nas duas configurações e comparar os intervalos de z, não os ln L. Se os
   intervalos coincidirem, a produção pode rodar na configuração barata com a
   comparação de hipóteses feita à parte, nos poucos pontos que importam.
   **Este é o caminho que eu investigaria primeiro, porque pode reduzir o
   problema em uma ordem de grandeza.**
3. **Quantos passos são necessários?** Não foi medido. Exige um piloto curto para
   estimar o tempo de autocorrelação nas 19 dimensões.
4. **Uma cadeia ou várias partidas?** As partidas diferentes paralelizam entre
   si e dão diagnóstico de convergência; isso muda o dimensionamento.

### 3.3 O que falta implementar

| item | estado |
|---|---|
| marginalização do fundo **por observação**, ponta a ponta | **rodada**, com fechamento e independência medidos (§2.2) |
| alvo por observação como **objetivo otimizável** | **implementado**, `scripts/alvo_por_observacao.py` e `reotimiza.py --por-observacao`. Sem essa opção a reotimização roda no alvo **coadicionado**, que não é o alvo escolhido. Validado ponta a ponta a 26 bandas: calibração de 56 avaliações, simplex e sonda final, com o alvo registrado na saída |
| alvo por observação no **amostrador** | **bloqueado por uma decisão de desenho, não por código.** `WorkerPool.evaluate` chama `problem.evaluate(..., return_pulse=True)` em todas as avaliações (`mcmc_fit.py:2150`, `:2169`, `:2262`) e `pool.model` no melhor ponto (`:2432`). Com treze observações há **treze pulsos e treze modelos**, e qual deles registrar não é questão de implementação: a rota recusa explicitamente em vez de devolver o da primeira observação como se fosse o do conjunto |
| propagação da incerteza do **alinhamento** | offsets validados em resolução inteira; sub-bin **não propagado** como incerteza |
| resposta temporal **sub-bin** do instrumento | **ausente do modelo**. O boxcar hoje implementado é **identidade em M = 5**, medido bit a bit, e passa a agir só a partir de M = 7 nas FULLWINDOW e M = 11 nas LARGEWINDOW. Um operador temporal consistente é **teste separado, não feito** |
| **correlação** entre latentes de bins que dividem canal | **medida** e dentro do orçamento (§2.2); **não resolvida** |
| extração **recentrada** | autorizada, **aguarda XREDUX** |
| diagnóstico **sem núcleo** (pile-up) | autorizado, **aguarda extração anular no XREDUX** |
| decisão do limiar de **flare** | **medido** (`docs/diagnostico_flare.json`): nas doze observações a mediana da taxa de partículas vai de 0,07 a 0,20 ct/s e o limiar de referência de 0,4 ct/s descarta **zero** exposição — é inerte. A `0402851001` tem mediana **2,125 ct/s**, 18× as outras, e perde **77 a 80%** da exposição em qualquer limiar de 0,2 a 1,0: não é flare isolado, é nível alto em toda a observação, e o limiar não a conserta. A decisão sobre ela é de **tratamento do fundo**, e o alvo por observação já a isola num latente próprio |

### 3.3-bis O orçamento, medido nos dois lados

| medida | estação | nó |
|---|---|---|
| avaliação, configuração aprovada | 938,2 s | **1420,9 s** (`node30`) |
| avaliação, configuração barata | 34,7 s | 62,5 s (`node30`), 46,9 s (`node26`) |
| ln P, as duas configurações | — | **um ulp** de diferença entre máquinas (`…028` contra `…029`); exato **dentro** de cada uma |

Escalabilidade (`node26`, 64 núcleos físicos + SMT): degradação de 4,2% em 64
motores simultâneos e de **81,8% em 128**, onde o throughput sobe apenas 14,6%.
A produção usa **64 trabalhadores por nó**, não 128.

Daí a lacuna que o planejamento precisava ver. Com 64 caminhantes, red/blue em
duas metades e o custo medido, cabem em quatro dias:

| configuração | iterações em 4 dias |
|---|---|
| aprovada | **≈ 117** |
| barata | ≈ 3540 |

**O que isto mede é o custo por iteração, não a suficiência.** Usar o
`burnIn = 5000` do pedido canônico como denominador foi erro meu de
enquadramento: esse valor é herdado de corridas no alvo **coadicionado** e não é
requisito do alvo novo. A métrica que decide é **amostras efetivas por hora** —
ESS por parâmetro sobre tempo de parede, com o ESS vindo do tempo de
autocorrelação medido nas próprias cadeias —, e ela só se obtém nos pilotos.

O que fica estabelecido é o custo por iteração e como ele responde a threads e
workers: com o laço de fases paralelizado, o melhor par medido leva de 122 para
**227 iterações** em quatro dias, fator 1,9. Threads reduzem latência e não
criam trabalho por hora.

Para a reotimização de C, com 1420,9 s e 133 avaliações de custo fixo: três
rodadas de 150 dão **9,6 dias** por semente, 1 × 500 dá 10,4 e 2 × 500 passa dos
14 dias da partição.

### 3.4 O que falta no cluster

| item | estado |
|---|---|
| ambiente | `~/magnus_tempo` montado e testado: Spack GCC 11.4, venv com numpy 2.4.6 e scipy 1.17.1 |
| **`bootstrap.sh`** | **corrigido**: raiz por marcador (`engine/main.cpp` + `scripts/mcmc_fit.py`), carrega `gcc@11.4.0`, confere a versão maior, compila e verifica que o binário reconhece `--observation-phase-offset` |
| **`--exclusive`** | **inviável em `ep-short`**: os seis nós estão sempre parcialmente ocupados e a fila estimou **13 dias** |
| **epílogo de fábrica** | faz `pkill -KILL -U` ao fim de **qualquer** job do usuário no nó. Com produção em vários nós isto é um risco de **derrubar as próprias cadeias**. Precisa de um nó por job e `--exclude`, ou de um job único multi-nó |
| scripts de submissão (`job_*.slurm`) | `job_frenteC_reotimizacao.slurm` **reescrito** para o alvo por observação, com ψ₀ 480, tabela 115200, M = 5, resolução efetiva zero e as aproximações pendentes declaradas. **Não submetido.** Os outros seguem anteriores à validação |
| **outra sessão submetendo pelo mesmo usuário** | em 10/10/2026 havia os jobs 47472 (`nodenv3`) e 47473 (`nodenv2`) de outra sessão. Com o epílogo de fábrica isto torna a coincidência de nó um risco **concreto**, e o lançador traz `--exclude` comentado para ser ajustado antes de submeter |
| pipeline por observação no cluster | funciona, via **cache de metadados** com caminhos reenraizados (a reconstrução não existe lá e não deve). Os treze npz com os campos `fundo_*` tiveram de ser copiados: o primeiro job morreu em dez segundos com a mensagem certa, de que o produto v1 não os traz |
| **throughput com 128 workers** | **medido** (§3.3-bis): escala até 64, quebra em 128 por SMT |

### 3.5 Caminho mínimo até a produção, na ordem

0. **confirmar `A` no canto fino**, completando os contrastes dos três vetores
   — em curso, job 47228, ~5,7 h;
1. **testar se o orçamento de precisão vale para z** (item 3.2.2) — barato, e
   pode mudar tudo o que vem depois;
2. decidir **M** à luz desse teste;
3. rodar a **marginalização por observação** ponta a ponta e verificar que ela
   não muda a configuração escolhida;
4. **piloto curto** para medir autocorrelação e dimensionar;
5. reescrever os **lançadores** para a configuração escolhida, com `--exclude` e
   um job por nó;
6. consertar o **`bootstrap.sh`** para carregar GCC 11;
7. **revalidar** a configuração nos novos melhores pontos e numa pequena
   seleção de pontos distintos do piloto; se o orçamento falhar fora das
   sementes, ajustar a configuração;
8. submeter a produção, dimensionada pelo piloto.

**P1 continua bloqueando produção física e intervalos de z publicáveis** até que
a precisão escolhida seja aprovada explicitamente. O que mudou esta semana é que
agora existe uma configuração com erro medido e uma referência contra a qual
medi-lo — o que não existia.

### 3.6 Pendências de segurança, separadas da ciência

- a **senha do cluster** foi colada em texto puro numa sessão anterior e **precisa
  ser trocada**;
- o **token do Zenodo** ficou exposto e **precisa ser revogado** em
  `zenodo.org/account/settings/applications`.

Nenhuma das duas é algo que eu possa fazer.
