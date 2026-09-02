# MAGNUS

Atmosfera magnetizada de estrela de nêutrons, resolvida em ângulo.

O MAGNUS escreve o que o `nsmaxg` não dá: um solucionador de transporte
radiativo em plasma magnetizado que entrega **I(E, μ, θ_B)** — a intensidade
por ângulo de emissão, sem a qual não há perfil de pulso calculado, só
parametrizado.

O plano está em **[PLANO.md](PLANO.md)**; o original, em
[docs/Atmosfera de Dois Modos.pdf](docs/).

## Relação com o PULSARIS

O MAGNUS é a outra versão do PULSARIS. `engine/main.cpp` partiu de uma **cópia
byte a byte** do motor dele no commit `4df2ab5`
(`md5 eea8f774fe2aae55323d13fbdf53697d`), e já divergiu: o leitor de tabela de
atmosfera, o eixo θ_B e a geometria dipolar entraram nesta cópia e não naquela.

**O PULSARIS não se toca.** O que o MAGNUS devolve para ele é um arquivo — a
tabela de intensidade —, não código.

Fica fora da cópia o `instrument_data/` (136 MB). Só o estágio 4 precisa da
resposta instrumental; até lá, aponta-se para a cópia do PULSARIS.

## Uso

```bash
make engine       # compila build/magnus_engine
make auditoria    # o aferidor do estágio 0: sigma T^4, frações e opacidades
make test         # os portões: leitor de tabela e transporte analítico
```

Os dois saem com zero. O estágio 0 está fechado: os quatro gabaritos que
reprovavam o teste de sigma T^4 são os modelos de superfície dipolar do Ho — não
são gabaritos de atmosfera local, e o aferidor agora os lista com a razão.

## A tabela de intensidade

O formato de intercâmbio do MAGNUS é `I(E, μ, θ_B)`, gravado como lg da razão
para um corpo negro isotrópico de mesma T_ef, em cinco eixos
`(lg T, lg g, θ_B, μ, lg E)`. Escrita por `scripts/tabela_intensidade.py`, lida
pelo motor:

```bash
build/magnus_engine --spectral-grid \
  --atmosphere-table build/gabarito_corpo_negro.magnus \
  --magnetic-colatitude 0
```

O eixo θ_B é o que o formato de cinco colunas do X-PSI não tem, e sem ele dois
pontos quentes em colatitudes diferentes usariam o mesmo feixe: num dipolo,
colatitude magnética de 30° e 120° dão θ_B de 16,10° e 40,89°. O motor calcula
esse ângulo ponto a ponto e o relata por ponto quente no JSON.

Uma tabela de zeros reproduz o motor sem atmosfera **bit a bit** — é a
propriedade que faz o modelo aninhar.

## O solucionador

`atmosfera/transporte.py` é o núcleo numérico: quadratura de Gauss–Legendre,
Feautrier tridiagonal por Thomas, o operador Λ explícito, o Milne cinza por
fator de Eddington variável, e espalhamento coerente com ALI mais aceleração de
Ng. Ele é medido contra soluções exatas, e não contra si mesmo:

| medida | exato | medido |
|---|---|---|
| q(0), função de Hopf | 0,577350 | 0,577356 |
| q(∞) | 0,710446 | 0,710443 |
| fluxo constante em profundidade | constante | 3×10⁻¹⁰ pico a pico |
| S(0)/√ε, espalhamento coerente | 1 | 1,0002 a 1,0006 |

Roda em Python porque roda **fora** do laço de verossimilhança: produz uma
tabela, uma vez, que o motor em C++ depois consome.

`atmosfera/estrutura.py` é a atmosfera do estágio 1 — sem campo, hidrogênio
totalmente ionizado, livre-livre com Gaunt de Elwert–Born, Thomson, correção de
Unsöld–Lucy e o operador de Compton de Kompaneets em forma conservativa. Ela
conserva sigma T_ef^4 a 6×10⁻⁶ e é numericamente convergida, mas **o portão
contra as tabelas `nsx` não fechou**: 20% a 102% contra um alvo de 5%.

O que falta ficou reduzido a uma coisa só, e ela é grande: **linearização
completa conjunta em profundidade e frequência** para o termo de Compton. Duas
formas mais baratas foram testadas com a temperatura congelada e as duas
divergem — a perturbação é amplificada pelo número de espalhamentos antes de ser
amortecida. Por isso `compton` vem desligado por padrão: o solucionador que roda
é o coerente, que converge. O operador de Kompaneets fica no lugar, com cinco
testes que o prendem.

```bash
python3 scripts/tabela_estagio1.py --temperaturas 6.0 --gravidades 14.3
build/magnus_engine --spectral-grid --atmosphere-table build/estagio1.magnus
```

A cadeia inteira funciona ponta a ponta — solucionador, formato, motor.

`atmosfera/magnetizada.py` é o estágio 2: os dois modos normais do plasma
magnetizado saem de um **autoproblema exato** do tensor dielétrico (nenhuma
fórmula de memória), as opacidades cíclicas carregam a supressão do elétron e a
ressonância do próton, e o tensor de Rosseland (K₀, K₁) se compara direto com as
colunas do Potekhin. Primeiro dia: aninhamento em B → 0 a 3×10⁻⁷, K₀/K₁
**dentro do alvo de 20%** em lg B = 12 (razões 0,88–1,11), e o **transporte de
dois modos** (`polarized_feautrier`: canais modo × ângulo com τ próprio,
acoplamento de posto 3 pelas componentes cíclicas) rodando a atmosfera inteira:
aninha no estágio 1 a ~1% quando B → 0, conserva σT_ef⁴ a 10⁻⁵, e os primeiros
espectros magnetizados ficam a ~30% do `nsmaxg` em lg B = 12 — antes do Gaunt
quantizante e da ionização parcial, nomeados no PLANO.md.

## O que já está no disco

| | |
|---|---|
| `atmosphere_data/nsmaxg_ho/` | 26 espectros de Ho, Potekhin & Chabrier, com procedência e hash. Contém `nsmaxg_HB1350ThB00g1438` — lg B = 13,50 e lg g = 14,38, que é a RBS 1223 com três casas |
| `atmosphere_data/potekhin_magnetic_h/` | 47 tabelas de EOS e opacidade de Rosseland do H magnetizado, lg B de 10,5 a 15,0 |
| `atmosphere_data/magnetic_anisotropy.csv` | derivada das anteriores pelo PULSARIS: a razão K₀/K₁ na fotosfera |
| `engine/main.cpp` | o traçado de raios relativístico que consome a intensidade |
