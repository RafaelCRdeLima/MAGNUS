# Dados de atmosfera

Este diretório guarda a **procedência** dos dados de terceiros que o
solucionador de atmosferas lê. Os dados em si não são redistribuídos no
repositório; são de outros autores e devem ser citados como tal. Para
obtê-los:

    python3 scripts/baixar_dados_terceiros.py

O script baixa cada arquivo do sítio original e confere o `sha256` contra o
valor registrado no `PROVENIENCIA.json` de cada pasta. Um hash diferente
interrompe o script, porque a fonte pode ter mudado.

## De onde veio cada coisa

### `potekhin_magnetic_h/`

Equação de estado e opacidades de Rosseland para hidrogênio parcialmente
ionizado em campo magnético forte, de **A. Y. Potekhin & G. Chabrier**
(Neutron Star Group, Ioffe Institute).

- Página: <http://www.ioffe.ru/astro/NSG/Hmagnet/hmagtab.html>
- Baixado de: <http://www.ioffe.ru/astro/NSG/Hmagnet/hmagnet.tar.gz>
- `sha256` do tar: `e1affeb26e4133effef801346e0d976bb68b783acb839ed0a464780231f58dae`
- 47 arquivos, lg B de 10,5 a 15,0
- Hash de cada arquivo em `potekhin_magnetic_h/PROVENIENCIA.json`

**Citar:** Potekhin A. Y., Chabrier G., 2003, ApJ, 585, 955
(arXiv:astro-ph/0212062); e Potekhin A. Y., Chabrier G., 2004, ApJ, 600, 317.

As duas colunas de opacidade são `lg(K0)`, ao longo de **B**, e `lg(K1)`,
perpendicular. A relação angular é dos próprios autores:

    1/K(theta) = cos²(theta)/K0 + sin²(theta)/K1

`atmosfera/magnetizada.py` compara o seu tensor de Rosseland (K₀, K₁) com
essas colunas; `tests/test_magnetizada.py` faz disso um teste, pulado se os
arquivos não estiverem no disco.

### `pc03_hmagnet/`

As tabelas lg B = 13,0 e 13,5 do mesmo conjunto, descomprimidas, mais
`hmn13_5.dat.gz`. `atmosfera/atomico.py` lê daqui as frações de ionização
`x(H)`, `x(H0)`, `x(H2)` que validam e substituem a fração neutra do modelo
próprio. O script de download as extrai do tar acima.

### `van_hoof/gauntff.dat`

Fatores de Gaunt livre-livre termicamente mediados, não relativísticos, de
**van Hoof et al. 2014**, MNRAS, 444, 420. Grade de 81 pontos em
lg(γ²) × 146 em lg(u). Baixado de <https://data.nublado.org/gauntff/gauntff.dat>.
Sem o arquivo, `atmosfera/estrutura.py` cai no Gaunt de Elwert–Born.

### `nsmaxg_ho/`

Vinte e seis espectros NSMAXG de **W. C. G. Ho, A. Y. Potekhin & G. Chabrier**
(modelo `nsmaxg` do XSPEC), usados apenas como gabaritos por
`scripts/auditoria_gabaritos.py` (conservação de σT⁴, comparação de espectros).
Os links de dados da HEASARC estavam fora do ar quando foram obtidos; o
`PROVENIENCIA.json` registra o `sha256` do zip e de cada arquivo, e o script de
download aceita o zip por `--nsmaxg-zip`.

## Arquivos derivados, opcionais

Dois arquivos derivados que o motor sabe ler **não** fazem parte do
repositório e não são necessários para os ajustes com tabela de atmosfera:

- `magnetic_anisotropy.csv`: a razão `a = K0/K1` na fotosfera, em
  (lg T, lg B, lg g), derivada das tabelas de Potekhin & Chabrier fora do
  MAGNUS. Só o modelo cinza de dois modos do motor a usa; `mcmc_fit.py` a
  passa ao motor apenas se ela existir.
- `nsmaxg_hydrogen.bin` / `.json`: os espectros NSMAXG reempacotados como
  backend espectral alternativo (`--nsmaxg-table`); igualmente opcional.

## O que essas tabelas dão e o que não dão

As tabelas de Potekhin & Chabrier dão a estrutura angular da opacidade, o que
faltava ao modelo cinza para produzir um feixe que não fosse sempre escurecido
no bordo. São médias de **Rosseland**, integradas em frequência e sobre os dois
modos de polarização; não dão a dependência espectral de cada modo. É por isso
que o MAGNUS resolve a atmosfera de dois modos por conta própria
(`atmosfera/magnetizada.py`) e usa essas tabelas como referência da opacidade
média, não como fonte do espectro.
