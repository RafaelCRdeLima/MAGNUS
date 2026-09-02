# Dados de atmosfera

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

As duas colunas que interessam são `lg(K0)`, a opacidade ao longo de **B**, e
`lg(K1)`, a perpendicular. A relação angular é dos próprios autores:

    1/K(theta) = cos²(theta)/K0 + sin²(theta)/K1

### `magnetic_anisotropy.csv`

Derivado dos arquivos acima por `scripts/build_magnetic_anisotropy.py`, que
resolve a fotosfera pela condição `P = g/K0` — os dois lados vindos da mesma
linha da tabela, sem parâmetro novo — e tabela `a = K0/K1` em (lg T, lg B, lg g).

## Por que isto está aqui

O modelo cinza de dois modos do motor produz o feixe só pela estratificação em
profundidade, e por isso dá **escurecimento de bordo para qualquer valor dos
seus parâmetros**. Os dados da RBS 1223 pedem o contrário. O que faltava era a
opacidade depender do ângulo em relação a **B**, e é isso que estas tabelas dão.

## O que elas não dão

São médias de **Rosseland**: integradas em frequência e sobre os dois modos de
polarização. Dão a estrutura angular, não a dependência espectral monocromática
de cada modo. Um cálculo completo — NSX, de Ho & Lai (2001) — resolveria as
duas; estas tabelas resolvem uma.

E há um resultado medido que convém não esquecer: com a anisotropia da
fotosfera da RBS 1223, `a = 1,47`, o feixe fica **menos** escurecido mas ainda
não vira leque. O mecanismo só produz máximo fora da normal a partir de
`a ≈ 3`. Ou a média de Rosseland subestima a anisotropia monocromática em
0,5 keV, ou o leque que os dados pedem vem de outro lugar.

## O que se procurou e não serviu

- **`nsmax` / `nsmaxg` / `nsx` do XSPEC** (os modelos do próprio Ho, já
  declarados no `model.dat` da HEASoft): são espectros, não padrões de feixe.
  Servem a ajuste espectral, não a traçado de raios. E os links de dados que a
  documentação da HEASARC indica estão fora do ar.
- **Tabelas do X-PSI no Zenodo** (`nsx_H_v171019.out`): têm a dependência
  angular no formato certo — (logT, logg, mu, logE) — mas são **não
  magnéticas**, hidrogênio e hélio totalmente ionizados, feitas para os
  milissegundo do NICER. Erradas para uma XDINS de 10¹³ G.
