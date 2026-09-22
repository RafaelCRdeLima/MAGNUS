# Tabelas de intensidade produzidas

- `magnus_HB1350_completa.magnus` — a grade principal: lg B = 13,5, lg g = 14,38,
  12 temperaturas (o eixo exato do `nsmaxg_HB1350ThB00g1438` do Ho) × 6 ângulos
  θ_B (0° a 75°) × 8 μ × 160 energias. 72 modelos convergidos. Foi a primeira
  tabela usada nos ajustes da RBS 1223 (campo fixo em lg B = 13,5).
- `magnus_HB1350_completa_estilo_ho.in` — a mesma grade exportada no formato
  `.in` do Ho (fatia θ_B = 0), nas unidades dele — intercambiável a 0,1%.
- `magnus_HB1350_completa_xpsi.txt` — exportação em cinco colunas do X-PSI.
- `estagio2_B135_thetab.magnus` — a grade menor (3 T × 6 θ_B) do primeiro
  varrimento em θ_B.

Regeneráveis por `scripts/tabela_estagio2.py` (~75 s por modelo) e
`scripts/exporta_formatos.py`.

## Grades com eixo de campo (formato MAGNUSI2)

- `magnus_campoBg.magnus` — lg B = {12,5; 13,0; 13,5; 13,8} × 12 T (5,7–6,8) ×
  lg g {13,8; 14,0; 14,2; 14,4; 14,6} × θ_B {0, 30, 60} × 8 μ × 160 E, totalmente
  ionizada (`scripts/tabela_campo_g_grade.py`, 720 modelos, ~2,6 h com 5 workers).
  **Superada**: com a feição ciclotron dentro da tabela, 4 nós em B a 0,3–0,5 dex
  geram duas depressões espúrias na interpolação (`scripts/compara_tabelas_B.py`
  reproduz a figura que mostra isso).
- `magnus_campoBg_denso.magnus` — **a tabela de produção da RBS 1223 (13/09/2026)**:
  lg B = 13,20…13,90 a 0,05 dex (15 nós) × 7 T (5,7–6,3) × lg g {14,0; 14,2; 14,4}
  × θ_B {0, 30, 60} × 8 μ × 160 E (`scripts/tabela_campo_g_denso.py`, 945 modelos,
  ~3,7 h com 5 workers, pior erro de fluxo 1,5 %). Fora da grade o motor grampeia.
- `magnus_campoBg_denso_ext.magnus` — a densa estendida (14/09/2026): lg B 13,20…14,00
  (17 nós) × 7 T × lg g {14,0…14,8} (5 nós) × θ_B {0, 30, 60}; mescla de
  `magnus_campoBg_denso` com as fatias novas (`scripts/mesclar_tabelas.py`). Usada
  só na investigação do modelo γ (que foge para g > 14,4); o fiducial usa a densa.

## Tabelas v2 (16/09/2026): livre-livre com logaritmo de Coulomb quantizante

Todas as tabelas acima foram geradas com o Gaunt não magnético em todas as componentes
cíclicas do livre-livre. A validação contra as médias de Rosseland de Potekhin & Chabrier
2003 e as opacidades monocromáticas de Suleimanov, Potekhin & Werner 2009 mostrou o modo X
3–6 vezes transparente demais em lg B ≥ 13; a correção (Eq. 44 de PC03, `quantizing_coulomb_ratio`
em `atmosfera/magnetizada.py`) e a aceleração de Ng na iteração de temperatura entram nestas
tabelas, geradas com 300 iterações. **São as tabelas de produção a partir de 16/09/2026**; as
anteriores (campoBg, denso, denso_ext, denso_ffq, denso_atom) ficam como registro.

- `magnus_campoBg_denso_v2.magnus`: mesmos eixos da densa (lg B 13,20–13,90 a 0,05 dex, 7 T em
  5,7–6,3, lg g {14,0; 14,2; 14,4}, θ_B {0, 30, 60}, 8 μ, 160 E), hidrogênio totalmente ionizado,
  sem polarização do vácuo (ramo ainda não validado contra as referências publicadas).
- `magnus_campoBg_denso_v2_atom.magnus`: idem com ionização parcial (fração neutra de PC03 em todo
  o eixo de B e ligado-livre atômico de primeira passada), para medir o efeito dos átomos.
- As versões intermediárias de 15/09 (livre-livre corrigido, 180 iterações, sem Ng) não são
  distribuídas; foram superadas pelas v2.

## Tabela v3 (18/09/2026): a v2 com os eixos estendidos, ainda SEM vácuo

`magnus_campoBg_denso_v3.magnus` é a **v2 estendida**, não uma física nova. Foi
montada por `exploracoes/estende_v2.sh`, que gera as fatias que faltavam com
`scripts/tabela_campo_g_denso.py` **sem `--vacuo`** e as une à v2 com
`scripts/mesclar_tabelas.py`. As células em comum com a v2 são byte a byte
idênticas; qualquer um pode conferir comparando as duas nos nós compartilhados.

Eixos: 17 lg B (13,20–14,00 a 0,05 dex) × 9 lg T (5,500–6,301) × 5 lg g
(14,0–14,8 a 0,2) × 3 θ_B × 8 μ × 160 E, ou seja **2295 modelos**.

Os três eixos foram estendidos porque o ajuste encostava nos limites da v2: o
campo parava em 13,871 contra o teto de 13,90; a gravidade pedia lg g = 14,59
contra o teto de 14,4; e o equador, em 0,3 T_polo = 0,45 MK, caía abaixo do piso
de 0,5 MK, com 11 por cento da área extrapolada.

**O que ela NÃO tem:** polarização do vácuo. Herda isso da v2. Em consequência a
rotulagem de modos por elipticidade, que só age quando as amplitudes do vácuo são
recalculadas por profundidade, fica sem efeito nela. A interpolação em energia
reescalada pelo ciclotron é do MOTOR (`engine/main.cpp`), não da tabela, e vale
para qualquer uma.

Chamar a v3 de "tabela corrigida" sem qualificar induz ao erro: ela corrige o
livre-livre, não o vácuo.

## Tabela v4, COM vácuo (pronta em 22/09/2026)

`magnus_campoBg_vacuo_v4.magnus`, a primeira com polarização do vácuo, rotulagem
de modos por elipticidade e integrador direto. **17 lg B × 9 lg T × 9 lg g × 3 θ_B
× 8 μ × 160 E = 4131 modelos**, 21,2 MB, montada dos dois blocos com
`scripts/mesclar_tabelas.py`. O passo em lg g é 0,1, o dobro da resolução da v3.

Qualidade, do erro de conservação de fluxo por modelo: mediana 1,0 %, p90 3,1 %,
1,7 % dos modelos acima de 10 % e 5 acima de 30 %. A cauda ruim está toda no canto
quente e de campo alto (lg B ≥ 13,9 com lg T ≥ 6,20, quase sempre em lg g 14,1 e
θ_B oblíquo), fora da região que o ajuste da RBS 1223 usa.

15,8 % das células saem grampeadas em |lg w| = 30, **todas acima de 3 keV**, onde a
razão contra o Planck estoura na cauda de Wien; a v3 grampeia os mesmos 15,85 %.
Dentro de 0,15–1,0 keV são 0,03 %.

### Quanto o vácuo muda, medido contra a v3

Em lg g 14,4, lg T 5,9–6,1, na banda 0,15–1,0 keV:

| lg B | mediana | p90 | fração > 12 % |
|---|---|---|---|
| 13,30 | 0,2 % | 2,4 % | 0,8 % |
| 13,55 | 0,5 % | 9,7 % | 7,9 % |
| 13,85 | **5,0 %** | **70 %** | **30 %** |
| 14,00 | 7,3 % | 126 % | 37,5 % |

O campo polar ajustado da RBS 1223 é lg B ≈ 13,85, então **o vácuo não é desprezível
para esta fonte**. A estimativa pela Eq. 5 de van Adelsberg & Lai (2006), que sugeria
efeito pequeno abaixo de B_l, subestima; o que vale é esta medida direta.

## Grades com vácuo (produzidas em 19–22/09/2026)

As primeiras tabelas COM polarização do vácuo, rotulagem por elipticidade e
integrador direto. Dois blocos que se mesclam no fim, 17 lg B × 9 lg T × 9 lg g
× 3 θ_B = **4131 modelos**, com passo 0,1 em lg g (o dobro da resolução da v3,
porque o ajuste cai entre dois nós de 0,2):

| Bloco | lg g | Modelos |
|---|---|---|
| A | 14,0 14,2 14,4 14,6 14,8 | 2295 |
| B | 14,1 14,3 14,5 14,7 | 1836 |

Custo medido: 0,43 GB de pico e ~5,3 h por modelo num núcleo do Epyc do sci-com,
contra 1,6 h no desktop. Ver `cluster/scicom/job_grade.slurm`.
