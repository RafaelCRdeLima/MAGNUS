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
