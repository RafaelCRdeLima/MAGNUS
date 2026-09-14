# Tabelas de intensidade produzidas

- `magnus_HB1350_completa.magnus` — a grade principal: lg B = 13,5, lg g = 14,38,
  12 temperaturas (o eixo exato do `nsmaxg_HB1350ThB00g1438` do Ho) × 6 ângulos
  θ_B (0° a 75°) × 8 μ × 160 energias. 72 modelos convergidos. É a tabela usada
  no ajuste da RBS 1223 (lnL = 52 441,9).
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
  geram duas depressões espúrias na interpolação (ver
  `exploracoes/tabela_densa_2026-09-13/compara_tabelas_B.png`).
- `magnus_campoBg_denso.magnus` — **a tabela de produção da RBS 1223 (13/09/2026)**:
  lg B = 13,20…13,90 a 0,05 dex (15 nós) × 7 T (5,7–6,3) × lg g {14,0; 14,2; 14,4}
  × θ_B {0, 30, 60} × 8 μ × 160 E (`scripts/tabela_campo_g_denso.py`, 945 modelos,
  ~3,7 h com 5 workers, pior erro de fluxo 1,5 %). Fora da grade o motor grampeia.
- `magnus_campoBg_denso_ext.magnus` — a densa estendida (14/09/2026): lg B 13,20…14,00
  (17 nós) × 7 T × lg g {14,0…14,8} (5 nós) × θ_B {0, 30, 60}; mescla de
  `magnus_campoBg_denso` com as fatias novas (`scripts/mesclar_tabelas.py`). Usada
  só na investigação do modelo γ (que foge para g > 14,4); o fiducial usa a densa.
