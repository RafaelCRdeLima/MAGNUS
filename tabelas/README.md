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
