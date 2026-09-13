#!/usr/bin/env python3
"""Grade MAGNUS DENSA em lg B (0,05 dex) para a RBS 1223.

Motivo (13/09/2026): a feição do ciclotron do próton está DENTRO da tabela
(o gerador refina 60 energias em torno de E_cp em cada nó de B) e é profunda
(w cai a 0,17 em lg B = 13,5 e a 0,03 em 13,8). Com nós a 0,3–0,5 dex, a
interpolação linear em lg w no B ajustado (13,6–13,7) produz DUAS depressões
espúrias — nas energias dos nós vizinhos — e nenhuma em E_cp(B). Com 0,05 dex
(E_cp varia 12 % entre nós, bem menos que a largura da feição) a interpolação
volta a ser fiel, e a linha gaussiana deixa de compensar um artefato.

Eixos restritos ao que a estrela usa: lg B cobre o polo (13,6–13,8) e o
equador dipolar (B_p/2, ~13,4); lg T de 5,7 a 6,3 (T_p ~ 1,25 MK = 6,10;
T_min = 0,3 T_p já fica abaixo da grade e é grampeado, como antes); lg g em
torno do implicado pelos ajustes (14,24–14,28).

Uso:
    OMP_NUM_THREADS=2 python3 scripts/tabela_campo_g_denso.py \\
        --saida build/magnus_campoBg_denso.magnus --workers 5
Retomável pelo checkpoint (.ckpt.npz) como o gerador original.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import tabela_campo_g_grade as grade  # noqa: E402

grade.LOG_B = [round(13.20 + 0.05 * k, 2) for k in range(15)]   # 13.20 ... 13.90
grade.LOG_T = [5.699, 5.799, 5.898, 6.000, 6.114, 6.204, 6.301]
grade.LOG_G = [14.0, 14.2, 14.4]
grade.THETA_B_DEG = [0.0, 30.0, 60.0]

if __name__ == "__main__":
    if not any(a.startswith("--saida") for a in sys.argv[1:]):
        sys.argv += ["--saida", str(ROOT / "build" / "magnus_campoBg_denso.magnus")]
    print(f"grade densa: {len(grade.LOG_B)} B x {len(grade.LOG_T)} T x "
          f"{len(grade.LOG_G)} g x {len(grade.THETA_B_DEG)} theta_B = "
          f"{len(grade.LOG_B)*len(grade.LOG_T)*len(grade.LOG_G)*len(grade.THETA_B_DEG)} modelos",
          flush=True)
    grade.main()
