#!/usr/bin/env python3
"""Manifesto de um job, exigido pelo plano de tres frentes.

Pre-requisito 2 do plano: "Criar um manifesto raiz e um por job: identificador
de frente, commit, hashes, compilador/flags, ambiente, configuracao efetiva,
dados, resposta, periodo, offsets, resolucao temporal efetiva e caminho de
saida."

Eu nao tinha nenhum: os jobs 47139 e 47228 rodaram sem manifesto, e a
rastreabilidade deles depende de eu lembrar o que sincronizei. Isto conserta,
inclusive retroativamente, porque os hashes das entradas continuam verificaveis.

Uso: python3 scripts/manifesto_do_job.py --frente A --job 47228 \
        --config "psi0 480, tabela 115200, M=5 e M=15" --vetores "A"
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))


def sha(c):
    h = hashlib.sha256()
    with open(c, "rb") as fh:
        for p in iter(lambda: fh.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.run(("git",) + args, cwd=RAIZ, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frente", required=True, choices=["A", "B", "C"])
    ap.add_argument("--job", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--vetores", nargs="+", required=True)
    ap.add_argument("--no", default=None)
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()
    import por_observacao as PO
    obss = PO.prepara(verbose=False)
    d = {
      "frente": a.frente, "job": a.job, "gerado_em": datetime.now(timezone.utc).isoformat(),
      "configuracao_efetiva": a.config, "vetores": a.vetores, "no": a.no,
      "codigo": {
        "commit": git("rev-parse", "HEAD"),
        "arvore_limpa": git("status", "--porcelain") == "",
        "sujo_em": [l for l in (git("status", "--porcelain") or "").splitlines()][:20],
      },
      "executavel_local": {
        "caminho": "build/magnus_engine",
        "sha256": sha(RAIZ / "build/magnus_engine")
            if (RAIZ / "build/magnus_engine").is_file() else None},
      "ambiente_local": {"python": platform.python_version(),
                         "plataforma": platform.platform()},
      "entradas_cientificas": [
        {"obsid": o["obsid"],
         "npz": o["npz"], "sha256_npz": o.get("sha256_npz"),
         "perfil_csv": o["perfil_csv"], "sha256_perfil_csv": o.get("sha256_perfil_csv"),
         "perfil_rmfbin": o["perfil_rmfbin"],
         "sha256_perfil_rmfbin": o.get("sha256_perfil_rmfbin"),
         "periodo_s": o["periodo"],
         "offset_bins": o["desloc_v3_bins"],
         "offset_ciclos": PO.offset_em_ciclos(o["desloc_v3_bins"]),
         "resolucao_declarada_us": o["resolucao_us"],
         "exposicao_s": o["exposicao"], "contagens": o["contagens"]}
        for o in obss],
      "convencoes": {
        "resolucao_temporal_EFETIVA_us": 0.0,
        "por_que": ("o motor nao suaviza abaixo de um bin de fase; a 32 fases as "
                    "resolucoes de 73400 e 47700 us sao inertes, medido com "
                    "diferenca 0,000e+00, e em grade fina elas LIGAM, logo os "
                    "testes de refino as zeram"),
        "offset_desloca": "a FONTE, nunca o fundo nem a exposicao",
        "referencia_global_de_fase": PO.REFERENCIA,
        "fundo": {"a0_por_keV": 0.01, "beta": 0.0,
                  "familia": "proporcional a largura do canal",
                  "papel_das_contagens": "verossimilhanca da medicao, nao priori"},
      },
      "o_que_este_manifesto_NAO_cobre": [
        "o ambiente do CLUSTER: compilador, venv e hashes de la sao registrados "
        "pelo proprio job, e este manifesto e o lado local",
        "a identidade do executavel remoto, que e recompilado la com Spack GCC "
        "11.4 e portanto tem outro sha256 por construcao",
      ],
    }
    saida = Path(a.saida) if a.saida else RAIZ / (
        "docs/manifestos/frente%s_job%s.json" % (a.frente, a.job))
    saida.parent.mkdir(parents=True, exist_ok=True)
    if saida.exists():
        raise SystemExit("RECUSANDO SOBRESCREVER %s: o plano pede recusa por "
                         "padrao. Apague ou use --saida" % saida)
    saida.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    print("manifesto em %s" % saida)
    print("  commit %s, arvore limpa: %s" % (d["codigo"]["commit"][:8],
                                             d["codigo"]["arvore_limpa"]))
    print("  %d entradas cientificas com hash" % len(d["entradas_cientificas"]))


if __name__ == "__main__":
    main()
