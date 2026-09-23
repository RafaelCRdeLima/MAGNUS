#!/usr/bin/env python3
"""Gera e verifica o manifesto das tabelas de intensidade.

As tabelas grandes nao vivem no git: elas sao depositadas no Zenodo, com DOI
proprio, e o que o repositorio versiona e este manifesto. A identidade de cada
tabela fica registrada pelo sha256, de modo que um resultado possa dizer
exatamente com qual arquivo foi produzido, e qualquer pessoa possa conferir.

Uso:
    python3 scripts/tabelas_manifesto.py gerar      # (re)escreve tabelas/MANIFEST.json
    python3 scripts/tabelas_manifesto.py verificar  # confere o que esta no disco
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
TABELAS = RAIZ / "tabelas"
MANIFESTO = TABELAS / "MANIFEST.json"

# A ordem dos eixos depende da versao do formato. O MAGNUSI1 nao tem eixo de
# campo (B fixo); o MAGNUSI2 acrescentou lg B na frente.
EIXOS = {
    "MAGNUSI1": ["lgT", "lgg", "thetaB", "mu", "lgE"],
    "MAGNUSI2": ["lgB", "lgT", "lgg", "thetaB", "mu", "lgE"],
}

# Eixos cujo produto conta um "modelo" de atmosfera (mu e energia sao saida)
EIXOS_MODELO = ("lgB", "lgT", "lgg", "thetaB")


def ler_cabecalho(caminho: Path) -> dict:
    """Le o magic e os eixos de uma tabela .magnus."""
    dados = caminho.read_bytes()
    magic = dados[:8].decode("ascii", "replace").strip()
    nomes = EIXOS.get(magic)
    if nomes is None:
        raise ValueError(f"formato desconhecido em {caminho.name}: {magic!r}")

    off, eixos = 8, {}
    for nome in nomes:
        (n,) = struct.unpack_from("<i", dados, off)
        if not 0 < n < 100_000 or off + 4 + 4 * n > len(dados):
            raise ValueError(f"eixo {nome} invalido em {caminho.name}")
        v = np.frombuffer(dados, dtype="<f4", count=n, offset=off + 4)
        off += 4 + 4 * n
        passo = np.diff(v)
        eixos[nome] = {
            "n": int(n),
            "min": round(float(v[0]), 6),
            "max": round(float(v[-1]), 6),
            "passo": (round(float(passo[0]), 6)
                      if len(passo) and np.allclose(passo, passo[0], atol=1e-4)
                      else None),
        }

    modelos = 1
    for nome in EIXOS_MODELO:
        if nome in eixos:
            modelos *= eixos[nome]["n"]

    return {"formato": magic, "eixos": eixos, "modelos": modelos,
            "bytes_cabecalho": off}


def sha256(caminho: Path, bloco: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for pedaco in iter(lambda: f.read(bloco), b""):
            h.update(pedaco)
    return h.hexdigest()


def descrever(caminho: Path) -> dict:
    info = ler_cabecalho(caminho)
    info["arquivo"] = str(caminho.relative_to(TABELAS))
    info["bytes"] = caminho.stat().st_size
    info["sha256"] = sha256(caminho)
    return info


def coletar() -> list[dict]:
    arquivos = sorted(TABELAS.rglob("*.magnus"))
    saida = []
    for a in arquivos:
        try:
            saida.append(descrever(a))
        except ValueError as e:
            print(f"  ignorado: {e}", file=sys.stderr)
    return saida


def gerar() -> int:
    itens = coletar()

    # Preserva as notas escritas a mao entre execucoes: o manifesto guarda a
    # identidade, o README guarda a fisica, e a nota liga os dois.
    antigas, deposito = {}, None
    if MANIFESTO.exists():
        antigo = json.loads(MANIFESTO.read_text(encoding="utf-8"))
        antigas = {t["arquivo"]: t.get("nota", "") for t in antigo.get("tabelas", [])}
        deposito = antigo.get("deposito")

    for t in itens:
        t["nota"] = antigas.get(t["arquivo"], "")

    doc = {
        "descricao": (
            "Identidade das tabelas de intensidade do MAGNUS. A fisica de cada "
            "uma esta em tabelas/README.md; aqui ficam formato, eixos e sha256, "
            "para que um resultado possa nomear sem ambiguidade a tabela usada."
        ),
        "gerado_por": "scripts/tabelas_manifesto.py",
        "unidades": {
            "lgB": "log10(B / G)",
            "lgT": "log10(T_eff / K)",
            "lgg": "log10(g / cm s^-2)",
            "thetaB": "graus entre o campo e a normal",
            "mu": "cosseno do angulo de emissao",
            "lgE": "log10(E / keV)",
        },
        "deposito": deposito or {
            "_": "Preencher apos publicar. Ver tabelas/zenodo/publicar.py",
            "doi_conceito": "",
            "doi_versao": "",
            "registro": "",
        },
        "tabelas": itens,
    }
    MANIFESTO.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    print(f"escrito {MANIFESTO.relative_to(RAIZ)} com {len(itens)} tabelas")
    return 0


def verificar() -> int:
    if not MANIFESTO.exists():
        print(f"sem manifesto em {MANIFESTO}", file=sys.stderr)
        return 2

    doc = json.loads(MANIFESTO.read_text(encoding="utf-8"))
    esperado = {t["arquivo"]: t for t in doc["tabelas"]}
    presentes = {str(a.relative_to(TABELAS)) for a in TABELAS.rglob("*.magnus")}

    falhas = 0
    for nome, t in sorted(esperado.items()):
        caminho = TABELAS / nome
        if not caminho.exists():
            print(f"  AUSENTE   {nome}  (baixar: scripts/tabelas_baixar.py)")
            falhas += 1
            continue
        if caminho.stat().st_size != t["bytes"]:
            print(f"  TAMANHO   {nome}  {caminho.stat().st_size} != {t['bytes']}")
            falhas += 1
            continue
        atual = sha256(caminho)
        if atual != t["sha256"]:
            print(f"  SHA256    {nome}")
            print(f"            disco    {atual}")
            print(f"            esperado {t['sha256']}")
            falhas += 1
        else:
            print(f"  ok        {nome}")

    for nome in sorted(presentes - set(esperado)):
        print(f"  NAO LISTADO  {nome}  (rode 'gerar')")
        falhas += 1

    print()
    print("tudo confere" if falhas == 0 else f"{falhas} problema(s)")
    return 1 if falhas else 0


if __name__ == "__main__":
    acao = sys.argv[1] if len(sys.argv) > 1 else "verificar"
    if acao == "gerar":
        sys.exit(gerar())
    if acao == "verificar":
        sys.exit(verificar())
    print(__doc__)
    sys.exit(2)
