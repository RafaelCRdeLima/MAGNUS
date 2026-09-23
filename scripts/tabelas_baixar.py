#!/usr/bin/env python3
"""Baixa as tabelas do deposito Zenodo e confere o sha256 do manifesto.

As tabelas grandes nao estao no git. Depois de clonar o repositorio, ou ao
montar o ambiente num cluster novo, rode isto para obter os arquivos.

Uso:
    python3 scripts/tabelas_baixar.py                 # baixa o que falta
    python3 scripts/tabelas_baixar.py --tudo          # rebaixa tudo
    python3 scripts/tabelas_baixar.py --doi 10.5281/zenodo.XXXXXXX
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
TABELAS = RAIZ / "tabelas"
MANIFESTO = TABELAS / "MANIFEST.json"


def sha256(caminho: Path, bloco: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for pedaco in iter(lambda: f.read(bloco), b""):
            h.update(pedaco)
    return h.hexdigest()


def registro(doi: str) -> dict:
    """Resolve um DOI do Zenodo no registro da API."""
    ident = doi.rstrip("/").rsplit(".", 1)[-1]
    url = f"https://zenodo.org/api/records/{ident}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def baixar(url: str, destino: Path) -> None:
    parcial = destino.with_suffix(destino.suffix + ".parcial")
    with urllib.request.urlopen(url, timeout=600) as r, parcial.open("wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        feito = 0
        while True:
            pedaco = r.read(1 << 20)
            if not pedaco:
                break
            f.write(pedaco)
            feito += len(pedaco)
            if total:
                print(f"\r    {100*feito/total:5.1f}%  {feito/1e6:7.1f} / {total/1e6:.1f} MB",
                      end="", flush=True)
        print()
    parcial.replace(destino)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--doi", help="sobrepoe o DOI gravado no manifesto")
    p.add_argument("--tudo", action="store_true", help="rebaixa mesmo o que ja confere")
    args = p.parse_args()

    doc = json.loads(MANIFESTO.read_text(encoding="utf-8"))
    esperado = {t["arquivo"]: t for t in doc["tabelas"]}

    doi = args.doi or doc.get("deposito", {}).get("doi_conceito", "")
    if not doi:
        print("O manifesto ainda nao tem DOI: o deposito nao foi publicado.", file=sys.stderr)
        print("Publique com tabelas/zenodo/publicar.py e preencha o bloco", file=sys.stderr)
        print("'deposito' em tabelas/MANIFEST.json, ou passe --doi.", file=sys.stderr)
        return 2

    print(f"resolvendo {doi} ...")
    reg = registro(doi)
    print(f"  {reg['metadata']['title'][:70]}")
    print(f"  versao {reg['metadata'].get('version', '?')}, {len(reg.get('files', []))} arquivos")
    print()

    remotos = {f["key"]: f for f in reg.get("files", [])}
    falhas = 0

    for nome, t in sorted(esperado.items()):
        destino = TABELAS / nome
        if destino.exists() and not args.tudo:
            if destino.stat().st_size == t["bytes"] and sha256(destino) == t["sha256"]:
                print(f"  ok        {nome}")
                continue
            print(f"  DIFERE    {nome}, rebaixando")

        chave = Path(nome).name
        if chave not in remotos:
            # Nem toda tabela do manifesto precisa estar no deposito: as pequenas
            # continuam versionadas no git.
            if destino.exists():
                print(f"  local     {nome}  (nao esta no deposito)")
            else:
                print(f"  AUSENTE   {nome}  (nem no disco, nem no deposito)")
                falhas += 1
            continue

        destino.parent.mkdir(parents=True, exist_ok=True)
        print(f"  baixando  {nome}")
        baixar(remotos[chave]["links"]["self"], destino)

        atual = sha256(destino)
        if atual != t["sha256"]:
            print(f"  SHA256 NAO CONFERE apos baixar {nome}")
            print(f"            obtido   {atual}")
            print(f"            esperado {t['sha256']}")
            falhas += 1
        else:
            print(f"  ok        {nome}")

    print()
    print("tudo confere" if falhas == 0 else f"{falhas} problema(s)")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
