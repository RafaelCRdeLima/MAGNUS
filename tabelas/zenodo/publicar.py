#!/usr/bin/env python3
"""Prepara (e, so com --publicar, publica) o deposito das tabelas no Zenodo.

As tabelas tem DOI proprio, separado do DOI do codigo. Sao coisas diferentes:
o codigo e GPL-3.0 e muda toda semana; as tabelas sao dados, CC-BY-4.0, custam
horas de CPU para regenerar e precisam ser citaveis por si.

SEGURANCA: publicar no Zenodo cunha um DOI e e IRREVERSIVEL. Por isso este
script nunca publica sozinho. Sem --publicar ele cria o rascunho, envia os
arquivos, grava os metadados e para, deixando o link para voce conferir no
navegador e apertar o botao. Teste antes com --sandbox.

Uso:
    export ZENODO_TOKEN=...                       # token com escopo deposit:write
    python3 tabelas/zenodo/publicar.py --sandbox                  # ensaio
    python3 tabelas/zenodo/publicar.py                            # rascunho real
    python3 tabelas/zenodo/publicar.py --publicar                 # cunha o DOI
    python3 tabelas/zenodo/publicar.py --nova-versao 12345678     # versao seguinte

Token: https://zenodo.org/account/settings/applications/tokens/new/
       (sandbox: https://sandbox.zenodo.org/account/settings/applications/tokens/new/)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("falta o requests:  pip install requests")

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
METADADOS = AQUI / "deposito_tabelas.json"


def api(sandbox: bool) -> str:
    return "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"


def checar(r: requests.Response, o_que: str) -> dict:
    if r.status_code >= 400:
        print(f"falhou: {o_que}  [{r.status_code}]", file=sys.stderr)
        try:
            print(json.dumps(r.json(), indent=1, ensure_ascii=False), file=sys.stderr)
        except ValueError:
            print(r.text[:800], file=sys.stderr)
        sys.exit(1)
    return r.json() if r.content else {}


def arquivos_do_deposito() -> list[Path]:
    """O que vai no deposito, lido do proprio manifesto."""
    doc = json.loads((RAIZ / "tabelas" / "MANIFEST.json").read_text(encoding="utf-8"))
    cfg = json.loads(METADADOS.read_text(encoding="utf-8"))
    quer = set(cfg["arquivos"])

    nomes = {t["arquivo"] for t in doc["tabelas"]}
    faltando = quer - nomes
    if faltando:
        sys.exit(f"nao estao no manifesto: {sorted(faltando)}")

    caminhos = [RAIZ / "tabelas" / n for n in sorted(quer)]
    ausentes = [p for p in caminhos if not p.exists()]
    if ausentes:
        sys.exit(f"nao estao no disco: {[p.name for p in ausentes]}")

    # O manifesto e o README vao junto: sem eles o deposito e um binario opaco.
    caminhos.append(RAIZ / "tabelas" / "MANIFEST.json")
    caminhos.append(RAIZ / "tabelas" / "README.md")
    return caminhos


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sandbox", action="store_true",
                   help="usa sandbox.zenodo.org; nenhum DOI real e cunhado")
    p.add_argument("--publicar", action="store_true",
                   help="publica de fato. IRREVERSIVEL: cunha o DOI")
    p.add_argument("--nova-versao", metavar="ID",
                   help="cria uma nova versao de um deposito ja publicado")
    args = p.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("defina ZENODO_TOKEN (escopo deposit:write e deposit:actions)")

    base = api(args.sandbox)
    par = {"access_token": token}
    cfg = json.loads(METADADOS.read_text(encoding="utf-8"))
    meta = cfg["metadata"]

    caminhos = arquivos_do_deposito()
    total = sum(c.stat().st_size for c in caminhos)
    print(f"deposito: {'SANDBOX' if args.sandbox else 'ZENODO REAL'}")
    print(f"titulo  : {meta['title']}")
    print(f"arquivos: {len(caminhos)}, {total/1e6:.1f} MB")
    for c in caminhos:
        print(f"   {c.stat().st_size/1e6:8.2f} MB  {c.name}")
    print()

    # 1. rascunho
    if args.nova_versao:
        r = requests.post(f"{base}/deposit/depositions/{args.nova_versao}/actions/newversion",
                          params=par, timeout=120)
        dep = checar(r, "criar nova versao")
        dep_id = dep["links"]["latest_draft"].rstrip("/").rsplit("/", 1)[-1]
        dep = checar(requests.get(f"{base}/deposit/depositions/{dep_id}",
                                  params=par, timeout=60), "ler rascunho")
        # a nova versao herda os arquivos antigos; limpa para subir os atuais
        for f in dep.get("files", []):
            checar(requests.delete(f"{base}/deposit/depositions/{dep_id}/files/{f['id']}",
                                   params=par, timeout=120), f"remover {f['filename']}")
        print(f"nova versao a partir de {args.nova_versao}: rascunho {dep_id}")
    else:
        dep = checar(requests.post(f"{base}/deposit/depositions", params=par,
                                   json={}, timeout=120), "criar rascunho")
        dep_id = dep["id"]
        print(f"rascunho criado: {dep_id}")

    bucket = dep["links"]["bucket"]

    # 2. arquivos
    for c in caminhos:
        print(f"  enviando {c.name} ...", flush=True)
        with c.open("rb") as fh:
            checar(requests.put(f"{bucket}/{c.name}", data=fh, params=par,
                                timeout=3600), f"enviar {c.name}")

    # 3. metadados
    checar(requests.put(f"{base}/deposit/depositions/{dep_id}", params=par,
                        json={"metadata": meta}, timeout=120), "gravar metadados")
    print("metadados gravados")

    # 4. publicacao, so se pedida
    url_html = f"https://{'sandbox.' if args.sandbox else ''}zenodo.org/deposit/{dep_id}"
    if not args.publicar:
        print()
        print("RASCUNHO PRONTO, NAO PUBLICADO.")
        print(f"  confira em: {url_html}")
        print("  para cunhar o DOI, rode de novo com --publicar")
        print("  (ou publique pelo navegador, que da no mesmo)")
        return 0

    pub = checar(requests.post(f"{base}/deposit/depositions/{dep_id}/actions/publish",
                               params=par, timeout=300), "publicar")
    print()
    print("PUBLICADO.")
    print(f"  DOI da versao : {pub.get('doi')}")
    print(f"  DOI conceito  : {pub.get('conceptdoi')}")
    print(f"  registro      : {pub['links'].get('record_html')}")
    print()
    print("Agora: ponha esses DOIs em tabelas/README.md e em CITATION.cff")
    print("(campo references / related-identifiers).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
