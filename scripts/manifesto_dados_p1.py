#!/usr/bin/env python3
"""Manifesto dos dados nao versionados da frente P1.

Exigencia do revisor em 06/10/2026: "dados nao versionados precisam constar no
manifesto com caminhos, hashes e procedencia".

A frente P1 de 06/10 e autocontida de proposito: o controle analitico nao usa
dado observacional nenhum, so a solucao fechada da esfera isotermica. O que ela
usa de nao versionado sao os BINARIOS compilados, e e isso que o manifesto
registra, mais a conferencia de que o binario de producao nao mudou.

Uso: python3 scripts/manifesto_dados_p1.py
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
#: sha256 do binario de producao ANTES de a macro MAGNUS_SEM_MAIN entrar, medido
#: em 06/10/2026 e repetido depois: o pre-processador devolve o mesmo arquivo
#: quando a macro nao esta definida, logo o binario tem de sair identico.
SHA_PRODUCAO_ANTES = ("65a505254b3517cdd3913365dd37edc6c170618734393a2831"
                      "ab4310412781c4")


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    entradas = []
    for rel, papel, proc in (
        ("build/magnus_engine", "quadratura de superficie, binario de PRODUCAO",
         "compilado de engine/main.cpp pelo alvo 'engine' do Makefile"),
        ("build/magnus_diag_imagem", "controle no plano da imagem, EXCLUSIVO do "
         "diagnostico", "compilado de engine/diag_plano_imagem.cpp, que inclui "
         "main.cpp com MAGNUS_SEM_MAIN, pelo alvo 'diag-imagem'"),
    ):
        caminho = RAIZ / rel
        if not caminho.is_file():
            entradas.append({"caminho": rel, "estado": "AUSENTE", "papel": papel})
            continue
        entradas.append({
            "caminho": rel, "papel": papel, "procedencia": proc,
            "sha256": sha256(caminho),
            "bytes": caminho.stat().st_size,
            "versionado": False,
            "reproduzivel_por": "make %s" % ("engine" if "engine" in rel
                                             else "diag-imagem")})
    producao = next((e for e in entradas
                     if e["caminho"] == "build/magnus_engine"), None)
    conferencia = {
        "sha256_esperado_do_binario_de_producao": SHA_PRODUCAO_ANTES,
        "sha256_medido": producao.get("sha256") if producao else None,
        "identico": bool(producao and producao.get("sha256") == SHA_PRODUCAO_ANTES),
        "por_que_importa": ("a frente P1 acrescentou a macro MAGNUS_SEM_MAIN em "
                            "engine/main.cpp para o executavel exclusivo poder "
                            "reaproveitar o arquivo; sem a macro o "
                            "pre-processador devolve o mesmo arquivo, logo o "
                            "binario de producao tem de sair byte a byte igual")}
    saida = {
        "o_que_e": "manifesto dos dados nao versionados da frente P1 de 06/10/2026",
        "data": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ,
                                 capture_output=True, text=True).stdout.strip(),
        "dado_observacional_usado": ("NENHUM: o controle e a solucao fechada da "
                                     "esfera isotermica e isotropica, e nao usa "
                                     "evento, resposta nem fundo"),
        "entradas": entradas,
        "conferencia_do_binario_de_producao": conferencia,
        "artefatos_gerados": [
            "docs/incidente_resolucao_2026-10-04/plano_imagem_p1.json",
            "docs/incidente_resolucao_2026-10-04/mecanismo_p1.json",
            "docs/incidente_resolucao_2026-10-04/por_imagem_p1.json"],
    }
    destino = RAIZ / "docs/incidente_resolucao_2026-10-04/manifesto_dados_p1.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n")
    print("binario de producao identico ao de antes: %s"
          % conferencia["identico"])
    for e in entradas:
        print("  %-28s %s" % (e["caminho"], e.get("sha256", "AUSENTE")[:16]))
    print("gravado em %s" % destino)


if __name__ == "__main__":
    main()
