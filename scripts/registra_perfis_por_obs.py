#!/usr/bin/env python3
"""Registra os 13 perfis POR OBSERVAÇÃO da reconstrução no manifesto do MAGNUS.

Necessário porque `FitProblem.worker_command` resolve `--instrument-profile` e
`--instrument-response` pelo manifesto, com caminhos relativos à raiz do MAGNUS
ou do PULSARIS, e não aceita caminho absoluto no pedido. Os perfis da
reconstrução vivem fora das duas árvores, então são copiados para
`instrumento/instrument_data/profiles/` com sha256 registrado.

Cada entrada leva a resolução temporal DAQUELA observação, que é o ponto: o
perfil combinado declarava 73400 para as treze, e seis são 47700.

O manifesto antigo é preservado em `manifest.json.antes_por_obs`, e entradas
existentes com o mesmo id são substituídas, não duplicadas.

Uso: python3 scripts/registra_perfis_por_obs.py [--conferir]
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RECON = Path(os.environ.get(
    "MAGNUS_RECONSTRUCAO",
    Path.home() / "Codes/XMM-NEWTON/reconstrucao_5906a57"))
PERFIS = RAIZ / "instrumento/instrument_data/profiles"
MANIFESTO = PERFIS / "manifest.json"


def _sem_raiz(caminho):
    """O caminho com a raiz conhecida trocada por marcador.

    `<XMM>/produtos/...` em vez de `/home/<usuario>/Codes/XMM-NEWTON/...`. Quem
    le sabe de onde veio sem que o manifesto publique a arvore de quem rodou.
    """
    import os
    t = str(caminho)
    for raiz, marca in (("Codes/XMM-NEWTON", "<XMM>"),
                        ("Codes/PULSARIS", "<PULSARIS>"),
                        ("Codes/MAGNUS", "<MAGNUS>")):
        i = t.find(raiz)
        if i >= 0:
            return marca + t[i + len(raiz):]
    return t.replace(os.path.expanduser("~"), "~")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for pedaco in iter(lambda: fh.read(1 << 20), b""):
            h.update(pedaco)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conferir", action="store_true",
                    help="so lista o que faria, sem copiar nem gravar")
    args = ap.parse_args()
    indice = json.loads((RECON / "indice.json").read_text())
    atual = json.loads(MANIFESTO.read_text())
    lista = atual if isinstance(atual, list) else atual.get("profiles", [])
    por_id = {p.get("id"): p for p in lista}
    novos, registro = [], []
    for s in indice["sets"]:
        ev = RECON / s["files"]["events"]
        perfil = ev.parent / "profile"
        csvs = sorted(perfil.glob("*.csv"))
        bins = sorted(perfil.glob("*.rmfbin"))
        if not (csvs and bins):
            print("SEM PERFIL para %s" % s["obsid"]); continue
        pid = csvs[0].stem
        destino_csv = PERFIS / csvs[0].name
        destino_bin = PERFIS / bins[0].name
        entrada = {
            "id": pid,
            "label": "XMM-Newton / EPN %s obs %s" % (s["submode"], s["obsid"]),
            "instrument": "EPN %s" % s["submode"],
            "energy_min_keV": 0.15, "energy_max_keV": 12.0,
            "time_resolution_us": float(s["time_resolution_us"]),
            "dead_time_us": 0.0,
            "calibration": ("resposta da observacao %s, reconstrucao %s, "
                            "commit %s" % (s["obsid"], RECON.name,
                                           indice.get("environment", {}).get("commit", "?"))),
            "source_url": "https://www.cosmos.esa.int/web/xmm-newton",
            "profile_file": str(destino_csv.relative_to(RAIZ)),
            "response_file": str(destino_bin.relative_to(RAIZ)),
            "redistribution": "full_sparse_ogip_rmf",
            "raw_files": [],
            "procedencia": {
                "obsid": s["obsid"], "set_id": s["set_id"],
                "manifest_sha256": s.get("manifest_sha256"),
                "sas_rand_seed": s.get("sas_rand_seed"),
                # PROCEDENCIA SEM CAMINHO ABSOLUTO. O manifesto vai ao
                # repositorio publico, e caminho absoluto ali e o nome de
                # usuario e a arvore de diretorios de quem rodou: informacao de
                # infraestrutura, nao de dado. Em 11/10/2026 isso barrou a
                # publicacao de 107 commits em dev/publicar.sh, com razao.
                # O que identifica a origem e o sha256 logo abaixo; a raiz vira
                # marcador, e o resto do caminho continua legivel.
                "origem_csv": _sem_raiz(csvs[0]),
                "origem_rmfbin": _sem_raiz(bins[0]),
                "sha256_csv": sha256(csvs[0]), "sha256_rmfbin": sha256(bins[0])},
        }
        novos.append((csvs[0], destino_csv, bins[0], destino_bin, entrada))
        registro.append({"id": pid, "obsid": s["obsid"],
                         "time_resolution_us": entrada["time_resolution_us"],
                         "ja_existia": pid in por_id})
    print("%-50s %10s %10s" % ("id", "res us", "ja existia"))
    for x in registro:
        print("%-50s %10.0f %10s" % (x["id"], x["time_resolution_us"],
                                     x["ja_existia"]))
    if args.conferir:
        print("\n--conferir: nada copiado nem gravado")
        return
    shutil.copy2(MANIFESTO, PERFIS / "manifest.json.antes_por_obs")
    for src_csv, dst_csv, src_bin, dst_bin, entrada in novos:
        shutil.copy2(src_csv, dst_csv)
        shutil.copy2(src_bin, dst_bin)
        por_id[entrada["id"]] = entrada
    saida = list(por_id.values())
    if isinstance(atual, dict):
        atual["profiles"] = saida
        MANIFESTO.write_text(json.dumps(atual, indent=2, ensure_ascii=False) + "\n")
    else:
        MANIFESTO.write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n")
    print()
    print("%d perfis registrados, manifesto com %d entradas; backup em "
          "manifest.json.antes_por_obs" % (len(novos), len(saida)))


if __name__ == "__main__":
    main()
