#!/usr/bin/env python3
"""Baixa e verifica os dados de terceiros que o MAGNUS consome e que NAO fazem
parte deste repositorio (sao de outros autores; ver os PROVENIENCIA.json em
atmosphere_data/*/ e cite-os).

  potekhin_magnetic_h/  EOS e opacidades de Rosseland do H magnetizado,
                        Potekhin & Chabrier 2003, 2004 (Ioffe). Tar publico.
  pc03_hmagnet/         as tabelas lg B = 13,0 e 13,5 do mesmo tar, descomprimidas,
                        que o modulo de atmosferas le diretamente.
  van_hoof/gauntff.dat  fatores de Gaunt livre-livre, van Hoof et al. 2014.
  nsmaxg_ho/            espectros NSMAXG de Ho, Potekhin & Chabrier (XSPEC). Sem
                        URL estavel: passe o zip com --nsmaxg-zip e ele e conferido
                        pelo sha256 registrado. Opcional (so a auditoria usa).

Uso:
    python3 scripts/baixar_dados_terceiros.py            # Ioffe + van Hoof
    python3 scripts/baixar_dados_terceiros.py --nsmaxg-zip ~/Downloads/nsmaxg.in.zip
Cada arquivo baixado e conferido contra o sha256 do PROVENIENCIA.json; um hash
diferente aborta, porque a fonte pode ter mudado.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "atmosphere_data"

IOFFE_TAR = "http://www.ioffe.ru/astro/NSG/Hmagnet/hmagnet.tar.gz"
VAN_HOOF = "https://data.nublado.org/gauntff/gauntff.dat"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def baixar(url: str) -> bytes:
    print(f"baixando {url} ...", flush=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def confere(nome: str, data: bytes, esperado: str | None) -> None:
    got = sha256(data)
    if esperado and got != esperado:
        raise SystemExit(f"{nome}: sha256 {got} difere do registrado {esperado}; fonte mudou?")
    print(f"  ok {nome} ({len(data)} bytes, sha256 {got[:12]}...)")


def potekhin() -> None:
    dest = DATA / "potekhin_magnetic_h"
    prov = json.load(open(dest / "PROVENIENCIA.json"))
    tar_bytes = baixar(prov["origem"].get("arquivo_baixado", IOFFE_TAR))
    confere("hmagnet.tar.gz", tar_bytes, prov["origem"].get("sha256_do_tar"))
    wanted = prov["arquivos"]
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        for name, info in wanted.items():
            if name not in members:
                print(f"  AVISO: {name} nao esta no tar")
                continue
            raw = tar.extractfile(members[name]).read()
            plain = gzip.decompress(raw) if name.endswith(".gz") else raw
            esperado = info.get("sha256_descomprimido")
            if esperado and sha256(plain) != esperado:
                raise SystemExit(f"{name}: conteudo difere do registrado")
            (dest / name).write_bytes(raw)
        # pc03_hmagnet: as duas tabelas descomprimidas e o hmn13_5 que o
        # modulo de atmosferas le diretamente.
        pc = DATA / "pc03_hmagnet"
        for name in ("hmag13_0.dat.gz", "hmag13_5.dat.gz"):
            if name in members:
                (pc / name[:-3]).write_bytes(gzip.decompress(tar.extractfile(members[name]).read()))
        if "hmn13_5.dat.gz" in members:
            (pc / "hmn13_5.dat.gz").write_bytes(tar.extractfile(members["hmn13_5.dat.gz"]).read())
    print(f"  {len(wanted)} tabelas em {dest.relative_to(ROOT)}; lg B 13,0/13,5 em pc03_hmagnet/")


def van_hoof() -> None:
    dest = DATA / "van_hoof"
    prov = json.load(open(dest / "PROVENIENCIA.json"))
    url = prov.get("origem", VAN_HOOF).split(",")[0].strip()
    data = baixar(url if url.startswith("http") else VAN_HOOF)
    confere("gauntff.dat", data, prov.get("sha256"))
    (dest / "gauntff.dat").write_bytes(data)


def nsmaxg(zip_path: Path) -> None:
    dest = DATA / "nsmaxg_ho"
    prov = json.load(open(dest / "PROVENIENCIA.json"))
    data = zip_path.read_bytes()
    confere(zip_path.name, data, prov["origem"].get("sha256_do_zip"))
    wanted = prov["arquivos"]
    n = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for member in z.namelist():
            name = Path(member).name
            if name not in wanted:
                continue
            raw = z.read(member)
            esperado = wanted[name].get("sha256_descomprimido")
            if esperado and sha256(gzip.decompress(raw)) != esperado:
                raise SystemExit(f"{name}: conteudo difere do registrado")
            (dest / name).write_bytes(raw)
            n += 1
    print(f"  {n}/{len(wanted)} espectros NSMAXG em {dest.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nsmaxg-zip", type=Path, help="zip dos espectros NSMAXG (XSPEC), opcional")
    ap.add_argument("--so-nsmaxg", action="store_true", help="nao baixar Ioffe/van Hoof")
    a = ap.parse_args()
    if not a.so_nsmaxg:
        potekhin()
        van_hoof()
    if a.nsmaxg_zip:
        nsmaxg(a.nsmaxg_zip)
    print("pronto.")


if __name__ == "__main__":
    main()
