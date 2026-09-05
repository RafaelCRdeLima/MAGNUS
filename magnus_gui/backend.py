"""Ponte com o motor: monta a linha de comando do magnus_engine e lê a saída.

A GUI não faz física — ela orquestra. Aqui mora a tradução de um ``ModelState``
(o que a tela mostra) para os argumentos do ``build/magnus_engine
--spectral-grid`` e o parse do JSON de volta.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "build" / "magnus_engine"
KT_KEV_PER_MK = 0.08617333262145


def default_table() -> str:
    for cand in ("tabelas/magnus_campoB.magnus", "build/campo_B13_5.magnus"):
        p = ROOT / cand
        if p.is_file():
            return str(p)
    return ""


@dataclass
class Spot:
    theta: float = 30.0
    phi: float = 0.0
    radius: float = 12.0
    kt_kev: float = 0.12


@dataclass
class ModelState:
    """Espelha o pedido de um modelo — o que a aba Modelo edita."""
    mass: float = 1.4
    radius: float = 12.0
    distance: float = 0.4
    inclination: float = 45.0
    period: float = 10.3131714585
    energy_min: float = 0.15
    energy_max: float = 1.2
    energy_bins: int = 40
    time_bins: int = 32
    max_images: int = 2
    # atmosfera
    atmosphere_table: str = field(default_factory=default_table)
    log_field: float = 13.5
    magnetic_colatitude: float = 60.0
    magnetic_azimuth: float = 0.0
    base_kt_kev: float = 0.09          # T do polo (T_p)
    temperature_peaking: float = 0.3   # concentração a
    temperature_min_frac: float = 0.3
    atmosphere_fraction: float = 1.0   # espessura f (1=cheia, 0=camadas)
    blackbody_spots: bool = False
    # linhas
    line1: tuple = (0.0, 0.3, 0.1)     # (profundidade, centro, largura); prof 0 = desligada
    line2: tuple = (0.0, 0.6, 0.1)
    # superfície
    spots: list = field(default_factory=lambda: [Spot()])

    def command(self) -> list[str]:
        c = [str(ENGINE), "--spectral-grid",
             "--mass", f"{self.mass}", "--radius", f"{self.radius}",
             "--inclination", f"{self.inclination}", "--period", f"{self.period}",
             "--distance", f"{self.distance}",
             "--energy-min", f"{self.energy_min}", "--energy-max", f"{self.energy_max}",
             "--energy-bins", f"{self.energy_bins}", "--time-bins", f"{self.time_bins}",
             "--samples", "81", "--max-images", f"{self.max_images}",
             "--instrument", "ideal"]
        if self.atmosphere_table:
            c += ["--atmosphere-table", self.atmosphere_table,
                  "--magnetic-field", f"{10.0 ** self.log_field}",
                  "--magnetic-colatitude", f"{self.magnetic_colatitude}",
                  "--magnetic-azimuth", f"{self.magnetic_azimuth}"]
        if self.base_kt_kev > 0:
            c += ["--base-kt-kev", f"{self.base_kt_kev}"]
        if self.temperature_peaking > 0:
            c += ["--temperature-peaking", f"{self.temperature_peaking}",
                  "--temperature-min-frac", f"{self.temperature_min_frac}"]
        if self.atmosphere_fraction < 1.0:
            c += ["--atmosphere-fraction", f"{self.atmosphere_fraction}"]
        if self.blackbody_spots:
            c += ["--blackbody-spots"]
        d1, e1, w1 = self.line1
        if d1 > 0:
            c += ["--line-energy", f"{e1}", "--line-width", f"{w1}", "--line-depth", f"{d1}"]
        d2, e2, w2 = self.line2
        if d2 > 0:
            c += ["--line2-energy", f"{e2}", "--line2-width", f"{w2}", "--line2-depth", f"{d2}"]
        for s in self.spots:
            mk = s.kt_kev / KT_KEV_PER_MK
            c += ["--spot", f"{s.theta},{s.phi},{s.radius},{mk}"]
        return c

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def run_spectral_grid(state: ModelState, timeout: float = 60.0) -> dict:
    """Roda o motor e devolve o JSON da grade espectral (ou levanta)."""
    if not ENGINE.is_file():
        raise FileNotFoundError(f"motor não compilado: {ENGINE} (rode `make engine`)")
    out = subprocess.run(state.command(), capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "motor falhou sem mensagem")
    for line in reversed(out.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError("motor não devolveu JSON")


def read_events_meta(path: str) -> dict:
    """Lê o cabeçalho ``#`` de uma lista de eventos pulsaris."""
    meta = {}
    try:
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            if line.startswith("#") and "=" in line:
                k, v = line[1:].split("=", 1)
                meta[k.strip()] = v.strip()
            elif not line.startswith("#"):
                break
    except OSError:
        pass
    return meta


def build_fit_request(state: ModelState, flags: dict, meta: dict, mcmc: dict) -> dict:
    """Monta o pedido que o ``scripts/mcmc_fit.py`` entende, a partir do modelo,
    das caixas 'ajustar' (flags) e da configuração de MCMC."""
    req = {
        "instrument": meta.get("instrument", "ideal"),
        "period": float(meta.get("period_s", state.period)),
        "distance": state.distance,
        "mass": state.mass, "radius": state.radius,
        "inclination": state.inclination, "phaseOffset": 0.0,
        "energyMin": state.energy_min, "energyMax": state.energy_max,
        "energyBins": state.energy_bins, "phaseBins": state.time_bins,
        "maxImages": state.max_images,
        "spotCount": max(1, len(state.spots)),
        "spots": [{"theta": s.theta, "radius": s.radius,
                   "temperatureKev": s.kt_kev, "phi": s.phi} for s in state.spots] or
                 [{"theta": 90.0, "radius": 0.5, "temperatureKev": 0.03, "phi": 0.0}],
        "nh": float(meta.get("nh", 0.0229)), "fitNh": False,
        "walkers": int(mcmc.get("walkers", 40)),
        "workers": int(mcmc.get("workers", 4)),
        "iterations": int(mcmc.get("iterations", 400)),
        "burnIn": int(mcmc.get("burnIn", 150)),
        "seed": int(mcmc.get("seed", 2026)),
        "priors": {},
    }
    if state.atmosphere_table:
        req["atmosphereTable"] = state.atmosphere_table
        req["magneticColatitude"] = state.magnetic_colatitude
        req["magneticAzimuth"] = state.magnetic_azimuth
        req["magneticField"] = 10.0 ** state.log_field
        req["baseTemperature"] = state.base_kt_kev
        req["baseTemperatureRange"] = [0.03, 0.20]
        req["temperaturePeaking"] = state.temperature_peaking
        req["atmosphereFraction"] = state.atmosphere_fraction
        if state.blackbody_spots:
            req["blackbodySpots"] = True
    d1, e1, w1 = state.line1
    if d1 > 0 or flags.get("fitLine"):
        req["lineEnergy"] = e1; req["lineWidth"] = w1; req["lineDepth"] = max(d1, 0.5)
    # caixas 'ajustar'
    for key in ("fitMagneticField", "fitBaseTemperature", "fitTemperaturePeaking",
                "fitAtmosphereFraction", "fitMagneticColatitude", "fitMagneticAzimuth",
                "fitLine"):
        if flags.get(key):
            req[key] = True
    if flags.get("fitMR"):
        req["priors"]["mass"] = [0.8, 2.3]
        req["priors"]["radius"] = [10.0, 15.0]
    else:
        # mcmc_fit deixa M e R livres por padrão; congela-os quando não pedidos.
        req["fixed"] = {"mass": state.mass, "radius": state.radius}
    return req


def folded_and_spectrum(grid: dict):
    """Extrai (fase, curva de luz), (energia, espectro) do JSON da grade."""
    import numpy as np
    nt = grid.get("time_bins") or len(grid.get("time_s", []))
    ne = grid["energy_bins"] if "energy_bins" in grid else len(grid["energy_centers_keV"])
    dr = np.array(grid["detected_count_rate"], float).reshape(nt, ne)
    energy = np.array(grid["energy_centers_keV"], float)
    phase = (np.arange(nt) + 0.5) / nt
    return phase, dr.sum(1), energy, dr.sum(0)
