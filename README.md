# MAGNUS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22760147.svg)](https://doi.org/10.5281/zenodo.22760147)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**MAGNUS** is a forward model and Bayesian inference code for the thermal
X-ray emission of strongly magnetized neutron stars. It was written to
analyse the X-ray Dim Isolated Neutron Stars (the "Magnificent Seven") observed
by *XMM-Newton*, but nothing in it is specific to those sources.

The pipeline has three layers, each of which can be used on its own:

1. **Magnetized atmosphere solver** (`atmosfera/`, Python). A plane-parallel,
   two-mode (ordinary and extraordinary) radiative-transfer solver for a
   hydrogen atmosphere in a field of 10^12 to 10^14 G. Feautrier scheme with
   accelerated Lambda iteration, exact normal modes from the dielectric tensor,
   electron and proton cyclotron opacities, free-free with thermally averaged
   Gaunt factors, Compton scattering (Kompaneets, conservative form). Its
   product is a **specific-intensity table** I(E, μ, θ_B) on a grid of
   (lg B, lg T_eff, lg g), i.e. the emergent beam as a function of photon
   energy, emission angle and angle between the field and the surface normal.
2. **Relativistic ray-tracing engine** (`engine/main.cpp`, C++20). Exact
   Schwarzschild light bending and gravitational redshift, a two-pole surface
   temperature law (Pérez-Azorín et al. 2006) with an arbitrary tilt of the
   second pole, a surface field that is uniform, a flat dipole or the
   Schwarzschild-corrected dipole (Ginzburg & Ozernoy 1964), local
   proton-cyclotron absorption tied to the local field, an effective-thickness
   parameter for the atmosphere, interstellar absorption and instrument
   response folding. It produces phase-resolved spectra and pulse profiles
   from the intensity table.
3. **Bayesian fitting** (`scripts/mcmc_fit.py`). Affine-invariant ensemble
   MCMC (Goodman & Weare 2010) over phase-energy event data, with checkpoints,
   resumption, stuck-walker diagnostics and Gelman-Rubin statistics. The C++
   engine stays resident in memory and evaluates the likelihood.

## What MAGNUS computes and what it imports

In one sentence: MAGNUS is a generator of angle-resolved magnetized
atmospheres that imports tabulated atomic microphysics and exports the
specific intensity that its own relativistic ray tracer consumes.

**Computed by MAGNUS.** The atmosphere itself: the hydrostatic structure, the
normal modes from an exact eigenproblem of the dielectric tensor (plasma plus
Euler-Heisenberg vacuum polarization), the electron- and proton-cyclotron,
free-free and scattering opacities, the two-mode Feautrier transfer with
accelerated Lambda iteration, the temperature correction and the emergent
intensity I(E, μ, θ_B). Every table in `tabelas/` was produced this way. No
external model supplies the spectrum.

**Imported from other authors.** Tabulated microphysics, at well-defined
points of the calculation:

- free-free Gaunt factors (van Hoof et al. 2014); without the file the solver
  falls back to its own Elwert-Born approximation;
- the neutral fraction x(H) of partially ionized hydrogen (Potekhin & Chabrier
  2003) at lg B = 13.0 and 13.5, which replaces the code's own Saha estimate
  where the table exists;
- the Rosseland opacities K0 and K1 of Potekhin & Chabrier, used only to
  validate the code's own Rosseland tensor;
- NSMAXG spectra (Ho, Potekhin & Chabrier), used only as benchmarks.

**Not yet in the production tables.** The grids in `tabelas/` used for fitting
are fully ionized hydrogen: the partial-ionization and bound-free machinery
exists in `atmosfera/atomico.py` but is not switched on in those grids. In
production, the only third-party input that enters the emergent spectrum is
the Gaunt-factor table.

**Downstream.** The C++ engine consumes MAGNUS tables only; the optional
grey-model anisotropy file and the NSMAXG backend are not used in the fits.
The MCMC driver consumes the engine.

The physics and the numerical checks are documented, in Portuguese, in the
docstrings and in the `README.md` files of each directory.

## Layout

| path | contents |
|---|---|
| `engine/main.cpp` | the ray-tracing engine (single file, no dependencies) |
| `atmosfera/` | the atmosphere solver: `transporte.py` (radiative transfer), `estrutura.py` (structure, non-magnetic stage), `magnetizada.py` (two-mode magnetized transfer), `atomico.py` (ionization and bound-free) |
| `scripts/tabela_*.py` | drivers that build intensity tables (single models, dense grids in B, T and g); `mesclar_tabelas.py` merges grids |
| `scripts/tabela_intensidade.py` | the table format (MAGNUSI2) reader and writer |
| `scripts/mcmc_fit.py` | the MCMC driver; `cornerplot*.py`, `curva_e_espectro.py`, `figuras_publicacao.py` plot its results |
| `scripts/coadicionar.py` | co-adds phase-energy event lists from several observations |
| `scripts/baixar_dados_terceiros.py` | downloads and verifies the third-party data (see below) |
| `tabelas/` | intensity tables already computed (see `tabelas/README.md`) |
| `atmosphere_data/` | provenance of the third-party data the solver reads |
| `tests/` | unit tests: analytic transfer solutions, table reader, atomic physics, two-mode transfer |
| `magnus_gui/` | a Qt (PySide6) desktop front end for the fits (optional) |
| `identity/` | logo and visual identity |

## Building and testing

Requirements: `g++` with C++20, Python 3.10 or newer with `numpy`, `scipy`
and `matplotlib`. The GUI additionally needs `PySide6`.

```bash
make engine        # builds build/magnus_engine
make test          # unit tests (the ones that need third-party data are skipped if it is absent)
```

## Optional external inputs

MAGNUS runs without any external table. The inputs below are **optional**:
each one refines or checks a specific piece of the calculation, and each is
the work of other authors, so none of them is redistributed here. Use the
original tables from the links below, or any equivalent table you have access
to in the same format, and cite their authors. The files are expected under
`atmosphere_data/` in the layout described in
[atmosphere_data/README.md](atmosphere_data/README.md); the file
`PROVENIENCIA.json` in each folder records the exact versions we used, with
their SHA-256.

| input | what it changes | source |
|---|---|---|
| free-free Gaunt factors, `van_hoof/gauntff.dat` | replaces the built-in Elwert-Born approximation in the free-free opacity; used for all tables in `tabelas/` | van Hoof et al. 2014, MNRAS 444, 420: <https://data.nublado.org/gauntff/> |
| EOS and Rosseland opacities of magnetized hydrogen, `potekhin_magnetic_h/`, `pc03_hmagnet/` | neutral fraction x(H) at lg B = 13.0 and 13.5 for the partial-ionization branch; K0, K1 for validating the Rosseland tensor | Potekhin & Chabrier 2003, ApJ 585, 955; 2004, ApJ 600, 317: <http://www.ioffe.ru/astro/NSG/Hmagnet/> |
| NSMAXG model spectra, `nsmaxg_ho/` | benchmarks only (`scripts/auditoria_gabaritos.py`) and the optional `--nsmaxg-table` engine backend, neither used in the fits | Ho, Potekhin & Chabrier 2008, ApJS 178, 102; XSPEC model page: <https://heasarc.gsfc.nasa.gov/xanadu/xspec/models/nsmaxg.html> |

For convenience, `scripts/baixar_dados_terceiros.py` fetches the first two
from the original sites and verifies their hashes; the NSMAXG spectra must be
obtained by the user.

## Running a fit

`scripts/mcmc_fit.py` takes a phase-energy event list (CSV) and a JSON request
describing the model, the priors and the sampler. Instrument responses and the
interstellar-absorption table are read from the directory named by the
environment variable `MAGNUS_INSTRUMENT_DIR`, which follows the
`instrument_data/` layout of PULSARIS (a `profiles/manifest.json` pointing to
sparse RMF/ARF files and `absorption/tbabs_wilm.csv`). The reduction pipeline
that produces those files from *XMM-Newton* observation data files is not part
of this repository.

```bash
export MAGNUS_INSTRUMENT_DIR=/path/to/instrument_data_root
python3 scripts/mcmc_fit.py --events events.csv --request request.json > result.json
```

## Origin

MAGNUS was developed by Rafael C. R. de Lima (Universidade do Estado de Santa
Catarina, UDESC). The ray-tracing engine started from the engine of PULSARIS,
the author's earlier pulse-profile code, and diverged from it with the
atmosphere-table backend, the field axis and the dipole geometry.

## License and disclaimer

MAGNUS is free software, released under the GNU General Public License,
version 3 (see [LICENSE](LICENSE)). You are free to use, study, modify and
redistribute it under the terms of that license.

The code is provided **as is**, without warranty of any kind. The authors are
not responsible for bugs, for errors in the implementation, or for results
obtained with it. Model atmospheres, radiative transfer and relativistic ray
tracing involve many physical approximations and numerical choices, and the
code is under active development. Anyone using MAGNUS for scientific work is
expected to verify, for their own application, both the physics and the
numerics, and is welcome to report problems by opening an issue.

## Citing MAGNUS

If MAGNUS contributes to a publication, please cite it. A paper describing the
code and its first application is in preparation; until it appears, cite the
archived release on Zenodo, using the metadata in [CITATION.cff](CITATION.cff)
(GitHub's "Cite this repository" button formats it for you):

> de Lima, R. C. R. (2026). MAGNUS: forward model and Bayesian inference for
> the thermal X-ray emission of magnetized neutron stars (v0.2.0). Zenodo.
> https://doi.org/10.5281/zenodo.22844147

The concept DOI [10.5281/zenodo.22760147](https://doi.org/10.5281/zenodo.22760147)
always resolves to the latest archived version. Please also cite the
third-party data you use through MAGNUS (see above).
