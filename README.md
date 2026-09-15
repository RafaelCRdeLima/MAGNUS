# MAGNUS

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

## Third-party data

The solver reads tables computed by other groups. They are **not
redistributed** here; the script below downloads them from the original
sites and checks their SHA-256 against the values recorded in
`atmosphere_data/*/PROVENIENCIA.json`. Please cite the original authors.

```bash
python3 scripts/baixar_dados_terceiros.py
```

- Potekhin & Chabrier (2003, 2004): equation of state and Rosseland opacities
  of partially ionized hydrogen in strong magnetic fields (Ioffe Institute).
- van Hoof et al. (2014): thermally averaged free-free Gaunt factors.
- Ho, Potekhin & Chabrier: NSMAXG model atmosphere spectra (XSPEC), used only
  as benchmarks by `scripts/auditoria_gabaritos.py`. There is no stable
  download URL; pass the zip with `--nsmaxg-zip`.

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
repository itself, using the metadata in [CITATION.cff](CITATION.cff)
(GitHub's "Cite this repository" button formats it for you):

> de Lima, R. C. R. (2026). MAGNUS: forward model and Bayesian inference for
> the thermal X-ray emission of magnetized neutron stars. Version 0.1.0.
> https://github.com/RafaelCRdeLima/MAGNUS

A Zenodo DOI for the archived releases will be added here as soon as it is
minted. Please also cite the third-party data you use through MAGNUS (see
above).
