# Simulated Scenarios Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pipeline's input `.pkl` from a *scenario* (a GW skymap, a list of pointings, and MC IRF files) with no real DL3, and turn it into **expected** (median-background) 2D/3D upper limits, so the method can be studied on simulations and, in later plans, on tiling strategies.

**Architecture:** Everything downstream of the input `.pkl` (`simulate.perform_n_simulations[_3d]`, `run_fit_ul[_3d]`) already works on a simulated dataset. Notebook 3 builds that dataset from real runs: their IRFs, pointing, livetime, and a ring background (cell 29). The new `gwuls/scenario.py` builds the same `.pkl` from a YAML scenario. Each pointing gets its own IRF file, and its background comes from the IRF file or from a separate background file. The GW-map preparation (notebook cells 7–13) moves into `utils.gw_on_geom` so the real and simulated paths share it. A thin CLI, `scripts/expected_ul.py`, runs one scenario end to end and is the unit that cluster batch jobs and later tiling studies loop over.

**Tech Stack:** Python 3.11, gammapy 2.1, ligo.skymap, healpy, PyYAML (already installed), pytest (new, dev only).

**Spec:** none written. This plan takes its requirements from the user's request ("a tool to do data simulations… then tiling studies"), from `docs/pipeline.md`, and from `README.md` (Status, Open items). Decisions made here in the user's absence are listed under *Assumptions* and need their review.

## Assumptions (review these first)

1. **"Data simulation" means simulating the IACT observation, not the GW event.** The inputs are a skymap (real, or an O5 observing-scenario one), pointings (sky position, start time after the merger, duration), and IRF FITS produced from MC on the cluster (lstchain full-enclosure: `EFFECTIVE AREA`, `ENERGY DISPERSION`, `PSF`, optionally `BACKGROUND`).
2. **Background:** from the IRF file if it has a `BACKGROUND` HDU, otherwise from a per-pointing `bkg_path` (e.g. a `pybkgmodel` file from real runs at a similar zenith). MC proton backgrounds are often too poor to use, which is why `bkg_path` exists. The background is treated as known, the same as in the current simulation dataset.
3. **No observed Λ.** Scenarios give *expected* limits: Λ* = median background Λ (`lambda_real = lambda_bkg_m`, `significance = 0`). That is the same Λ* that S240615dg used (it under-fluctuated), so the mirror test in Task 4 can compare like with like.
4. **Choosing IRFs by zenith/NSB is out of scope.** Each pointing names its IRF file explicitly. Automating that choice belongs to the tiling plan.

## Global Constraints

- Do not change the `.pkl` contract: keys `simulate._KEYS_COMMON + _KEYS_2D + _KEYS_3D + _KEYS_MODEL` plus `t_merger_iso`, `energy_edges`, `spectral_index`, `e_ref`, `source_name`, `type_obs`; `validate_simulation_input(data, mode="3d")` must pass.
- `run_gti_s` holds seconds since the merger, one `(n_intervals, 2)` array per pointing (`simulate.gti_seconds_since` convention).
- The distance grid starts at 1 Mpc, never 0 (`pipeline.md` §3.4): `r_grid = np.linspace(1, 10000, 1000)`.
- Defaults match notebook 3: `binsz=0.08`, `size_fov=2.5°`, `correlation_radius=0.1°`, reco 0.6–20 TeV at 4.5 bins/decade, true 0.05–100 TeV, `resolution_hp_to_grid=1024`, 95% mask, Γ = 2, E0 = 1 TeV, AIR projection, ICRS.
- No new runtime dependencies. Add `pytest` to `environment.yml` only.
- Code goes through git. Data and outputs stay ignored (`data/tmp/`, `outputs/`).

## Review Focus

1. **GW 95% bins outside every pointing's FoV** (always true for partial tiling). Λ must stay finite: those bins have zero exposure and are excluded by the safe mask. The limit should then show a ceiling, `c < 1` or a "no limit" bracket warning, instead of crashing. Tested in Task 2 (`test_lambda_finite_with_unobserved_mask`) and Task 3 (`test_expected_ul_toy_runs`).
2. **A pointing outside the geometry, or a geometry too small for the pointings.** Expect a `ValueError` naming the pointing, not a silently empty dataset. Task 2.
3. **An IRF file without `BACKGROUND` and no `bkg_path`.** Expect a `ValueError`, not a zero background that makes every TS infinite. Task 2.
4. **Pointings that overlap.** Exposure must add up in the stacked dataset, so Λ sees the deeper exposure. Task 2 (`test_overlap_exposure_adds`).
5. **A rerun with the same scenario.** The cached `.pkl` must be reused only if the scenario has not changed. Expected behaviour: the hash of the YAML content goes into the file name. Task 3.

---

### Task 1: `utils.gw_on_geom`, the GW map on any WCS geometry

**Files:**
- Modify: `gwuls/utils.py` (append one function)
- Create: `tests/test_gw_on_geom.py`
- Modify: `environment.yml` (add `- pytest`)

**Interfaces:**
- Produces: `gw_on_geom(file_lvk: str, geom: WcsGeom, r_grid: np.ndarray, resolution_hp_to_grid: int = 1024, credible: float = 0.95, cache_dir: Path | None = paths.GW_INPUT_DIR) -> dict` with keys `prob_gw` (ny, nx, floored at 1e-20), `prob_gw_integrated` (unfloored), `mask_threshold_95` (bool, ny×nx), `lvk_prob_hp`, `r_grid`, `cdf_table` (ny, nx, n_r), `cdf_valid`, `t_merger_iso` (str, UTC isot from the skymap `gps_time`), `lvk_prob_2d`, `threshold_maps`.

- [ ] **Step 1: Write the failing tests.** Use `data/gw_input/S240615dg.fits` (tracked) with a small geometry: centre (7.53, 45.81), `binsz=0.2`, width 3.6°, one energy axis 0.6–20 TeV; `r_grid = np.linspace(1, 10000, 200)`; `cache_dir=None`.

```python
def test_shapes_and_contract():
    out = utils.gw_on_geom(SKYMAP, geom, r_grid, cache_dir=None)
    ny, nx = geom.to_image().data_shape
    assert out["prob_gw"].shape == (ny, nx) == out["mask_threshold_95"].shape
    assert out["cdf_table"].shape == (ny, nx, r_grid.size)
    assert out["prob_gw"].min() >= 1e-20 and out["mask_threshold_95"].any()
    assert out["t_merger_iso"].startswith("2024-06-15")

def test_mask_probability_not_above_credible():
    out = utils.gw_on_geom(SKYMAP, geom, r_grid, cache_dir=None)
    assert out["prob_gw_integrated"][out["mask_threshold_95"]].sum() <= 0.95 + 0.02

def test_cache_roundtrip(tmp_path):
    a = utils.gw_on_geom(SKYMAP, geom, r_grid, cache_dir=tmp_path)
    b = utils.gw_on_geom(SKYMAP, geom, r_grid, cache_dir=tmp_path)   # loaded from cache
    np.testing.assert_array_equal(a["cdf_table"], b["cdf_table"])
```

- [ ] **Step 2: Run and see them fail.** `conda run -n gw-gamma-global-uls pytest tests/test_gw_on_geom.py -v` should fail with `AttributeError: ... no attribute 'gw_on_geom'`.
- [ ] **Step 3: Implement `gw_on_geom`.** Move the non-Dirac path of notebook 3 cells 7, 9 (`extract_geom_coords`), 11 and 13 into it unchanged: `read_sky_map(..., distances=True)`, `healpix2map`, `get_hp_map_thresholds`, `integrate_hp_on_wcs`, `compute_threshold_masks`, `map_hp_pixels_to_wcs_bins(fill_empty=True)`, then `compute_distance_cdf_table(normalize=True)` with the `distance_cdf_cache_key`/`save`/`load` cache when `cache_dir` is not None. Don't port the Dirac-delta mode; notebook 3 keeps its own branch for it.
- [ ] **Step 4: Run the tests and see them pass.** Same command, 3 passed.
- [ ] **Step 5: Commit.** `git add gwuls/utils.py tests/test_gw_on_geom.py environment.yml && git commit -m "Add utils.gw_on_geom: GW map, 95% mask and distance CDFs on any WCS geometry"`

---

### Task 2: `gwuls/scenario.py`, from a scenario to the input `.pkl`

**Files:**
- Create: `gwuls/scenario.py`
- Create: `tests/toy_irf.py` (helper, not a test)
- Create: `tests/test_scenario.py`

**Interfaces:**
- Consumes: `utils.gw_on_geom` (Task 1); `simulate.validate_simulation_input`.
- Produces:
  - `@dataclass Pointing: ra_deg: float, dec_deg: float, t_start_s: float, duration_s: float, irf_path: str, bkg_path: str | None = None` (`t_start_s` is seconds after the merger)
  - `@dataclass Scenario: name: str, skymap_path: str, pointings: list[Pointing], center_deg: tuple[float, float] | None = None, width_deg: float | None = None`, plus the analysis settings from Global Constraints as fields with those defaults. If `center_deg` is None, use the mean of the pointings (via a unit-vector mean, so it is safe at RA 0/360). If `width_deg` is None, use the largest pointing separation from the centre plus `size_fov`, times 2, plus `2·binsz`.
  - `load_scenario(path: str) -> Scenario` (YAML: top-level fields plus a `pointings:` list of mappings; relative paths resolve against the YAML's directory)
  - `scenario_hash(path: str) -> str` (first 10 hex characters of the sha1 of the YAML bytes)
  - `build_simulation_input(scn: Scenario, out_path: str, n_containment: int = 100, seed: int = 0) -> dict` writes the `.pkl` and returns `validate_simulation_input(..., mode="3d")`'s info dict.
- `tests/toy_irf.py`: `write_toy_irf(path, with_bkg=True) -> str`. Writes a full-enclosure IRF FITS: `EffectiveAreaTable2D` flat at 1e5 m² over offsets 0–3°, `EnergyDispersion2D.from_gauss` (σ = 0.1), a Gaussian `PSF3D` (σ = 0.1°), and, if `with_bkg`, a flat `Background3D` (1e-6 s⁻¹ MeV⁻¹ sr⁻¹). Each goes through `.to_table_hdu()` into one `HDUList`. It must load with `gammapy.irf.load_irf_dict_from_file`.

- [ ] **Step 1: Write the failing tests.** Use the toy IRF, the S240615dg skymap, `binsz=0.2`, `n_containment=5`, and two 1800 s pointings 1.5° apart near (7.53, 45.81), starting at 3600 s and 7200 s.

```python
def test_pkl_contract(tmp_path):
    info = scenario.build_simulation_input(scn, tmp_path / "s.pkl", n_containment=5)
    data = pickle.load(open(tmp_path / "s.pkl", "rb"))
    assert len(data["datasets_runs"]) == 2
    np.testing.assert_allclose(data["run_gti_s"][1], [[7200, 9000]])
    assert data["t_merger_iso"].startswith("2024-06-15")
    assert 0 < info["containment_factor"] <= 1.5

def test_exposure_outside_fov_is_zero(tmp_path): ...      # stacked exposure summed over energy: 0 at a bin > size_fov from both pointings, > 0 at each pointing
def test_overlap_exposure_adds(tmp_path): ...             # two identical pointings: exposure at the centre = 2× that of one
def test_lambda_finite_with_unobserved_mask(tmp_path):    # Review Focus 1
    # build with one pointing covering part of M95; perform_n_simulations(n_sim=3, amplitude=0)
    assert np.isfinite(np.load(out)["lambda_data"]).all()
def test_pointing_outside_geometry_raises(): ...          # width_deg=3, pointing 5° away -> ValueError naming the pointing
def test_missing_background_raises(tmp_path): ...         # write_toy_irf(with_bkg=False), bkg_path=None -> ValueError
def test_load_scenario_yaml(tmp_path): ...                # YAML with relative irf_path -> absolute; field values round-trip
```

- [ ] **Step 2: Run and see them fail.** `pytest tests/test_scenario.py -v` should fail with `ModuleNotFoundError: gwuls.scenario`.
- [ ] **Step 3: Implement `scenario.py`.** `build_simulation_input` follows notebook 3 cells 20–22, 29, 31 and 39, without the real-data parts:
  - `WcsGeom.create(skydir=center, binsz, width=(w, w), proj="AIR", frame="icrs", axes=[energy])`, then `utils.gw_on_geom` with the default `cache_dir`.
  - One `Observation.create(pointing=FixedPointingInfo(...), livetime=duration_s·s, irfs=irf_dict, location=EarthLocation.of_site("Roque de los Muchachos"), reference_time=Time(t_merger_iso) + t_start_s·s)` per pointing, then `MapDatasetMaker(selection=["background","psf","edisp","exposure"])` and `SafeMaskMaker(["offset-max"], offset_max=size_fov)`; stack them.
  - Load `irf_dict` with `load_irf_dict_from_file(irf_path)`, updated from `bkg_path` when given. Raise if `"bkg"` is missing. Raise if a pointing is closer than `size_fov` to the geometry edge.
  - Estimators and containment as in cells 22 and 31. Build the containment model from gammapy objects directly, with no notebook globals.
  - `type_obs = "sim"`, `run_gti_s[i] = np.array([[t_start_s, t_start_s + duration_s]])`.
  - Write the `.pkl` with every key from Global Constraints, then return `validate_simulation_input(data, "3d")`.
- [ ] **Step 4: Run the tests and see them pass.** Same command, 7 passed.
- [ ] **Step 5: Commit.** `git add gwuls/scenario.py tests/toy_irf.py tests/test_scenario.py && git commit -m "Add gwuls.scenario: build the simulation input from skymap + pointings + MC IRFs"`

---

### Task 3: Expected upper limits for one scenario (`scripts/expected_ul.py`)

**Files:**
- Modify: `gwuls/scenario.py` (add one function)
- Create: `scripts/expected_ul.py`
- Create: `scenarios/toy.yaml` (the Task 2 toy setup, with IRFs written by `tests/toy_irf.py`)
- Test: `tests/test_scenario.py` (append)

**Interfaces:**
- Consumes: `build_simulation_input`, `load_scenario`, `scenario_hash` (Task 2); `simulate.perform_n_simulations`, `run_fit_ul`, `run_fit_ul_3d`, `summarize_upper_limits`, `EmissionModelInjection`.
- Produces: `expected_upper_limits(path_pkl: str, n_bkg: int = 5000, which: tuple[str, ...] = ("2d", "3d"), fit_kwargs: dict | None = None, emission_model: EmissionModelInjection | None = None, n_jobs: int = 1, tmp_dir: str) -> dict`. The dict has keys `lambda_bkg_median`, `lambda_bkg_split` (two half-sample medians), `ul_2d` and `ul_3d` (the `run_fit_ul[_3d]` dicts, without the per-realisation arrays), `ul_model` if `emission_model` is set, and `frac_gw_in_mask`.
- CLI: `python scripts/expected_ul.py SCENARIO.yaml [--n-bkg 5000] [--which 2d,3d,model] [--max-sims 6000] [--precision 0.01] [--n-jobs -1] [--overwrite]`. It writes `data/tmp/<name>_<hash>.pkl` (reused unless `--overwrite`) and `outputs/results/<name>/<name>_<hash>_expected_ul.json` (via `json.dump(..., default=str)`).

- [ ] **Step 1: Write the failing test** (tiny budget, so it runs in minutes):

```python
def test_expected_ul_toy_runs(tmp_path):
    scenario.build_simulation_input(scn, tmp_path / "s.pkl", n_containment=5)
    res = scenario.expected_upper_limits(str(tmp_path / "s.pkl"), n_bkg=60,
            fit_kwargs=dict(n_scout=150, n_refine=150, max_sims=300, precision=0.1),
            tmp_dir=str(tmp_path))
    assert np.isfinite(res["lambda_bkg_median"])
    assert res["ul_2d"]["amp_ul"] > 0 and res["ul_3d"]["lum_ul"] > 0

def test_cli_reuses_pkl(tmp_path): ...   # run CLI twice on scenarios/toy.yaml with tiny flags; the pkl mtime does not change; editing the YAML changes the hash and the file name
```

- [ ] **Step 2: Run and see it fail** with `AttributeError: expected_upper_limits`.
- [ ] **Step 3: Implement it.** Run `perform_n_simulations(n_bkg, 0.0, seed=0, store_ts_maps=False, store_stats=False)`; set `lambda_bkg_m = median`; call `run_fit_ul` and `run_fit_ul_3d` with `lambda_real=lambda_bkg_m, significance=0.0`, `seed=1_000_000` (`pipeline.md` §6.5), and `cache_path` under `tmp_dir`. The CLI is argparse calling `load_scenario`, `build_simulation_input` and `expected_upper_limits`, and nothing else.
- [ ] **Step 4: Run the tests and see them pass.** `pytest tests/ -v`: everything passes.
- [ ] **Step 5: Commit.** `git add gwuls/scenario.py scripts/expected_ul.py scenarios/toy.yaml tests/test_scenario.py && git commit -m "Add expected-UL runner for simulated scenarios"`

---

### Task 4: Shared GW prep in notebook 3, and the S240615dg mirror (acceptance test)

**Files:**
- Modify: `notebooks/3_upper_limits.ipynb` (cells 11 and 13: the non-Dirac branch calls `utils.gw_on_geom`)
- Create: `scenarios/S240615dg_mirror.yaml`
- Modify: `docs/pipeline.md` (new §14 "Simulated scenarios": what is simulated, the assumptions above, the mirror result); `README.md` (one line under Notebooks/Layout)

**Interfaces:**
- Consumes: `utils.gw_on_geom` (Task 1), `scripts/expected_ul.py` (Task 3).

- [ ] **Step 1: Swap the notebook's GW prep and check that nothing moved** (locally, needs only the skymap). Rerun cells 1–14. Expected: "Geometry covers X% of the GW", the mask pixel count, "CDF table shape …, valid bins …" and the hottest-bin distance line are identical to the outputs stored in the committed notebook (`git diff` on those output lines shows no change).
- [ ] **Step 2: Write the mirror scenario (on the cluster).** Five pointings, one per run 17821–17825: `RA_PNT`, `DEC_PNT`, `t_start_s` = run start − merger, `duration_s` = `LIVETIME` (from the notebook's `obs_table` printout). `irf_path` = each run's DL3 file, `bkg_path` = `bkg_pybkgmodel_3d_<obs_id>.fits`. `center_deg = (7.53, 45.81)` and `width_deg = 3.575`, the same as notebook 3.
- [ ] **Step 3: Run it.** `python scripts/expected_ul.py scenarios/S240615dg_mirror.yaml --which 2d,3d --n-jobs -1`. Expected: the 2D limit is within ±15% of 2.34e-12 cm⁻² s⁻¹ TeV⁻¹ and the 3D limit within ±15% of 3.85e45 erg/s (`README` Status; the real run also used Λ* = background median). What differs is the template background against the ring background, and one GTI per run against the real GTIs. Outside ±15%, stop and investigate before any tiling work: compare κ (0.682) and the background Λ median (−4.07).
- [ ] **Step 4: Document.** Write §14 in `pipeline.md` with the mirror numbers and add a row to the §12 validation table.
- [ ] **Step 5: Commit.** `git add notebooks/3_upper_limits.ipynb scenarios/S240615dg_mirror.yaml docs/pipeline.md README.md && git commit -m "Notebook 3 uses gw_on_geom; S240615dg mirror scenario validates the simulation path"`

---

## Roadmap: later plans, each built on this one

Each item gets its own plan once this one is done. The order is a recommendation.

1. **Validate the method with pseudo-experiments (next).** Add `pseudo_observation(pkl, seed, injection)` and run the full chain hundreds of times with a known truth. Checks: coverage (the fraction of pseudo-experiments whose UL ≥ the true L should be ≥ 95%), the effect of the power-constrained Λ* choice, Λ = max against a summed (Bayes-factor) statistic, and the per-position statistic of guidelines Step 8 (`pipeline.md` §11.5). Known truth should also settle the 17% trials-factor puzzle (§7.6). Without this, no tiling result can be trusted.
2. **Tiling studies.** Turn tilepy schedules (the `tilepyenv` env exists) into scenario YAMLs and compare strategies: number of tiles against depth per tile, and delay. Partial coverage leaves the 95% UL undefined (the coverage cap, §7.2), so the metric has to change. The natural one is the curve P_det(L) with its ceiling c = covered probability, plus L at a fixed fraction of c. Choosing that metric is a decision for that plan.
3. **O5 forecasts.** Loop over observing-scenario BNS skymaps with realistic delays and get the distribution of expected limits against the emission model. The 3D emission-model limit becomes physically meaningful here: BNS with jets, not BBH.
4. **Emission-model systematics.** Loop over the jet variants already cached via `model_dir_3d` (§11.5).
5. **S241125n.** The Dirac-delta mode (§3.5) already fixes the sky at one position, with no trials factor. Pinning the sky at the candidate counterpart and drawing distance and angle from the PE conditional on that position gives a model limit for it with little new code. It does not need this plan.
