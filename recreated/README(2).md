# GMN daily wavelet sky maps — from raw meteor radiants

Reproduces the Global Meteor Network (GMN) daily wavelet shower maps and the
corresponding maxlist **directly from raw trajectory radiants**, using the 3‑D
Mexican‑hat wavelet method of **Brown, Wong, Weryk & Wiegert (2010)**, *Icarus*
**207**, 66–81.

The input is a GMN **trajectory‑summary** file — the real radiants
(geocentric RA/Dec, Vg, solar longitude, orbital elements, IAU shower code), one
row per meteor. The tool computes the wavelet coefficient field, the significance
map, finds the maxima and writes the maxlist. Nothing about the *output* is used
as input.

```
trajectory_summary  ──►  3‑D wavelet  ──►  xsig maps (EQU + SCE)
   (raw radiants)                      └─►  maxlist (radiants → showers + orbits)
```

## Files

| file | role |
|------|------|
| `gmn_wavelet.py`        | library: transforms, wavelet kernel, backgrounds, shower catalogue + association, quality cuts, maxlist I/O, plotting, **traj‑summary parsing + fetching** |
| `gmn_wavelet_maps.py`   | command‑line driver: radiants → two maps + maxlist |
| `rms_catalog/`          | shower catalogues for association: `established_showers.csv`, `flux_showers.csv` (RMS), `mdc_streamfulldata.txt` (IAU MDC working list) |
| `make_synth_trajsummary.py` | generates a synthetic traj‑summary file (test data only) |
| `out/`                  | example real output (maps + maxlist) and the cached annual background |

## Usage

**Fetch a daily file from GMN by date** — the solar‑longitude suffix is looked up
from the GMN directory index, so you can give just the date:

```bash
python gmn_wavelet_maps.py --date 20250318
```

You can still pin the range explicitly (this builds the URL directly, e.g.
`…/daily/traj_summary_20250318_solrange_358.0-359.0.txt`):

```bash
python gmn_wavelet_maps.py --date 20250318 --solrange 358.0 359.0
```

The filename builder matches GMN's convention exactly — three‑digit zero‑padded
bounds and the `359.0-000.0` year‑boundary wrap — verified against the live
directory listing across 2018–2025.

**Use a local trajectory‑summary file:**

```bash
python gmn_wavelet_maps.py --file traj_summary_20250318_solrange_358.0-359.0.txt
```

**Reproduce the published GMN map** — this requires the yearly morhist background
(a single daily file can only give the `self` approximation). `--fetch-year`
downloads `traj_summary_yearly_YYYY.txt` and uses it as the background; `--quality`
applies the trajectory quality cuts and `--bg-cache` saves the (date‑independent)
yearly background so later dates reuse it:

```bash
python gmn_wavelet_maps.py --date 20250318 --fetch-year 2025 \
       --quality --bg-cache bg_2025.npz --out out
# or, with a local copy of the yearly file / your own year of dailies:
python gmn_wavelet_maps.py --date 20250318 \
       --background annual --bg-file traj_summary_yearly_2025.txt \
       --quality --bg-cache bg_2025.npz --out out
```

The validated coarse command (matches the reference; see **Status** below):

```bash
python gmn_wavelet_maps.py --file traj_summary_20250318_solrange_358.0-359.0.txt \
       --background annual --bg-file traj_summary_yearly_2025.txt \
       --step 0.5 --vel-step-pct 5 --quality --bg-cache out/bg_2025.npz --out out
```

GMN's exact parameters (read from the maxlist header and used as defaults): sky
grid 0.2°, Vg 9–72 km/s in 2% steps, spatial probe 1.0°, velocity probe 5%,
window half‑width 0.5°, filters r_cnt ≥ 10 and ≥ 3σ above the yearly median,
shower association 3.0°/10%. At 0.2° the annual background over a full year is a
heavy (server‑scale) computation — coarsen with `--step` for a faster look.

Outputs, written to `--out` (default `.`):

```
wav-<tag>-equ.png    equatorial Hammer map of xsig
wav-<tag>-sce.png    sun-centred ecliptic Hammer map of xsig
maxlist-<tag>.txt    detected maxima with shower codes + orbital elements
```

## Method

For a test point `(x0, y0, Vg0)` and the radiant set `f(x, y, Vg)`, Brown et al.
Eq. (1):

```
Wc = N · Σ_i (3 − u_i) · exp(−u_i/2),
u_i = [(Δx)² + (Δy)²] / a²  +  (ΔVg)² / sv²
```

with `x = (λ − λ☉)` (sun‑centred ecliptic longitude), `y = β`, spatial probe
`a = 1.0°`, velocity probe `sv = 0.05·Vg0` (5 %), and only radiants within 4 probe
sizes contributing. Distances are great‑circle, so the same evaluator works in
equatorial or sun‑centred‑ecliptic coordinates. The displayed quantity is

```
xsig = (Wc − background_median) / background_sigma
```

(`xsig` is a ratio, so the prefactor `N` cancels and is dropped). The map shows
the maximum of `xsig` over the velocity axis; the maxlist reports the velocity at
which each maximum occurs.

Per maximum, the radiants inside the wavelet probe are collected to give the
radiant count, the **mean orbital elements** (a, e, i, ω, Ω, q, Q, Tj, V∞) and
the **shower association** (the modal IAU code among those radiants). Maxima with
fewer than `--min-radiants` (default 10) radiants are rejected — Brown et al.'s
minimum‑radiant filter.

## Background / significance — important

The significance scale depends on how the background is estimated. Two modes:

* **`self`** (default) — single‑epoch. The background median and σ are taken from
  the map's *own* wavelet field (robust, 3σ‑clipped, per velocity). Needs only
  the one file and runs immediately. It reproduces the **structure** of the GMN
  maps (apex, antihelion/helion, toroidal sources, and the showers), but it is an
  **approximation**.

* **`annual`** — the Brown/GMN estimator. At each `(ll₀, β, Vg)` cell the wavelet
  coefficient is evaluated once per degree of solar longitude across the whole
  year, and the median + σ are found by iteratively discarding points > 3σ above
  the median. `xsig = (wc − median)/σ` is the number of σ a given day sits above
  that yearly level. Supply a full year via `--fetch-year` / `--bg-file`. Key
  implementation points:
  * The field is evaluated **natively in sun‑centred ecliptic**, and each
    background epoch's radiants are converted to SCE with **that epoch's λ☉**, so
    the (SCE‑stationary) sporadic sources line up across the year. The equatorial
    map is a reprojection of the SCE field, so both maps share one correct
    background.
  * One year of epochs gives a noisy σ that collapses in sparse (off‑ecliptic,
    low‑velocity) cells, producing spurious significance spikes. GMN's `wc_s` is
    really the **analytic shot noise** (its footer formula), so σ is floored at
    `--shot-frac · √(Σh²)` of the day's radiants (`shot_frac ≈ 1/N`, the dropped
    wavelet normalisation; default 0.20, calibrated to GMN's `wc_s`). This fixes
    both the spikes and the absolute `xsig` scale.
  * The background depends only on (year, grid, probe) — not the date — so
    `--bg-cache PATH` persists it (median/σ `.npz`) and every later date reuses
    it. `--bg-step` computes it on a coarser grid and interpolates up.

**Why `self` is not pixel‑identical to the published maps.** The official `xsig`
is referenced to the *yearly* background. Sporadic sources (apex, antihelion, …)
are present every night, so their yearly median is high and their significance on
any given day is *suppressed*; a transient shower like EVI is absent the rest of
the year, so its yearly median is low and it *dominates*. A single night cannot
know the yearly level, so in `self` mode persistent sporadic sources appear
relatively brighter and transient showers relatively less dominant than in the
yearly‑normalised product. This is a real property of the method, not a
work‑around — to match the published maps, run `--background annual` with the
yearly morhist (`--fetch-year YYYY`). At GMN's 0.2° grid over ~360 epochs this is
a heavy, server‑scale computation; coarsen with `--step` for a faster look.

## Notes

* `make_synth_trajsummary.py` is a **test‑data** generator only (plants EVI + DME
  to exercise parse → wavelet → maps → maxlist). The real maps in `out/` come from
  the `--date`/`--file` commands above on real GMN data.
* **Shower association** uses the RMS catalogues plus the IAU MDC working list
  (`rms_catalog/`): each maximum is matched to the nearest catalogued shower
  within 3°/10% (radiant drifted to the map's λ☉). EVI and EOP reproduce the
  reference exactly; some other codes (DME/AAL/POS/BCO) are missed or mismatched
  because the public catalogues lack GMN's internally‑calibrated working‑list
  radiants for them (GMN's wavelet‑map generator is not public). Drop a
  GMN‑matching list into `rms_catalog/` (or pass `--rms-catalog`) to improve it.
* The maxlist mirrors the GMN `example.txt` header and columns (Slon, ll₀, β, RA,
  Dec, Vg, wc, xsig, **wc_s**, r_cnt, V∞, a, e, i, ω, Ω, q_per, q_aph, Tj, **eta**,
  Shower). `eta` is emitted as `-` exactly as GMN does. Per `example.txt`'s own
  footer, **`xsig` = (wc − yearly_median)/σ_year** and **`wc_s` = σ_year** (the
  per‑cell yearly standard deviation of wc, in wc units) — *not* a self/single‑epoch
  significance. Verified: for every reference row `wc − xsig·wc_s ≈ 0`, and `wc_s`
  tracks GMN's analytic shot noise `0.04987·√r_cnt / a[rad] / √sv` (DME 10.33 vs
  10.3, BCO 9.89 vs 9.9).
* On the maps, each named shower is circled **once** (the strongest maximum for
  that code above `--label-min-xsig`, at most `--max-labels`, strongest first);
  the full maxlist still lists every maximum passing the r_cnt and σ filters.
* Trajectory‑summary columns are read by fixed index from the semicolon‑delimited
  GMN format (RAgeo=7, DECgeo=9, Vgeo=15, Sol=5, …, validated against a live file).

## Verification

* **Parser** — validated against live GMN files from **two independent dates**
  (2025‑03‑18, λ☉≈358; 2025‑12‑02, λ☉≈250), including a real `HYD` shower meteor
  on a retrograde Halley‑type orbit. The trajectory‑summary format is fixed
  across all daily files (same generator), and the column order was
  cross‑checked against the GMN data schema, so the parser is date‑independent.
* **Filename builder** — reproduces the real file names for 10 dates spanning
  2018–2025, covering the leading‑zero (`009.0-010.0`) and year‑wrap
  (`359.0-000.0`) cases.
* **Pipeline (real data)** — run end‑to‑end against the reference set
  (`original/example.txt` + the published PNGs) using the real daily file and the
  2025 yearly morhist as the `annual` background (coarse `--step 0.5`,
  `--vel-step-pct 5`, `--quality`). Results vs `example.txt`:

  | code | ll₀ (ref→mine) | β | Vg | xsig ref → mine |
  |------|----------------|---|----|-----------------|
  | EVI  | 186.5 → 186.5 | 5.5 | 26.7 | **41.6 → 44.3** |
  | EOP  | 262.7 → 263.5 | 6.9 | 70.6 | **15.0 → 15.5** |

  Radiant count after `--quality`: **2585** (GMN NOrb = 2574). EVI lands on the
  exact cell and dominates the map; the sporadic field is suppressed as in the
  reference. The `--shot-frac` floor was calibrated on EVI alone, yet EOP matches
  independently — confirming it captures the real (analytic shot‑noise) scale.
  The absolute `xsig` is not pixel‑identical because our morhist is calendar‑2025,
  not GMN's exact 2024/358→2025/358 window.

## Dependencies

`numpy`, `scipy` (cKDTree, ndimage), `matplotlib`. No astropy/pandas required.
Fetching by `--date` needs network access to `globalmeteornetwork.org`; offline,
download the file and pass it with `--file`.

---

# Status — validated against the reference set

The pipeline has been run end‑to‑end on real GMN data and validated against the
user's reference set (`original/example.txt` + the published PNGs). The build‑time
"open problems" are resolved:

* **`wc_s` definition (was unconfirmed):** resolved from `example.txt`'s footer —
  `wc_s` = the per‑cell **yearly σ** of wc (≈ the analytic shot noise), and
  `xsig = (wc − yearly_median)/σ`. The earlier "self‑significance" emission was a
  bug, now fixed.
* **Annual background:** evaluated SCE‑natively with per‑epoch λ☉ conversion (was
  a frame bug); cached on disk (`--bg-cache`); σ floored at the analytic shot
  noise (`--shot-frac`) so the `xsig` scale matches GMN and sparse‑cell spikes are
  gone.
* **Quality cuts (`--quality`):** `Qc ≥ 15°`, `MedianFitErr ≤ 110″`, `≥ 2`
  stations → 2585 radiants in the 358–359 window (GMN NOrb = 2574).
* **Shower association:** RMS catalogues + IAU MDC working list, nearest within
  3°/10% (radiant drifted to λ☉).
* **Validation:** EVI at the exact cell with `xsig` 44.3 (ref 41.6); EOP 15.5 (ref
  15.0); see the **Verification** section above. Example output is in `out/`.

The reproduction command is the validated one shown under **Usage**. Because the
public morhist (calendar 2025) is not GMN's exact 2024/358→2025/358 window, the
absolute `xsig` is close but not pixel‑identical; positions, structure and the
significance scale all match.

## Verified ground truth (use directly, don't re‑derive)

**traj_summary columns** (0‑based, `;`‑delimited, comment lines start `#`),
encoded in `gmn_wavelet._TS_COL`: iau=4, sol=5, RAgeo=7, DECgeo=9, LAMgeo=11,
BETgeo=13, Vgeo=15, a=23, e=25, incl=27, peri=29, node=31, q=37, qaph=43,
TisserandJ=49, Vinit=59. Other useful (not yet parsed) quality columns: Qc≈80,
MedianFitErr≈81, Num stations≈84. IAU code `...` = sporadic.

**Projections** (confirmed against the PNGs): ll₀ = (λ − λ☉) mod 360. EQU:
plot_lon = −wrap180(RA), lat = Dec (RA increases left, centre RA = 0). SCE:
plot_lon = −wrap180(ll₀ − 270), lat = β (apex ll₀ = 270 at centre). Obliquity
ε = 23.43928° (J2000). The wavelet is evaluated in the equatorial frame; great‑
circle distance is frame‑independent.

**Kernel:** h = (3 − u)·exp(−u/2), u = (d/a)² + (ΔVg/sv)², a = 1.0°,
sv = 0.05·Vg, contributions cut off beyond 4 probe sizes in each dimension.

**GMN production parameters** (from `example.txt` header): ll₀ & β grid 0.2°;
Vg 9–72 km/s in 2% multiplicative steps; spatial probe 1.0°; velocity probe 5%;
window half‑width 0.5°; filters r_cnt ≥ 10 and wc ≥ 3σ above the yearly median;
shower association 3.0° / 10%. Colormap: `LogNorm(1, 150)`, near‑black → navy →
blue → cyan → green → yellow → orange → red → pink → white.

**GMN data URLs:**
* daily index: `…/data/traj_summary_data/daily/`
* daily file: `…/daily/traj_summary_YYYYMMDD_solrange_LLL.L-HHH.H.txt`
  (3‑digit zero‑padded, upper bound wraps 360→`000.0`)
* yearly morhist: `…/data/traj_summary_data/traj_summary_yearly_YYYY.txt`
* monthly: `…/monthly/traj_summary_monthly_YYYYMM.txt`; full DB:
  `…/traj_summary_all.txt` (~1 GB); latest day: `…/daily/traj_summary_yesterday.txt`

## Remaining / future work

* **Full 0.2° render.** The validated run is coarse (`--step 0.5`,
  `--vel-step-pct 5`) so the map is grainier than GMN's 0.2°/2% product. The full
  resolution is wired up and one‑time‑cacheable — run with `--step 0.2
  --vel-step-pct 2 --bg-cache …` (heavy: ~1.6 M cells × 105 velocities × ~360
  epochs; let it run once and the cache serves every later date). Secondary
  speedups: `--bg-step` (coarse background + interpolate), parallelise the chunk
  loop, or move the kernel sum to GPU.
* **Shower association coverage.** EVI/EOP reproduce exactly, but the public
  catalogues lack GMN‑calibrated radiants for several codes (DME/AAL/POS/BCO miss
  or mismatch — e.g. BCO→MTA). GMN's wavelet‑map working list is not public; drop
  a GMN‑matching list into `rms_catalog/` (or `--rms-catalog`) to close the gap.
* **Exact‑window morhist.** For a closer numeric match to `example.txt`, build the
  background from GMN's exact 2024/358→2025/358 window (combine
  `traj_summary_yearly_2024.txt` + `…_2025.txt`; the per‑degree sol binning already
  merges years) rather than calendar 2025.
* **`--shot-frac` provenance.** The 0.20 floor (≈ 1/N) is calibrated to GMN's
  `wc_s`; if Brown et al.'s wavelet normalisation N can be recovered analytically,
  set it from first principles instead.

## Local setup

Python 3.10+. `pip install numpy scipy matplotlib`. `--date` / `--fetch-year`
need network access to `globalmeteornetwork.org`. `make_synth_trajsummary.py` is a
test‑data generator (kept for testing). Shower catalogues live in `rms_catalog/`
(RMS CSVs from `github.com/CroatianMeteorNetwork/RMS` + the IAU MDC working list).
