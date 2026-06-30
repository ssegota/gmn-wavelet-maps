# GMN daily wavelet meteor-shower maps — from raw radiants

Reproduces the **Global Meteor Network (GMN)** daily wavelet shower maps and the
corresponding *maxlist* **directly from raw trajectory radiants**, using the 3-D
Mexican-hat wavelet method of **Brown, Wong, Weryk & Wiegert (2010)**, *Icarus*
**207**, 66–81.

Input: a GMN trajectory-summary file (one meteor per row — geocentric RA/Dec, Vg,
solar longitude, orbit, IAU code). Output: the equatorial and sun-centred-ecliptic
significance maps plus a maxlist of detected showers (position, significance,
radiant count, mean orbit, shower name).

## Repository layout

| path | contents |
|------|----------|
| [`recreated/`](recreated/) | the implementation: `gmn_wavelet.py` (library), `gmn_wavelet_maps.py` (CLI driver), `rms_catalog/` (shower catalogues), `out/` (example outputs) — see [`recreated/README(2).md`](recreated/README(2).md) for full usage |
| [`recreated/report/`](recreated/report/) | the write-up: **[`report.pdf`](recreated/report/report.pdf)** (methodology, validation, multi-shower examples, limitations, usage) and its LaTeX source + figures |
| [`original/`](original/) | the published GMN reference product for λ☉≈358.5° (maps + `example.txt` maxlist) used as the validation target |

## Quick start

```bash
cd recreated
# published-style product: yearly background + quality cuts + on-disk cache
python gmn_wavelet_maps.py --date 20250318 --fetch-year 2025 \
       --quality --bg-cache bg_2025.npz --out out
```

See [`recreated/README(2).md`](recreated/README(2).md) and the report for all flags,
modes (`self` vs `annual`), validation results and the caveats.

## Not included (git-ignored)

The yearly morhist (`traj_summary_yearly_*.txt`, ~0.9 GB), the cached background
(`*.npz`), and the copyrighted Brown et al. (2010) PDF are excluded. The yearly
file is re-fetched automatically with `--fetch-year`, and the cache is regenerated
on first run.

## Credits / data

- Method: Brown, P., Wong, D. K., Weryk, R. J., & Wiegert, P. (2010), *Icarus*, 207(1), 66–81. https://doi.org/10.1016/j.icarus.2009.11.015
- Data: the Global Meteor Network (https://globalmeteornetwork.org).
- Shower catalogue: RMS (CroatianMeteorNetwork/RMS) + the IAU Meteor Data Center working list.
