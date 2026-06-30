#!/usr/bin/env python3
"""
gmn_wavelet_maps.py
===================

Reproduce the GMN daily wavelet sky maps (equatorial + sun-centred ecliptic) and
the corresponding maxlist DIRECTLY FROM RAW METEOR RADIANTS, using the 3-D
Mexican-hat wavelet method of Brown et al. (2010) implemented in gmn_wavelet.py.

The input is one or more GMN *trajectory-summary* files (the real radiants:
geocentric RA/Dec, Vg, solar longitude, orbital elements, IAU shower code).
These can be a LOCAL file or fetched from the GMN data server by date + solar
longitude range.

Usage
-----
  # fetch a single daily file from GMN and make the maps (single-epoch 'self'
  # significance -- runs immediately, no year of data needed):
  python gmn_wavelet_maps.py --date 20250318 --solrange 358.0 359.0

  # use a local trajectory-summary file instead:
  python gmn_wavelet_maps.py --file traj_summary_20250318_solrange_358.0-359.0.txt

  # exact GMN significance (yearly per-cell background) from a local year of
  # daily files:
  python gmn_wavelet_maps.py --file target.txt --background annual \
        --bg-file "gmn_year/*.txt"

Background modes
----------------
  self   (default) -- significance from the single map's own wavelet field
                      (robust spatial median+sigma per velocity).  Needs only
                      the one file; APPROXIMATES the official product.
  annual           -- the exact Brown/GMN estimator: per-cell yearly median+sigma
                      (iterative 3-sigma clip) from a full year of daily files
                      supplied via --bg-file.  See the README for the caveat on
                      why 'self' differs from the published maps.

Outputs (written to --out, default '.'):
  wav-<tag>-equ.png    equatorial Hammer map of xsig
  wav-<tag>-sce.png    sun-centred ecliptic Hammer map of xsig
  maxlist-<tag>.txt    detected maxima with orbital elements (GMN-style)
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gmn_wavelet as gw


# --------------------------------------------------------------------------- #
#  Geometry helpers (projection conventions, confirmed against the GMN PNGs)
# --------------------------------------------------------------------------- #

def wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def grid_to_radec(frame, plon, plat, lambda_sun):
    """Map a regular plot grid (deg) to RA/Dec (deg) for the chosen frame.

    equatorial : plot_lon = -wrap180(RA)         -> RA increases to the left
    sce        : plot_lon = -wrap180(ll0 - 270)  -> apex (ll0=270) at centre
    """
    PL, PB = np.meshgrid(plon, plat)
    if frame == "equatorial":
        ra = (-PL) % 360.0
        dec = PB
    elif frame == "sce":
        ll0 = (270.0 - PL) % 360.0
        ra, dec = gw.sce_to_radec(ll0, PB, lambda_sun)
    else:
        raise ValueError(frame)
    return PL, PB, ra, dec


# --------------------------------------------------------------------------- #
#  Field evaluation  (computed once in sun-centred ecliptic; the equatorial map
#  is a reprojection of it, so both maps share one correct yearly background)
# --------------------------------------------------------------------------- #

def _sce_plot_grid(step):
    """Regular plot grid (deg).  plon -> ll0 = (270 - plon) % 360 (apex centred)."""
    plon = np.arange(-180.0 + step / 2, 180.0, step)
    plat = np.arange(-90.0 + step / 2, 90.0, step)
    return plon, plat


def _bilinear_sample(field, src_lon_deg, src_lat_deg, q_lon_deg, q_lat_deg):
    """Bilinearly sample field[lat, lon] (uniform axes) at query points.
    The longitude axis wraps over 360 deg; the latitude axis clamps."""
    from scipy.ndimage import map_coordinates
    dlon = src_lon_deg[1] - src_lon_deg[0]
    dlat = src_lat_deg[1] - src_lat_deg[0]
    ci = (np.asarray(q_lon_deg, float) - src_lon_deg[0]) / dlon
    ri = (np.asarray(q_lat_deg, float) - src_lat_deg[0]) / dlat
    ri = np.clip(ri, 0.0, field.shape[0] - 1.0)          # never wrap latitude
    coords = np.vstack([ri.ravel(), ci.ravel()])
    out = map_coordinates(field, coords, order=1, mode="grid-wrap")
    return out.reshape(np.shape(q_lon_deg))


def _annual_med_sig(test_xyz, bg_slon, vgrid, cfg, cache_path, meta):
    """Per-cell yearly (median, sigma) with optional on-disk caching.

    The annual background depends only on (bg source, grid, probe, vgrid) -- not
    on the target date -- so it is computed once and reused.  ``meta`` is a dict
    of identifying parameters stored alongside the arrays and re-checked on load.
    """
    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        ok = (d["med"].shape[0] == test_xyz.shape[0] and
              str(d.get("meta")) == str(meta))
        if ok:
            print(f"  loaded annual background from cache {cache_path}")
            return d["med"], d["sig"]
        print(f"  cache {cache_path} does not match current params; recomputing")
    if not bg_slon:
        raise SystemExit("annual background requested but no morhist available "
                         "(use --fetch-year or --bg-file, or a matching --bg-cache)")
    med, sig = gw.annual_background(test_xyz, bg_slon, vgrid, cfg, progress=True)
    if cache_path:
        np.savez(cache_path, med=med, sig=sig, meta=np.array(str(meta)))
        print(f"  saved annual background to cache {cache_path}")
    return med, sig


def evaluate_sce_field(tgt, bg_slon, lambda_sun, vgrid, cfg, step,
                       background, sigma_floor, bg_step=None, bg_cache=None,
                       bg_meta=None, shot_frac=0.20):
    """Compute the wavelet xsig field on a regular SCE plot grid.

    Returns a dict: plon_deg, plat_deg (axes), plon/plat (radians, for plotting),
    xsig, wc, vg (each shaped (nlat, nlon)), and sig (the per-cell yearly sigma at
    the velocity of the maximum, for the wc_s column; ``None`` in 'self' mode).
    """
    plon, plat = _sce_plot_grid(step)
    nlat, nlon = len(plat), len(plon)
    PL, PB = np.meshgrid(plon, plat)
    ll0 = (270.0 - PL) % 360.0
    test_xyz = gw.sphere_to_unit(ll0.ravel(), PB.ravel())   # SCE-native

    # target radiants in sun-centred ecliptic (this day's lambda_sun)
    tl0, tbet = gw.radec_to_sce(tgt["ra"], tgt["dec"], lambda_sun)
    tgt_xyz = gw.sphere_to_unit(tl0, tbet)
    wc, shot_today = gw.wavelet_and_shotnoise(test_xyz, tgt_xyz, tgt["vg"],
                                              vgrid, cfg)

    sig_map = None
    if background == "annual":
        if bg_step and abs(bg_step - step) > 1e-9:
            # compute the background on a coarse grid, interpolate up to `step`
            cplon, cplat = _sce_plot_grid(bg_step)
            CPL, CPB = np.meshgrid(cplon, cplat)
            cll0 = (270.0 - CPL) % 360.0
            cxyz = gw.sphere_to_unit(cll0.ravel(), CPB.ravel())
            cmed, csig = _annual_med_sig(cxyz, bg_slon, vgrid, cfg, bg_cache,
                                         bg_meta)
            nv = len(vgrid)
            cmed = cmed.reshape(len(cplat), len(cplon), nv)
            csig = csig.reshape(len(cplat), len(cplon), nv)
            med = np.empty((nlat, nlon, nv))
            sig = np.empty((nlat, nlon, nv))
            for j in range(nv):
                med[:, :, j] = _bilinear_sample(cmed[:, :, j], cplon, cplat,
                                                PL, PB)
                sig[:, :, j] = _bilinear_sample(csig[:, :, j], cplon, cplat,
                                                PL, PB)
            med = med.reshape(nlat * nlon, nv)
            sig = sig.reshape(nlat * nlon, nv)
        else:
            med, sig = _annual_med_sig(test_xyz, bg_slon, vgrid, cfg, bg_cache,
                                       bg_meta)
        # GMN's wc_s is the *analytic* shot noise (its footer formula), not the
        # empirical scatter; with the wavelet normalisation N dropped here, the
        # equivalent floor in our units is shot_frac * sqrt(sum h^2) of the day's
        # radiants (shot_frac ~ 1/N, calibrated once).  This both fixes the xsig
        # scale and removes the single-year small-number spikes in sparse cells.
        sig = np.maximum(sig, shot_frac * shot_today)
        xsig, vidx = gw.compute_xsig(wc, med, sig)
        sig_at = sig[np.arange(sig.shape[0]), vidx]
        sig_map = sig_at.reshape(nlat, nlon)
    else:  # self
        xsig, vidx, _, _ = gw.spatial_significance(wc, sigma_floor=sigma_floor)

    wc_max = wc[np.arange(wc.shape[0]), vidx]
    return {
        "plon_deg": plon, "plat_deg": plat,
        "plon": np.radians(plon), "plat": np.radians(plat),
        "xsig": xsig.reshape(nlat, nlon),
        "wc": wc_max.reshape(nlat, nlon),
        "vg": np.asarray(vgrid)[vidx].reshape(nlat, nlon),
        "sig": sig_map,
    }


def reproject_to_equ(sce, lambda_sun, step):
    """Reproject the SCE xsig field onto an equatorial plot grid.

    For each equatorial plot cell (plot_lon = -wrap180(RA)) compute its ll0 and
    bilinearly sample the SCE field, so the two maps show identical xsig at the
    same point on the sky (as in GMN's products).
    Returns (plon_rad, plat_rad, xsig_map).
    """
    eplon, eplat = _sce_plot_grid(step)
    EPL, EPB = np.meshgrid(eplon, eplat)
    ra = (-EPL) % 360.0
    dec = EPB
    lam, bet = gw.equatorial_to_ecliptic(ra, dec)
    ll0 = (lam - lambda_sun) % 360.0
    q_plon = wrap180(270.0 - ll0)            # SCE plot longitude of each ll0
    xsig = _bilinear_sample(sce["xsig"], sce["plon_deg"], sce["plat_deg"],
                            q_plon, bet)
    return np.radians(eplon), np.radians(eplat), xsig


# --------------------------------------------------------------------------- #
#  Rendering
# --------------------------------------------------------------------------- #

def overlay_maxima(ax, frame, maxima, lambda_sun, min_xsig):
    """Circle and label only maxima with an identified shower code (as the
    published GMN maps do), coloured by strength."""
    for m in maxima:
        code = m.get("code", "")
        if not code or m["xsig"] < min_xsig:
            continue
        ra, dec = gw.sce_to_radec(m["ll0"], m["bet"], lambda_sun)
        if frame == "equatorial":
            plon = np.radians(-wrap180(float(ra)))
            plat = np.radians(float(dec))
        else:
            plon = np.radians(-wrap180(m["ll0"] - 270.0))
            plat = np.radians(m["bet"])
        if m["xsig"] >= 25:
            col = "lime"
        elif m["xsig"] >= 10:
            col = "orange"
        else:
            col = "red"
        ax.scatter([plon], [plat], s=190, facecolors="none", edgecolors=col,
                   linewidths=1.2, zorder=6)
        ax.annotate(code, (plon, plat),
                    textcoords="offset points", xytext=(8, 8),
                    color=col, fontsize=8, fontweight="bold", zorder=7)


def overlay_candidates(ax, frame, cands, lambda_sun):
    """Circle unmatched significant peaks (no catalogue association) as shower
    candidates, in a distinct dashed white style with a '?' marker."""
    for m in cands:
        ra, dec = gw.sce_to_radec(m["ll0"], m["bet"], lambda_sun)
        if frame == "equatorial":
            plon = np.radians(-wrap180(float(ra)))
            plat = np.radians(float(dec))
        else:
            plon = np.radians(-wrap180(m["ll0"] - 270.0))
            plat = np.radians(m["bet"])
        ax.scatter([plon], [plat], s=150, facecolors="none", edgecolors="white",
                   linewidths=1.0, linestyle=(0, (3, 3)), zorder=6)
        ax.annotate("?", (plon, plat), textcoords="offset points",
                    xytext=(6, 6), color="white", fontsize=9,
                    fontweight="bold", zorder=7)


def render_map(frame, plon, plat, xsig_map, maxima, lambda_sun, n_orb,
               title, subtitle, out_png, label_min_xsig, candidates=None):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection="hammer")
    gw.plot_hammer(ax, plon, plat, xsig_map, vmin=1.0, vmax=150.0)
    overlay_maxima(ax, frame, maxima, lambda_sun, label_min_xsig)
    if candidates:
        overlay_candidates(ax, frame, candidates, lambda_sun)

    # degree graticule labels (longitude along equator, latitude on meridian)
    lon_deg = np.arange(-150, 151, 30)
    if frame == "sce":
        lon_lab = [f"{int((270 - d) % 360)}" for d in lon_deg]
    else:
        lon_lab = [f"{int((-d) % 360)}" for d in lon_deg]
    ax.set_xticks(np.radians(lon_deg))
    ax.set_xticklabels(lon_lab, color="0.65", fontsize=7)
    lat_deg = np.arange(-60, 61, 30)
    ax.set_yticks(np.radians(lat_deg))
    ax.set_yticklabels([f"{int(d):+d}" for d in lat_deg], color="0.65", fontsize=7)
    ax.grid(False)

    fig.text(0.02, 0.95, title, color="white", fontsize=12, family="monospace")
    fig.text(0.02, 0.915, f"NOrb = {n_orb}", color="white", fontsize=11,
             family="monospace")
    fig.text(0.5, 0.04, subtitle, color="white", fontsize=11, ha="center")

    sm = plt.cm.ScalarMappable(
        cmap=gw.make_gmn_colormap(),
        norm=__import__("matplotlib").colors.LogNorm(vmin=1.0, vmax=150.0))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("xsig value", color="0.8")
    cb.ax.yaxis.set_tick_params(color="0.6")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="0.8")

    fig.patch.set_facecolor("black")
    fig.savefig(out_png, facecolor="black", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
#  Input handling
# --------------------------------------------------------------------------- #

def load_target(args):
    """Return (radiants_dict, lambda_sun, sol_lo, sol_hi, tag, date_str)."""
    if args.date:
        if args.solrange:
            lo, hi = args.solrange
            url = gw.traj_summary_url(args.date, lo, hi)
            sol_lo, sol_hi = lo % 360.0, hi % 360.0
        else:
            print(f"Looking up daily file(s) for {args.date} in the GMN index ...")
            try:
                cands = gw.find_daily_file(args.date)
            except Exception as exc:                   # noqa: BLE001
                raise SystemExit(
                    f"Could not read the GMN directory index ({exc}).\n"
                    f"Pass --solrange LO HI explicitly, or download the file "
                    f"and use --file.")
            if not cands:
                raise SystemExit(f"No daily file found for {args.date}.")
            url = cands[0]
            m = re.search(r"solrange_([0-9.]+)-([0-9.]+)\.txt", url)
            sol_lo, sol_hi = float(m.group(1)), float(m.group(2))
            if len(cands) > 1:
                print(f"  {len(cands)} files for this date; using "
                      f"{url.rsplit('/', 1)[1]} (override with --solrange)")
        cache = os.path.join(args.out, url.rsplit("/", 1)[1])
        print(f"Fetching {url}")
        try:
            text = gw.fetch_traj_summary(url, cache_path=cache)
        except Exception as exc:                       # noqa: BLE001
            raise SystemExit(
                f"Could not fetch the file ({exc}).\n"
                f"If you are offline or behind a firewall, download it manually "
                f"and pass it with --file.")
        rad = gw.parse_traj_summary(text, is_text=True)
        date_str = args.date
        tag = f"{args.date}-sol{sol_lo:.0f}-{sol_hi:.0f}"
    else:
        files = []
        for f in args.file:
            files.extend(sorted(glob.glob(f)))
        if not files:
            raise SystemExit("no input files matched --file")
        print(f"Reading {len(files)} local file(s)")
        rad = gw.parse_traj_summary(files)
        base = os.path.basename(files[0])
        date_str = base.split("_")[2] if base.startswith("traj_summary_") else ""
        sol_lo = float(np.nanmin(rad["sol"]))
        sol_hi = float(np.nanmax(rad["sol"]))
        tag = os.path.splitext(base)[0].replace("traj_summary_", "") \
            if base else "local"

    # wrap-aware window midpoint (handles the 359->000 boundary)
    hi_eff = sol_hi if sol_hi >= sol_lo else sol_hi + 360.0
    lambda_sun = args.sollon if args.sollon is not None else \
        (0.5 * (sol_lo + hi_eff)) % 360.0
    return rad, lambda_sun, sol_lo, sol_hi, tag, date_str


def build_annual_bg(args, quality=None):
    """Build the per-solar-longitude radiant list for the annual background
    (the GMN morhist): one radiant set per integer degree of solar longitude,
    each converted to sun-centred ecliptic with that epoch's solar longitude so
    the (stationary) sporadic sources line up across the year.

    ``quality`` : optional (min_qc, max_fiterr, min_stations) applied to the
    background radiants too, so it is filtered consistently with the target.
    """
    if args.background != "annual":
        return None
    sources = []
    if args.fetch_year:
        url = (f"{gw.GMN_DAILY_BASE.rsplit('/', 1)[0]}/"
               f"traj_summary_yearly_{args.fetch_year}.txt")
        cache = os.path.join(args.out, f"traj_summary_yearly_{args.fetch_year}.txt")
        print(f"Fetching yearly morhist {url} (large; this is the background)")
        try:
            text = gw.fetch_traj_summary(url, cache_path=cache)
        except Exception as exc:                       # noqa: BLE001
            raise SystemExit(f"Could not fetch the yearly file ({exc}). "
                             f"Download it and pass it with --bg-file.")
        sources.append((text, True))
    for f in (args.bg_file or []):
        for p in sorted(glob.glob(f)):
            sources.append((p, False))
    if not sources:
        raise SystemExit("--background annual requires --bg-file or --fetch-year")
    print("Parsing morhist for the annual background ...")
    parts = [gw.parse_traj_summary(s, is_text=t) for s, t in sources]
    rad = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n_read = rad["ra"].size
    if quality is not None:
        keep = gw.quality_mask(rad, *quality)
        rad = {k: rad[k][keep] for k in rad}
        print(f"  quality cut: {rad['ra'].size} of {n_read} background radiants kept")
    sol_bin = np.floor(rad["sol"]).astype(int) % 360
    bg = []
    for b in np.unique(sol_bin):
        m = sol_bin == b
        lam_sun_b = float(np.mean(rad["sol"][m]))        # this epoch's lambda_sun
        ll0, bet = gw.radec_to_sce(rad["ra"][m], rad["dec"][m], lam_sun_b)
        xyz = gw.sphere_to_unit(ll0, bet)                # SCE-native, per epoch
        bg.append((xyz, rad["vg"][m]))
    print(f"  {rad['ra'].size} radiants over {len(bg)} solar-longitude samples")
    return bg


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="GMN 3-D wavelet sky maps + maxlist from raw radiants.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--date", help="UTC date YYYYMMDD; fetch from GMN "
                                    "(solrange auto-discovered if not given)")
    src.add_argument("--file", nargs="+",
                     help="local trajectory-summary file(s) / glob(s)")
    ap.add_argument("--solrange", nargs=2, type=float, metavar=("LO", "HI"),
                    help="solar-longitude range for --date, e.g. 358.0 359.0; "
                         "if omitted it is discovered from the GMN index")
    ap.add_argument("--sollon", type=float, default=None,
                    help="solar longitude of the map [deg] "
                         "(default: midpoint of the window)")
    ap.add_argument("--solwidth", type=float, default=0.5,
                    help="half-width of the radiant window [deg] (default 0.5)")
    ap.add_argument("--background", choices=["self", "annual"], default="self",
                    help="'annual' = exact GMN yearly morhist background "
                         "(needs --bg-file or --fetch-year); 'self' = single-file "
                         "approximation (default, does NOT match the GMN product)")
    ap.add_argument("--bg-file", nargs="+",
                    help="morhist / yearly file(s) or glob(s) for the annual "
                         "background, e.g. traj_summary_yearly_2025.txt")
    ap.add_argument("--fetch-year", type=int, metavar="YYYY",
                    help="download traj_summary_yearly_YYYY.txt from GMN and use "
                         "it as the annual background (implies --background annual)")
    ap.add_argument("--step", type=float, default=0.2,
                    help="sky-grid resolution [deg] (GMN uses 0.2; coarsen for "
                         "speed)")
    ap.add_argument("--bg-step", type=float, default=None,
                    help="resolution [deg] for the annual background (default: "
                         "= --step). If coarser, the background is computed on "
                         "that grid and bilinearly interpolated up to --step.")
    ap.add_argument("--bg-cache", default=None, metavar="PATH",
                    help="cache the annual background (median/sigma) in this .npz "
                         "and reuse it on later runs (it is date-independent)")
    ap.add_argument("--quality", action="store_true",
                    help="apply trajectory quality cuts before the wavelet "
                         "(tuned to approach GMN's radiant count)")
    ap.add_argument("--min-qc", type=float, default=15.0,
                    help="min convergence angle Qc [deg] for --quality")
    ap.add_argument("--max-fiterr", type=float, default=110.0,
                    help="max MedianFitErr [arcsec] for --quality "
                         "(default tuned so the 358-359 window ~= GMN's NOrb)")
    ap.add_argument("--min-stations", type=int, default=2,
                    help="min participating stations for --quality")
    ap.add_argument("--rms-catalog", nargs="+", default=None,
                    help="RMS shower-catalogue CSV(s) for shower association "
                         "(default: rms_catalog/*.csv next to this script)")
    ap.add_argument("--ang-thr", type=float, default=3.0,
                    help="shower-association radiant threshold [deg] (GMN: 3.0)")
    ap.add_argument("--vg-thr", type=float, default=10.0,
                    help="shower-association velocity threshold [%%] (GMN: 10)")
    ap.add_argument("--mark-unknown", action="store_true",
                    help="flag significant peaks with no catalogue match as shower "
                         "candidates: tag them UNK in the maxlist and circle the "
                         "strongest ones (white '?') on the maps")
    ap.add_argument("--candidate-min-xsig", type=float, default=8.0,
                    help="min xsig for an unmatched peak to be drawn as a "
                         "candidate on the maps (with --mark-unknown)")
    ap.add_argument("--max-candidates", type=int, default=8,
                    help="max unmatched-peak candidates to draw on the maps")
    ap.add_argument("--a-deg", type=float, default=1.0, help="spatial probe [deg]")
    ap.add_argument("--vel-frac", type=float, default=0.05, help="velocity probe")
    ap.add_argument("--vel-step-pct", type=float, default=2.0,
                    help="velocity-bin multiplicative step %% (GMN uses 2)")
    ap.add_argument("--vmin-kms", type=float, default=9.0)
    ap.add_argument("--vmax-kms", type=float, default=72.0)
    ap.add_argument("--sigma-floor", type=float, default=0.5,
                    help="min background sigma for 'self' mode")
    ap.add_argument("--shot-frac", type=float, default=0.20,
                    help="annual xsig: floor the yearly sigma at this fraction of "
                         "the daily shot noise sqrt(sum h^2) (~1/N, the dropped "
                         "wavelet normalisation; calibrated to GMN's wc_s)")
    ap.add_argument("--min-xsig", type=float, default=3.0,
                    help="sigma threshold above background median (GMN: 3.0)")
    ap.add_argument("--min-radiants", type=int, default=10,
                    help="min radiants in a maximum's probe (GMN: 10)")
    ap.add_argument("--label-min-xsig", type=float, default=8.0,
                    help="min xsig to label a named shower on the maps")
    ap.add_argument("--max-labels", type=int, default=6,
                    help="max number of shower labels to draw (strongest first), "
                         "matching GMN's selective labelling")
    ap.add_argument("--out", default=".", help="output directory")
    args = ap.parse_args(argv)

    if args.fetch_year:
        args.background = "annual"
    os.makedirs(args.out, exist_ok=True)

    cfg = gw.KernelConfig(a_deg=args.a_deg, vel_frac=args.vel_frac, n_probe=4.0)
    # multiplicative velocity grid (GMN: 9..72 km/s in 2% steps)
    ratio = 1.0 + args.vel_step_pct / 100.0
    nvel = int(np.floor(np.log(args.vmax_kms / args.vmin_kms) / np.log(ratio))) + 1
    vgrid = args.vmin_kms * ratio ** np.arange(nvel)
    vgrid = vgrid[vgrid <= args.vmax_kms + 1e-9]

    quality = (args.min_qc, args.max_fiterr, args.min_stations) \
        if args.quality else None

    # shower-association catalogue (RMS); default to rms_catalog/*.csv beside us
    cat_paths = args.rms_catalog
    if cat_paths is None:
        here = os.path.dirname(os.path.abspath(__file__))
        cdir = os.path.join(here, "rms_catalog")
        cat_paths = (sorted(glob.glob(os.path.join(cdir, "*.csv"))) +
                     sorted(glob.glob(os.path.join(cdir, "mdc_*.txt"))))
    catalog = gw.load_shower_catalog(*cat_paths) if cat_paths else []
    if catalog:
        print(f"Loaded {len(catalog)} showers from the RMS catalogue.")
    else:
        print("No RMS catalogue found; using modal IAU code as a proxy.")

    # --- load radiants & select the solar-longitude window -----------------
    rad, lambda_sun, sol_lo, sol_hi, tag, date_str = load_target(args)
    if rad["ra"].size == 0:
        raise SystemExit("no radiants parsed from the input")
    sel = np.abs(((rad["sol"] - lambda_sun + 180) % 360) - 180) <= args.solwidth
    tgt = {k: rad[k][sel] for k in rad}
    n_window = int(tgt["ra"].size)
    if quality is not None:
        keep = gw.quality_mask(tgt, *quality)
        tgt = {k: tgt[k][keep] for k in tgt}
    n_orb = int(tgt["ra"].size)
    qmsg = (f" -> {n_orb} after quality cut" if quality is not None else "")
    print(f"lambda_sun = {lambda_sun:.2f} deg ; window +/-{args.solwidth} deg ; "
          f"{n_window} radiants (of {rad['ra'].size} read){qmsg}")
    if n_orb < 10:
        print("WARNING: very few radiants in the window; map will be sparse.")

    if args.background == "self":
        print("\n*** 'self' single-file background does NOT reproduce the GMN\n"
              "    product (see image differences). For the published maps use the\n"
              "    yearly morhist: --background annual --fetch-year YYYY  (or\n"
              "    --bg-file traj_summary_yearly_YYYY.txt).\n")

    bg_meta = None
    bg_slon = None
    if args.background == "annual":
        bg_meta = dict(v=3, year=args.fetch_year, bg_file=args.bg_file,
                       step=(args.bg_step or args.step), a=args.a_deg,
                       vel_frac=args.vel_frac, nvel=len(vgrid),
                       quality=quality)
        # only parse the (large) morhist if the cache cannot serve this run
        cache_ok = False
        if args.bg_cache and os.path.exists(args.bg_cache):
            d = np.load(args.bg_cache, allow_pickle=True)
            cache_ok = str(d.get("meta")) == str(bg_meta)
            if cache_ok:
                print(f"Annual background cache {args.bg_cache} matches; "
                      f"skipping morhist parse.")
        if not cache_ok:
            bg_slon = build_annual_bg(args, quality=quality)

    # --- evaluate the field in SCE; reproject to equatorial ----------------
    print(f"Evaluating sun-centred ecliptic field ({args.background} background) ...")
    sce = evaluate_sce_field(tgt, bg_slon, lambda_sun, vgrid, cfg, args.step,
                             args.background, args.sigma_floor,
                             bg_step=args.bg_step, bg_cache=args.bg_cache,
                             bg_meta=bg_meta, shot_frac=args.shot_frac)
    print("Reprojecting to equatorial ...")
    pe_lon, pe_lat, xe = reproject_to_equ(sce, lambda_sun, args.step)
    ps_lon, ps_lat = sce["plon"], sce["plat"]
    xs, wcs, vgs = sce["xsig"], sce["wc"], sce["vg"]

    # --- detect maxima on the SCE map + attach orbits ----------------------
    plon_deg = sce["plon_deg"]
    bet_axis = sce["plat_deg"]
    maxima = gw.detect_maxima(plon_deg, bet_axis, xs, wc_map=wcs, vg_map=vgs,
                              min_xsig=args.min_xsig, footprint=7,
                              min_separation_deg=3.0)
    for m in maxima:
        m["ll0"] = (270.0 - m.pop("lon")) % 360.0
        m["bet"] = m.pop("lat")
    maxima.sort(key=lambda d: -d["xsig"])
    maxima = gw.aggregate_maxima(maxima, tgt, lambda_sun, cfg, catalog=catalog,
                                 ang_thr=args.ang_thr, vg_thr=args.vg_thr)
    n_before = len(maxima)
    maxima = [m for m in maxima if m.get("r_cnt", 0) >= args.min_radiants]
    # wc_s = yearly sigma of wc at each maximum (sampled from the annual
    # background); in 'self' mode fall back to GMN's analytic sigma from r_cnt.
    for m in maxima:
        if sce["sig"] is not None:
            pl = wrap180(-(m["ll0"] - 270.0))
            ilon = int(np.argmin(np.abs(plon_deg - pl)))
            ilat = int(np.argmin(np.abs(bet_axis - m["bet"])))
            m["wc_s"] = float(sce["sig"][ilat, ilon])
        else:
            m["wc_s"] = float(gw.analytic_wc_sigma(m.get("r_cnt", 0),
                                                   m["vg"], cfg))
    print(f"Maxima: {n_before} detected, {len(maxima)} pass the "
          f">= {args.min_radiants}-radiant filter.")

    # error #3 fix: label each named shower once (strongest), like the GMN maps
    best = {}
    for m in maxima:
        c = m.get("code", "")
        if not c or m["xsig"] < args.label_min_xsig:
            continue
        if c not in best or m["xsig"] > best[c]["xsig"]:
            best[c] = m
    labels = sorted(best.values(), key=lambda d: -d["xsig"])[:args.max_labels]
    if labels:
        print("Labeled showers: " + ", ".join(f"{m['code']}({m['xsig']:.0f})"
                                               for m in labels))

    # unmatched significant peaks = shower candidates (no catalogue association)
    candidates = None
    if args.mark_unknown:
        unk = [m for m in maxima if not m.get("code")
               and m["xsig"] >= args.candidate_min_xsig]
        candidates = sorted(unk, key=lambda d: -d["xsig"])[:args.max_candidates]
        n_unmatched = sum(1 for m in maxima if not m.get("code"))
        print(f"Unmatched peaks (shower candidates): {n_unmatched} total"
              f"; {len(candidates)} drawn (xsig >= {args.candidate_min_xsig}).")

    out_equ = os.path.join(args.out, f"wav-{tag}-equ.png")
    out_sce = os.path.join(args.out, f"wav-{tag}-sce.png")
    out_max = os.path.join(args.out, f"maxlist-{tag}.txt")

    render_map("equatorial", pe_lon, pe_lat, xe, labels, lambda_sun, n_orb,
               f"GMN Wavelet {tag}", "Equatorial", out_equ, args.label_min_xsig,
               candidates=candidates)
    render_map("sce", ps_lon, ps_lat, xs, labels, lambda_sun, n_orb,
               f"GMN Wavelet {tag}", "Sun-Centred Ecliptic", out_sce,
               args.label_min_xsig, candidates=candidates)

    bg_n = sum(x[0].shape[0] for x in bg_slon) if bg_slon else None
    gw.write_maxlist_full(out_max, maxima, cfg, lambda_sun, n_orb,
                          date=date_str, sol_lo=sol_lo, sol_hi=sol_hi,
                          background=args.background, bg_nradiants=bg_n,
                          min_radiants=args.min_radiants, n_sigma=args.min_xsig,
                          ll0_step=args.step, bet_step=args.step,
                          vg_lo=args.vmin_kms, vg_hi=args.vmax_kms,
                          vg_pct=args.vel_step_pct, half_width=args.solwidth,
                          mark_unknown=args.mark_unknown)

    # --- report ------------------------------------------------------------
    print(f"\nDetected {len(maxima)} maxima (xsig >= {args.min_xsig}). Strongest:")
    print(f"  {'ll0':>6} {'bet':>6} {'RA':>6} {'Dec':>6} {'Vg':>5} "
          f"{'xsig':>6} {'r_cnt':>5}  code")
    for m in maxima[:12]:
        ra, dec = gw.sce_to_radec(m["ll0"], m["bet"], lambda_sun)
        print(f"  {m['ll0']:6.1f} {m['bet']:6.1f} {float(ra):6.1f} "
              f"{float(dec):6.1f} {m['vg']:5.1f} {m['xsig']:6.1f} "
              f"{m.get('r_cnt',0):5d}  {m.get('code','')}")
    print(f"\nWrote:\n  {out_equ}\n  {out_sce}\n  {out_max}")
    return out_equ, out_sce, out_max


if __name__ == "__main__":
    main()
