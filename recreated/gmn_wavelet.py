"""
gmn_wavelet.py
==============

A faithful re-implementation of the 3D Mexican-hat wavelet meteor-shower search
of Brown, Wong, Weryk & Wiegert (2010), *Icarus* 207, 66-81 ("A meteoroid
stream survey using the Canadian Meteor Orbit Radar II"), adapted to the
single-solar-longitude "daily wavelet map" product of the Global Meteor Network
(GMN).

Brown et al. (2010), Eq. (1) defines the wavelet coefficient at a test point
(x0, y0, Vg0) for a radiant distribution f(x, y, Vg):

    Wc(x0,y0,Vg0) = N * INT f(x,y,Vg) *
                    [ 3 - ((x0-x)^2+(y0-y)^2)/a^2 - (Vg0-Vg)^2/sv^2 ] *
                    exp{ -0.5 * [ ((x0-x)^2+(y0-y)^2)/a^2 + (Vg0-Vg)^2/sv^2 ] }
                    dx dy dVg

with x = (lambda - lambda_sun)_g (sun-centred ecliptic longitude), y = beta_g
(ecliptic latitude), a the spatial probe size (deg) and sv the velocity probe.
The leading "3" is the dimensionality (2 spatial + 1 velocity) of the Mexican
hat.  For a discrete set of radiants the integral becomes a sum, and only
radiants within ~4 probe sizes contribute.

GMN parameters for the daily maps (read directly from the maxlist header):
    spatial probe a       = 1.0 deg
    velocity probe        = 5% of Vg0  (i.e. sv = 0.05 * Vg0, multiplicative)
    contributing radius   = 4 probe sizes in each dimension

The DISPLAYED quantity is ``xsig`` = (Wc - background_median) / background_sigma,
i.e. the number of standard deviations the wavelet coefficient sits above the
local background.  Two background estimators are provided (see ``compute_xsig``):

    'annual'    -- the exact Brown/GMN estimator: a per-cell robust median and
                   sigma obtained by iterative 3-sigma clipping of the wavelet
                   coefficient evaluated once per degree of solar longitude
                   through a full virtual year.  Requires the full multi-slon
                   radiant database (the "morhist" file).

    'reference' -- an analytic estimator (Campbell's theorem for a filtered
                   Poisson point process) that derives the per-cell background
                   mean and sigma from a high-statistics, stationary
                   sporadic-only reference radiant sample.  This reproduces the
                   key behaviour of the annual estimator -- significance is
                   suppressed in the busy sporadic sources and enhanced for
                   isolated peaks -- from a single epoch of data.

Because ``xsig`` is a ratio, it is invariant to the constant normalisation
prefactor N in Eq. (1); N is therefore irrelevant for reproducing the displayed
maps and is dropped.

The wavelet is a scalar field on the sphere; great-circle distance is
frame-independent, so the same machinery evaluates the field in equatorial
(RA, Dec) or sun-centred-ecliptic (ll0, beta) coordinates simply by supplying
the test grid and radiants in the chosen frame.

Author: re-implementation for S. Baressi Segota.
"""

from __future__ import annotations

import dataclasses
import os
import re
import urllib.request

import numpy as np

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #

OBLIQUITY_DEG = 23.43928        # mean obliquity of the ecliptic, J2000.0
OBLIQUITY = np.radians(OBLIQUITY_DEG)

# GMN's documented prefactor for the analytic single-wc standard deviation
# (footer of the maxlist: "Standard deviation of single wc is given by
#  0.04987 * sqrt(r_cnt) / angular probe size[rad] / sqrt(velocity probe[km/s])").
ANALYTIC_SIGMA_CONST = 0.04987


# --------------------------------------------------------------------------- #
#  Coordinate transforms  (all angles in degrees on the public API)
# --------------------------------------------------------------------------- #

def equatorial_to_ecliptic(ra_deg, dec_deg):
    """Equatorial (RA, Dec) -> ecliptic (lambda, beta), J2000.0.  Degrees."""
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    ce, se = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra) * ce + np.sin(dec) * se
    z = -np.cos(dec) * np.sin(ra) * se + np.sin(dec) * ce
    lam = np.degrees(np.arctan2(y, x)) % 360.0
    beta = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return lam, beta


def ecliptic_to_equatorial(lam_deg, beta_deg):
    """Ecliptic (lambda, beta) -> equatorial (RA, Dec), J2000.0.  Degrees."""
    lam = np.radians(np.asarray(lam_deg, float))
    beta = np.radians(np.asarray(beta_deg, float))
    ce, se = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    x = np.cos(beta) * np.cos(lam)
    y = np.cos(beta) * np.sin(lam) * ce - np.sin(beta) * se
    z = np.cos(beta) * np.sin(lam) * se + np.sin(beta) * ce
    ra = np.degrees(np.arctan2(y, x)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return ra, dec


def radec_to_sce(ra_deg, dec_deg, lambda_sun_deg):
    """Equatorial (RA, Dec) -> sun-centred ecliptic (ll0 = lambda - lambda_sun, beta)."""
    lam, beta = equatorial_to_ecliptic(ra_deg, dec_deg)
    ll0 = (lam - lambda_sun_deg) % 360.0
    return ll0, beta


def sce_to_radec(ll0_deg, beta_deg, lambda_sun_deg):
    """Sun-centred ecliptic (ll0, beta) -> equatorial (RA, Dec)."""
    lam = (np.asarray(ll0_deg, float) + lambda_sun_deg) % 360.0
    return ecliptic_to_equatorial(lam, beta_deg)


def sphere_to_unit(lon_deg, lat_deg):
    """Spherical (longitude, latitude in deg) -> (N, 3) unit Cartesian vectors."""
    lon = np.radians(np.asarray(lon_deg, float)).ravel()
    lat = np.radians(np.asarray(lat_deg, float)).ravel()
    cl = np.cos(lat)
    return np.column_stack((cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)))


# --------------------------------------------------------------------------- #
#  Wavelet kernel configuration
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class KernelConfig:
    """Parameters of the 3D Mexican-hat wavelet (Brown et al. 2010, Eq. 1)."""
    a_deg: float = 1.0          # spatial probe size (deg)            [GMN: 1.0]
    vel_frac: float = 0.05      # velocity probe = vel_frac * Vg0     [GMN: 0.05]
    n_probe: float = 4.0        # contributing radius, in probe sizes [paper: 4]

    @property
    def chord_cutoff(self) -> float:
        """Max chord length on the unit sphere for the spatial cutoff."""
        return 2.0 * np.sin(np.radians(self.n_probe * self.a_deg) / 2.0)


def analytic_wc_sigma(r_cnt, vg, cfg):
    """GMN's analytic standard deviation of a single wavelet coefficient.

    From the maxlist footer (see ``original/example.txt``):

        sigma_wc = 0.04987 * sqrt(r_cnt) / a[radians] / sqrt(sv[km/s])

    with the spatial probe ``a`` in radians and the velocity probe
    ``sv = vel_frac * vg`` in km/s.  This is the shot-noise level that the
    *empirical* yearly sigma (the ``wc_s`` column) closely tracks, and it lets us
    estimate ``wc_s`` (and a maximum's significance ``wc/sigma`` when the yearly
    median is ~0) without a full year of data.  Accepts scalars or arrays.
    """
    a_rad = np.radians(cfg.a_deg)
    sv = cfg.vel_frac * np.asarray(vg, float)
    r = np.asarray(r_cnt, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        sig = ANALYTIC_SIGMA_CONST * np.sqrt(r) / a_rad / np.sqrt(sv)
    return sig


# --------------------------------------------------------------------------- #
#  Core evaluator
# --------------------------------------------------------------------------- #

def _great_circle_deg(chord):
    """Chord length on unit sphere -> great-circle angle (deg)."""
    return np.degrees(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))


def _accumulate(test_xyz, rad_xyz, rad_v, vgrid, cfg, want_sq=False):
    """
    Evaluate, for every test point and every velocity bin, the sum over radiants
    of the 3D Mexican-hat kernel (and optionally the sum of its square).

    Returns
    -------
    sum_h  : (n_test, n_v) ndarray   -- sum_i h(test, radiant_i ; v)
    sum_h2 : (n_test, n_v) ndarray or None
    """
    from scipy.spatial import cKDTree

    n_test = test_xyz.shape[0]
    n_v = len(vgrid)
    a = cfg.a_deg

    sum_h = np.zeros((n_test, n_v), dtype=np.float64)
    sum_h2 = np.zeros((n_test, n_v), dtype=np.float64) if want_sq else None

    if rad_xyz.shape[0] == 0:
        return sum_h, sum_h2

    # Spatial neighbour search on the unit sphere (frame-independent).
    tree_test = cKDTree(test_xyz)
    tree_rad = cKDTree(rad_xyz)
    coo = tree_test.sparse_distance_matrix(
        tree_rad, max_distance=cfg.chord_cutoff, output_type="coo_matrix"
    )
    ti = coo.row                      # test-point index for each pair
    ri = coo.col                      # radiant index for each pair
    gc = _great_circle_deg(coo.data)  # great-circle separation (deg)

    if ti.size == 0:
        return sum_h, sum_h2

    u_sp = (gc / a) ** 2              # spatial part of u (fixed per pair)
    v_pair = rad_v[ri]               # radiant velocity for each pair
    spatial_ok = gc < (cfg.n_probe * a)

    for j, v0 in enumerate(vgrid):
        sv = cfg.vel_frac * v0                       # velocity probe (km/s)
        dv = v0 - v_pair
        mask = spatial_ok & (np.abs(dv) < cfg.n_probe * sv)
        if not np.any(mask):
            continue
        u = u_sp[mask] + (dv[mask] / sv) ** 2
        h = (3.0 - u) * np.exp(-0.5 * u)             # 3D Mexican hat
        idx = ti[mask]
        sum_h[:, j] = np.bincount(idx, weights=h, minlength=n_test)
        if want_sq:
            sum_h2[:, j] = np.bincount(idx, weights=h * h, minlength=n_test)

    return sum_h, sum_h2


def wavelet_coefficient(test_xyz, rad_xyz, rad_v, vgrid, cfg):
    """Wavelet coefficient Wc(test, v) for a radiant set.  (n_test, n_v) array."""
    sum_h, _ = _accumulate(test_xyz, rad_xyz, rad_v, vgrid, cfg, want_sq=False)
    return sum_h


def wavelet_and_shotnoise(test_xyz, rad_xyz, rad_v, vgrid, cfg):
    """Wavelet coefficient Wc and its per-cell Poisson shot-noise standard
    deviation sqrt(sum h^2).

    For a sum Wc = sum_i h_i over a Poisson radiant field, Campbell's theorem
    gives Var[Wc] = integral(density * h^2) ~ sum_i h_i^2, so sqrt(sum h^2) is the
    one-realisation shot-noise sigma of Wc in the *same units as Wc* (independent
    of the dropped normalisation prefactor N).  This is the floor GMN's empirical
    yearly sigma (the ``wc_s`` column) sits at; using it as a lower bound on the
    yearly sigma removes the small-number-statistics significance blow-ups that
    Brown et al. note in sparse (off-ecliptic, low-velocity) cells.
    """
    sum_h, sum_h2 = _accumulate(test_xyz, rad_xyz, rad_v, vgrid, cfg, want_sq=True)
    return sum_h, np.sqrt(sum_h2)


# --------------------------------------------------------------------------- #
#  Background estimators and xsig
# --------------------------------------------------------------------------- #

def annual_background(test_xyz, slon_radiants, vgrid, cfg, chunk=None,
                      n_clip=10, sigma_floor_frac=0.5, progress=False,
                      mem_budget_gb=0.4):
    """
    Exact Brown/GMN background: per-cell robust median and sigma of the wavelet
    coefficient evaluated once per solar-longitude sample through the year, with
    iterative 3-sigma clipping (discard points > median + 3 sigma until stable).
    See Brown et al. (2010), Section 2 (lines describing the "yearly median
    background"): the median is found by recursively discarding points more than
    3 sigma above the median until none remain.

    Parameters
    ----------
    test_xyz : (n_test, 3) unit vectors of the evaluation grid.
    slon_radiants : list of (rad_xyz, rad_v)
        One entry per solar-longitude sample (ideally ~360 for a full year).
    chunk : int
        Number of test points processed at once.  Memory ~ n_slon*chunk*n_v.
        For 360 slon, 26 velocities and chunk=4000 this is ~0.3 GB.

    Returns
    -------
    median : (n_test, n_v) ndarray
    sigma  : (n_test, n_v) ndarray
        The per-cell yearly median and standard deviation of Wc.  The sigma is
        floored at the *typical daily* Poisson shot-noise level
        sqrt(median_epochs(sum h^2)): with only one year of epochs the empirical
        scatter is noisy and collapses in sparse (off-ecliptic, low-velocity)
        cells, producing spurious significance spikes; the shot-noise floor (the
        level GMN's analytic ``wc_s`` tracks) is the principled lower bound and
        is in the same units as Wc, so it is independent of the dropped wavelet
        normalisation N.
    """
    n_test = test_xyz.shape[0]
    n_v = len(vgrid)
    n_s = len(slon_radiants)
    if chunk is None:
        # two (n_s, chunk, n_v) stacks (sum_h and sum_h^2) within the budget
        chunk = int(max(200, mem_budget_gb * 1e9 / (max(n_s, 1) * n_v * 8 * 2)))
    median = np.empty((n_test, n_v), dtype=np.float64)
    sigma = np.empty((n_test, n_v), dtype=np.float64)
    shot = np.empty((n_test, n_v), dtype=np.float64)   # typical daily shot sigma
    if progress:
        print(f"    annual background: {n_test} cells x {n_v} vel x {n_s} "
              f"epochs, chunk={chunk}", flush=True)

    for start in range(0, n_test, chunk):
        stop = min(start + chunk, n_test)
        tx = test_xyz[start:stop]
        stack = np.empty((n_s, stop - start, n_v), dtype=np.float64)
        stack2 = np.empty((n_s, stop - start, n_v), dtype=np.float64)
        for k, (rxyz, rv) in enumerate(slon_radiants):
            sh, sh2 = _accumulate(tx, rxyz, rv, vgrid, cfg, want_sq=True)
            stack[k] = sh
            stack2[k] = sh2

        med = np.median(stack, axis=0)
        sig = np.std(stack, axis=0)
        keep = np.ones_like(stack, dtype=bool)
        for _ in range(n_clip):
            hi = med + 3.0 * sig
            keep = stack <= hi
            clipped = np.where(keep, stack, np.nan)
            new_med = np.nanmedian(clipped, axis=0)
            new_sig = np.nanstd(clipped, axis=0)
            if (np.allclose(new_med, med, atol=1e-9, equal_nan=True) and
                    np.allclose(new_sig, sig, atol=1e-9, equal_nan=True)):
                med, sig = new_med, new_sig
                break
            med, sig = new_med, new_sig
        median[start:stop] = med
        sigma[start:stop] = sig
        # Background Poisson shot noise: Var[Wc] = E[sum h^2], estimated by the
        # MEAN of sum h^2 over the *non-outlier* (sporadic-only) epochs -- the
        # mean (not the median, which is ~0 for cells active only a few days a
        # year) correctly captures the day-to-day scatter at transient-shower
        # cells.  This is the level GMN's wc_s sits at and the floor that removes
        # the single-year small-number significance spikes.
        sh2_bg = np.where(keep, stack2, np.nan)
        shot[start:stop] = np.sqrt(np.nanmean(sh2_bg, axis=0))
        if progress:
            print(f"    annual background: {stop}/{n_test} cells", flush=True)

    # sigma >= background shot noise (the dominant, principled floor), with a
    # small global backstop for cells that are essentially empty all year.
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    shot = np.where(np.isfinite(shot), shot, 0.0)
    sigma = np.maximum(sigma, shot)
    pos = shot[shot > 0]
    backstop = sigma_floor_frac * float(np.median(pos)) if pos.size else 1e-6
    sigma = np.maximum(sigma, backstop)
    return median, sigma


def reference_background(test_xyz, ref_xyz, ref_v, n_target, vgrid, cfg,
                         var_floor=0.4):
    """
    Analytic background from a high-statistics sporadic reference sample, via
    Campbell's theorem for a filtered Poisson point process.

    If radiants are a Poisson sample of an intensity proportional to the
    reference density, then for a target sample of ``n_target`` radiants drawn
    from the same spatial/velocity distribution:

        E[Wc_bg]   = (n_target / n_ref) * sum_ref h
        Var[Wc_bg] = (n_target / n_ref) * sum_ref h^2     (peaked-kernel approx.)

    A small additive ``var_floor`` (an irreducible isotropic background-noise
    variance) regularises cells where the reference sample is locally sparse --
    e.g. the converging meridians at the poles or the anti-apex direction.  It
    plays the role of Brown et al.'s minimum-radiant requirement, preventing the
    small-number-statistics spikes they note are otherwise common in the
    anti-apex region, and keeps the displayed significance on a physical scale.

    Returns
    -------
    mean  : (n_test, n_v) ndarray
    sigma : (n_test, n_v) ndarray
    """
    n_ref = ref_xyz.shape[0]
    scale = n_target / float(n_ref)
    sum_h, sum_h2 = _accumulate(test_xyz, ref_xyz, ref_v, vgrid, cfg, want_sq=True)
    mean = scale * sum_h
    var = scale * sum_h2 + var_floor
    sigma = np.sqrt(np.maximum(var, 1e-12))
    return mean, sigma


def compute_xsig(wc_grid, bg_mean, bg_sigma):
    """
    xsig per (test, velocity), then collapse to the maximum over velocity.

    Returns
    -------
    xsig_max : (n_test,) ndarray   -- displayed significance
    v_index  : (n_test,) ndarray   -- velocity-bin index of the maximum
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        xsig = (wc_grid - bg_mean) / bg_sigma
    xsig = np.where(np.isfinite(xsig), xsig, -np.inf)
    v_index = np.argmax(xsig, axis=1)
    xsig_max = xsig[np.arange(xsig.shape[0]), v_index]
    xsig_max = np.where(np.isfinite(xsig_max), xsig_max, 0.0)
    return xsig_max, v_index


def spatial_significance(wc_grid, n_clip=10, sigma_floor=0.5):
    """
    Single-epoch ("self") background: estimate significance from one map's own
    wavelet-coefficient field, with NO multi-year data.

    Rationale -- the sporadic complex is present every night, so the typical
    (sporadic) wavelet level at a given velocity is well approximated by the
    robust spatial median of Wc over the sky at that velocity; transient showers
    rise above it.  For each velocity column we take the iterative 3-sigma-clipped
    median and standard deviation over all sky cells, then

        xsig(cell, v) = (Wc - median_v) / sigma_v ,   xsig_max = max over v.

    ``sigma_floor`` is a minimum noise level that prevents blow-ups in nearly
    empty velocity bins (where the clipped sigma would otherwise collapse) -- the
    single-epoch analogue of Brown et al.'s minimum-radiant requirement.

    NOTE: this approximates, but does not equal, the official GMN ``xsig`` which
    is referenced to the *yearly* per-cell median (see ``annual_background``).
    Persistent sporadic sources are suppressed less strongly here than in the
    yearly-normalised product.

    Returns
    -------
    xsig_max : (n_test,) ndarray
    v_index  : (n_test,) ndarray
    median   : (n_v,) ndarray
    sigma    : (n_v,) ndarray
    """
    wc = np.asarray(wc_grid, float)
    med = np.median(wc, axis=0)
    sig = np.std(wc, axis=0)
    for _ in range(n_clip):
        hi = med + 3.0 * sig
        lo = med - 3.0 * sig
        clipped = np.where((wc > hi) | (wc < lo), np.nan, wc)
        new_med = np.nanmedian(clipped, axis=0)
        new_sig = np.nanstd(clipped, axis=0)
        if (np.allclose(new_med, med, atol=1e-9) and
                np.allclose(new_sig, sig, atol=1e-9)):
            med, sig = new_med, new_sig
            break
        med, sig = new_med, new_sig

    sig = np.maximum(sig, sigma_floor)
    with np.errstate(invalid="ignore", divide="ignore"):
        xsig = (wc - med[None, :]) / sig[None, :]
    xsig = np.where(np.isfinite(xsig), xsig, -np.inf)
    v_index = np.argmax(xsig, axis=1)
    xsig_max = xsig[np.arange(xsig.shape[0]), v_index]
    xsig_max = np.where(np.isfinite(xsig_max), xsig_max, 0.0)
    return xsig_max, v_index, med, sig


def xsig_field(target_xyz, target_v, ref_xyz, ref_v, test_xyz, vgrid, cfg,
               var_floor=0.4):
    """
    Convenience driver: wavelet coefficient of the target, background from the
    reference sample, and the resulting max-over-velocity xsig.

    Returns
    -------
    dict with keys: xsig, wc_max, vg_at_max, wc_grid
    """
    wc_grid = wavelet_coefficient(test_xyz, target_xyz, target_v, vgrid, cfg)
    bg_mean, bg_sigma = reference_background(
        test_xyz, ref_xyz, ref_v, n_target=target_xyz.shape[0], vgrid=vgrid, cfg=cfg,
        var_floor=var_floor,
    )
    xsig, vidx = compute_xsig(wc_grid, bg_mean, bg_sigma)
    wc_max = wc_grid[np.arange(wc_grid.shape[0]), vidx]
    return {
        "xsig": xsig,
        "wc_max": wc_max,
        "vg_at_max": np.asarray(vgrid)[vidx],
        "wc_grid": wc_grid,
    }


# --------------------------------------------------------------------------- #
#  Maxima detection (Brown et al. 2010, Section 3)
# --------------------------------------------------------------------------- #

def detect_maxima(lon_grid, lat_grid, xsig_map, wc_map=None, vg_map=None,
                  min_xsig=3.0, footprint=5, min_separation_deg=2.0):
    """
    Locate local maxima of a 2-D xsig map (regular grid in lon/lat degrees).

    A point is a maximum if it equals the maximum within a square footprint and
    exceeds ``min_xsig``.  Near-duplicate maxima within ``min_separation_deg``
    are merged (strongest kept).  Returns a list of dicts sorted by descending
    xsig.
    """
    from scipy.ndimage import maximum_filter

    local_max = (xsig_map == maximum_filter(xsig_map, size=footprint))
    peaks = local_max & (xsig_map >= min_xsig)
    iy, ix = np.where(peaks)

    cand = []
    for y, x in zip(iy, ix):
        cand.append({
            "lon": float(lon_grid[x]),
            "lat": float(lat_grid[y]),
            "xsig": float(xsig_map[y, x]),
            "wc": float(wc_map[y, x]) if wc_map is not None else np.nan,
            "vg": float(vg_map[y, x]) if vg_map is not None else np.nan,
        })
    cand.sort(key=lambda d: -d["xsig"])

    kept = []
    for c in cand:
        clon = np.radians(c["lon"]); clat = np.radians(c["lat"])
        dup = False
        for k in kept:
            klon = np.radians(k["lon"]); klat = np.radians(k["lat"])
            cosd = (np.sin(clat) * np.sin(klat) +
                    np.cos(clat) * np.cos(klat) * np.cos(clon - klon))
            if np.degrees(np.arccos(np.clip(cosd, -1, 1))) < min_separation_deg:
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


# --------------------------------------------------------------------------- #
#  Maxlist I/O  (the GMN "search" text format, as in example.txt)
# --------------------------------------------------------------------------- #

def parse_maxlist(path):
    """
    Parse a GMN maxlist file.  Returns (header_dict, list_of_rows).
    Each row is a dict with the numeric columns plus 'code' and 'name'.
    """
    header = {}
    rows = []
    with open(path) as fh:
        lines = [ln.rstrip("\n").rstrip("\r") for ln in fh]

    for i, ln in enumerate(lines):
        if "Radiants in slon window" in ln and i + 1 < len(lines):
            p = lines[i + 1].split()
            if len(p) >= 3:
                header["start"] = p[0]
                header["end"] = p[1]
                header["n_radiants"] = int(p[2])
        if "spatial probe size" in ln and i + 1 < len(lines):
            p = lines[i + 1].split()
            if len(p) >= 3:
                header["slon_halfwidth"] = float(p[0])
                header["a_deg"] = float(p[1])
                header["vel_frac_pct"] = float(p[2])

    cols = ["slon", "ll0", "bet", "ra", "dec", "vg", "wc", "xsig",
            "wc_s", "r_cnt", "vinf", "a", "e", "incl", "omega",
            "ascnod", "q_per", "q_aph", "tj", "eta"]
    for ln in lines:
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        p = ln.split()
        try:
            float(p[0])
        except (ValueError, IndexError):
            continue
        if len(p) < 20:
            continue
        rec = {}
        for k, name in enumerate(cols):
            try:
                rec[name] = float(p[k])
            except ValueError:
                rec[name] = np.nan
        rec["r_cnt"] = int(rec["r_cnt"])
        rec["code"] = p[20] if len(p) > 20 else ""
        rec["name"] = " ".join(p[21:]) if len(p) > 21 else ""
        rows.append(rec)
    return header, rows


_MAXLIST_HEADER = """\
#
# Readmaxlist ver gmn_wavelet.py reimplementation
#
# Start year/sl   End year/sl   Radiants in slon window
   {start:>12}  {end:>12}  {n:>16d}
#
# slon window-half width [deg]  spatial probe size [deg]   velocity probe size [%]
{hw:>16.1f} {a:>26.1f} {vf:>28.1f}
#
# Filters used
# Min # of radiants used to calculate wc   Number of sigma wc must be above median
{minr:>20d} {nsig:>43.1f}
#
#  Slon   ll_0   Beta     RA    Dec     Vg       wc    xsig  r_cnt  Shower association
"""


def write_maxlist(path, maxima, cfg, lambda_sun, n_radiants,
                  start="0000/000", end="0000/000", min_radiants=10, n_sigma=3.0):
    """Write detected maxima (in SCE: each must have ll0, bet, vg, xsig, wc, r_cnt)
    to a GMN-style maxlist text file."""
    with open(path, "w") as fh:
        fh.write(_MAXLIST_HEADER.format(
            start=start, end=end, n=n_radiants,
            hw=cfg.a_deg * 0 + 0.5, a=cfg.a_deg, vf=cfg.vel_frac * 100.0,
            minr=min_radiants, nsig=n_sigma))
        for m in maxima:
            ra, dec = sce_to_radec(m["ll0"], m["bet"], lambda_sun)
            fh.write(
                f"{lambda_sun:7.1f} {m['ll0']:6.1f} {m['bet']:6.1f} "
                f"{float(ra):6.1f} {float(dec):6.1f} {m['vg']:6.1f} "
                f"{m['wc']:8.1f} {m['xsig']:7.1f} {m.get('r_cnt', 0):5d}  "
                f"{m.get('code', '')}\n")


# --------------------------------------------------------------------------- #
#  GMN trajectory-summary input  (the RAW radiants -- the real pipeline input)
# --------------------------------------------------------------------------- #

GMN_DAILY_BASE = ("https://globalmeteornetwork.org/data/"
                  "traj_summary_data/daily")

# 0-based column indices in the semicolon-delimited GMN traj_summary files.
_TS_COL = {
    "iau":   4,    # IAU shower code (3-letter) or '...' for sporadic
    "sol":   5,    # solar longitude [deg]
    "ra":    7,    # geocentric radiant RA  [deg]
    "dec":   9,    # geocentric radiant Dec [deg]
    "lam":  11,    # geocentric ecliptic longitude [deg]
    "bet":  13,    # geocentric ecliptic latitude  [deg]
    "vg":   15,    # geocentric velocity [km/s]
    "a":    23,    # semi-major axis [AU]
    "e":    25,    # eccentricity
    "incl": 27,    # inclination [deg]
    "peri": 29,    # argument of perihelion [deg]
    "node": 31,    # ascending node [deg]
    "q":    37,    # perihelion distance [AU]
    "qaph": 43,    # aphelion distance [AU]
    "tj":   49,    # Tisserand parameter wrt Jupiter
    "vinf": 59,    # initial (pre-atmospheric) velocity [km/s]
    "qc":   80,    # Qc, convergence angle [deg]            (quality)
    "fiterr": 81,  # MedianFitErr [arcsec]                  (quality)
    "nstat": 84,   # number of participating stations       (quality)
}


def traj_summary_url(date, sol_lo, sol_hi, base=GMN_DAILY_BASE):
    """Build the URL of a GMN daily trajectory-summary file.

    ``date`` is 'YYYYMMDD' (str) or anything with ``strftime``; ``sol_lo`` and
    ``sol_hi`` are the solar-longitude range bounds (deg).

    The published file names zero-pad the bounds to three integer digits and one
    decimal, and wrap 360 -> 000 at the year boundary, e.g.
        traj_summary_20250318_solrange_358.0-359.0.txt
        traj_summary_20190330_solrange_009.0-010.0.txt   (leading zeros!)
        traj_summary_20190320_solrange_359.0-000.0.txt   (wrap)
    """
    if hasattr(date, "strftime"):
        date = date.strftime("%Y%m%d")
    lo = sol_lo % 360.0
    hi = sol_hi % 360.0
    return f"{base}/traj_summary_{date}_solrange_{lo:05.1f}-{hi:05.1f}.txt"


def find_daily_file(date, base=GMN_DAILY_BASE, sol_lo=None):
    """Discover the actual daily-file URL(s) for a date from the GMN directory
    index, so the caller does not have to know the exact solar-longitude suffix
    (which is GMN-assigned, zero-padded, and wraps at 360).

    ``date`` is 'YYYYMMDD' or a strftime-able object.  Returns a list of full
    URLs whose names match that date (usually one; occasionally two when a date
    straddles two integer solar-longitude bins).  If ``sol_lo`` is given, a file
    whose lower bound rounds to it is preferred.  Requires network access.
    """
    if hasattr(date, "strftime"):
        date = date.strftime("%Y%m%d")
    index_html = fetch_traj_summary(base.rstrip("/") + "/")
    pat = re.compile(
        rf"traj_summary_{date}_solrange_([0-9]+\.[0-9])-([0-9]+\.[0-9])\.txt")
    seen, urls = set(), []
    for lo_s, hi_s in pat.findall(index_html):
        name = f"traj_summary_{date}_solrange_{lo_s}-{hi_s}.txt"
        if name in seen:
            continue
        seen.add(name)
        urls.append((float(lo_s), f"{base.rstrip('/')}/{name}"))
    if not urls:
        return []
    if sol_lo is not None:
        urls.sort(key=lambda t: abs(t[0] - (sol_lo % 360.0)))
    else:
        urls.sort(key=lambda t: t[0])
    return [u for _, u in urls]


def fetch_traj_summary(url, cache_path=None, timeout=120, force=False):
    """Download a trajectory-summary file and return its text.

    If ``cache_path`` is given the file is cached on disk and re-used on
    subsequent calls (unless ``force``).  Network access to
    globalmeteornetwork.org is required; in restricted sandboxes this will fail
    and the caller should fall back to a local ``--file``.
    """
    if cache_path and os.path.exists(cache_path) and not force:
        with open(cache_path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    req = urllib.request.Request(url, headers={"User-Agent": "gmn-wavelet/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


def parse_traj_summary(source, is_text=False):
    """Parse one or more GMN trajectory-summary files (the raw radiants).

    ``source`` is a file path, an already-loaded text string (set
    ``is_text=True``), or a list mixing either.  Returns a dict of numpy arrays:

        ra, dec, vg, sol, lam, bet            -- geocentric radiant + velocity
        a, e, incl, peri, node, q, qaph, tj   -- orbital elements
        vinf                                  -- initial velocity
        iau                                   -- IAU 3-letter code (str array;
                                                 '' for sporadic)

    Comment lines (starting with '#') and malformed rows are skipped.
    """
    if isinstance(source, (list, tuple)):
        chunks = []
        for s in source:
            chunks.append(s if is_text else _read_text(s))
        text = "\n".join(chunks)
    else:
        text = source if is_text else _read_text(source)

    keys = list(_TS_COL.keys())
    # Quality columns are parsed leniently (NaN on failure) so a missing/odd
    # quality value never discards an otherwise-good radiant; the explicit
    # quality filter (see quality_mask) handles NaNs.
    soft_keys = {"qc", "fiterr", "nstat"}
    out = {k: [] for k in keys}
    maxidx = max(_TS_COL.values())
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = ln.split(";")
        if len(parts) <= maxidx:
            continue
        try:
            for k in keys:
                tok = parts[_TS_COL[k]].strip()
                if k == "iau":
                    out[k].append("" if (tok in ("", "...", "nan")) else tok)
                elif k in soft_keys:
                    try:
                        out[k].append(float(tok))
                    except ValueError:
                        out[k].append(np.nan)
                else:
                    out[k].append(float(tok))
        except (ValueError, IndexError):
            for k in keys:           # roll back this partial row
                if len(out[k]) > 0 and len(out[k]) > min(len(out[j]) for j in keys):
                    out[k].pop()
            continue

    res = {k: (np.array(out[k], dtype=object) if k == "iau"
               else np.asarray(out[k], dtype=float)) for k in keys}
    return res


def _read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def quality_mask(radiants, min_qc=None, max_fiterr=None, min_stations=None):
    """Boolean mask selecting radiants that pass the trajectory-quality cuts.

    GMN filters its solutions before the wavelet search, but the exact public
    thresholds are not published, so these knobs are tuned empirically (see the
    driver's ``--quality`` defaults) to reproduce the published radiant count.

        min_qc        : minimum convergence angle Qc [deg] (well-converged geometry)
        max_fiterr    : maximum MedianFitErr [arcsec]      (good astrometric fit)
        min_stations  : minimum number of participating stations

    NaN quality values fail a cut only when that cut is active.
    """
    n = radiants["ra"].size
    keep = np.ones(n, dtype=bool)
    if min_qc is not None:
        qc = radiants.get("qc")
        if qc is not None:
            keep &= np.isfinite(qc) & (qc >= min_qc)
    if max_fiterr is not None:
        fe = radiants.get("fiterr")
        if fe is not None:
            keep &= np.isfinite(fe) & (fe <= max_fiterr)
    if min_stations is not None:
        ns = radiants.get("nstat")
        if ns is not None:
            keep &= np.isfinite(ns) & (ns >= min_stations)
    return keep


# --------------------------------------------------------------------------- #
#  IAU shower catalogue (RMS) + radiant association
# --------------------------------------------------------------------------- #

def _num(tok, default=0.0):
    try:
        return float(tok)
    except (ValueError, TypeError):
        return default


def _parse_rms_shower_line(ln):
    """One RMS shower-CSV row (established_showers / flux_showers; pipe format:
    IAUNo|Code|Name|SolBeg|SolMax|SolEnd|RA|dRA|Dec|dDec|Vg|dVg|...)."""
    p = [c.strip() for c in ln.split("|")]
    if len(p) < 12:
        return None
    code = p[1]
    if not code or len(code) > 4 or not code.isalpha():
        return None
    return {"code": code, "name": p[2], "sol_max": _num(p[4], np.nan),
            "ra": _num(p[6], np.nan), "dra": _num(p[7]),
            "dec": _num(p[8], np.nan), "ddec": _num(p[9]),
            "vg": _num(p[10], np.nan), "dvg": _num(p[11]),
            "n": _num(p[12] if len(p) > 12 else "", 0.0)}


def _parse_mdc_shower_line(ln):
    """One IAU MDC ``streamfulldata`` row (quoted pipe format:
    LP|IAUNo|AdNo|Code|Name|activity|s|LaSun|Ra|De|dRa|dDe|Vg|...|N|...)."""
    p = [c.strip().strip('"').strip() for c in ln.split("|")]
    if len(p) < 13:
        return None
    code = p[3]
    if not code or len(code) > 4 or not code.isalpha():
        return None
    return {"code": code, "name": p[4], "sol_max": _num(p[7], np.nan),
            "ra": _num(p[8], np.nan), "dra": _num(p[10]),
            "dec": _num(p[9], np.nan), "ddec": _num(p[11]),
            "vg": _num(p[12], np.nan), "dvg": 0.0,
            "n": _num(p[19] if len(p) > 19 else "", 0.0)}


def load_shower_catalog(*paths):
    """Load shower catalogue(s) for radiant association.

    Handles both the RMS pipe-CSV format (``established_showers.csv`` /
    ``flux_showers.csv``) and the IAU MDC ``streamfulldata`` quoted-pipe format.
    Returns a list of dicts with keys ``code, name, sol_max, ra, dra, dec, ddec,
    vg, dvg`` (drift terms per degree of solar longitude).

    The MDC list has several solutions per code; the one with the most
    contributing orbits (``N``) is kept.  RMS entries (GMN-calibrated) override
    MDC entries for the same code, so well-determined radiants win.
    """
    rms, mdc = {}, {}
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                s = ln.lstrip()
                if s.startswith("#") or s.startswith(":") or "|" not in ln:
                    continue
                is_mdc = s.startswith('"')
                rec = (_parse_mdc_shower_line(ln) if is_mdc
                       else _parse_rms_shower_line(ln))
                if rec is None or not (np.isfinite(rec["ra"]) and
                                       np.isfinite(rec["dec"]) and
                                       np.isfinite(rec["vg"])):
                    continue
                rec["src"] = "mdc" if is_mdc else "rms"
                table = mdc if is_mdc else rms
                cur = table.get(rec["code"])
                if cur is None or rec["n"] >= cur["n"]:
                    table[rec["code"]] = rec       # keep best-observed solution
    merged = dict(mdc)
    merged.update(rms)                              # RMS (GMN) overrides MDC
    return list(merged.values())


def shower_radiant_at(rec, lambda_sun):
    """Drift a catalogue shower's reference radiant to solar longitude
    ``lambda_sun``: RA/Dec/Vg = value_at_max + drift * (lambda_sun - sol_max),
    using the shortest angular path in solar longitude.  Returns (ra, dec, vg)."""
    sol_max = rec["sol_max"]
    if not np.isfinite(sol_max):
        return rec["ra"], rec["dec"], rec["vg"]
    dlam = ((lambda_sun - sol_max + 180.0) % 360.0) - 180.0
    ra = (rec["ra"] + rec["dra"] * dlam) % 360.0
    dec = rec["dec"] + rec["ddec"] * dlam
    vg = rec["vg"] + rec["dvg"] * dlam
    return ra, dec, vg


def associate_shower(ra, dec, vg, lambda_sun, catalog,
                     ang_thr=3.0, vg_thr=10.0):
    """Associate a radiant (RA, Dec, Vg at solar longitude ``lambda_sun``) to a
    catalogue shower, requiring both the radiant within ``ang_thr`` deg
    (great-circle) and the velocity within ``vg_thr`` percent -- GMN's documented
    3.0 deg / 10 % test.  Among the candidates that pass, the one with the
    smallest *combined* normalised distance is returned (radiant and velocity
    weighted equally), with GMN-calibrated (RMS) entries preferred on near-ties.
    Returns the matching record or None.
    """
    if not catalog:
        return None
    p = sphere_to_unit([ra], [dec])[0]
    best, best_d = None, np.inf
    for rec in catalog:
        sra, sdec, svg = shower_radiant_at(rec, lambda_sun)
        if not np.isfinite(svg) or svg <= 0:
            continue
        dvg = abs(vg - svg)
        if dvg > 0.01 * vg_thr * svg:
            continue
        q = sphere_to_unit([sra], [sdec])[0]
        sep = np.degrees(np.arccos(np.clip(np.dot(p, q), -1.0, 1.0)))
        if sep > ang_thr:
            continue
        d = np.hypot(sep / ang_thr, dvg / (0.01 * vg_thr * svg))
        if rec.get("src") == "rms":
            d -= 1e-3                              # prefer GMN-calibrated radiants
        if d < best_d:
            best, best_d = rec, d
    return best


# --------------------------------------------------------------------------- #
#  Per-maximum orbit aggregation + full-format maxlist
# --------------------------------------------------------------------------- #

def aggregate_maxima(maxima, radiants, lambda_sun, cfg, catalog=None,
                     ang_thr=3.0, vg_thr=10.0):
    """For each detected maximum, gather the radiants inside its wavelet probe
    and attach the radiant count, mean orbital elements and the shower
    association (the GMN ``Shower association`` column).

    If a ``catalog`` (from :func:`load_shower_catalog`) is given, each maximum is
    associated to the nearest IAU MDC shower within ``ang_thr`` deg / ``vg_thr``
    percent (GMN's 3.0 deg / 10 % test) and the full name is attached; otherwise
    (or when no catalogue match is found) the modal IAU code of the radiants in
    the probe is used as a proxy.

    ``maxima``   : list of dicts each with at least 'll0', 'bet', 'vg'.
    ``radiants`` : dict from ``parse_traj_summary`` (the target-window radiants).
    Returns the same list with extra keys added in place.
    """
    rl0, rbet = radec_to_sce(radiants["ra"], radiants["dec"], lambda_sun)
    rxyz = sphere_to_unit(rl0, rbet)
    rvg = radiants["vg"]
    orb_keys = ["vinf", "a", "e", "incl", "peri", "node", "q", "qaph", "tj"]

    for m in maxima:
        mxyz = sphere_to_unit([m["ll0"]], [m["bet"]])[0]
        cosd = np.clip(rxyz @ mxyz, -1.0, 1.0)
        gc = np.degrees(np.arccos(cosd))
        sv = cfg.vel_frac * m["vg"]
        sel = (gc < cfg.n_probe * cfg.a_deg) & \
              (np.abs(rvg - m["vg"]) < cfg.n_probe * sv)
        m["r_cnt"] = int(np.count_nonzero(sel))
        for k in orb_keys:
            vals = radiants[k][sel]
            vals = vals[np.isfinite(vals)]
            m[k] = float(np.mean(vals)) if vals.size else np.nan

        m["code"], m["name"] = "", ""
        if catalog:
            # GMN associates only to the catalogue (3 deg / 10 %); unmatched
            # maxima are left blank, as in example.txt.
            ra, dec = sce_to_radec(m["ll0"], m["bet"], lambda_sun)
            rec = associate_shower(float(ra), float(dec), m["vg"], lambda_sun,
                                   catalog, ang_thr=ang_thr, vg_thr=vg_thr)
            if rec is not None:
                m["code"], m["name"] = rec["code"], rec["name"]
        else:
            # no catalogue available: fall back to the modal IAU code (proxy)
            codes = [c for c in radiants["iau"][sel] if c]
            if codes:
                uniq, counts = np.unique(np.array(codes), return_counts=True)
                m["code"] = str(uniq[int(np.argmax(counts))])
    return maxima


_FULL_HEADER = """\
#
# Wavelet maxlist generated by gmn_wavelet.py (Brown et al. 2010 3D wavelet)
# using {bg} background{bgn}
#
# Start year/sl   End year/sl   Radiants in slon window
{startsl:>14} {endsl:>13} {n:>22d}
#
# Shower association ang_thr[deg]  vg_thr[%]
{aang:>26.1f} {avg:>10.1f}
#
# slon min   slon max   slon step   ll_0 min   ll_0 max   ll_0 step   bet min   bet max   bet step   vg min   vg max   vg step [%]
{sol:>10.1f} {sol:>10.1f} {slstep:>10.1f} {llo:>10.1f} {lhi:>10.1f} {lstep:>10.1f} {blo:>9.1f} {bhi:>9.1f} {bstep:>9.1f} {vlo:>8.1f} {vhi:>8.1f} {vpct:>11.1f}
#
# slon window-half width [deg]   spatial probe size [deg]   velocity probe size [%]
{half:>14.1f} {a:>27.1f} {vf:>26.1f}
#
# Filters used
# Min # of radiants used to calculate wc   Number of sigma wc must be above median
{minr:>18d} {nsig:>43.1f}
#
#  Slon   ll_0   Beta     RA    Dec     Vg       wc    xsig   wc_s  r_cnt   vinf      a      e   incl  omega ascnod  q_per  q_aph     Tj    eta  Shower association
"""


# Column definitions, mirroring the footer of GMN's example.txt.  Note in
# particular that xsig is the per-cell yearly significance and wc_s is the
# per-cell yearly standard deviation of wc (NOT a self/single-epoch quantity).
_MAXLIST_FOOTER = """\
#
# Slon : Solar longitude [deg]
# ll_0 : Sun centred ecliptic longitude [deg]
# Beta : Ecliptic latitude [deg]
# RA : Right ascension [deg]
# Dec : Declination [deg]
# vg : Geocentric velocity [km/s]
# wc : Wavelet coefficient
# xsig : Number of std devs wc is above the median at ll_0, beta, and vg through the year
# wc_s : Standard deviation of the wavelet coefficient at ll_0, beta and vg through the year
# r_cnt : Number of radiants used to calculate the wavelet coefficient
#
# Standard deviation of single wc is given by:
# {const} * sqrt(r_cnt) / angular probe size[radians] / sqrt(velocity probe size[km/s])
# (spatial probe = {a:.1f} deg, velocity probe = {vf:.1f} %)
"""


def write_maxlist_full(path, maxima, cfg, lambda_sun, n_radiants, *,
                       date="", sol_lo=0.0, sol_hi=0.0, background="self",
                       min_radiants=10, n_sigma=3.0, bg_nradiants=None,
                       ll0_step=0.2, bet_step=0.2, vg_lo=9.0, vg_hi=72.0,
                       vg_pct=2.0, half_width=0.5, ang_thr=3.0, vg_thr=10.0,
                       start_sl="", end_sl=""):
    """Write detected maxima in the GMN maxlist format (mirrors the uploaded
    ``example.txt`` header and columns, including ``wc_s`` and ``eta``).
    ``eta`` is emitted as '-' as GMN does; ``wc_s`` is the single-epoch (self)
    significance when available, else equal to ``xsig``."""
    def g(m, k):
        v = m.get(k, np.nan)
        return v if (v is not None and np.isfinite(v)) else float("nan")

    bgn = f" of {bg_nradiants} radiants" if bg_nradiants else ""
    with open(path, "w") as fh:
        fh.write(_FULL_HEADER.format(
            bg=background, bgn=bgn,
            startsl=start_sl or "-", endsl=end_sl or "-", n=n_radiants,
            aang=ang_thr, avg=vg_thr, sol=lambda_sun, slstep=1.0,
            llo=0.0, lhi=359.9, lstep=ll0_step,
            blo=-89.9, bhi=89.9, bstep=bet_step,
            vlo=vg_lo, vhi=vg_hi, vpct=vg_pct,
            half=half_width, a=cfg.a_deg, vf=cfg.vel_frac * 100.0,
            minr=min_radiants, nsig=n_sigma))
        for m in maxima:
            ra, dec = sce_to_radec(m["ll0"], m["bet"], lambda_sun)
            wc_s = m.get("wc_s", m.get("xsig", np.nan))
            shower = m.get("code", "")
            if m.get("name"):
                shower = f"{shower} {m['name']}".strip()
            fh.write(
                f"{lambda_sun:7.1f} {m['ll0']:6.1f} {m['bet']:6.1f} "
                f"{float(ra):6.1f} {float(dec):6.1f} {m['vg']:6.1f} "
                f"{m['wc']:8.1f} {m['xsig']:7.1f} {wc_s:6.1f} "
                f"{m.get('r_cnt', 0):6d} "
                f"{g(m,'vinf'):6.2f} {g(m,'a'):6.2f} {g(m,'e'):6.2f} "
                f"{g(m,'incl'):6.2f} {g(m,'peri'):6.2f} {g(m,'node'):6.2f} "
                f"{g(m,'q'):6.2f} {g(m,'qaph'):6.2f} {g(m,'tj'):6.1f} "
                f"{'-':>6}  {shower}\n")
        fh.write(_MAXLIST_FOOTER.format(const=ANALYTIC_SIGMA_CONST,
                                        a=cfg.a_deg, vf=cfg.vel_frac * 100.0))



def make_gmn_colormap():
    """LinearSegmentedColormap approximating the GMN wavelet colour scale used
    with a logarithmic norm over xsig in [1, 150]:
    near-black -> navy -> blue -> cyan -> green -> yellow -> orange -> red ->
    pink -> white."""
    from matplotlib.colors import LinearSegmentedColormap
    stops = [
        (0.00, (0.00, 0.00, 0.12)),   # xsig ~ 1   near-black navy
        (0.10, (0.00, 0.00, 0.45)),   #            dark blue
        (0.22, (0.00, 0.05, 0.85)),   # xsig ~ 3   blue
        (0.36, (0.00, 0.45, 1.00)),   #            azure
        (0.48, (0.00, 0.90, 0.95)),   # xsig ~ 10  cyan
        (0.58, (0.20, 1.00, 0.40)),   #            green
        (0.68, (0.85, 1.00, 0.10)),   #            yellow-green
        (0.74, (1.00, 0.85, 0.00)),   # xsig ~ 40  yellow
        (0.82, (1.00, 0.45, 0.00)),   #            orange
        (0.89, (1.00, 0.05, 0.05)),   #            red
        (0.95, (1.00, 0.40, 0.80)),   # xsig ~ 110 pink
        (1.00, (1.00, 1.00, 1.00)),   # xsig = 150 white
    ]
    cmap = LinearSegmentedColormap.from_list(
        "gmn_wavelet", [(p, c) for p, c in stops])
    cmap.set_under((0.0, 0.0, 0.0))
    cmap.set_bad((0.0, 0.0, 0.0))
    return cmap


def plot_hammer(ax, plot_lon_grid, lat_grid, xsig_map, *,
                vmin=1.0, vmax=150.0, cmap=None, graticule_step=30):
    """
    Render an xsig map in a matplotlib Hammer projection with a logarithmic
    colour norm and a dashed graticule, styled like the GMN daily maps.

    ``plot_lon_grid`` must already be in *plot* longitude (radians, -pi..pi) and
    ``lat_grid`` in radians (-pi/2..pi/2).  ``xsig_map`` has shape
    (len(lat_grid), len(plot_lon_grid)).
    """
    from matplotlib.colors import LogNorm

    if cmap is None:
        cmap = make_gmn_colormap()

    ax.set_facecolor("black")
    LON, LAT = np.meshgrid(plot_lon_grid, lat_grid)
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=False)
    mesh = ax.pcolormesh(LON, LAT, np.clip(xsig_map, vmin * 0.5, vmax),
                         cmap=cmap, norm=norm, shading="gouraud", rasterized=True)

    # Dashed graticule.
    gl = np.radians(np.arange(-180, 181, graticule_step))
    gb = np.radians(np.arange(-90, 91, graticule_step))
    fine = np.radians(np.linspace(-90, 90, 181))
    finel = np.radians(np.linspace(-180, 180, 361))
    for lon in gl:
        ax.plot(np.full_like(fine, lon), fine, color="0.5", lw=0.4,
                ls=(0, (4, 4)), alpha=0.55, zorder=3)
    for lat in gb:
        ax.plot(finel, np.full_like(finel, lat), color="0.5", lw=0.4,
                ls=(0, (4, 4)), alpha=0.55, zorder=3)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(colors="0.7", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("0.4")
    return mesh
