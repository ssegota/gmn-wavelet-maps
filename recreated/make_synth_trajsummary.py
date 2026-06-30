#!/usr/bin/env python3
"""Generate a SYNTHETIC GMN trajectory-summary file in the real semicolon format,
for testing gmn_wavelet_maps.py without network access.  Radiants: an EVI cluster
(tagged EVI), the sporadic complex, an isolated DME-like southern shower, and an
isotropic floor.  NOT real data -- only for exercising the pipeline."""
import numpy as np
import gmn_wavelet as gw

RNG = np.random.default_rng(7)
LSUN = 358.5

# (name, ll0, bet, sig_deg, Vg, sigVg, n, iau)
SRC = [
    ("EVI",        186.5,   5.5, 1.2, 26.7, 1.0, 230, "EVI"),
    ("DME",        283.6, -76.5, 1.5, 36.7, 1.2,  60, "DME"),
    ("antihelion", 197.0,   0.0, 13.0, 30.0, 6.0, 650, ""),
    ("helion",     343.0,   0.0, 13.0, 33.0, 6.0, 360, ""),
    ("north_apex", 270.0,  19.0, 15.0, 61.0, 6.0, 320, ""),
    ("south_apex", 270.0, -19.0, 15.0, 61.0, 6.0, 220, ""),
    ("n_toroidal", 270.0,  56.0, 9.0, 37.0, 5.0, 170, ""),
    ("s_toroidal", 270.0, -56.0, 9.0, 37.0, 5.0,  90, ""),
]
N_ISO = 520

ll0, bet, vg, iau = [], [], [], []
for name, cl, cb, sg, vm, vs, n, code in SRC:
    cosb = np.cos(np.radians(np.clip(cb, -85, 85)))
    ll0.append((cl + RNG.normal(0, sg, n) / cosb) % 360.0)
    bet.append(np.clip(cb + RNG.normal(0, sg, n), -89.9, 89.9))
    vg.append(np.clip(RNG.normal(vm, vs, n), 9.0, 72.0))
    iau.append(np.array([code] * n, dtype=object))
# isotropic floor
ll0.append(RNG.uniform(0, 360, N_ISO))
bet.append(np.degrees(np.arcsin(RNG.uniform(-1, 1, N_ISO))))
vg.append(np.clip(RNG.normal(42, 14, N_ISO), 9.0, 72.0))
iau.append(np.array([""] * N_ISO, dtype=object))

ll0 = np.concatenate(ll0); bet = np.concatenate(bet)
vg = np.concatenate(vg); iau = np.concatenate(iau)
ra, dec = gw.sce_to_radec(ll0, bet, LSUN)
lam, beta = gw.equatorial_to_ecliptic(ra, dec)
sol = LSUN + RNG.normal(0, 0.25, len(ra))
n = len(ra)

# crude but finite orbital elements (not physically exact; test only)
a = np.clip(RNG.normal(2.5, 1.0, n), 0.6, 30)
e = np.clip(RNG.normal(0.7, 0.15, n), 0.05, 0.99)
incl = np.clip(np.abs(RNG.normal(20, 25, n)), 0, 180)
peri = RNG.uniform(0, 360, n)
node = (sol + RNG.normal(0, 1, n)) % 360
q = a * (1 - e)
qaph = a * (1 + e)
tj = np.clip(RNG.normal(3.0, 1.5, n), -2, 9)
vinf = vg + RNG.uniform(0.5, 1.5, n)

NF = 86
with open("synth_traj.txt", "w") as fh:
    fh.write("# SYNTHETIC trajectory summary (test data; not real)\n")
    fh.write("# columns mirror the GMN traj_summary semicolon layout\n")
    fh.write("# " + "-" * 40 + "\n")
    for i in range(n):
        f = ["0"] * NF
        f[0] = f"synthetic_{i:06d}"
        f[4] = iau[i] if iau[i] else "..."
        f[5] = f"{sol[i]:.6f}"
        f[7] = f"{ra[i]:.5f}"
        f[9] = f"{dec[i]:.5f}"
        f[11] = f"{lam[i]:.5f}"
        f[13] = f"{beta[i]:.5f}"
        f[15] = f"{vg[i]:.5f}"
        f[23] = f"{a[i]:.6f}"
        f[25] = f"{e[i]:.6f}"
        f[27] = f"{incl[i]:.6f}"
        f[29] = f"{peri[i]:.6f}"
        f[31] = f"{node[i]:.6f}"
        f[37] = f"{q[i]:.6f}"
        f[43] = f"{qaph[i]:.6f}"
        f[49] = f"{tj[i]:.6f}"
        f[59] = f"{vinf[i]:.5f}"
        fh.write("; ".join(f) + "\n")
print(f"wrote synth_traj.txt with {n} radiants")
