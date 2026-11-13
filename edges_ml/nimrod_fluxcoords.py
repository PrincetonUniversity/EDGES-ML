"""
Flux-aligned (straight-field-line) coordinates for NIMROD equilibria.

This module is a port of M3D-C1's m3dc1.flux_coordinates.flux_coordinates()
to an evaluation-agnostic interface: the equilibrium magnetic field is passed
in as a vectorised callback eval_b(R, Z) -> (B_R, B_Z, B_phi).  It has no
dependency on nimpy, so it can be tested with analytic fields.  It is used by
NIMRODAdapter._get_flux_coordinates() to build the grid of the poloidal mode
spectra.

NIMROD does not provide a poloidal flux function, so psi (relative to the
magnetic axis) is reconstructed by integrating the equilibrium field along
straight rays that start at the axis:

    dpsi/drho = R * (B_Z * cos - B_R * sin)          (R -> 1 for 'lin')

which follows the M3D-C1 convention B_R = -psi_Z / R, B_Z = psi_R / R.
B_phi must be supplied in the M3D-C1 toroidal-angle convention.

Stage 1 (same structure as M3D-C1)
    * poloidal rays at theta_k = 2*pi*k/m (no duplicate end point); theta = 0
      at the outboard midplane, increasing CLOCKWISE in the (R, Z) plane
      (cos(-theta), sin(-theta)); each ray ends at 99.9 % of the distance to
      the domain bounding box, or earlier where the field evaluation returns
      more than two consecutive NaNs (the ray left the domain);
    * radial targets uniform in psi_N, linspace(1e-4, 1, n);
    * psi_N = 1 is the LCFS.  The first estimate is the smallest (over all
      rays) of the largest |psi| reached along each ray.  This equals the
      limiter value for limited plasmas and approaches psi at the X-point
      from above for diverted plasmas (along the ray through the X-point the
      flux decreases again inside the private flux region).  The estimate is
      refined with X-points found by a poloidal-field-null finder, and can be
      overridden (see "Heuristics" below).
    * on each ray, psi is inverted with the quadratic that is exactly
      consistent with the trapezoidal integration of psi (no additional field
      evaluations are needed).

Stage 2 (faithful port of M3D-C1, including its two numerical quirks)
    * dl from periodic central differences (as fpylib.deriv on the wrapped
      arrays);
    * dtheta/dl for 'pest' (|I|/(R^2 Bp)), 'boozer' (B^2/Bp), 'hamada' (1/Bp);
      'canonical' is identical to PEST after the per-surface normalisation;
    * cumulative trapezoidal integration WITHOUT the closing segment, while
      the normalisation uses the closed rectangle sum (quirk 1);
    * R, Z are resampled onto the uniform theta grid with LINEAR
      interpolation, clipping theta to the theta_sfl range (quirk 2).

Absolute values are used for dtheta/dl.  M3D-C1 only accepts sign
combinations for which its signed expressions are positive, so wherever
M3D-C1 succeeds the result is identical; elsewhere this version does not
abort.

The geometric poloidal angle is available as an explicit fallback only.

For 'lin' (cylindrical) geometry I = B_phi and R -> 1 are used, so that
q = (1/L) * closed-integral(B_phi / Bp dl) is the cylindrical safety factor.

Heuristics (X-point / LCFS search)
    * automatic X-point guesses: the |Bp| minimum in the upper and in the
      lower half (relative to the axis), between 35 % and 97 % of each ray's
      length, accepted if it is below 25 % of the median |Bp|;
    * a null returned by find_null() is accepted if it lies farther than 10 %
      of the median ray length from the axis, is not a duplicate, has |Bp|
      below 5 % of the median |Bp|, psi(X) has the same sign as the edge flux,
      and |psi(X)| does not exceed the ray estimate;
    * the accepted candidate with the smallest |psi| defines psi_LCFS.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

from .utils import printwarn, printnote, normalize_fcoords


TWO_PI = 2.0 * np.pi
PSIN_MIN = 1.0e-4


@dataclass
class NimrodFluxCoordinates:
    """
    Flux coordinate system, mirroring the attributes of fpy.flux_coordinates
    that are needed for mode spectra.

    rpath, zpath      : [n_theta, n_psi]  (R, Z) of the grid points
    psi_norm          : [n_psi]           normalised poloidal flux
    theta             : [n_theta]         2*pi*arange(n_theta)/n_theta
    q                 : [n_psi]           safety factor (M3D-C1 sign convention)
    fcoords           : str               coordinate system actually used
    axis              : [2]               magnetic axis (R, Z)
    psi_lcfs          : float             psi_LCFS - psi_axis (Wb/rad)
    lcfs_source       : str               which candidate defined psi_lcfs
    omega             : [n_theta, n_psi]  zeta - phi (Boozer / Hamada), or None
    n_clipped         : int               uniform-theta points outside the
                                          theta_sfl range (clipped, quirk 2)
    n_unreached       : int               (theta, psi) targets not reached
                                          along their ray (clipped to the end)
    n_failed_surfaces : int               surfaces with invalid field values
                                          (kept at their geometric angle)
    """
    rpath: np.ndarray
    zpath: np.ndarray
    psi_norm: np.ndarray
    theta: np.ndarray
    q: np.ndarray
    fcoords: str
    axis: np.ndarray
    psi_lcfs: float
    lcfs_source: str
    omega: Optional[np.ndarray] = None
    n_clipped: int = 0
    n_unreached: int = 0
    n_failed_surfaces: int = 0

    @property
    def m(self):
        return self.rpath.shape[0]

    @property
    def n(self):
        return self.rpath.shape[1]

    @property
    def psi_n(self):
        return self.psi_norm


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------

def _cumtrapz(y, x):
    """Cumulative trapezoidal integral, starting at 0."""
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    if out.size > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def _ray_lengths(axis, bbox, co, sn, margin=0.999):
    """Distance from the axis to the bounding box along each ray (times *margin*)."""
    rmin, rmax, zmin, zmax = bbox
    r0, z0 = float(axis[0]), float(axis[1])
    with np.errstate(divide='ignore', invalid='ignore'):
        tr = np.where(co > 0, (rmax - r0) / co,
                      np.where(co < 0, (rmin - r0) / co, np.inf))
        tz = np.where(sn > 0, (zmax - z0) / sn,
                      np.where(sn < 0, (zmin - z0) / sn, np.inf))
    return margin * np.minimum(tr, tz)


def _valid_extent(g, max_gap=2):
    """
    Returns (last, g_filled):
      last     index of the last valid sample before the first run of more
               than *max_gap* consecutive NaNs (= the ray left the domain);
      g_filled copy of g[:last+1] in which isolated NaNs (failed point
               evaluations) are linearly interpolated.
    """
    g = np.asarray(g, dtype=float)
    bad = ~np.isfinite(g)
    last = -1
    run = 0
    for i in range(g.size):
        if bad[i]:
            run += 1
            if run > max_gap:
                break
        else:
            run = 0
            last = i
    if last < 1:
        return last, g[:max(last + 1, 0)].copy()
    gg = g[:last + 1].copy()
    b = ~np.isfinite(gg)
    if b.any():
        idx = np.arange(last + 1)
        gg[b] = np.interp(idx[b], idx[~b], gg[~b])
    return last, gg


def _cumulative_angle(d, dl):
    """
    Faithful port of the M3D-C1 angle integration (per column):
        theta[0] = 0
        theta[i] = theta[i-1] + dl[i]*d[i]/2 + dl[i-1]*d[i-1]/2
        theta   /= sum(d*dl) / (2*pi)
    The closing segment is not included in the cumulative sum, while the
    normalisation uses the closed sum (M3D-C1 quirk, kept on purpose).
    """
    w = d * dl
    th = np.zeros_like(w)
    th[1:] = np.cumsum(0.5 * (w[1:] + w[:-1]), axis=0)
    f = np.sum(w, axis=0) / TWO_PI
    with np.errstate(divide='ignore', invalid='ignore'):
        return th / f[None, :]


# ---------------------------------------------------------------------------
# Stage 1: psi along rays
# ---------------------------------------------------------------------------

def _psi_along_rays(eval_b, axis, co, sn, lengths, nrho, torgeom):
    """
    Evaluates B on all rays in a single eval_b() call and integrates psi
    along each ray.  The points are ordered ray by ray (contiguous along each
    ray), which helps evaluators that reuse logical-coordinate guesses from
    the previous point.  (The NIMROD adapter calls nimpy without an EvalGrid
    cache for this point set.)
    """
    m = co.size
    frac = np.linspace(0.0, 1.0, nrho)
    rho = lengths[:, None] * frac[None, :]
    R = axis[0] + rho * co[:, None]
    Z = axis[1] + rho * sn[:, None]

    br, bz, _ = eval_b(R.ravel(), Z.ravel())
    br = np.asarray(br, dtype=float).reshape(R.shape)
    bz = np.asarray(bz, dtype=float).reshape(R.shape)

    reff = R if torgeom else np.ones_like(R)
    g = reff * (bz * co[:, None] - br * sn[:, None])

    psi = np.full_like(g, np.nan)
    last = np.full(m, -1, dtype=int)
    for j in range(m):
        lj, gj = _valid_extent(g[j])
        last[j] = lj
        if lj >= 1:
            g[j, :lj + 1] = gj
            psi[j, :lj + 1] = _cumtrapz(gj, rho[j, :lj + 1])

    return {
        "rho": rho, "R": R, "Z": Z, "g": g, "psi": psi, "last": last,
        "bp": np.sqrt(br ** 2 + bz ** 2), "frac": frac,
    }


def _psi_at_point(eval_b, axis, rz, torgeom, nsamp=801):
    """
    psi (relative to the axis) at an arbitrary point, by ray integration.
    Returns NaN if the straight ray from the axis leaves the domain.
    """
    d = np.asarray(rz[:2], dtype=float) - np.asarray(axis[:2], dtype=float)
    L = float(np.hypot(d[0], d[1]))
    if L <= 0:
        return 0.0
    c, s = d / L
    rho = np.linspace(0.0, L, nsamp)
    R = axis[0] + rho * c
    Z = axis[1] + rho * s
    br, bz, _ = eval_b(R, Z)
    reff = R if torgeom else np.ones_like(R)
    g = reff * (np.asarray(bz, dtype=float) * c - np.asarray(br, dtype=float) * s)
    last, gg = _valid_extent(g)
    if last < nsamp - 1:
        return np.nan
    return float(np.sum(0.5 * (gg[1:] + gg[:-1]) * np.diff(rho)))


def _auto_xpoint_guesses(rays, axis):
    """
    Coarse |Bp| minima in the upper and lower half, away from the axis
    (35 % - 97 % of the ray length, below 25 % of the median |Bp|).
    """
    bp, R, Z, last, frac = rays["bp"], rays["R"], rays["Z"], rays["last"], rays["frac"]
    mask = np.zeros(bp.shape, dtype=bool)
    for j, lj in enumerate(last):
        if lj >= 1:
            mask[j, :lj + 1] = True
    mask &= (frac[None, :] >= 0.35) & (frac[None, :] <= 0.97) & np.isfinite(bp)
    if not mask.any():
        return []
    med = float(np.nanmedian(bp[mask]))
    guesses = []
    for half in (Z > axis[1], Z < axis[1]):
        mm = mask & half
        if not mm.any():
            continue
        idx = int(np.argmin(np.where(mm, bp, np.inf)))
        j, k = np.unravel_index(idx, bp.shape)
        if bp[j, k] < 0.25 * med:
            guesses.append([float(R[j, k]), float(Z[j, k])])
    return guesses


def _determine_psi_lcfs(rays, eval_b, axis, torgeom, find_null,
                        xpoint_guesses, psi_override):
    """
    Returns (psi_lcfs - psi_axis, description of the source).

    Candidates: the ray estimate (smallest over rays of the largest |psi|
    reached) and the flux at accepted X-points; the smallest |psi| wins.
    *psi_override* (magnitude) replaces the result; its sign follows the
    edge flux.
    """
    psi, last = rays["psi"], rays["last"]
    valid = [j for j in range(psi.shape[0]) if last[j] >= 1]
    if not valid:
        raise ValueError("No valid poloidal rays; cannot reconstruct psi.")

    edge = np.array([psi[j, last[j]] for j in valid])
    edge = edge[np.isfinite(edge)]
    if edge.size == 0:
        raise ValueError("No valid poloidal rays; cannot reconstruct psi.")
    sign = float(np.sign(np.median(edge)))
    if sign == 0.0:
        raise ValueError("Poloidal flux does not change away from the axis.")

    if psi_override is not None:
        return float(sign * abs(float(psi_override))), "config (nimrod_psi_lcfs)"

    # Smallest over rays of the largest |psi| reached along each ray.
    ray_max = []
    for j in valid:
        seg = sign * psi[j, :last[j] + 1]
        if np.any(np.isfinite(seg)):
            ray_max.append(float(np.nanmax(seg)))
    ray_max = np.asarray(ray_max)
    ray_max = ray_max[np.isfinite(ray_max) & (ray_max > 0)]
    if ray_max.size == 0:
        raise ValueError("Poloidal flux does not increase away from the axis.")
    psi_b = sign * float(np.min(ray_max))
    candidates = [(abs(psi_b), psi_b, "ray maxima (domain boundary / separatrix estimate)")]

    if find_null is not None:
        guesses = [list(g)[:2] for g in (xpoint_guesses or [])]
        guesses += _auto_xpoint_guesses(rays, axis)

        valid_bp = [rays["bp"][j, :last[j] + 1] for j in valid]
        med_bp = float(np.nanmedian(np.concatenate(valid_bp))) if valid_bp else np.nan
        med_len = float(np.nanmedian(rays["rho"][:, -1]))

        found = []
        for gss in guesses:
            try:
                rz = find_null([float(gss[0]), float(gss[1]), 0.0])
            except Exception:
                rz = None
            if rz is None:
                continue
            rz = np.asarray(rz, dtype=float)
            if rz.size < 2 or not np.all(np.isfinite(rz[:2])):
                continue
            # Reject the magnetic axis itself and duplicates.
            if np.hypot(rz[0] - axis[0], rz[1] - axis[1]) < 0.1 * med_len:
                continue
            if any(np.hypot(rz[0] - f[0], rz[1] - f[1]) < 1e-3 * med_len for f in found):
                continue
            # Must be a genuine poloidal field null.
            br, bz, _ = eval_b(np.array([rz[0]]), np.array([rz[1]]))
            bpx = float(np.hypot(np.asarray(br, dtype=float)[0],
                                 np.asarray(bz, dtype=float)[0]))
            if not np.isfinite(bpx) or (np.isfinite(med_bp) and bpx > 0.05 * med_bp):
                continue
            found.append(rz)
            psi_x = _psi_at_point(eval_b, axis, rz, torgeom)
            if not np.isfinite(psi_x) or np.sign(psi_x) != sign:
                continue
            if abs(psi_x) <= abs(psi_b) * (1.0 + 1e-6):
                candidates.append((abs(psi_x), psi_x,
                                   f"X-point at (R,Z)=({rz[0]:.4f}, {rz[1]:.4f})"))

    _, psi_s, source = min(candidates, key=lambda c: c[0])
    return float(psi_s), source


def _invert_rays(rays, psi_lcfs, targets):
    """
    rho at which psi_N reaches each target value on every ray (first
    crossing).  Within a sample interval psi is the quadratic that is exactly
    consistent with the trapezoidal integration of the (linear) integrand.

    Returns (rho[m, n_targets], number of unreachable targets).  Unreachable
    targets are clipped to the last valid sample of the ray.
    """
    rho, g, psi, last = rays["rho"], rays["g"], rays["psi"], rays["last"]
    m = rho.shape[0]
    nt = targets.size
    out_all = np.zeros((m, nt))
    unreached = 0

    for j in range(m):
        lj = last[j]
        if lj < 1:
            unreached += nt
            continue
        r = rho[j, :lj + 1]
        h = r[1] - r[0]
        pn = psi[j, :lj + 1] / psi_lcfs
        gn = g[j, :lj + 1] / psi_lcfs
        runmax = np.maximum.accumulate(pn)
        k = np.searchsorted(runmax, targets, side='left')

        out = np.empty(nt)
        over = k > lj
        low = k <= 0
        mid = ~over & ~low
        unreached += int(np.count_nonzero(over))
        out[over] = r[-1]
        out[low] = 0.0

        if np.any(mid):
            kk = k[mid]
            lo = kk - 1
            t = targets[mid]
            p_lo, p_hi = pn[lo], pn[kk]
            g_lo, g_hi = gn[lo], gn[kk]
            a = (g_hi - g_lo) / (2.0 * h)
            b = g_lo
            c = p_lo - t
            disc = b * b - 4.0 * a * c
            with np.errstate(divide='ignore', invalid='ignore'):
                denom = b + np.sqrt(np.maximum(disc, 0.0))
                s_quad = -2.0 * c / denom
                dp = p_hi - p_lo
                s_lin = np.where(dp > 0, h * (t - p_lo) / dp, 0.0)
            ok = (disc >= 0) & (denom > 0) & np.isfinite(s_quad) \
                & (s_quad >= -1e-12 * h) & (s_quad <= h * (1.0 + 1e-9))
            s = np.clip(np.where(ok, s_quad, s_lin), 0.0, h)
            out[mid] = r[lo] + s

        out_all[j] = out

    return out_all, unreached


# ---------------------------------------------------------------------------
# Stage 2: straight-field-line angle (port of M3D-C1)
# ---------------------------------------------------------------------------

def _sfl_stage2(r, z, br, bz, bphi, fcoords, torgeom, period, fac):
    """
    Returns (theta_c, theta_pest, q, good) on the Stage-1 grid, where
    theta_c is the straight-field-line angle of the requested system and
    good[j] is False for surfaces with non-finite angles.
    """
    rp = r if torgeom else np.ones_like(r)
    ix = r * bphi if torgeom else bphi
    bp = np.sqrt(br ** 2 + bz ** 2)

    # 3-point periodic derivative with respect to the index (= fpyl.deriv on
    # the wrapped arrays for uniform spacing).
    dr = 0.5 * (np.roll(r, -1, axis=0) - np.roll(r, 1, axis=0))
    dz = 0.5 * (np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0))
    dl = np.sqrt(dr ** 2 + dz ** 2)

    with np.errstate(divide='ignore', invalid='ignore'):
        fjr2 = ix / (rp ** 2 * bp)
        d_pest = np.abs(fjr2)
        if fcoords == "boozer":
            d = (bp ** 2 + (ix / rp) ** 2) / bp
        elif fcoords == "hamada":
            d = 1.0 / bp
        else:
            d = d_pest
        # M3D-C1: fjr2 = -fac*ix/(rp^2 bp);  q = sum(fjr2*dl)/period
        q = -fac * np.sum(fjr2 * dl, axis=0) / period

    theta_c = _cumulative_angle(d, dl)
    theta_p = theta_c if d is d_pest else _cumulative_angle(d_pest, dl)

    good = np.all(np.isfinite(theta_c), axis=0) & np.all(np.isfinite(theta_p), axis=0)
    return theta_c, theta_p, q, good


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_flux_coordinates(eval_b: Callable,
                           axis,
                           bbox,
                           points: int,
                           fcoords: str = "pest",
                           torgeom: bool = True,
                           period: float = TWO_PI,
                           find_null: Optional[Callable] = None,
                           xpoint_guesses=None,
                           psi_lcfs=None,
                           ray_samples: Optional[int] = None,
                           theta_oversample: int = 1,
                           include_omega: bool = False,
                           verbose: bool = True) -> NimrodFluxCoordinates:
    """
    Builds flux coordinates for an axisymmetric equilibrium.

    Parameters
    ----------
    eval_b : callable(R, Z) -> (B_R, B_Z, B_phi)
        Vectorised evaluation of the axisymmetric EQUILIBRIUM field on 1D point
        arrays.  B_phi must use the M3D-C1 toroidal angle convention.  Points
        outside the domain must return NaN.
    axis : (R0, Z0[, ...])
        Magnetic axis.
    bbox : (rmin, rmax, zmin, zmax)
        Bounding box of the computational domain.
    points : int
        Number of poloidal AND radial points (like 'points' in M3D-C1).
    fcoords : 'pest' | 'boozer' | 'hamada' | 'canonical' | 'geometric'
    torgeom : bool
        True for toroidal geometry ('tor'), False for 'lin'.
    period : float
        Toroidal period length (2*pi for 'tor', per_length for 'lin').
    find_null : callable([R, Z, 0]) -> [R, Z, ...] or None, optional
        Poloidal field null finder (e.g. a nimpy.fsa.find_pf_null wrapper).
    xpoint_guesses : list of [R, Z], optional
        Initial guesses for X-points (in addition to an automatic search).
    psi_lcfs : float, optional
        Override for psi_LCFS - psi_axis (only the magnitude is used).
    ray_samples : int, optional
        Samples per ray for the psi integration (default 4 * points; at
        least 50 are always used).
    theta_oversample : int
        Stage-1 oversampling factor in theta (1 = faithful M3D-C1 algorithm).
        The output always has *points* poloidal points.
    include_omega : bool
        Also return omega = zeta - phi for Boozer / Hamada coordinates.
    verbose : bool
        Print progress notes and warnings.

    Returns
    -------
    NimrodFluxCoordinates
    """
    fcoords = normalize_fcoords(fcoords)
    points = int(points)
    if points < 4:
        raise ValueError("points must be at least 4.")
    s = max(1, int(theta_oversample))
    m1 = s * points
    nrho = max(50, int(ray_samples or 4 * points))
    axis = np.asarray(axis, dtype=float).ravel()[:2]

    theta1 = TWO_PI * np.arange(m1) / m1
    co = np.cos(-theta1)
    sn = np.sin(-theta1)
    lengths = _ray_lengths(axis, bbox, co, sn)
    if not np.all(np.isfinite(lengths) & (lengths > 0)):
        raise ValueError("Magnetic axis lies outside the domain bounding box.")

    if verbose:
        printnote(f"NIMROD flux coordinates: {fcoords}, {points}x{points} points, "
                  f"{m1} rays x {nrho} samples.")

    # --- Stage 1 -----------------------------------------------------------
    rays = _psi_along_rays(eval_b, axis, co, sn, lengths, nrho, torgeom)

    psi_s, source = _determine_psi_lcfs(rays, eval_b, axis, torgeom, find_null,
                                        xpoint_guesses, psi_lcfs)
    # M3D-C1: fac = -1 if psi_s < psi_0 else +1
    fac = 1.0 if psi_s > 0 else -1.0
    if verbose:
        printnote(f"NIMROD flux coordinates: psi_LCFS - psi_axis = {psi_s:.6e} "
                  f"({source}).")

    psi_norm = np.linspace(PSIN_MIN, 1.0, points)
    rho_s, unreached = _invert_rays(rays, psi_s, psi_norm)
    if unreached and verbose:
        printwarn(f"NIMROD flux coordinates: {unreached} (theta, psi) targets could "
                  "not be reached along their ray and were clipped to the last "
                  "valid point.")

    r1 = axis[0] + rho_s * co[:, None]
    z1 = axis[1] + rho_s * sn[:, None]

    br, bz, bphi = eval_b(r1.ravel(), z1.ravel())
    br = np.asarray(br, dtype=float).reshape(r1.shape)
    bz = np.asarray(bz, dtype=float).reshape(r1.shape)
    bphi = np.asarray(bphi, dtype=float).reshape(r1.shape)

    # --- Stage 2 -----------------------------------------------------------
    sfl_kind = "pest" if fcoords in ("pest", "canonical", "geometric") else fcoords
    theta_c, theta_p, q, good = _sfl_stage2(r1, z1, br, bz, bphi, sfl_kind,
                                            torgeom, period, fac)

    theta = TWO_PI * np.arange(points) / points

    if fcoords == "geometric":
        if verbose:
            printwarn("NIMROD flux coordinates: using the GEOMETRIC poloidal angle. "
                      "Poloidal spectra depend on the angle definition; this is a "
                      "fallback for special cases only.")
        return NimrodFluxCoordinates(
            rpath=r1[::s].copy(), zpath=z1[::s].copy(), psi_norm=psi_norm,
            theta=theta, q=q, fcoords="geometric", axis=axis, psi_lcfs=psi_s,
            lcfs_source=source, omega=None, n_unreached=unreached,
        )

    want_omega = bool(include_omega) and fcoords in ("boozer", "hamada")
    # zeta - phi = q * (theta_flux - theta_PEST) along the field lines.
    omega_src = q[None, :] * (theta_c - theta_p) if want_omega else None

    rpath = np.empty((points, points))
    zpath = np.empty((points, points))
    omega = np.zeros((points, points)) if want_omega else None
    fp_tol = 1e-10 * TWO_PI
    n_clipped = 0
    n_failed = 0

    for j in range(points):
        if not good[j]:
            n_failed += 1
            rpath[:, j] = r1[::s, j]
            zpath[:, j] = z1[::s, j]
            continue
        th = np.maximum.accumulate(theta_c[:, j])
        n_clipped += int(np.count_nonzero(theta > th[-1] + fp_tol))
        n_clipped += int(np.count_nonzero(theta < th[0] - fp_tol))
        tc = np.clip(theta, th[0], th[-1])
        rpath[:, j] = np.interp(tc, th, r1[:, j])
        zpath[:, j] = np.interp(tc, th, z1[:, j])
        if want_omega:
            omega[:, j] = np.interp(tc, th, omega_src[:, j])

    if n_failed and verbose:
        printwarn(f"NIMROD flux coordinates: {n_failed} surface(s) had invalid field "
                  "values; their geometric-angle positions were kept.")

    if verbose:
        qa = q[np.isfinite(q)]
        if qa.size:
            q95 = float(np.interp(0.95, psi_norm, q))
            printnote(f"NIMROD flux coordinates: q(axis)={qa[0]:.3f}, q95={q95:.3f}, "
                      f"q(LCFS)={qa[-1]:.3f}.")

    return NimrodFluxCoordinates(
        rpath=rpath, zpath=zpath, psi_norm=psi_norm, theta=theta, q=q,
        fcoords=fcoords, axis=axis, psi_lcfs=psi_s, lcfs_source=source,
        omega=omega, n_clipped=n_clipped, n_unreached=unreached,
        n_failed_surfaces=n_failed,
    )
