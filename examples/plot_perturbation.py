#!/usr/bin/env python3
"""
Plot the mode perturbation and its poloidal spectrum from an edges-ml
'reduced_h5' file (similar to m3dc1.eigenfunction.eigenfunction()).

Figures
-------
1. Perturbation in the R-Z plane
   perturbations/<entry>/2d_fields/<field> on grids/2d_grid
   (or, with --fluxgrid, spectra/<field>/eigenfunction on the flux-aligned
   grid spectra/<field>/R, Z; requires store_fluxgrid=True at export time).
2. Poloidal spectrum |xi_m|(psi_N) from spectra/<field>/amplitude.

Usage
-----
    python plot_perturbation.py FILE.h5 --list
    python plot_perturbation.py FILE.h5 [--entry n05] [--field p] [--show-res]
                                        [--norm] [--fluxgrid] [--save DIR]

Examples
--------
    python examples/plot_perturbation.py datasets/file_m3dc1.h5 --list
    python examples/plot_perturbation.py datasets/file_m3dc1.h5 --entry n05 --field p --show-res
    python examples/plot_perturbation.py datasets/file_m3dc1.h5 --norm --fluxgrid --save plots/
"""

import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.tri as mtri


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return value.astype(str)
    return value


def list_contents(h5):
    """Prints the perturbation entries, their 2D fields and spectra."""
    if "perturbations" not in h5:
        print("No 'perturbations' group in this file.")
        return
    for entry, grp in h5["perturbations"].items():
        f2d = list(grp["2d_fields"].keys()) if "2d_fields" in grp else []
        spec = list(grp["spectra"].keys()) if "spectra" in grp else []
        print(f"{entry:>12s}:  2d_fields={f2d}  spectra={spec}")


def pick_entry(h5, entry):
    entries = list(h5["perturbations"].keys())
    if not entries:
        raise SystemExit("The file contains no perturbation entries.")
    if entry is None:
        return entries[0]
    if entry not in entries:
        raise SystemExit(f"Entry '{entry}' not found. Available: {entries}")
    return entry


def pick_field(grp, field):
    """Default field: first quantity with a spectrum, else first 2D field."""
    if field is not None:
        return field
    if "spectra" in grp and len(grp["spectra"]):
        return list(grp["spectra"].keys())[0]
    if "2d_fields" in grp and len(grp["2d_fields"]):
        name = list(grp["2d_fields"].keys())[0]
        for suffix in ("_R", "_Z", "_phi"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name
    raise SystemExit("This entry contains neither 2D fields nor spectra.")


def toroidal_mode_number(grp, entry, field):
    """n from the spectrum attributes, else from the entry name ('n05')."""
    try:
        n = int(grp["spectra"][field].attrs["n"])
        if n >= 0:
            return n
    except Exception:
        pass
    digits = "".join(c for c in entry.split("_")[-1] if c.isdigit())
    return int(digits) if digits else None


def masked_triangulation(R, Z, max_edge_factor=4.0):
    """
    Delaunay triangulation of scattered points.  Triangles with an edge much
    longer than the typical edge (concave regions outside the wall) are masked.
    """
    tri = mtri.Triangulation(R, Z)
    x = R[tri.triangles]
    y = Z[tri.triangles]
    edges = np.sqrt((x - np.roll(x, 1, axis=1)) ** 2 + (y - np.roll(y, 1, axis=1)) ** 2)
    longest = edges.max(axis=1)
    tri.set_mask(longest > max_edge_factor * np.median(longest))
    return tri


def symmetric_levels(values, nlevels=101):
    vmax = np.nanmax(np.abs(values))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    return np.linspace(-vmax, vmax, nlevels)


def scalar_formatter():
    sfmt = ticker.ScalarFormatter()
    sfmt.set_powerlimits((-3, 4))
    return sfmt


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_rz(h5, grp, field, component, n, norm, cmap, use_fluxgrid):
    """Perturbation in the R-Z plane."""
    fig, ax = plt.subplots()
    unit = ""

    if use_fluxgrid:
        if "spectra" not in grp or field not in grp["spectra"]:
            raise SystemExit(f"No spectrum for '{field}' in this entry.")
        sg = grp["spectra"][field]
        if not all(k in sg for k in ("eigenfunction", "R", "Z")):
            raise SystemExit("The flux grid was not stored (export with "
                             "perturbations/spectrum/store_fluxgrid=True).")
        ef = np.asarray(sg["eigenfunction"][()], dtype=float)
        R = np.asarray(sg["R"][()], dtype=float)
        Z = np.asarray(sg["Z"][()], dtype=float)
        # Close the poloidal direction (theta = 2 pi == theta = 0), as in eigenfunction().
        ef = np.vstack([ef, ef[:1]])
        R = np.vstack([R, R[:1]])
        Z = np.vstack([Z, Z[:1]])
        if norm:
            ef = ef / np.nanmax(np.abs(ef))
        cont = ax.contourf(R, Z, ef, levels=symmetric_levels(ef), cmap=cmap)
        label = field
    else:
        if "2d_fields" not in grp:
            raise SystemExit("This entry has no 2D fields.")
        f2d = grp["2d_fields"]
        name = field
        if name not in f2d:
            name = f"{field}_{component}"
        if name not in f2d:
            raise SystemExit(f"2D field '{field}' not found. Available: {list(f2d.keys())}")

        values = np.asarray(f2d[name][()], dtype=float).ravel()
        unit = str(_decode(f2d[name].attrs.get("unit", "")))

        grid = np.asarray(h5["grids/2d_grid"][()], dtype=float)
        if grid.ndim != 2 or grid.shape[1] < 2 or grid.shape[0] != values.size:
            raise SystemExit(f"grids/2d_grid has shape {grid.shape}, which does not "
                             f"match the {values.size} values of '{name}'.")
        R, Z = grid[:, 0], grid[:, 1]

        good = np.isfinite(values)
        if norm:
            values = values / np.nanmax(np.abs(values[good]))
        tri = masked_triangulation(R[good], Z[good])
        cont = ax.tricontourf(tri, values[good], levels=symmetric_levels(values[good]),
                              cmap=cmap)
        label = name

    if norm:
        unit = "a.u."
    cbar = fig.colorbar(cont, ax=ax, format=scalar_formatter())
    cbar.set_label(f"$\\delta$ {label}" + (f" ({unit})" if unit else ""))

    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.5)
    if n is not None:
        ax.set_title(f"n={n}")
    fig.tight_layout()
    return fig


def resonant_surfaces(h5, n):
    """psi_N of the rational surfaces q = m/n from the equilibrium q profile."""
    try:
        psin = np.asarray(h5["grids/1d_grid"][()], dtype=float)
        q = np.asarray(h5["equilibrium/flux_averages/q"][()], dtype=float)
    except KeyError:
        print("Warning: equilibrium q profile not found; resonant surfaces not shown.")
        return []
    good = np.isfinite(q) & np.isfinite(psin)
    psin, q = psin[good], q[good]
    if psin.size < 2 or n is None or n == 0:
        return []
    qa = np.abs(q)
    out = []
    for qres in np.arange(np.ceil(qa.min() * n), np.floor(qa.max() * n) + 1) / n:
        # first crossing (robust also for non-monotonic q)
        idx = np.where(np.diff(np.sign(qa - qres)) != 0)[0]
        if idx.size == 0:
            continue
        i = idx[0]
        p = psin[i] + (qres - qa[i]) * (psin[i + 1] - psin[i]) / (qa[i + 1] - qa[i])
        out.append((qres, int(round(qres * n)), p))
    return out


def plot_spectrum(h5, grp, field, n, norm, nummodes, show_res, xlimits):
    """Poloidal spectrum |xi_m|(psi_N)."""
    if "spectra" not in grp or field not in grp["spectra"]:
        print(f"No spectrum for '{field}' in this entry (vector fields have none).")
        return None
    sg = grp["spectra"][field]
    psin = np.asarray(sg["psi_norm"][()], dtype=float)
    m = np.asarray(sg["m"][()], dtype=int)
    amp = np.asarray(sg["amplitude"][()], dtype=float)

    fac = 1.0 / np.nanmax(amp) if norm else 1.0
    mmax = np.nanmax(amp, axis=1)
    largest = set(np.argsort(np.nan_to_num(mmax, nan=-np.inf))[-nummodes:]) if nummodes > 0 else set()

    fig, ax = plt.subplots()
    for j in range(amp.shape[0]):
        if j in largest:
            ax.plot(psin, fac * amp[j], label=f"m={m[j]}")
        else:
            ax.plot(psin, fac * amp[j])

    if show_res:
        cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ytop = 0.98 * ax.get_ylim()[1]
        for i, (qres, mres, p) in enumerate(resonant_surfaces(h5, n)):
            ax.axvline(p, linestyle="--", color=cols[i % len(cols)])
            ax.annotate(f"q={qres:g}, m={mres}", xy=(p, ytop), rotation=90,
                        va="top", ha="left")

    ax.set_xlabel(r"$\psi_N$")
    ax.set_ylabel(f"{field} spectrum" + (" (a. u.)" if norm else ""))
    ax.set_xlim(left=xlimits[0], right=xlimits[1])
    ax.yaxis.set_major_formatter(scalar_formatter())
    ax.grid(True, alpha=0.5)

    fcoords = _decode(sg.attrs.get("fcoords", ""))
    title = f"n={n}" if n is not None else ""
    if fcoords:
        title += f"  ({fcoords})"
    ax.set_title(title.strip())
    if largest:
        ax.legend(ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", help="edges-ml reduced_h5 output file")
    p.add_argument("--list", action="store_true", help="list entries / fields and exit")
    p.add_argument("--entry", default=None, help="perturbation entry, e.g. 'n05' (default: first)")
    p.add_argument("--field", default=None, help="quantity, e.g. 'p' (default: first with a spectrum)")
    p.add_argument("--component", default="R", choices=("R", "Z", "phi"),
                   help="component for vector fields (default: R)")
    p.add_argument("--fluxgrid", action="store_true",
                   help="plot the eigenfunction on the flux-aligned grid (needs store_fluxgrid)")
    p.add_argument("--norm", action="store_true", help="normalize to unity")
    p.add_argument("--nummodes", type=int, default=10, help="number of modes in the legend")
    p.add_argument("--show-res", action="store_true", help="mark resonant surfaces q = m/n")
    p.add_argument("--xlim", type=float, nargs=2, default=[None, None], metavar=("MIN", "MAX"),
                   help="psi_N limits of the spectrum plot")
    p.add_argument("--cmap", default="jet", help="colormap of the R-Z plot")
    p.add_argument("--save", default=None, metavar="DIR", help="save PNGs to DIR instead of showing")
    args = p.parse_args(argv)

    with h5py.File(args.file, "r") as h5:
        if args.list:
            list_contents(h5)
            return
        if "perturbations" not in h5:
            raise SystemExit("No 'perturbations' group in this file.")

        entry = pick_entry(h5, args.entry)
        grp = h5["perturbations"][entry]
        field = pick_field(grp, args.field)
        n = toroidal_mode_number(grp, entry, field)
        print(f"Plotting entry '{entry}', field '{field}' (n={n}).")

        figs = {
            "rz": plot_rz(h5, grp, field, args.component, n, args.norm,
                          args.cmap, args.fluxgrid),
            "spectrum": plot_spectrum(h5, grp, field, n, args.norm, args.nummodes,
                                      args.show_res, args.xlim),
        }

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.file))[0]
        for kind, fig in figs.items():
            if fig is None:
                continue
            out = os.path.join(args.save, f"{stem}_{entry}_{field}_{kind}.png")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"Saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
