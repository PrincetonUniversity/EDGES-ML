"""
edges_ml.browser  –  static HTML browser for simulation directories.

This module is intentionally dependency free (Python standard library only) so
that it can be executed on a login node / laptop without the heavy simulation
post-processing stack (fpy, nimpy, h5py, imas, ...).

It crawls the same directory layout that `edges_ml.build_dataset` expects:

    M3D-C1 :  <...>/<model_dir>/<mode_dir>/C1.h5
    NIMROD :  <...>/<model_dir>/<mode_dir>/nimhist.bin  (+ nimrod.in)

and produces a single, self-contained HTML file that lets a user browse the
directory tree, inspect the physics model of every simulation, and read the
code input namelists.  A "Plots" tab is already present; it stays empty (with
a placeholder note) unless image files are discovered (``--scan-plots``), so
that plotting can be added later without touching the HTML template.

Keeping the page small
----------------------
The embedded data model is *packed* before it is written:

* namelists are shipped as raw text only and parsed in the browser on demand
  (the parsed parameter table used to roughly double the payload),
* identical file contents are stored once in a shared text pool,
* nodes are stored as a flat array of arrays with parent indices; absolute
  paths, descendant counts and inherited summaries are reconstructed in the
  browser instead of being repeated for every entry,
* repeated row labels are replaced by indices into a label dictionary,
* long numbered file series (``dumpgll.00001`` ...) are collapsed into one row,
* the resulting JSON is gzip + base64 encoded and inflated in the page with
  ``DecompressionStream`` (use ``--no-compress`` if that is not available).

Typical use
-----------
    $ python -m edges_ml.browser /scratch/mastu_runs -o simulations.html
    $ python make_browser.py --m3dc1 /scratch/m3dc1 --nimrod /scratch/nimrod

Python API
----------
    from edges_ml import build_browser
    build_browser([{"code": "m3dc1", "directories": ["/scratch/runs"]}],
                  output="simulations.html")
"""

import argparse
import base64
import gzip
import json
import os
import re
import sys
import html as _html
import datetime as _dt
from collections import OrderedDict
from pathlib import Path
from string import Template


__all__ = [
    "build_browser",
    "discover_simulations",
    "parse_namelist",
    "describe_simulation",
    "render_html",
    "main",
]


# ===========================================================================
# CONSTANTS
# ===========================================================================

CODE_LABELS = {
    "m3dc1":  "M3D-C1",
    "nimrod": "NIMROD",
}

# Order used to encode the code of a node as a small integer.
CODE_ORDER = ["", "m3dc1", "nimrod"]

# File that marks a directory as a simulation (mode) directory.
SIM_MARKERS = {
    "m3dc1":  "C1.h5",
    "nimrod": "nimhist.bin",
}

# Input/namelist files that are offered in the "Inputs" tab.
DOC_PATTERNS = {
    "m3dc1": ["C1input", "C1input*", "*.in", "*.namelist", "README*"],
    "nimrod": ["nimrod.in", "nimeq.in", "fluxgrid.in", "oculus.in", "nimset.in",
               "*.in", "README*"],
}

# The file that is used to determine the physics model of a simulation.
PRIMARY_INPUT = {
    "m3dc1":  ["C1input"],
    "nimrod": ["nimrod.in", "nimeq.in"],
}

PLOT_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".pdf"}
PLOT_SUBDIRS  = ("", "plots", "figures", "figs", "images")

SKIP_DOC_SUFFIXES = {".h5", ".bin", ".bp", ".nc", ".so", ".o", ".a", ".png",
                     ".jpg", ".jpeg", ".pdf", ".gz", ".tar", ".zip", ".npz"}

DEFAULT_MAX_DOC_BYTES = 256 * 1024     # text shown per input file
HARD_DOC_SIZE_LIMIT   = 8 * 1024 * 1024
DEFAULT_MAX_FILES     = 200            # entries shown in the file listing
COLLAPSE_THRESHOLD    = 5              # collapse numbered series of >= N files

KIND_ORDER = {"root": 0, "folder": 1, "model": 2, "simulation": 3}
KIND_CODES = {"root": 0, "folder": 1, "model": 2, "simulation": 3}

_TRUE_WORDS  = {".true.", ".t.", "t", "true", "yes", "y", "on"}
_FALSE_WORDS = {".false.", ".f.", "f", "false", "no", "n", "off"}


# ===========================================================================
# SMALL HELPERS
# ===========================================================================

def _natural_key(name):
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r'(\d+)', str(name))]


def _human_size(nbytes):
    try:
        n = float(nbytes)
    except (TypeError, ValueError):
        return ""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            if unit == "B":
                return "%d B" % int(n)
            return "%.1f %s" % (n, unit)
        n /= 1024.0
    return ""


def _mtime(path):
    try:
        return int(path.stat().st_mtime)
    except OSError:
        return 0


def _mtime_str(path):
    stamp = _mtime(path)
    if not stamp:
        return ""
    return _dt.datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M")


def _is_relative_to(path, other):
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _toroidal_from_name(name):
    """Extracts the toroidal mode number from a directory name such as 'n40'."""
    m = re.match(r'^n(\d+)$', str(name))
    if m:
        return int(m.group(1))
    m = re.match(r'^n(\d+)[^0-9]', str(name))
    if m:
        return int(m.group(1))
    return None


# ===========================================================================
# FORTRAN NAMELIST PARSING
# ===========================================================================

_ASSIGN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_%]*(?:\s*\([^)]*\))?)\s*=\s*")


def _strip_comment(line):
    out = []
    quote = None
    for ch in line:
        if quote is not None:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            continue
        if ch in ("!", "#"):
            break
        out.append(ch)
    return "".join(out)


def _quote_spans(line):
    spans = []
    quote = None
    start = 0
    for i, ch in enumerate(line):
        if quote is None:
            if ch in ("'", '"'):
                quote = ch
                start = i
        elif ch == quote:
            spans.append((start, i))
            quote = None
    if quote is not None:
        spans.append((start, len(line)))
    return spans


def _in_spans(pos, spans):
    for a, b in spans:
        if a <= pos <= b:
            return True
    return False


def clean_value(value):
    """Removes surrounding quotes and trailing separators from a namelist value."""
    v = str(value).strip().rstrip(",").strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        v = v[1:-1]
    return v.strip()


def parse_namelist(text):
    """
    Parses Fortran namelist-ish input (M3D-C1 'C1input', NIMROD 'nimrod.in', ...).

    Returns
    -------
    groups : OrderedDict
        {group_name: OrderedDict(key -> raw value string)}
    flat : OrderedDict
        {lowercase key without array index -> cleaned value string}
    """
    groups = OrderedDict()
    current = None
    last = None

    for raw in str(text).splitlines():
        line = _strip_comment(raw).strip()
        if not line:
            continue

        # --- group markers: &group ... / , $group ... $end ---------------
        while line[:1] in ("&", "$"):
            m = re.match(r'[&$]([A-Za-z0-9_]*)\s*', line)
            if not m:
                break
            name = m.group(1)
            line = line[m.end():].strip()
            if name.lower() == "end" or name == "":
                current = None
            else:
                current = name
                groups.setdefault(current, OrderedDict())
            if not line:
                break

        if not line or line in ("/", "&end", "$end"):
            continue
        if line.endswith("/"):
            line = line[:-1].strip()
        if not line:
            continue

        gname = current or "(global)"
        groups.setdefault(gname, OrderedDict())

        spans = _quote_spans(line)
        matches = [m for m in _ASSIGN_RE.finditer(line)
                   if not _in_spans(m.start(), spans)]

        if not matches:
            # Continuation of the previous (array) value.
            if last is not None:
                g, k = last
                groups[g][k] = (groups[g][k] + " " + line.rstrip(",")).strip()
            continue

        if matches[0].start() > 0 and last is not None:
            extra = line[:matches[0].start()].strip().rstrip(",").strip()
            if extra:
                g, k = last
                groups[g][k] = (groups[g][k] + " " + extra).strip()

        for i, m in enumerate(matches):
            key = m.group(1).strip()
            end = matches[i + 1].start() if (i + 1) < len(matches) else len(line)
            val = line[m.end():end].strip().rstrip(",").strip()
            groups[gname][key] = val
            last = (gname, key)

    flat = OrderedDict()
    for gname, kv in groups.items():
        for key, val in kv.items():
            base = key.split("(")[0].strip().lower()
            flat[base] = clean_value(val)

    return groups, flat


# --- typed accessors -------------------------------------------------------

def _raw(params, key):
    v = params.get(key)
    return None if v is None else clean_value(v)


def _first_token(value):
    if value is None:
        return None
    tok = str(value).replace(",", " ").split()
    return tok[0] if tok else None


def _int(params, key):
    tok = _first_token(_raw(params, key))
    if tok is None:
        return None
    try:
        return int(float(tok.replace("d", "e").replace("D", "e")))
    except ValueError:
        return None


def _float(params, key):
    tok = _first_token(_raw(params, key))
    if tok is None:
        return None
    try:
        return float(tok.replace("d", "e").replace("D", "e"))
    except ValueError:
        return None


def _bool(params, key):
    tok = _raw(params, key)
    if tok is None:
        return None
    s = str(tok).strip().lower()
    if s in _TRUE_WORDS:
        return True
    if s in _FALSE_WORDS:
        return False
    try:
        return float(s) != 0.0
    except ValueError:
        return None


def _str(params, key):
    v = _raw(params, key)
    return None if v is None else str(v).strip().lower()


def _yesno(flag):
    if flag is None:
        return "unknown"
    return "yes" if flag else "no"


# ===========================================================================
# PHYSICS MODEL DESCRIPTION
# ===========================================================================

M3DC1_NUMVAR = {
    1: "reduced MHD (numvar=1)",
    2: "two-field MHD (numvar=2)",
    3: "full MHD (numvar=3)",
}

M3DC1_EXTRA_KEYS = [
    ("ntor",           "Toroidal mode number (ntor)"),
    ("nplanes",        "Toroidal planes (nplanes)"),
    ("itor",           "Toroidal geometry (itor)"),
    ("gyro",           "Gyroviscosity (gyro)"),
    ("db_fac",         "Two-fluid parameter (db_fac)"),
    ("kinetic",        "Kinetic particles (kinetic)"),
    ("idens",          "Density evolved (idens)"),
    ("ipres",          "Pressure model (ipres)"),
    ("ipressplit",     "Pressure split (ipressplit)"),
    ("iadapt",         "Adaptive mesh (iadapt)"),
    ("irmp",           "RMP coils (irmp)"),
    ("extsubtract",    "External field subtracted (extsubtract)"),
    ("icsubtract",     "Coil field subtracted (icsubtract)"),
    ("ikprad",         "Impurity radiation (ikprad)"),
    ("iresfunc",       "Resistivity model (iresfunc)"),
    ("etar",           "Resistivity (etar)"),
    ("eta_fac",        "Resistivity factor (eta_fac)"),
    ("amu",            "Viscosity (amu)"),
    ("amupar",         "Parallel viscosity (amupar)"),
    ("denm",           "Density diffusion (denm)"),
    ("kappat",         "Thermal conductivity (kappat)"),
    ("kappar",         "Parallel conductivity (kappar)"),
    ("dt",             "Time step (dt)"),
    ("ntimemax",       "Number of time steps (ntimemax)"),
    ("ntimepr",        "Output frequency (ntimepr)"),
    ("mesh_filename",  "Mesh file"),
    ("gfile",          "Equilibrium (gfile)"),
    ("iread_eqdsk",    "Read EQDSK (iread_eqdsk)"),
    ("iread_dskbal",   "Read dskbal (iread_dskbal)"),
    ("bzero",          "Vacuum field (bzero)"),
    ("rzero",          "Major radius (rzero)"),
]

NIMROD_OHMS = {
    "mhd":       "resistive MHD",
    "hall":      "Hall MHD",
    "mhd&hall":  "MHD + Hall",
    "2fl":       "two-fluid",
}

NIMROD_CONTINUITY = {
    "none":       "no density evolution",
    "fix profile": "fixed density profile",
    "n=0 only":   "n=0 density only",
    "full":       "full continuity",
}

NIMROD_PMODEL = {
    "isothermal":    "isothermal",
    "adiabat":       "adiabatic",
    "aniso1":        "anisotropic conduction",
    "aniso_tdep":    "anisotropic, T-dependent conduction",
    "aniso_plltdep": "anisotropic, T-dependent parallel conduction",
}

NIMROD_EXTRA_KEYS = [
    ("nonlinear",    "Nonlinear"),
    ("ohms",         "Ohm's law (ohms)"),
    ("continuity",   "Continuity model"),
    ("separate_pe",  "Separate electron pressure"),
    ("p_model",      "Pressure/conduction model"),
    ("advect",       "Advection (advect)"),
    ("lphi",         "Toroidal resolution exponent (lphi)"),
    ("nphi",         "Toroidal planes (nphi)"),
    ("zperiod",      "Toroidal periodicity (zperiod)"),
    ("lin_nmax",     "Max linear mode (lin_nmax)"),
    ("nlayers",      "Fourier layers (nlayers)"),
    ("poly_degree",  "Polynomial degree"),
    ("mx",           "Blocks in x (mx)"),
    ("my",           "Blocks in y (my)"),
    ("nxbl",         "Block decomposition (nxbl)"),
    ("nybl",         "Block decomposition (nybl)"),
    ("dt",           "Time step (dt)"),
    ("dtm",          "Maximum time step (dtm)"),
    ("tmax",         "Final time (tmax)"),
    ("nstep",        "Number of steps (nstep)"),
    ("elecd",        "Resistivity / magnetic diffusivity (elecd)"),
    ("eta_model",    "Resistivity model (eta_model)"),
    ("kin_visc",     "Kinematic viscosity (kin_visc)"),
    ("iso_visc",     "Isotropic viscosity (iso_visc)"),
    ("par_visc",     "Parallel viscosity (par_visc)"),
    ("gyr_visc",     "Gyroviscosity (gyr_visc)"),
    ("k_perp",       "Perpendicular conduction (k_perp)"),
    ("k_pll",        "Parallel conduction (k_pll)"),
    ("nd_diff",      "Density diffusivity (nd_diff)"),
    ("neoe_flag",    "Neoclassical electron flag"),
    ("neoi_flag",    "Neoclassical ion flag"),
    ("eq_flow",      "Equilibrium flow (eq_flow)"),
    ("transfer_eq",  "Equilibrium transferred (transfer_eq)"),
    ("dump_name",    "Dump name"),
    ("eqfile",       "Equilibrium file"),
]


def _extra_rows(params, keys):
    rows = []
    for key, label in keys:
        val = _raw(params, key)
        if val is None or val == "":
            continue
        rows.append([label, val])
    return rows


def _describe_m3dc1(params, dir_name=None):
    rows = []
    summary_bits = []

    linear = _int(params, "linear")
    if linear is not None:
        txt = "linear" if linear else "nonlinear"
        rows.append(["Evolution", "%s (linear=%d)" % (txt, linear)])
        summary_bits.append(txt.capitalize())

    two_fluid = _int(params, "itwo_fluid")
    db_fac = _float(params, "db_fac")
    if two_fluid:
        fluid = "two-fluid"
    elif db_fac:
        fluid = "two-fluid"
    else:
        fluid = "single-fluid"
    rows.append(["Fluid model", fluid])
    summary_bits.append(fluid)

    numvar = _int(params, "numvar")
    if numvar is not None:
        model = M3DC1_NUMVAR.get(numvar, "numvar=%d" % numvar)
        rows.append(["MHD model", model])
        summary_bits.append(model)

    eqsub = _int(params, "eqsubtract")
    if eqsub is not None:
        rows.append(["Equilibrium subtracted", _yesno(bool(eqsub))])

    icomplex = _int(params, "icomplex")
    if icomplex is not None:
        rows.append(["Complex (single-n) representation", _yesno(bool(icomplex))])

    ntor = _int(params, "ntor")
    if ntor is None and dir_name is not None:
        ntor = _toroidal_from_name(dir_name)
    if ntor is not None:
        summary_bits.append("n=%d" % ntor)

    rows.extend(_extra_rows(params, M3DC1_EXTRA_KEYS))

    summary = ""
    if summary_bits:
        head = " ".join(summary_bits[:-1]) if len(summary_bits) > 1 else ""
        if summary_bits and summary_bits[-1].startswith("n="):
            summary = (head + ", " + summary_bits[-1]).strip(", ")
        else:
            summary = " ".join(summary_bits)
    return summary, rows


def _describe_nimrod(params, dir_name=None):
    rows = []
    summary_bits = []

    nonlinear = _bool(params, "nonlinear")
    if nonlinear is not None:
        txt = "nonlinear" if nonlinear else "linear"
        rows.append(["Evolution", txt])
        summary_bits.append(txt.capitalize())

    ohms = _str(params, "ohms")
    if ohms:
        desc = NIMROD_OHMS.get(ohms, ohms)
        rows.append(["Ohm's law", "%s (ohms='%s')" % (desc, ohms)])
        summary_bits.append(desc)
    else:
        summary_bits.append("MHD")

    cont = _str(params, "continuity")
    if cont:
        desc = NIMROD_CONTINUITY.get(cont, cont)
        rows.append(["Continuity", "%s (continuity='%s')" % (desc, cont)])
        summary_bits.append(desc)

    sep_pe = _bool(params, "separate_pe")
    if sep_pe is not None:
        rows.append(["Separate electron pressure", _yesno(sep_pe)])
        if sep_pe:
            summary_bits.append("separate p_e")

    pmodel = _str(params, "p_model")
    if pmodel:
        desc = NIMROD_PMODEL.get(pmodel, pmodel)
        rows.append(["Pressure / conduction model", "%s (p_model='%s')" % (desc, pmodel)])
        summary_bits.append(desc)

    ntor = _toroidal_from_name(dir_name) if dir_name else None
    if ntor is not None:
        summary_bits.append("n=%d" % ntor)

    rows.extend(_extra_rows(params, NIMROD_EXTRA_KEYS))

    summary = ", ".join([b for b in summary_bits if b])
    return summary, rows


def describe_simulation(code, params, dir_name=None):
    """
    Returns (summary_string, [[label, value], ...]) describing the physics model.
    """
    if code == "m3dc1":
        return _describe_m3dc1(params, dir_name)
    if code == "nimrod":
        return _describe_nimrod(params, dir_name)
    return "", []


# ===========================================================================
# FILE SYSTEM CRAWLING
# ===========================================================================

def _iter_simulation_dirs(root, code):
    """
    Yields the simulation (mode) directories below *root* for the given code,
    mirroring the rules used by edges_ml.group_simulation_directories().
    """
    marker = SIM_MARKERS.get(code)
    if marker is None:
        return

    try:
        candidates = sorted(Path(root).rglob(marker))
    except OSError:
        candidates = []

    for hit in candidates:
        if any(part.startswith("base_") for part in hit.parts):
            continue
        if any(part.startswith(".") for part in hit.parts[:-1]):
            continue
        mode_dir = hit.parent

        if code == "nimrod":
            if not (mode_dir / "nimrod.in").exists():
                continue
            if not re.match(r'^n(\d+|ln)', mode_dir.name):
                continue

        yield mode_dir


def discover_simulations(sources):
    """
    Returns a sorted list of (code, root_path, simulation_path) tuples.
    """
    out = []
    seen = set()
    for code, root in _normalize_sources(sources):
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        root = root.resolve()
        for sim in _iter_simulation_dirs(root, code):
            key = (code, str(sim))
            if key in seen:
                continue
            seen.add(key)
            out.append((code, root, sim))
    out.sort(key=lambda t: (str(t[1]), _natural_key(str(t[2]))))
    return out


def _normalize_sources(sources):
    """
    Accepts the same source dictionaries as edges_ml.build_dataset, plain
    strings / Paths, or (code, directory) tuples.  Returns [(code, dir), ...].
    """
    norm = []
    if sources is None:
        return norm
    if isinstance(sources, (str, Path)):
        sources = [sources]

    for src in sources:
        if isinstance(src, (str, Path)):
            for code in SIM_MARKERS:
                norm.append((code, Path(src)))
            continue
        if isinstance(src, (tuple, list)) and len(src) == 2:
            code, directory = src
            code = str(code).lower()
            if code in ("auto", "any", "*"):
                for c in SIM_MARKERS:
                    norm.append((c, Path(directory)))
            else:
                norm.append((code, Path(directory)))
            continue
        if isinstance(src, dict):
            code = str(src.get("code", "auto")).lower()
            dirs = src.get("directories") or src.get("dirs") or []
            if isinstance(dirs, (str, Path)):
                dirs = [dirs]
            codes = list(SIM_MARKERS) if code in ("auto", "any", "*") else [code]
            for c in codes:
                for d in dirs:
                    norm.append((c, Path(d)))
            continue
    return norm


# ===========================================================================
# DOCUMENT / PLOT / FILE COLLECTION
# ===========================================================================

def _read_document(path, max_bytes):
    """
    Reads one input file.  Only the raw text is kept - the parameter table is
    produced in the browser, which keeps the embedded payload small.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > HARD_DOC_SIZE_LIMIT:
        return None

    try:
        with open(str(path), "r", errors="replace") as fh:
            text = fh.read(max_bytes + 1)
    except OSError:
        return None

    truncated = len(text) > max_bytes
    if truncated:
        text = text[:max_bytes]

    return {
        "name":      path.name,
        "size":      int(size),
        "truncated": bool(truncated),
        "text":      text,
        "readme":    path.name.lower().startswith("readme"),
    }


def _collect_documents(directory, code, max_bytes, extra_files=()):
    """Collects the input/namelist files of a directory (primary input first)."""
    found = OrderedDict()

    for name in PRIMARY_INPUT.get(code, []):
        p = directory / name
        if p.is_file():
            found[p.name] = p

    for pattern in DOC_PATTERNS.get(code, ["*.in", "README*"]):
        try:
            hits = sorted(directory.glob(pattern), key=lambda q: _natural_key(q.name))
        except OSError:
            hits = []
        for p in hits:
            if not p.is_file():
                continue
            if p.suffix.lower() in SKIP_DOC_SUFFIXES:
                continue
            found.setdefault(p.name, p)

    for p in extra_files:
        p = Path(p)
        if p.is_file():
            found.setdefault(p.name, p)

    docs = []
    for name, path in found.items():
        doc = _read_document(path, max_bytes)
        if doc is not None:
            docs.append(doc)
    return docs


def _primary_params(code, documents):
    """Returns the flat parameter dict of the primary input file (if any)."""
    wanted = [n.lower() for n in PRIMARY_INPUT.get(code, [])]
    ordered = sorted(
        documents,
        key=lambda d: wanted.index(d["name"].lower()) if d["name"].lower() in wanted else 99,
    )
    for doc in ordered:
        if doc.get("readme") or "=" not in doc.get("text", ""):
            continue
        try:
            _groups, flat = parse_namelist(doc["text"])
        except Exception:
            continue
        if flat:
            return flat, doc["name"]
    return OrderedDict(), None


def _collect_plots(directory, output_dir, scan):
    if not scan:
        return []
    plots = []
    seen = set()
    for sub in PLOT_SUBDIRS:
        d = directory / sub if sub else directory
        if not d.is_dir():
            continue
        try:
            entries = sorted(d.iterdir(), key=lambda q: _natural_key(q.name))
        except OSError:
            continue
        for p in entries:
            if not p.is_file() or p.suffix.lower() not in PLOT_SUFFIXES:
                continue
            if str(p) in seen:
                continue
            seen.add(str(p))
            try:
                src = os.path.relpath(str(p), str(output_dir))
            except ValueError:
                src = p.as_uri()
            plots.append({
                "name": (("%s/%s" % (sub, p.name)) if sub else p.name),
                "src":  src.replace(os.sep, "/"),
            })
    return plots


_DIGITS_RE = re.compile(r'\d+')


def _list_files(directory, limit):
    """
    Compact directory listing.

    Rows are ``[name, size, mtime]`` with ``size = -1`` for directories and
    ``size = None`` for informational rows; long numbered series are collapsed
    into a single row ``[pattern, total_size, newest_mtime, count]``.
    """
    if limit is not None and limit <= 0:
        return []

    try:
        entries = sorted(directory.iterdir(), key=lambda q: _natural_key(q.name))
    except OSError:
        return []

    dir_rows = []
    files = []
    for p in entries:
        try:
            st = p.stat()
        except OSError:
            st = None
        stamp = int(st.st_mtime) if st else 0
        if p.is_dir():
            dir_rows.append([p.name + "/", -1, stamp])
        else:
            files.append((p.name, int(st.st_size) if st else 0, stamp))

    groups = OrderedDict()
    for name, size, stamp in files:
        groups.setdefault(_DIGITS_RE.sub("#", name), []).append((name, size, stamp))

    file_rows = []
    for pattern, items in groups.items():
        if len(items) >= COLLAPSE_THRESHOLD:
            file_rows.append([pattern,
                              sum(i[1] for i in items),
                              max(i[2] for i in items),
                              len(items)])
        else:
            for name, size, stamp in items:
                file_rows.append([name, size, stamp])

    rows = dir_rows + file_rows
    if limit is not None and len(rows) > limit:
        hidden = len(rows) - limit
        rows = rows[:limit]
        rows.append(["... (%d more entries not listed)" % hidden, None, 0])
    return rows


def _find_gamma_file(directory):
    """Locates an M3D-C1 'gamma' growth-rate table (same heuristic as the adapter)."""
    try:
        entries = sorted(directory.iterdir(), key=lambda q: _natural_key(q.name))
    except OSError:
        return None
    for f in entries:
        if not f.is_file():
            continue
        if f.suffix.lower() not in ("", ".txt", ".dat", ".out"):
            continue
        try:
            if f.stat().st_size > 4 * 1024 * 1024:
                continue
            with open(str(f), "r", errors="replace") as fh:
                head = "".join([fh.readline() for _ in range(5)])
        except OSError:
            continue
        if "gamma" in head and "sig_gamma" in head:
            return f
    return None


# ===========================================================================
# TREE CONSTRUCTION
# ===========================================================================

def _new_node(kind, name, path):
    return {
        "kind":      kind,
        "name":      str(name),
        "path":      str(path),
        "code":      None,
        "summary":   "",
        "info":      [],
        "physics":   [],
        "documents": [],
        "plots":     [],
        "files":     [],
        "children":  [],
    }


def _ensure_folder_node(path, root_path, root_node, registry):
    if path == root_path:
        return root_node
    node = registry.get(path)
    if node is not None:
        return node
    parent = _ensure_folder_node(path.parent, root_path, root_node, registry)
    node = _new_node("folder", path.name, path)
    node["info"] = [["Last modified", _mtime_str(path)]]
    parent["children"].append(node)
    registry[path] = node
    return node


def _attach_model_info(node, code, opts):
    """Turns a folder node into a model node and attaches its own input files."""
    if node.get("_model_done"):
        return
    node["_model_done"] = True
    if node["kind"] == "folder":
        node["kind"] = "model"
    node["code"] = node.get("code") or code

    directory = Path(node["path"])
    extra = []
    if code == "m3dc1":
        gamma = _find_gamma_file(directory)
        if gamma is not None:
            extra.append(gamma)

    node["documents"] = _collect_documents(directory, code, opts["max_doc_bytes"],
                                           extra_files=extra)
    node["plots"] = _collect_plots(directory, opts["output_dir"], opts["scan_plots"])
    node["files"] = _list_files(directory, opts["max_files"])

    info = [["Last modified", _mtime_str(directory)]]
    if extra:
        info.append(["Growth-rate file", extra[0].name])
    node["info"] = info + [r for r in node["info"] if r[0] != "Last modified"]


def _simulation_node(mode_dir, code, opts):
    node = _new_node("simulation", mode_dir.name, mode_dir)
    node["code"] = code

    docs = _collect_documents(mode_dir, code, opts["max_doc_bytes"])
    node["documents"] = docs
    node["plots"] = _collect_plots(mode_dir, opts["output_dir"], opts["scan_plots"])
    node["files"] = _list_files(mode_dir, opts["max_files"])

    params, primary_name = _primary_params(code, docs)
    summary, physics = describe_simulation(code, params, mode_dir.name)
    node["summary"] = summary
    node["physics"] = physics

    info = [["Last modified", _mtime_str(mode_dir)]]

    ntor = _int(params, "ntor") if code == "m3dc1" else None
    if ntor is None:
        ntor = _toroidal_from_name(mode_dir.name)
    if ntor is not None:
        info.append(["Toroidal mode number", str(ntor)])

    if primary_name:
        info.append(["Primary input file", primary_name])

    marker = SIM_MARKERS.get(code)
    if marker:
        mpath = mode_dir / marker
        if mpath.is_file():
            try:
                info.append([marker, "%s  (%s)" % (_human_size(mpath.stat().st_size),
                                                   _mtime_str(mpath))])
            except OSError:
                pass

    if code == "m3dc1":
        n_time = len(list(mode_dir.glob("time_*.h5")))
        info.append(["Time slice files (time_*.h5)", str(n_time)])
        info.append(["equilibrium.h5", _yesno((mode_dir / "equilibrium.h5").is_file())])
    elif code == "nimrod":
        h5_dumps = [p for p in mode_dir.glob("dumpgll.*.h5")]
        bin_dumps = [p for p in mode_dir.glob("dumpgll.*")
                     if re.match(r'^dumpgll\.\d+$', p.name)]
        info.append(["HDF5 dumps (dumpgll.*.h5)", str(len(h5_dumps))])
        info.append(["Binary dumps (dumpgll.NNNNN)", str(len(bin_dumps))])
        info.append(["energy.bin", _yesno((mode_dir / "energy.bin").is_file())])
        info.append(["discharge.bin", _yesno((mode_dir / "discharge.bin").is_file())])

    node["info"] = info
    return node


def _finalize(node):
    """Recursively sorts children and counts simulations / models."""
    sims = 0
    models = 0
    for child in node["children"]:
        s, m = _finalize(child)
        sims += s
        models += m

    if node["kind"] == "simulation":
        sims += 1
    if node["kind"] == "model":
        models += 1

    node["children"].sort(key=lambda c: (KIND_ORDER.get(c["kind"], 9),
                                         _natural_key(c["name"])))
    node.pop("_model_done", None)
    return sims, models


def _build_tree(sources, opts):
    registry = {}
    roots = []
    seen = set()
    stats = {"roots": 0, "models": 0, "simulations": 0, "by_code": {}}

    for code, raw_root in _normalize_sources(sources):
        root_path = Path(raw_root).expanduser()
        if not root_path.is_dir():
            if not opts["quiet"]:
                print("warning: directory not found, skipping: %s" % root_path)
            continue
        root_path = root_path.resolve()

        root_node = registry.get(root_path)
        if root_node is None:
            root_node = _new_node("root", str(root_path), root_path)
            root_node["info"] = [["Last modified", _mtime_str(root_path)]]
            registry[root_path] = root_node
            roots.append(root_node)

        for mode_dir in _iter_simulation_dirs(root_path, code):
            key = (code, str(mode_dir))
            if key in seen:
                continue
            seen.add(key)

            model_path = mode_dir.parent
            if model_path == root_path or not _is_relative_to(model_path, root_path):
                model_node = root_node
            else:
                model_node = _ensure_folder_node(model_path, root_path, root_node,
                                                 registry)
            _attach_model_info(model_node, code, opts)

            sim_node = _simulation_node(mode_dir, code, opts)
            model_node["children"].append(sim_node)

            stats["simulations"] += 1
            stats["by_code"][code] = stats["by_code"].get(code, 0) + 1

    for root_node in roots:
        _sims, models = _finalize(root_node)
        stats["models"] += models

    stats["roots"] = len(roots)
    return roots, stats


# ===========================================================================
# PAYLOAD PACKING
# ===========================================================================

def _pack_payload(roots, stats, title):
    """
    Converts the rich node tree into the compact structure that is embedded in
    the page (see the module docstring for the rationale).
    """
    labels = []
    label_index = {}
    texts = []
    text_index = {}
    nodes = []

    def label_id(text):
        key = str(text)
        idx = label_index.get(key)
        if idx is None:
            idx = len(labels)
            labels.append(key)
            label_index[key] = idx
        return idx

    def text_id(text):
        key = text if isinstance(text, str) else str(text)
        idx = text_index.get(key)
        if idx is None:
            idx = len(texts)
            texts.append(key)
            text_index[key] = idx
        return idx

    def pack_rows(rows):
        return [[label_id(r[0]), r[1]] for r in rows if r[1] not in (None, "")]

    def pack_doc(doc):
        entry = [doc["name"], text_id(doc.get("text", "")), int(doc.get("size", 0))]
        if doc.get("truncated"):
            entry.append(1)
        return entry

    def walk(node, parent):
        index = len(nodes)
        entry = [
            parent,
            KIND_CODES.get(node["kind"], 1),
            node["path"] if node["kind"] == "root" else node["name"],
            CODE_ORDER.index(node["code"]) if node.get("code") in CODE_ORDER else 0,
            node.get("summary") or "",
            pack_rows(node.get("info") or []),
            pack_rows(node.get("physics") or []),
            [pack_doc(d) for d in (node.get("documents") or [])],
            [[p["name"], p["src"]] for p in (node.get("plots") or [])],
            node.get("files") or [],
        ]
        while len(entry) > 3 and not entry[-1]:
            entry.pop()
        nodes.append(entry)
        for child in node["children"]:
            walk(child, index)

    for root in roots:
        walk(root, -1)

    return {
        "v":           2,
        "title":       str(title),
        "generated":   _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sep":         os.sep,
        "codes":       CODE_ORDER,
        "code_labels": CODE_LABELS,
        "stats":       stats,
        "labels":      labels,
        "texts":       texts,
        "nodes":       nodes,
    }


def _encode_payload(payload, compress):
    """Returns (embedded_text, encoding_name, raw_bytes, encoded_bytes)."""
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    raw_bytes = len(raw.encode("utf-8"))

    if not compress:
        safe = raw.replace("<", "\\u003c").replace("&", "\\u0026")
        return safe, "json", raw_bytes, len(safe.encode("utf-8"))

    blob = gzip.compress(raw.encode("utf-8"), 9)
    b64 = base64.b64encode(blob).decode("ascii")
    wrapped = "\n".join(b64[i:i + 120] for i in range(0, len(b64), 120))
    return wrapped, "gzip+base64", raw_bytes, len(wrapped)


# ===========================================================================
# HTML RENDERING
# ===========================================================================

_STYLE = """
:root{
  --bg:#f5f6f8; --panel:#ffffff; --line:#d8dce3; --text:#1d2430; --muted:#6b7280;
  --accent:#1f6feb; --accent-soft:#e8f0fe; --m3dc1:#8250df; --nimrod:#0f7b6c;
  --header-bg:#ffffff; --sidebar-bg:#ffffff; --input-bg:#ffffff;
  --chip-bg:#ffffff; --doctab-bg:#ffffff; --pre-bg:#0f172a; --pre-text:#e2e8f0;
  --note-bg:#fff8e1; --note-border:#f0d99b; --note-text:#7a5b00;
  --gallery-bg:#ffffff;
}
[data-theme="dark"]{
  --bg:#1e2128; --panel:#262b35; --line:#3a4050; --text:#d4d8e0; --muted:#8b95a8;
  --accent:#4d9ef7; --accent-soft:#1a2d45; --m3dc1:#b08aff; --nimrod:#3ecfb0;
  --header-bg:#262b35; --sidebar-bg:#262b35; --input-bg:#2e3340;
  --chip-bg:#2e3340; --doctab-bg:#2e3340; --pre-bg:#0d1117; --pre-text:#e2e8f0;
  --note-bg:#2e2a1a; --note-border:#6b5a1e; --note-text:#e0c97a;
  --gallery-bg:#2e3340;
}
*{box-sizing:border-box;}
html,body{height:100%;margin:0;}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
     font-size:14px;color:var(--text);background:var(--bg);}
code,pre,.mono{font-family:"SFMono-Regular",Menlo,Consolas,"Liberation Mono",monospace;}
header{display:flex;align-items:center;gap:16px;flex-wrap:wrap;
       padding:10px 18px;background:var(--header-bg);border-bottom:1px solid var(--line);}
header h1{font-size:16px;margin:0;font-weight:600;}
header .stats{color:var(--muted);font-size:12px;}
header .spacer{flex:1;}
header input[type=search]{padding:6px 10px;border:1px solid var(--line);border-radius:6px;
       width:260px;font-size:13px;background:var(--input-bg);color:var(--text);}
header button{padding:5px 10px;border:1px solid var(--line);border-radius:6px;
       background:var(--chip-bg);color:var(--text);
       cursor:pointer;font-size:12px;}
header button:hover{background:var(--accent-soft);border-color:var(--accent);}
#layout{display:flex;height:calc(100vh - 49px);}
#sidebar{width:380px;min-width:240px;max-width:60%;overflow:auto;background:var(--sidebar-bg);
         border-right:1px solid var(--line);padding:10px 6px 40px 6px;resize:horizontal;}
#detail{flex:1;overflow:auto;padding:18px 24px 60px 24px;}
ul.tree,ul.tree ul{list-style:none;margin:0;padding-left:14px;}
ul.tree{padding-left:2px;}
.row{display:flex;align-items:center;gap:6px;padding:2px 6px;border-radius:5px;cursor:pointer;
     white-space:nowrap;}
.row:hover{background:var(--accent-soft);}
.row.selected{background:var(--accent);color:#fff;}
.row.selected .count,.row.selected .badge{background:rgba(255,255,255,.25);color:#fff;}
.toggle{width:12px;display:inline-block;color:var(--muted);font-size:11px;}
.row.selected .toggle{color:#fff;}
.icon{width:16px;text-align:center;}
.label{overflow:hidden;text-overflow:ellipsis;}
.count{font-size:11px;background:#eceff4;color:var(--muted);border-radius:9px;padding:0 6px;}
.badge{font-size:11px;border-radius:9px;padding:1px 7px;background:#eceff4;color:var(--muted);}
[data-theme="dark"] .count,[data-theme="dark"] .badge{background:#3a4050;color:var(--muted);}
.badge.code-m3dc1{background:#f1e9ff;color:var(--m3dc1);}
.badge.code-nimrod{background:#e4f5f1;color:var(--nimrod);}
[data-theme="dark"] .badge.code-m3dc1{background:#2d1f4a;color:var(--m3dc1);}
[data-theme="dark"] .badge.code-nimrod{background:#0f2e28;color:var(--nimrod);}
.detail-head{margin-bottom:14px;}
.detail-head h1{font-size:22px;margin:4px 0 6px 0;}
.crumbs{font-size:12px;color:var(--muted);}
.crumbs span{cursor:pointer;}
.crumbs span:hover{text-decoration:underline;color:var(--accent);}
.crumbs .sep{cursor:default;padding:0 4px;}
.badges{display:flex;gap:8px;flex-wrap:wrap;margin:4px 0;}
.summary{font-size:15px;margin:8px 0 2px 0;}
.path{font-size:12px;color:var(--muted);word-break:break-all;margin:2px 0 0 0;}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin:14px 0 0 0;}
.tabs button{border:1px solid transparent;border-bottom:none;background:none;padding:7px 14px;
             cursor:pointer;font-size:13px;color:var(--muted);border-radius:6px 6px 0 0;}
.tabs button.active{background:var(--panel);border-color:var(--line);color:var(--text);
                    font-weight:600;margin-bottom:-1px;}
.panel{background:var(--panel);border:1px solid var(--line);border-top:none;
       border-radius:0 0 8px 8px;padding:16px;}
h2.section{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
           margin:18px 0 8px 0;}
h2.section:first-child{margin-top:0;}
table.kv{border-collapse:collapse;width:100%;margin-bottom:6px;}
table.kv td,table.kv th{border-bottom:1px solid var(--line);padding:5px 8px;
                        vertical-align:top;text-align:left;font-size:13px;}
table.kv td:first-child{color:var(--muted);width:34%;white-space:nowrap;}
table.kv th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em;}
table.kv td.v{font-family:"SFMono-Regular",Menlo,Consolas,monospace;word-break:break-all;}
.chips{display:flex;flex-wrap:wrap;gap:6px;}
.chip{border:1px solid var(--line);border-radius:14px;padding:3px 10px;cursor:pointer;
      background:var(--chip-bg);color:var(--text);font-size:12px;}
.chip:hover{border-color:var(--accent);background:var(--accent-soft);}
.docbar{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;}
.doctab{border:1px solid var(--line);background:var(--doctab-bg);color:var(--text);
        border-radius:6px;padding:4px 10px;cursor:pointer;font-size:12px;}
.doctab.active{background:var(--accent);border-color:var(--accent);color:#fff;}
pre.raw{background:var(--pre-bg);color:var(--pre-text);padding:12px;border-radius:6px;
        overflow:auto;max-height:70vh;font-size:12px;line-height:1.45;}
.muted{color:var(--muted);}
.note{background:var(--note-bg);border:1px solid var(--note-border);border-radius:6px;
      padding:12px;font-size:13px;color:var(--note-text);}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;}
.gallery figure{margin:0;border:1px solid var(--line);border-radius:8px;overflow:hidden;
                background:var(--gallery-bg);}
.gallery img{width:100%;display:block;background:var(--gallery-bg);}
.gallery figcaption{font-size:12px;padding:6px 8px;color:var(--muted);word-break:break-all;}
.filter{margin:0 0 8px 0;padding:5px 9px;border:1px solid var(--line);border-radius:6px;
        width:260px;font-size:12px;background:var(--input-bg);color:var(--text);}
footer{padding:10px 18px;color:var(--muted);font-size:11px;}
"""

_SCRIPT = r"""
(function () {
  "use strict";

  /* --------------------------------------------------------- dark mode */

  var THEME_KEY = "edges-ml-theme";

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    var btn = document.getElementById("theme-toggle");
    if (btn) { btn.textContent = theme === "dark" ? "\u2600\uFE0F Light" : "\uD83C\uDF19 Dark"; }
  }

  function toggleTheme() {
    var cur = document.documentElement.getAttribute("data-theme") || "light";
    var next = cur === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem(THEME_KEY, next); } catch (e) {}
  }

  (function initTheme() {
    var saved = null;
    try { saved = localStorage.getItem(THEME_KEY); } catch (e) {}
    var preferred = saved || ((window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches) ? "dark" : "light");
    applyTheme(preferred);
  })();

  document.getElementById("theme-toggle").addEventListener("click", toggleTheme);

  /* ------------------------------------------------------------ helpers */

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) { e.className = cls; }
    if (text !== undefined && text !== null) { e.textContent = text; }
    return e;
  }

  function humanSize(n) {
    if (n === null || n === undefined || n < 0) { return ""; }
    var units = ["B", "KB", "MB", "GB", "TB"], i = 0, v = Number(n);
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
    return (i === 0 ? String(Math.round(v)) : v.toFixed(1)) + " " + units[i];
  }

  function fmtTime(t) {
    if (!t) { return ""; }
    var d = new Date(t * 1000);
    function p(x) { return (x < 10 ? "0" : "") + x; }
    return d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate()) +
           " " + p(d.getHours()) + ":" + p(d.getMinutes());
  }

  function message(text, isError) {
    var host = document.getElementById("detail");
    host.innerHTML = "";
    host.appendChild(el("p", isError ? "note" : "muted", text));
  }

  /* ----------------------------------------------- payload (lazy inflate) */

  function base64ToBytes(b64) {
    var bin = atob(String(b64).replace(/\s+/g, ""));
    var out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) { out[i] = bin.charCodeAt(i); }
    return out;
  }

  function loadPayload() {
    var holder = document.getElementById("edges-data");
    var enc = holder.getAttribute("data-encoding") || "json";
    if (enc !== "gzip+base64") {
      return Promise.resolve(JSON.parse(holder.textContent));
    }
    if (typeof DecompressionStream === "undefined") {
      return Promise.reject(new Error(
        "This browser cannot inflate the embedded data (DecompressionStream is " +
        "missing). Use a recent Firefox/Chrome/Edge/Safari, or regenerate the " +
        "page with --no-compress."));
    }
    var stream = new Blob([base64ToBytes(holder.textContent)]).stream()
      .pipeThrough(new DecompressionStream("gzip"));
    return new Response(stream).text().then(function (txt) { return JSON.parse(txt); });
  }

  /* ------------------------------------------- namelist parser (client) */

  function stripComment(line) {
    var out = "", quote = null;
    for (var i = 0; i < line.length; i++) {
      var ch = line.charAt(i);
      if (quote !== null) { out += ch; if (ch === quote) { quote = null; } continue; }
      if (ch === "'" || ch === '"') { quote = ch; out += ch; continue; }
      if (ch === "!" || ch === "#") { break; }
      out += ch;
    }
    return out;
  }

  function quoteSpans(line) {
    var spans = [], quote = null, start = 0;
    for (var i = 0; i < line.length; i++) {
      var ch = line.charAt(i);
      if (quote === null) {
        if (ch === "'" || ch === '"') { quote = ch; start = i; }
      } else if (ch === quote) { spans.push([start, i]); quote = null; }
    }
    if (quote !== null) { spans.push([start, line.length]); }
    return spans;
  }

  function inSpans(pos, spans) {
    for (var i = 0; i < spans.length; i++) {
      if (spans[i][0] <= pos && pos <= spans[i][1]) { return true; }
    }
    return false;
  }

  function cleanValue(value) {
    var v = String(value).trim().replace(/,+$/, "").trim();
    if (v.length >= 2 && v.charAt(0) === v.charAt(v.length - 1) &&
        (v.charAt(0) === "'" || v.charAt(0) === '"')) {
      v = v.slice(1, -1);
    }
    return v.trim();
  }

  var ASSIGN_RE = /([A-Za-z_][A-Za-z0-9_%]*(?:\s*\([^)]*\))?)\s*=\s*/g;
  var GROUP_RE = /^[&$]([A-Za-z0-9_]*)\s*/;

  function parseNamelist(text) {
    var rows = [], index = {}, current = null, last = null;
    var lines = String(text).split(/\r?\n/);

    function put(group, key, value) {
      var k = group + "\u0000" + key;
      if (index[k] === undefined) { index[k] = rows.length; rows.push([key, value, group]); }
      else { rows[index[k]][1] = value; }
      last = k;
    }

    function append(extra) {
      if (last === null || index[last] === undefined || !extra) { return; }
      var r = rows[index[last]];
      r[1] = (r[1] + " " + extra).trim();
    }

    for (var li = 0; li < lines.length; li++) {
      var line = stripComment(lines[li]).trim();
      if (!line) { continue; }

      while (line.charAt(0) === "&" || line.charAt(0) === "$") {
        var gm = GROUP_RE.exec(line);
        if (!gm) { break; }
        var gname = gm[1];
        line = line.slice(gm[0].length).trim();
        current = (gname === "" || gname.toLowerCase() === "end") ? null : gname;
        if (!line) { break; }
      }

      if (!line || line === "/" || line === "&end" || line === "$end") { continue; }
      if (line.charAt(line.length - 1) === "/") { line = line.slice(0, -1).trim(); }
      if (!line) { continue; }

      var group = current || "(global)";
      var spans = quoteSpans(line);
      var hits = [], m;
      ASSIGN_RE.lastIndex = 0;
      while ((m = ASSIGN_RE.exec(line)) !== null) {
        if (!inSpans(m.index, spans)) {
          hits.push({start: m.index, end: ASSIGN_RE.lastIndex, key: m[1].trim()});
        }
      }

      if (!hits.length) { append(line.replace(/,+$/, "").trim()); continue; }
      if (hits[0].start > 0) {
        append(line.slice(0, hits[0].start).trim().replace(/,+$/, "").trim());
      }
      for (var h = 0; h < hits.length; h++) {
        var stop = (h + 1 < hits.length) ? hits[h + 1].start : line.length;
        put(group, hits[h].key,
            line.slice(hits[h].end, stop).trim().replace(/,+$/, "").trim());
      }
    }

    return rows.map(function (r) { return [r[0], cleanValue(r[1]), r[2]]; });
  }

  /* ---------------------------------------------------------------- boot */

  message("Loading simulation index\u2026");
  loadPayload().then(boot, function (err) {
    message(String((err && err.message) || err), true);
  });

  function boot(DATA) {
    var SEP = DATA.sep || "/";
    var LABELS = DATA.labels || [];
    var TEXTS = DATA.texts || [];
    var CODES = DATA.codes || [""];
    var KINDS = ["root", "folder", "model", "simulation"];

    var NODES = [];
    (DATA.nodes || []).forEach(function (r, i) {
      var parent = r[0];
      var node = {
        id: i,
        parent: parent,
        kind: KINDS[r[1] || 0] || "folder",
        name: r[2] || "",
        code: CODES[r[3] || 0] || null,
        summary: r[4] || "",
        info: r[5] || [],
        physics: r[6] || [],
        docs: r[7] || [],
        plots: r[8] || [],
        files: r[9] || [],
        children: [],
        sims: 0,
        models: 0,
        depth: 0
      };
      if (parent >= 0) {
        node.path = NODES[parent].path + SEP + node.name;
        node.depth = NODES[parent].depth + 1;
        NODES[parent].children.push(node);
      } else {
        node.path = node.name;
      }
      NODES.push(node);
    });

    for (var i = NODES.length - 1; i >= 0; i--) {
      var n = NODES[i];
      if (n.kind === "simulation") { n.sims += 1; }
      if (n.kind === "model") { n.models += 1; }
      if (n.parent >= 0) {
        NODES[n.parent].sims += n.sims;
        NODES[n.parent].models += n.models;
      }
    }

    var ROOTS = NODES.filter(function (x) { return x.parent < 0; });
    var expanded = {}, current = null, activeTab = "Overview";

    NODES.forEach(function (x) {
      expanded[x.id] = (x.kind === "root") || (x.depth <= 1 && x.kind === "folder");
    });

    function parentOf(node) {
      return node.parent >= 0 ? NODES[node.parent] : null;
    }

    function codeLabel(code) {
      return (DATA.code_labels && DATA.code_labels[code]) ? DATA.code_labels[code] : code;
    }

    function kindLabel(kind) {
      return {root: "Root directory", folder: "Directory",
              model: "Model", simulation: "Simulation"}[kind] || kind;
    }

    function iconFor(node) {
      if (node.kind === "simulation") { return "\u25CF"; }
      if (node.kind === "model") { return "\u25A3"; }
      return "\u25B0";
    }

    /* a model/folder inherits the description of its first simulation */
    function reference(node) {
      if (node._ref !== undefined) { return node._ref; }
      var found = null;
      if (node.kind === "simulation") {
        found = node;
      } else {
        for (var k = 0; k < node.children.length; k++) {
          var cand = reference(node.children[k]);
          if (cand && (cand.summary || cand.physics.length)) { found = cand; break; }
        }
      }
      node._ref = found;
      return found;
    }

    function summaryOf(node) {
      if (node.summary) { return node.summary; }
      var ref = reference(node);
      return ref ? ref.summary : "";
    }

    /* ---------------------------------------------------------- tree */

    function query() {
      return document.getElementById("search").value.trim().toLowerCase();
    }

    function matches(node, q) {
      if (!q) { return true; }
      if (node._hay === undefined) {
        node._hay = (node.name + " " + node.path + " " + summaryOf(node)).toLowerCase();
      }
      return node._hay.indexOf(q) >= 0;
    }

    function subtreeMatches(node, q) {
      if (matches(node, q)) { return true; }
      for (var k = 0; k < node.children.length; k++) {
        if (subtreeMatches(node.children[k], q)) { return true; }
      }
      return false;
    }

    function isOpen(node, q) {
      return q ? true : !!expanded[node.id];
    }

    function renderTree() {
      var q = query();
      var host = document.getElementById("tree");
      host.innerHTML = "";
      var ul = el("ul", "tree");
      ROOTS.forEach(function (node) {
        var li = renderNode(node, q);
        if (li) { ul.appendChild(li); }
      });
      if (!ul.childNodes.length) {
        host.appendChild(el("p", "muted", "No entries match the current filter."));
        return;
      }
      host.appendChild(ul);
    }

    function renderNode(node, q) {
      if (q && !subtreeMatches(node, q)) { return null; }

      var li = el("li");
      var kids = node.children;
      var row = el("div", "row kind-" + node.kind);
      if (current === node.id) { row.className += " selected"; }

      var tog = el("span", "toggle", kids.length ? (isOpen(node, q) ? "\u25BE" : "\u25B8") : "");
      tog.addEventListener("click", function (ev) {
        ev.stopPropagation();
        if (!kids.length) { return; }
        expanded[node.id] = !isOpen(node, q);
        renderTree();
      });
      row.appendChild(tog);
      row.appendChild(el("span", "icon", iconFor(node)));

      var label = el("span", "label", node.name);
      label.title = node.path;
      row.appendChild(label);

      if (node.kind === "simulation" && node.code) {
        row.appendChild(el("span", "badge code-" + node.code, codeLabel(node.code)));
      } else if (node.sims) {
        row.appendChild(el("span", "count", String(node.sims)));
      }

      row.addEventListener("click", function () { select(node.id); });
      li.appendChild(row);

      if (kids.length && isOpen(node, q)) {
        var ul = el("ul");
        kids.forEach(function (c) {
          var x = renderNode(c, q);
          if (x) { ul.appendChild(x); }
        });
        li.appendChild(ul);
      }
      return li;
    }

    function select(id) {
      var node = NODES[id];
      if (!node) { return; }
      current = id;
      var p = parentOf(node);
      while (p) { expanded[p.id] = true; p = parentOf(p); }
      renderTree();
      renderDetail(node);
      if (window.history && window.history.replaceState) {
        window.history.replaceState(null, "", "#n" + id);
      }
      var sel = document.querySelector(".row.selected");
      if (sel && sel.scrollIntoView) { sel.scrollIntoView({block: "nearest"}); }
    }

    /* -------------------------------------------------------- detail */

    function kvTable(rows, headers) {
      var t = el("table", "kv");
      if (headers) {
        var thead = el("thead"), htr = el("tr");
        headers.forEach(function (h) { htr.appendChild(el("th", null, h)); });
        thead.appendChild(htr);
        t.appendChild(thead);
      }
      var tb = el("tbody");
      rows.forEach(function (r) {
        var tr = el("tr");
        r.forEach(function (cell, i) {
          tr.appendChild(el("td", i === 0 ? null : "v",
                            String(cell === null || cell === undefined ? "" : cell)));
        });
        tb.appendChild(tr);
      });
      t.appendChild(tb);
      return t;
    }

    function crumbs(node) {
      var chain = [], p = node;
      while (p) { chain.unshift(p); p = parentOf(p); }
      var box = el("div", "crumbs");
      chain.forEach(function (x, i) {
        if (i) { box.appendChild(el("span", "sep", "/")); }
        var s = el("span", null, x.name);
        s.addEventListener("click", function () { select(x.id); });
        box.appendChild(s);
      });
      return box;
    }

    function infoRows(node) {
      var rows = [];
      if (node.code) { rows.push(["Code", codeLabel(node.code)]); }
      rows.push([node.kind === "root" ? "Root directory" : "Path", node.path]);
      node.info.forEach(function (r) { rows.push([LABELS[r[0]] || "", r[1]]); });
      if (node.kind !== "simulation") {
        rows.push(["Simulations below", String(node.sims)]);
        if (node.models) { rows.push(["Models below", String(node.models)]); }
      }
      return rows;
    }

    function fileRows(node) {
      return node.files.map(function (f) {
        var size = f[1], count = f[3], text;
        if (count) { text = count + " files, " + humanSize(size); }
        else if (size === null || size === undefined) { text = ""; }
        else if (size < 0) { text = "directory"; }
        else { text = humanSize(size); }
        return [f[0], text, fmtTime(f[2])];
      });
    }

    function overviewPanel(node) {
      var panel = el("div", "panel");

      panel.appendChild(el("h2", "section", "Metadata"));
      panel.appendChild(kvTable(infoRows(node)));

      var phys = node.physics, from = null;
      if (!phys.length) {
        var ref = reference(node);
        if (ref && ref.physics.length) { phys = ref.physics; from = ref; }
      }
      if (phys.length) {
        panel.appendChild(el("h2", "section",
          from ? ("Physical model (from " + from.name + ")") : "Physical model"));
        panel.appendChild(kvTable(phys.map(function (r) {
          return [LABELS[r[0]] || "", r[1]];
        })));
      }

      if (node.children.length) {
        panel.appendChild(el("h2", "section", "Contents"));
        var chips = el("div", "chips");
        node.children.forEach(function (c) {
          var txt = c.name;
          if (c.kind !== "simulation" && c.sims) { txt += "  (" + c.sims + ")"; }
          var b = el("button", "chip", txt);
          b.title = summaryOf(c) || c.path;
          b.addEventListener("click", function () { select(c.id); });
          chips.appendChild(b);
        });
        panel.appendChild(chips);
      }

      if (node.files.length) {
        panel.appendChild(el("h2", "section", "Files in this directory"));
        panel.appendChild(kvTable(fileRows(node), ["Name", "Size", "Modified"]));
      }
      return panel;
    }

    function docParams(d) {
      if (d._p === undefined) {
        d._p = /^readme/i.test(d[0]) ? [] : parseNamelist(TEXTS[d[1]] || "");
      }
      return d._p;
    }

    function inputsPanel(node) {
      var panel = el("div", "panel");
      var docs = node.docs;
      if (!docs.length) {
        panel.appendChild(el("p", "muted",
          "No input files were found in this directory."));
        return panel;
      }

      var state = {idx: 0, raw: false, filter: ""};

      function draw() {
        panel.innerHTML = "";

        var bar = el("div", "docbar");
        docs.forEach(function (d, i) {
          var b = el("button", "doctab" + (i === state.idx ? " active" : ""), d[0]);
          b.title = node.path + SEP + d[0];
          b.addEventListener("click", function () {
            state.idx = i; state.filter = ""; draw();
          });
          bar.appendChild(b);
        });
        panel.appendChild(bar);

        var doc = docs[state.idx];
        var text = TEXTS[doc[1]] || "";

        panel.appendChild(el("p", "muted",
          node.path + SEP + doc[0] + "  \u2022  " + humanSize(doc[2] || 0) +
          (doc[3] ? "  \u2022  (display truncated)" : "")));

        var toggle = el("button", "chip",
          state.raw ? "Show parsed parameters" : "Show raw file");
        toggle.addEventListener("click", function () { state.raw = !state.raw; draw(); });
        panel.appendChild(toggle);

        var params = state.raw ? [] : docParams(doc);

        if (state.raw || !params.length) {
          var pre = el("pre", "raw", text);
          panel.appendChild(pre);
          if (!state.raw && !params.length) {
            panel.insertBefore(el("p", "muted",
              "No key = value assignments were detected; showing the raw file."), pre);
          }
          return;
        }

        var filt = el("input", "filter");
        filt.type = "search";
        filt.placeholder = "Filter parameters\u2026";
        filt.value = state.filter;
        filt.addEventListener("input", function () {
          state.filter = filt.value.toLowerCase();
          drawTable();
        });
        panel.appendChild(filt);

        var holder = el("div");
        panel.appendChild(holder);

        function drawTable() {
          holder.innerHTML = "";
          var rows = params.filter(function (r) {
            if (!state.filter) { return true; }
            return (r[0] + " " + r[1] + " " + r[2]).toLowerCase()
                     .indexOf(state.filter) >= 0;
          });
          if (!rows.length) {
            holder.appendChild(el("p", "muted", "No parameter matches the filter."));
            return;
          }
          holder.appendChild(kvTable(rows, ["Parameter", "Value", "Namelist group"]));
        }
        drawTable();
      }

      draw();
      return panel;
    }

    function plotsPanel(node) {
      var panel = el("div", "panel");
      if (!node.plots.length) {
        var note = el("div", "note");
        note.textContent = "No plots are attached to this entry yet. " +
          "Plot support is prepared: regenerate the browser with --scan-plots once " +
          "figures (png/svg/jpg) have been placed in the simulation directory or in " +
          "a 'plots/' subdirectory.";
        panel.appendChild(note);
        return panel;
      }
      var gal = el("div", "gallery");
      node.plots.forEach(function (p) {
        var name = p[0], src = p[1];
        var fig = el("figure");
        if (/\.pdf$/i.test(src)) {
          var a = el("a", null, name);
          a.href = src;
          a.target = "_blank";
          fig.appendChild(a);
        } else {
          var img = document.createElement("img");
          img.src = src;
          img.alt = name;
          img.loading = "lazy";
          var link = document.createElement("a");
          link.href = src;
          link.target = "_blank";
          link.appendChild(img);
          fig.appendChild(link);
        }
        fig.appendChild(el("figcaption", null, name));
        gal.appendChild(fig);
      });
      panel.appendChild(gal);
      return panel;
    }

    function renderDetail(node) {
      var host = document.getElementById("detail");
      host.innerHTML = "";
      if (!node) {
        host.appendChild(el("p", "muted", "Select an entry in the tree."));
        return;
      }

      var head = el("div", "detail-head");
      head.appendChild(crumbs(node));
      head.appendChild(el("h1", null, node.name));

      var badges = el("div", "badges");
      badges.appendChild(el("span", "badge", kindLabel(node.kind)));
      if (node.code) {
        badges.appendChild(el("span", "badge code-" + node.code, codeLabel(node.code)));
      }
      if (node.sims && node.kind !== "simulation") {
        badges.appendChild(el("span", "badge",
          node.sims + " simulation" + (node.sims === 1 ? "" : "s")));
      }
      head.appendChild(badges);

      var sum = summaryOf(node);
      if (sum) { head.appendChild(el("p", "summary", sum)); }
      head.appendChild(el("p", "path", node.path));
      host.appendChild(head);

      var names = ["Overview", "Inputs", "Plots"];
      var counts = {
        "Overview": "",
        "Inputs": node.docs.length ? " (" + node.docs.length + ")" : "",
        "Plots": node.plots.length ? " (" + node.plots.length + ")" : ""
      };
      if (names.indexOf(activeTab) < 0) { activeTab = "Overview"; }

      var tabs = el("div", "tabs");
      names.forEach(function (nm) {
        var b = el("button", nm === activeTab ? "active" : null, nm + counts[nm]);
        b.addEventListener("click", function () { activeTab = nm; renderDetail(node); });
        tabs.appendChild(b);
      });
      host.appendChild(tabs);

      if (activeTab === "Inputs") { host.appendChild(inputsPanel(node)); }
      else if (activeTab === "Plots") { host.appendChild(plotsPanel(node)); }
      else { host.appendChild(overviewPanel(node)); }
    }

    /* ------------------------------------------------------ controls */

    function setAll(flag) {
      NODES.forEach(function (x) { expanded[x.id] = flag; });
      renderTree();
    }

    document.getElementById("search").addEventListener("input", renderTree);
    document.getElementById("expand-all").addEventListener("click", function () {
      setAll(true);
    });
    document.getElementById("collapse-all").addEventListener("click", function () {
      setAll(false);
      ROOTS.forEach(function (x) { expanded[x.id] = true; });
      renderTree();
    });

    renderTree();
    var hash = (window.location.hash || "").replace("#n", "");
    var start = parseInt(hash, 10);
    if (!isNaN(start) && NODES[start]) { select(start); }
    else if (NODES.length) { select(0); }
    else { renderDetail(null); }
  }
})();
"""

_PAGE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>$style</style>
</head>
<body>
<header>
  <h1>$title</h1>
  <span class="stats">$stats</span>
  <span class="spacer"></span>
  <input id="search" type="search" placeholder="Filter by name, path or model...">
  <button id="expand-all" type="button">Expand all</button>
  <button id="collapse-all" type="button">Collapse all</button>
  <button id="theme-toggle" type="button">&#127769; Dark</button>
</header>
<div id="layout">
  <nav id="sidebar"><div id="tree"></div></nav>
  <main id="detail"></main>
</div>
<footer>Generated by edges-ml on $generated &mdash; re-run the generator after adding new data.</footer>
<script id="edges-data" type="application/octet-stream" data-encoding="$encoding">$data</script>
<script>$script</script>
</body>
</html>
""")


def _stats_line(stats):
    bits = ["%d root director%s" % (stats["roots"], "y" if stats["roots"] == 1 else "ies"),
            "%d model(s)" % stats["models"],
            "%d simulation(s)" % stats["simulations"]]
    for code, n in sorted(stats.get("by_code", {}).items()):
        bits.append("%d %s" % (n, CODE_LABELS.get(code, code)))
    return "  \u2022  ".join(bits)


def render_html(payload, title, compress=True):
    """Renders the complete self-contained page for a packed *payload*."""
    text, encoding, _raw_bytes, _enc_bytes = _encode_payload(payload, compress)
    return _PAGE.safe_substitute(
        title=_html.escape(str(title)),
        style=_STYLE,
        script=_SCRIPT,
        stats=_html.escape(_stats_line(payload["stats"])),
        generated=_html.escape(payload["generated"]),
        encoding=encoding,
        data=text,
    )


# ===========================================================================
# PUBLIC API
# ===========================================================================

def build_browser(sources,
                  output="simulations.html",
                  title="edges-ml simulation browser",
                  scan_plots=False,
                  max_doc_bytes=DEFAULT_MAX_DOC_BYTES,
                  max_files=DEFAULT_MAX_FILES,
                  json_output=None,
                  compress=True,
                  quiet=False):
    """
    Crawls *sources* and writes a self-contained HTML browser to *output*.

    Parameters
    ----------
    sources : list
        Either the source dictionaries used by `edges_ml.build_dataset`
        ({"code": "m3dc1", "directories": [...]}), plain directory paths
        (both codes are then searched), or (code, directory) tuples.
    output : str or Path
        Destination HTML file.
    scan_plots : bool
        If True, image files found in the simulation directory (or in a
        'plots/' / 'figures/' subdirectory) are linked in the "Plots" tab.
    max_doc_bytes : int
        Maximum number of characters shown per input file.
    max_files : int
        Maximum number of entries shown in the per-directory file listing
        (0 disables the listing completely).
    json_output : str or Path or None
        Optional path for dumping the embedded (packed) data model as JSON.
    compress : bool
        Embed the data gzip+base64 encoded (default).  Set to False to embed
        plain JSON, e.g. for browsers without ``DecompressionStream``.

    Returns
    -------
    Path
        The path of the written HTML file.
    """
    output = Path(output).expanduser()
    if output.parent and str(output.parent) not in ("", "."):
        output.parent.mkdir(parents=True, exist_ok=True)

    opts = {
        "max_doc_bytes": int(max_doc_bytes),
        "max_files":     int(max_files),
        "scan_plots":    bool(scan_plots),
        "output_dir":    output.resolve().parent,
        "quiet":         bool(quiet),
    }

    tree, stats = _build_tree(sources, opts)
    payload = _pack_payload(tree, stats, title)
    _text, encoding, raw_bytes, enc_bytes = _encode_payload(payload, compress)

    output.write_text(render_html(payload, title, compress=compress), encoding="utf-8")

    if json_output:
        jpath = Path(json_output).expanduser()
        if jpath.parent and str(jpath.parent) not in ("", "."):
            jpath.parent.mkdir(parents=True, exist_ok=True)
        jpath.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                         encoding="utf-8")

    if not quiet:
        print("edges-ml browser: %s" % _stats_line(stats))
        print("Payload: %s packed JSON -> %s embedded (%s)"
              % (_human_size(raw_bytes), _human_size(enc_bytes), encoding))
        try:
            print("Wrote %s (%s)" % (output.resolve(),
                                     _human_size(output.stat().st_size)))
        except OSError:
            print("Wrote %s" % output.resolve())
        if json_output:
            print("Wrote %s" % Path(json_output).resolve())
        if not stats["simulations"]:
            print("warning: no simulations were found - check the given directories.")

    return output


# ===========================================================================
# COMMAND LINE INTERFACE
# ===========================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="edges-ml-browser",
        description="Generate a static HTML browser for M3D-C1 / NIMROD "
                    "simulation directories.",
        epilog="Examples:\n"
               "  python -m edges_ml.browser /scratch/runs -o simulations.html\n"
               "  python -m edges_ml.browser --m3dc1 /data/m3dc1 --nimrod /data/nimrod\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("roots", nargs="*", metavar="DIR",
                        help="Root directories to crawl (both codes are searched).")
    parser.add_argument("--m3dc1", action="append", default=[], metavar="DIR",
                        help="Root directory crawled for M3D-C1 runs only.")
    parser.add_argument("--nimrod", action="append", default=[], metavar="DIR",
                        help="Root directory crawled for NIMROD runs only.")
    parser.add_argument("-o", "--output", default="simulations.html",
                        help="Output HTML file (default: simulations.html).")
    parser.add_argument("-t", "--title", default="edges-ml simulation browser",
                        help="Page title.")
    parser.add_argument("--scan-plots", action="store_true",
                        help="Link image files found next to the simulations "
                             "(disabled by default; the Plots tab then shows a "
                             "placeholder).")
    parser.add_argument("--max-doc-bytes", type=int, default=DEFAULT_MAX_DOC_BYTES,
                        help="Maximum characters embedded per input file "
                             "(default: %d)." % DEFAULT_MAX_DOC_BYTES)
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES,
                        help="Maximum entries in the per-directory file listing, "
                             "0 disables it (default: %d)." % DEFAULT_MAX_FILES)
    parser.add_argument("--no-compress", dest="compress", action="store_false",
                        help="Embed plain JSON instead of gzip+base64 (bigger "
                             "file, but works without DecompressionStream).")
    parser.add_argument("--json", dest="json_output", default=None, metavar="FILE",
                        help="Also dump the embedded (packed) data model as JSON.")
    parser.add_argument("--open", dest="open_browser", action="store_true",
                        help="Open the generated page in a web browser.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress progress output.")

    args = parser.parse_args(argv)

    sources = []
    for d in args.roots:
        sources.append({"code": "auto", "directories": [d]})
    for d in args.m3dc1:
        sources.append({"code": "m3dc1", "directories": [d]})
    for d in args.nimrod:
        sources.append({"code": "nimrod", "directories": [d]})

    if not sources:
        parser.error("no directories given - pass one or more DIR arguments, "
                     "or use --m3dc1 / --nimrod.")

    out = build_browser(
        sources,
        output=args.output,
        title=args.title,
        scan_plots=args.scan_plots,
        max_doc_bytes=args.max_doc_bytes,
        max_files=args.max_files,
        json_output=args.json_output,
        compress=args.compress,
        quiet=args.quiet,
    )

    if args.open_browser:
        try:
            import webbrowser
            webbrowser.open(Path(out).resolve().as_uri())
        except Exception as exc:  # pragma: no cover
            print("Could not open a web browser: %s" % exc)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
