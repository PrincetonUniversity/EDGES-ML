import os
import re
import sys
import json
import math
import h5py
import shlex
import hashlib
import shutil
import importlib
import subprocess
import numpy as np
from typing import Union
import matplotlib.path as mpltPath
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from termcolor import colored

# ---------------------------------------------------------------------------
# EXTERNAL DEPENDENCIES
# These assume the user has the M3D-C1 Python environment installed.
# ---------------------------------------------------------------------------
try:
    import fpy
    from m3dc1.eval_field import eval_field
    from m3dc1.plot_mesh import get_points_inside_inner_wall
    from m3dc1.flux_average import flux_average
    from m3dc1.gamma_file import Gamma_file
    from m3dc1.get_timetrace import get_timetrace
    from m3dc1.eval_field import get_shape
    from m3dc1.convert2imas import convert2imas
    from m3dc1.convert2imas_imaspy import convert2imas_imaspy
    from m3dc1.get_time_of_slice import get_time_of_slice
    from m3dc1.read_h5 import readParameter
except ImportError:
    print("Warning: fpy or m3dc1 modules not found. M3D-C1 extraction will fail if called.")

try:
    from nimpy.eval_nimrod import EvalNimrod
except ImportError:
    print("Warning: nimpy module not found. NIMROD extraction will fail if called.")


def printwarn(string):
    print(colored(string, 'yellow'))
    return


def printerr(string):
    print(colored(string, 'red'))
    return


def printnote(string):
    print(colored(string, 'blue'))
    return


# ===========================================================================
# QUANTITY REGISTRIES & LABELS
# These dictionaries map short code-specific names to human-readable labels
# or descriptions. They define the universe of possible quantities the
# user can request in their ML configuration.
# ===========================================================================

FIELD_LABELS = {
    'j': ('current density', r'A/m$^2$'),
    'ni': ('ion density', r'particles/m$^3$'),
    'ne': ('electron density', r'particles/m$^3$'),
    'v': ('velocity', 'm/s'),
    'B': ('magnetic field strength', 'T'),
    'p': ('pressure', 'Pa'),
    'pi': ('ion pressure', 'Pa'),
    'pe': ('electron pressure', 'Pa'),
    'ti': ('ion temperature', 'eV'),
    'te': ('electron temperature', 'eV'),
    'A': ('vector potential', r'T$\cdot$m'),
    'gradA': ('grad vector potential', r'Tesla$\cdot$m / (m or rad)'),
    'E': ('electric field', 'V/m'),
    'alpha': (r'ballooning parameter $\alpha$', ''),
    'eta': ('resistivity', r'$\Omega$m'),
    'eta_spitzer': ('resistivity', r'$\Omega$m'),
    'shear': ('magnetic shear', ''),
    'psi': ('poloidal flux', 'Wb'),
    'ne/ng': ('$n_e / n_G$', ''),
    'S': ('Lundquist number $S$', '$S$'),
    'lundquist': ('Lundquist number $S$', '$S$')
}

exclude_for_flux_average = {'ne/ng', 'gradA'}

# Create a new dict, keeping only keys NOT in the exclusion set
FLUX_AVERAGE_LABELS = {k: v for k, v in FIELD_LABELS.items() if k not in exclude_for_flux_average}

# Add quantities that are uniquely available as 1D flux averages
FLUX_AVERAGE_ONLY_LABELS = {
    'q': ('safety factor', ''),
    'rho': ('normalized toroidal flux', ''),
    'flux_t': ('toroidal flux', 'Wb'),
    'flux_p': ('poloidal flux', 'Wb'),
    'Ip': ('plasma current', 'A'),
#    'jav': ('flux-surface averaged current density', r'A/m$^2$'),
#    'jelite': ('ELITE current density', r'A/m$^2$'),
#    'fs-area': ('flux surface area', r'm$^2$'),
#    'polarea': ('poloidal area', r'm$^2$'),
    'V': ('plasma volume', r'm$^3$'),
#    'nueff': ('effective collisionality', ''),
#    'collisionality': ('collisionality', ''),
#    'elongation': ('elongation', ''),
#    'dqdrho': ('dq/drho', ''),
    'f': ('f (RB_phi)', r'T$\cdot$m'),
    'ffprime': ("ff'", r'T$^2\cdot$m'),
#    'DS': ('Suydam parameter', ''),
#    'DM': ('Mercier parameter', ''),
#    'lambda': ('lambda', ''),
#    'beta_pol': ('poloidal beta', ''),
#    'alpha2': ('alpha2', ''),
#    'kappa_implied': ('implied elongation', ''),
#    'amu_implied': ('implied amu', '')
}
FLUX_AVERAGE_LABELS.update(FLUX_AVERAGE_ONLY_LABELS)


TIME_TRACE_LABELS = {
    'Ave_P': ('Average pressure', 'Pa'),
    'E_K3': ('Compressional kinetic energy', 'J'),
    'E_K3D': ('Compressional viscous dissipation', 'W'),
    'E_K3H': ('Compressional hyper-viscous dissipation', 'W'),
    'E_KP': ('Poloidal kinetic energy', 'J'),
    'E_KPD': ('Poloidal viscous dissipation', 'W'),
    'E_KPH': ('Poloidal hyper-viscous dissipation', 'W'),
    'E_KT': ('Toroidal kinetic energy', 'J'),
    'E_KTD': ('Toroidal viscous dissipation', 'W'),
    'E_KTH': ('Toroidal hyper-viscous dissipation', 'W'),
    'E_MP': ('Poloidal magnetic energy', 'J'),
    'E_MPC': ('Poloidal magnetic energy in conductor region', 'J'),
    'E_MPD': ('Poloidal resistive dissipation', 'W'),
    'E_MPH': ('Poloidal hyper-resistive dissipation', 'W'),
    'E_MPV': ('Poloidal magnetic energy in vacuum region', 'J'),
    'E_MT': ('Toroidal magnetic energy', 'J'),
    'E_MTC': ('Toroidal magnetic energy in conductor region', 'J'),
    'E_MTD': ('Toroidal resistive dissipation', 'W'),
    'E_MTH': ('Toroidal hyper-resistive dissipation', 'W'),
    'E_MTV': ('Toroidal magnetic energy in vacuum region', 'J'),
    'E_P': ('Total thermal energy', 'J'),
    'E_PD': ('Thermal dissipation (unused)', 'W'),
    'E_PE': ('Electron thermal energy', 'J'),
    'E_PH': ('Thermal hyper-dissipation (unused)', 'W'),
    'E_grav': ('Gravitational potential energy', 'J'),
    'Flux_kinetic': ('Kinetic-energy convection to wall', 'W'),
    'Flux_poynting': ('Poynting flux to wall', 'W'),
    'Flux_pressure': ('Pressure convection to wall', 'W'),
    'Flux_thermal': ('Heat flux to wall', 'W'),
    'IP_co': ('Plasma current (cosine-component)', 'A'),
    'IP_sn': ('Plasma current (sine-component)', 'A'),
    'M_IZ': ('Plasma current centroid', r'm'),
    'M_IZ_co': ('Plasma current (cosine-component) centroid', r'm'),
    'M_IZ_sn': ('Plasma current (sine-component) centroid', r'm'),
    'Parallel_viscous_heating': ('Parallel viscous heating', 'W'),
    'Particle_Flux_convective': ('Convective particle flux to wall', 'particles/s'),
    'Particle_Flux_diffusive': ('Diffusive particle flux to wall', 'particles/s'),
    'Particle_source': ('Particle source', 'particles/s'),
    'Torque_com': ('Compressional torque', r'N$\cdot$m'),
    'Torque_em': ('Electromagnetic torque', r'N$\cdot$m'),
    'Torque_gyro': ('Gyroviscous torque', r'N$\cdot$m'),
    'Torque_parvisc': ('Parallel viscous torque', r'N$\cdot$m'),
    'Torque_sol': ('Torque_sol', r'N$\cdot$m'),
    'Torque_visc': ('Viscous torque', r'N$\cdot$m'),
    'W_M': ('Stored magnetic energy', 'J'),
    'W_P': ('Stored thermal energy', 'J'),
    'Wall_Force_n0_x': (r'$n=0$ wall force in $R$ direction', 'N'),
    'Wall_Force_n0_x_halo': (r'$n=0$ halo force in $R$ direction', 'N'),
    'Wall_Force_n0_y': (r'$n=0$ wall force in $\phi$ direction', 'N'),
    'Wall_Force_n0_z': (r'$n=0$ wall force in $Z$ direction', 'N'),
    'Wall_Force_n0_z_halo': (r'$n=0$ halo force in $Z$ direction', 'N'),
    'Wall_Force_n1_x': (r'$n=1$ wall force in $R$ direction', 'N'),
    'Wall_Force_n1_y': (r'$n=1$ wall force in $\phi$ direction', 'N'),
    'sideways_force': (r'sideways force', 'N'),
    'angular_momentum': ('Angular momentum', r'kg$\cdot$m$^2$/s'),
    'angular_momentum_p': ('Angular momentum in plasma', r'kg$\cdot$m$^2$/s'),
    'area': ('Domain area', r'm$^2$'),
    'area_p': ('Plasma area', r'm$^2$'),
    'brem_rad': ('Bremsstrahlung radiated power', 'W'),
    'circulation': ('Circulation', r'm$^2$/s'),
    'dt': ('Time step', 's'),
    'electron_number': ('Number of electrons', 'particles'),
    'helicity': ('Magnetic helicity', r'Wb$^2$'),
    'i_control%err_i': ('Current control - integrated error', 'A'),
    'i_control%err_p_old': ('Current control - proportional error', 'A'),
    'ion_loss': ('Ionization power', 'W'),
    'kprad_dt': ('kprad integration time step (in seconds)', 's'),
    'kprad_n': ('Total impurities', 'particles'),
    'kprad_n0': ('Neutral impurities', 'particles'),
    'line_rad': ('Line radiated power', 'W'),
    'loop_voltage': ('Loop voltage', 'V'),
    'n_control%err_i': ('Density control - integrated error', 'particles'),
    'n_control%err_p_old': ('Density control - proportional error', 'particles'),
    'particle_number': ('Number of main ions', 'particles'),
    'particle_number_p': ('Number of main ions in plasma', 'particles'),
    'power_injected': ('Heat source field integrated over MHD region', ''),
    'psi0': ('Poloidal flux on-axis', r'T$\cdot$m$^2$'),
    'psi_lcfs': ('Poloidal flux at separatrix', r'T$\cdot$m$^2$'),
    'psimin': ('Minimum poloidal flux in plasma', r'T$\cdot$m$^2$'),
    'radiation': ('Radiated power', 'W'),
    'reconnected_flux': ('Reconnected flux', r'T$\cdot$m$^2$'),
    'reck_rad': ('Recombination radiated power (kinetic)', 'W'),
    'recp_rad': ('Recombination radiated power (thermal)', 'W'),
    'runaways': ('Number of runaway electrons', 'particles'),
    'temax': ('Extremum of temperature near-axis', 'eV'),
    'time': ('Time', 's'),
    'toroidal_current': ('Toroidal current', 'A'),
    'toroidal_current_p': ('Toroidal current in plasma', 'A'),
    'toroidal_current_w': ('Toroidal current in wall', 'A'),
    'toroidal_flux': ('Toroidal flux', r'T$\cdot$m$^2$'),
    'toroidal_flux_p': ('Toroidal flux in plasma', r'T$\cdot$m$^2$'),
    'volume': ('Domain volume', r'm$^3$'),
    'volume_p': ('Plasma volume', r'm$^3$'),
    'volume_pd': ('Volume of parallel diffusion', ''),
    'xmag': (r'$R$ of magnetic axis', 'm'),
    'xnull': ('$R$ of primary X-point', 'm'),
    'xnull2': ('$R$ of secondary X-point', 'm'),
    'zmag': ('$Z$ of magnetic axis', 'm'),
    'znull': ('$Z$ of primary X-point', 'm'),
    'znull2': ('$Z$ of secondary X-point', 'm'),
    'bharmonics': ('Magnetic energy', 'J'),
    'keharmonics': ('Kinetic energy', 'J'),
    'cauchy_fraction': ('cauchy_fraction', ''),
    'cloud_pel': ('Size of cloud over size of pellet', ''),
    'pellet_mix': ('Fraction of pellet that is D2', ''),
    'pellet_phi': ('Toroidal angle of pellet', 'radians'),
    'pellet_r': (r'$R$ location of pellet', 'm'),
    'pellet_rate': ('Impurity deposition rate', 'particles/s'),
    'pellet_rate_D2': ('D2 deposition rate', 'particles/s'),
    'pellet_ablrate': ('Pellet ablation rate', 'particles/s'),
    'pellet_var': ('Poloidal half-width of impurity cloud', 'm'),
    'pellet_var_tor': ('Toroidal half-width of impurity cloud', 'm'),
    'pellet_velphi': ('Toroidal velocity of pellet', 'm/s'),
    'pellet_velr': (r'$R$ velocity of pellet', 'm/s'),
    'pellet_velz': (r'$Z$ velocity of pellet', 'm/s'),
    'pellet_vx': (r'$X$ velocity of pellet', 'm/s'),
    'pellet_vy': (r'$Y$ velocity of pellet', 'm/s'),
    'pellet_z': (r'$Z$ location of pellet', 'm'),
    'r_p': ('Pellet radius', 'm'),
}


# ===========================================================================
# UNIT DICTIONARIES (To be populated later)
# ===========================================================================

# Code unit conversion formulas for 2D/3D fields
FIELD_UNITS_CODE = {k: 'dummy_conversion_formula' for k in FIELD_LABELS.keys()}

# Code unit conversion formulas for flux averages that don't exist as 2D/3D fields
FLUX_AVERAGE_ONLY_UNITS_CODE = {k: 'dummy_conversion_formula' for k in FLUX_AVERAGE_ONLY_LABELS.keys()}

# Code unit conversion formulas for time traces
TIME_TRACE_UNITS_CODE = {k: 'dummy_conversion_formula' for k in TIME_TRACE_LABELS.keys()}


# The AVAILABLE lists define what the script is officially allowed to process.
AVAILABLE_1D_PROFILES = ["q_profile"]
AVAILABLE_2D_FIELDS = list(FIELD_LABELS.keys())
AVAILABLE_3D_FIELDS = list(FIELD_LABELS.keys())
AVAILABLE_FLUX_AVERAGES = list(FLUX_AVERAGE_LABELS.keys())#["p", "q", "te", "ne", "ti", "ni", "eta", "j", "B"]
AVAILABLE_TIME_TRACES = list(TIME_TRACE_LABELS.keys())
AVAILABLE_INPUTS = ["Ip", "Bt"]


# ===========================================================================
# HELPERS & DATACLASSES
# ===========================================================================

def expand_request(selection, registry):
    """
    Helper function to parse the user's config request.
    If they ask for "all", it returns everything in the registry.
    If None, returns an empty list.
    """
    if selection is None:
        return []
    if selection == "all":
        return registry
    return selection

@dataclass
class GridSpec:
    """
    A data structure to define how a field should be spatially extracted.
    - type: 'rectangular' (fixed grid), 'native' (raw unstructured),
            or 'inside_wall' (unstructured, strictly inside the vessel).
    - resolution: Tuple defining the dimensions of the grid (if applicable).
    """
    type: str
    resolution: Union[tuple, int, None] = None

def generate_unique_filename(model_dir, code):
    """
    Generates a unique filename by combining the shortened parent directory
    and the full model directory name.

    Example input: .../varyped/varyped132543_700_kEFIT_Walter/1f_eqrotnc-ion_eta_x0.1
    Example output: varyped132543_700_1f_eqrotnc-ion_eta_x0.1_m3dc1.h5
    """
    # Shorten the parent directory name if it's very long (e.g., stripping _kEFIT_Walter)
    # We MUST keep the actual model directory name fully intact 
    # so we don't lose crucial scan parameters (like eta_x0.1)

    parts = [p for p in model_dir.parts if p not in ('.', '..', '/', '\\')]
    parts = [p for p in parts if p and not p.endswith(':')]

    full_name = "_".join(parts)
    filename = f"{full_name}_{code}.h5"

    # Ensure the filename does not exceed OS limits (typically 255 chars)
    if len(filename) > 240:
        path_hash = hashlib.md5(str(model_dir.resolve()).encode()).hexdigest()[:8]
        suffix = f"_{path_hash}_{code}.h5"
        allowed_len = 240 - len(suffix)
        truncated_name = full_name[-allowed_len:].lstrip('_')
        filename = f"{truncated_name}{suffix}"

    return filename

def compute_file_hash(filepath, algo='sha256'):
    """Computes the hash of a file for reproducibility tracking."""
    hash_func = hashlib.new(algo)
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except FileNotFoundError:
        return None

def auto_cast(val_str):
    """Attempts to cast text parameters to int or float. Everything else remains a string."""
    if not val_str:
        return ""

    try:
        return int(val_str)
    except ValueError:
        try:
            return float(val_str)
        except ValueError:
            return val_str


# ===========================================================================
# NIMROD -> IMAS SUPPORT (nimrod2imas)
#
# Unlike M3D-C1 (where m3dc1.convert2imas() writes ONE self-contained IMAS
# HDF5 file), 'nimrod2imas' is organised as three command-line tools that all
# write into a shared IMAS *data entry directory*:
#
#     <dbpath>/<dd>/<major_dd_version>/<pulse>/<run>/
#         master.h5, equilibrium.h5, core_profiles.h5, wall.h5,
#         mhd_linear_<occ>.h5, mhd_<occ>.h5, workflow.h5, dataset_fair.h5, ...
#
#   * input2imas  : GEQDSK + PEQDSK (+ nimrod.in/nimeq.in/oculus.in/fluxgrid.in)
#                   -> equilibrium, core_profiles, wall, mhd, summary,
#                      dataset_fair, workflow          [occurrence 0]
#   * dump2imas   : dumpgll.*.h5 -> equilibrium, core_profiles, edge_profiles,
#                   mhd_linear (per species), mhd (GGD) [occurrence occ_base+]
#   * gamma2imas  : energy.bin/logen.bin (+ nimhist.bin) -> growth rates and
#                   frequencies in mhd_linear.toroidal_mode[]
#
# ---------------------------------------------------------------------------
# TWO DISTINCT FAILURE MODES, WHICH MUST NOT BE CONFLATED
# ---------------------------------------------------------------------------
#  (a) The *scripts* cannot be located on disk        -> a PATH problem.
#      Fix with config["imas"]["nimrod2imas_path"] or $NIMROD2IMAS_DIR.
#
#  (b) The scripts ARE found, but importing/executing them fails because the
#      Python environment lacks their DEPENDENCIES -> an ENVIRONMENT problem.
#      The most common instance is "No module named 'imas'": dump2imas.py and
#      input2imas.py both do a top-level `import imas`, while gamma2imas.py
#      imports the IMAS helpers lazily and therefore imports cleanly even
#      without imas-python installed.
#      Fix by running the tools with an interpreter that has imas-python, via
#      config["imas"]["python"] and/or config["imas"]["setup_cmd"].
#
# discover_nimrod2imas() handles (a); load_nimrod2imas() handles (b); the two
# are reported separately so the guidance printed to the user is correct.
#
# ---------------------------------------------------------------------------
# EXECUTION MODES
# ---------------------------------------------------------------------------
# 'subprocess' (default whenever the current interpreter cannot import imas):
#     The tools are executed as real command-line programs with a chosen
#     interpreter and an optional shell prelude (e.g. "module load imas").
#     This is also the safer mode in general, because dump2imas.py installs a
#     process-wide RLIMIT_AS on startup that can never be raised again, and
#     several tools chdir() internally.
#
# 'inprocess':
#     module.main() is called directly with sys.argv temporarily patched. Only
#     selected when the crawler's own interpreter can import imas.
#
# LAYOUT NOTE: the nimrod2imas distribution is a FLAT directory of modules
# (nimrod2imas.py, dump2imas.py, input2imas.py, gamma2imas.py) with NO
# __init__.py. The tools internally do 'from nimrod2imas import entry_dir,
# open_dbentry, ...', which only resolves when that directory ITSELF is on
# sys.path / PYTHONPATH (so that nimrod2imas.py is importable as the module
# 'nimrod2imas'). Importing them as 'nimrod2imas.dump2imas' must be AVOIDED.
#
# IMPORTANT: all three tools compute the entry-directory version component as
# args.dd_version[0], so --dd-version must ALWAYS be supplied explicitly
# (passing None would raise a TypeError inside the tools).
#
# Occurrence layout used here (one IMAS entry per model directory):
#     occ 0                              -> input2imas (shared equilibrium)
#     occ occ_base + i*occ_stride        -> mode directory i (nXX)
# The stride leaves room for dump2imas' per-species mhd_linear occurrences
# (occ_base + species_index).
# ===========================================================================

_NIMROD2IMAS_TOOLS = ("dump2imas", "input2imas", "gamma2imas")

# Populated by discover_nimrod2imas()
_NIMROD2IMAS_DIR = None
_NIMROD2IMAS_SCRIPTS = {}
_NIMROD2IMAS_SEARCHED = []
_NIMROD2IMAS_DISCOVERED = False

# Populated by load_nimrod2imas()
_NIMROD2IMAS_MODULES = None
_NIMROD2IMAS_INJECTED = []
_NIMROD2IMAS_IMPORT_ERRORS = {}

# Cache for "does this interpreter have imas?" probes
_IMAS_PROBE_CACHE = {}


def _nimrod2imas_candidate_dirs(extra_path=None):
    """
    Ordered list of directories that may hold the nimrod2imas sources.

      - explicit 'extra_path' (config["imas"]["nimrod2imas_path"]; a file path
        is accepted and reduced to its parent directory)
      - $NIMROD2IMAS_DIR / $NIMROD2IMAS_PATH
      - <dir of this file>/nimrod2imas
      - <dir of this file>/../nimrod2imas          (typical sibling checkout)
      - <dir of this file>/../../nimrod2imas
      - <cwd>/nimrod2imas
      - <dir of this file>, <dir of this file>/..
    """
    candidates = []

    def _add(p):
        if not p:
            return
        try:
            p = Path(p).expanduser().resolve()
        except Exception:
            return
        if p.is_file():
            p = p.parent
        if p.is_dir() and p not in candidates:
            candidates.append(p)

    _add(extra_path)
    for env_key in ("NIMROD2IMAS_DIR", "NIMROD2IMAS_PATH"):
        _add(os.environ.get(env_key))

    here = Path(__file__).resolve().parent
    _add(here / "nimrod2imas")
    _add(here.parent / "nimrod2imas")
    _add(here.parent.parent / "nimrod2imas")
    _add(Path.cwd() / "nimrod2imas")
    _add(here)
    _add(here.parent)

    return candidates


def discover_nimrod2imas(extra_path=None, force=False):
    """
    Locate the nimrod2imas tool scripts WITHOUT importing them.

    Separating discovery from import is essential: a missing dependency such as
    imas-python must never be reported as "the scripts could not be found".

    Returns (tools_dir_or_None, {tool_name: script_path}).
    """
    global _NIMROD2IMAS_DIR, _NIMROD2IMAS_SCRIPTS, _NIMROD2IMAS_SEARCHED, _NIMROD2IMAS_DISCOVERED

    if _NIMROD2IMAS_DISCOVERED and not force:
        return _NIMROD2IMAS_DIR, dict(_NIMROD2IMAS_SCRIPTS)

    candidates = _nimrod2imas_candidate_dirs(extra_path)
    _NIMROD2IMAS_SEARCHED = [str(c) for c in candidates]

    best_dir = None
    best_scripts = {}
    best_count = 0

    for cand in candidates:
        scripts = {}
        for tool in _NIMROD2IMAS_TOOLS:
            script = cand / f"{tool}.py"
            if script.is_file():
                scripts[tool] = script
        if len(scripts) > best_count:
            best_dir = cand
            best_scripts = scripts
            best_count = len(scripts)
        if best_count == len(_NIMROD2IMAS_TOOLS):
            break

    _NIMROD2IMAS_DIR = best_dir
    _NIMROD2IMAS_SCRIPTS = best_scripts
    _NIMROD2IMAS_DISCOVERED = True

    return _NIMROD2IMAS_DIR, dict(_NIMROD2IMAS_SCRIPTS)


def _missing_module_from_exc(exc):
    """Returns the module name of a ModuleNotFoundError, else None."""
    if isinstance(exc, ModuleNotFoundError):
        name = getattr(exc, "name", None)
        if name:
            return str(name)
        m = re.search(r"No module named '([^']+)'", str(exc))
        if m:
            return m.group(1)
    return None


def load_nimrod2imas(extra_path=None, force=False):
    """
    Import the nimrod2imas command line modules into THIS interpreter.

    The tools directory is prepended to sys.path, then each tool is imported
    FLAT ('import dump2imas'), which is the only form compatible with the
    package's internal 'from nimrod2imas import ...' statements when there is
    no __init__.py. A dotted import is attempted only as a last resort (for
    installations that do provide a real package).

    A failure here does NOT mean the scripts are missing - see the module-level
    notes. Import errors are recorded in _NIMROD2IMAS_IMPORT_ERRORS so callers
    can distinguish a path problem from a dependency problem.

    Returns a dict {tool_name: module_or_None}.
    """
    global _NIMROD2IMAS_MODULES, _NIMROD2IMAS_INJECTED, _NIMROD2IMAS_IMPORT_ERRORS

    if _NIMROD2IMAS_MODULES is not None and not force:
        return _NIMROD2IMAS_MODULES

    tools_dir, scripts = discover_nimrod2imas(extra_path, force=force)

    injected = []
    if tools_dir is not None:
        s = str(tools_dir)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)
        injected.append(s)
    _NIMROD2IMAS_INJECTED = injected

    mods = {}
    errors = {}
    for tool in _NIMROD2IMAS_TOOLS:
        mod = None
        tool_errors = []
        # Flat import first (required for the __init__.py-less layout).
        for modname in (tool, f"nimrod2imas.{tool}"):
            try:
                cand_mod = importlib.import_module(modname)
            except Exception as exc:
                tool_errors.append((modname, exc))
                continue
            if not hasattr(cand_mod, "main"):
                tool_errors.append((modname, AttributeError("module has no main()")))
                continue
            mod = cand_mod
            break
        mods[tool] = mod
        if mod is None:
            errors[tool] = tool_errors

    _NIMROD2IMAS_MODULES = mods
    _NIMROD2IMAS_IMPORT_ERRORS = errors
    return mods


def _classify_nimrod2imas_import_errors():
    """
    Summarises why in-process imports failed.

    Returns (missing_dependencies:set, other_messages:list).
    'missing_dependencies' contains third-party module names such as 'imas'
    that the tool scripts need but that are absent from this interpreter.
    """
    missing = set()
    others = []
    for tool, errs in (_NIMROD2IMAS_IMPORT_ERRORS or {}).items():
        for modname, exc in errs:
            # Ignore the expected dotted-import failure of the flat layout.
            if modname.startswith("nimrod2imas."):
                continue
            dep = _missing_module_from_exc(exc)
            if dep and dep.split('.')[0] not in _NIMROD2IMAS_TOOLS:
                missing.add(dep.split('.')[0])
            else:
                others.append(f"{tool}: {type(exc).__name__}: {exc}")
    return missing, others


def _current_python_has_imas():
    """True if imas-python is importable in the interpreter running this script."""
    try:
        importlib.import_module("imas")
        return True
    except Exception:
        return False


def _run_module_main(module, argv, label="", quiet=False):
    """
    Runs the argparse-based main() of an external CLI module by temporarily
    patching sys.argv (and restoring the working directory afterwards, since
    some of the tools chdir internally). Returns True on success.
    """
    if module is None:
        return False

    label = label or (argv[0] if argv else "tool")
    old_argv = sys.argv
    old_cwd = os.getcwd()
    try:
        sys.argv = [str(a) for a in argv]
        rc = module.main()
        if isinstance(rc, int) and rc != 0:
            if not quiet:
                printwarn(f"{label} returned non-zero exit code {rc}.")
            return False
        return True
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return True
        if isinstance(code, int):
            if code != 0 and not quiet:
                printwarn(f"{label} exited with code {code}.")
            return code == 0
        if not quiet:
            printwarn(f"{label} exited: {code}")
        return False
    except Exception as exc:
        if not quiet:
            printerr(f"{label} failed: {type(exc).__name__}: {exc}")
        return False
    finally:
        sys.argv = old_argv
        try:
            os.chdir(old_cwd)
        except Exception:
            pass


def _nimrod2imas_env(tools_dir, extra_env=None):
    """
    Environment for subprocess execution: the tools directory must be on
    PYTHONPATH so the scripts' internal 'from nimrod2imas import ...' resolves.
    """
    env = dict(os.environ)
    if tools_dir is not None:
        prev = env.get("PYTHONPATH", "")
        parts = [str(tools_dir)] + [p for p in prev.split(os.pathsep) if p]
        # De-duplicate while preserving order
        seen = set()
        uniq = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        env["PYTHONPATH"] = os.pathsep.join(uniq)
    for k, v in (extra_env or {}).items():
        env[str(k)] = str(v)
    return env


def _compose_subprocess_cmd(python_exe, script, args, setup_cmd=None):
    """
    Builds the argv for running a tool script.

    If 'setup_cmd' is given it is executed first inside a login shell, which
    allows environment-module systems, e.g.
        setup_cmd = "module load imas"
    """
    base = [str(python_exe), str(script)] + [str(a) for a in args]
    if setup_cmd:
        inner = " ".join(shlex.quote(x) for x in base)
        return ["bash", "-lc", f"{setup_cmd} && exec {inner}"]
    return base


def _python_has_imas(python_exe, env=None, setup_cmd=None):
    """
    Probes whether the given interpreter (optionally after 'setup_cmd') can
    import imas-python. Result is cached per (interpreter, setup_cmd).
    """
    key = (str(python_exe), str(setup_cmd or ""))
    if key in _IMAS_PROBE_CACHE:
        return _IMAS_PROBE_CACHE[key]

    probe = "import imas, sys; sys.stdout.write(getattr(imas,'__version__','?'))"
    base = [str(python_exe), "-c", probe]
    if setup_cmd:
        inner = " ".join(shlex.quote(x) for x in base)
        cmd = ["bash", "-lc", f"{setup_cmd} && exec {inner}"]
    else:
        cmd = base

    ok = False
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        ok = (proc.returncode == 0)
    except Exception:
        ok = False

    _IMAS_PROBE_CACHE[key] = ok
    return ok


def _prepare_nimrod2imas_runtime(imas_cfg):
    """
    Resolves how the nimrod2imas tools will be executed for this run.

    Returns a runtime dict, or None if nothing can be executed (in which case a
    precise, actionable error has already been printed).
    """
    imas_cfg = dict(imas_cfg or {})

    tools_dir, scripts = discover_nimrod2imas(imas_cfg.get("nimrod2imas_path"))

    if tools_dir is None or not scripts:
        printerr(
            "Could not LOCATE the nimrod2imas scripts (dump2imas.py / input2imas.py / "
            "gamma2imas.py). This is a PATH problem.\n"
            "  Searched: " + ", ".join(_NIMROD2IMAS_SEARCHED) + "\n"
            "  Fix: set config['imas']['nimrod2imas_path'] to the directory holding those "
            "files, or export $NIMROD2IMAS_DIR."
        )
        return None

    missing_tools = [t for t in _NIMROD2IMAS_TOOLS if t not in scripts]
    if missing_tools:
        printwarn(
            f"nimrod2imas directory {tools_dir} is missing: "
            + ", ".join(f"{t}.py" for t in missing_tools)
        )

    requested_mode = str(imas_cfg.get("run_mode", "auto") or "auto").strip().lower()
    if requested_mode not in ("auto", "subprocess", "inprocess"):
        printwarn(f"Unknown imas run_mode '{requested_mode}'; using 'auto'.")
        requested_mode = "auto"

    setup_cmd = imas_cfg.get("setup_cmd") or os.environ.get("NIMROD2IMAS_SETUP_CMD") or None
    python_exe = (
        imas_cfg.get("python")
        or os.environ.get("NIMROD2IMAS_PYTHON")
        or sys.executable
    )
    env = _nimrod2imas_env(tools_dir, imas_cfg.get("env"))

    # ---- decide the execution mode ----
    modules = {}
    mode = requested_mode

    if requested_mode in ("auto", "inprocess"):
        modules = load_nimrod2imas(imas_cfg.get("nimrod2imas_path"))
        inproc_ok = _current_python_has_imas() and all(
            modules.get(t) is not None for t in scripts.keys()
        )
        if requested_mode == "inprocess":
            if not inproc_ok:
                missing, others = _classify_nimrod2imas_import_errors()
                printwarn(
                    "run_mode='inprocess' was requested but this interpreter cannot import "
                    "the tools"
                    + (f" (missing: {', '.join(sorted(missing))})" if missing else "")
                    + ". Falling back to subprocess execution."
                )
                mode = "subprocess"
            else:
                mode = "inprocess"
        else:
            mode = "inprocess" if inproc_ok else "subprocess"

    if mode == "subprocess":
        # Report *why* we are not running in-process, with the correct diagnosis.
        if _NIMROD2IMAS_IMPORT_ERRORS:
            missing, others = _classify_nimrod2imas_import_errors()
            if missing:
                printnote(
                    f"nimrod2imas scripts found in {tools_dir}, but this interpreter is "
                    f"missing: {', '.join(sorted(missing))}. Running the tools as "
                    "subprocesses instead (this is an ENVIRONMENT issue, not a path issue)."
                )
            for msg in others:
                printwarn(f"nimrod2imas import note: {msg}")

        if not bool(imas_cfg.get("ignore_imas_probe", False)):
            if not _python_has_imas(python_exe, env=env, setup_cmd=setup_cmd):
                printerr(
                    "The interpreter selected for nimrod2imas cannot import imas-python.\n"
                    f"  interpreter : {python_exe}\n"
                    f"  setup_cmd   : {setup_cmd or '(none)'}\n"
                    "  dump2imas.py and input2imas.py both require 'import imas' at module "
                    "level, so the conversion cannot proceed.\n"
                    "  Fix by ONE of:\n"
                    "    * run this crawler inside your IMAS environment "
                    "(e.g. after 'module load imas' / activating the conda env);\n"
                    "    * config['imas']['python']    = '/path/to/env/bin/python';\n"
                    "    * config['imas']['setup_cmd'] = 'module load imas'  "
                    "(executed in a login shell before the tool);\n"
                    "    * config['imas']['ignore_imas_probe'] = True to attempt anyway."
                )
                return None

    printnote(
        f"nimrod2imas: dir={tools_dir}; mode={mode}"
        + (f"; python={python_exe}" if mode == "subprocess" else "")
        + (f"; setup_cmd={setup_cmd!r}" if (mode == "subprocess" and setup_cmd) else "")
    )

    return {
        "dir": tools_dir,
        "scripts": scripts,
        "modules": modules,
        "mode": mode,
        "python": str(python_exe),
        "setup_cmd": setup_cmd,
        "env": env,
        "capture": bool(imas_cfg.get("capture_output", False)),
    }


def _nimrod2imas_tool_available(rt, tool):
    """True if the given tool can actually be executed in the resolved mode."""
    if rt is None:
        return False
    if rt["mode"] == "inprocess":
        return rt["modules"].get(tool) is not None
    return tool in rt["scripts"]


def _run_subprocess_tool(rt, tool, argv, label="", quiet=False):
    """
    Runs a nimrod2imas tool as a separate process.

    argv[0] is a cosmetic program name (kept so in-process and subprocess modes
    share the same argv construction); the real script path is substituted here.
    """
    script = rt["scripts"].get(tool)
    if script is None:
        if not quiet:
            printwarn(f"{label or tool}: script not found; skipping.")
        return False

    cmd = _compose_subprocess_cmd(rt["python"], script, argv[1:], setup_cmd=rt["setup_cmd"])
    capture = bool(rt["capture"] or quiet)

    try:
        if capture:
            proc = subprocess.run(cmd, env=rt["env"], capture_output=True, text=True)
            if proc.returncode != 0:
                if not quiet:
                    printerr(f"{label or tool} failed (exit {proc.returncode}).")
                    tail = "\n".join((proc.stderr or "").splitlines()[-40:])
                    if tail.strip():
                        print(tail)
                return False
            return True

        proc = subprocess.run(cmd, env=rt["env"])
        if proc.returncode != 0:
            if not quiet:
                printwarn(f"{label or tool} exited with code {proc.returncode}.")
            return False
        return True
    except FileNotFoundError as exc:
        if not quiet:
            printerr(f"{label or tool}: could not execute {cmd[0]!r}: {exc}")
        return False
    except Exception as exc:
        if not quiet:
            printerr(f"{label or tool} failed: {type(exc).__name__}: {exc}")
        return False


def _run_nimrod2imas_tool(rt, tool, argv, label="", quiet=False):
    """Dispatches a tool invocation to the resolved execution mode."""
    if rt is None:
        return False
    label = label or tool
    if rt["mode"] == "inprocess":
        return _run_module_main(rt["modules"].get(tool), argv, label=label, quiet=quiet)
    return _run_subprocess_tool(rt, tool, argv, label=label, quiet=quiet)


def _options_dict_to_cli(options):
    """
    Converts a {option_name: value} dictionary into a list of CLI tokens.
    Underscores become dashes, booleans become bare flags (True) or are
    dropped (False), lists are expanded into multiple values.
    """
    argv = []
    for key, val in (options or {}).items():
        flag = str(key) if str(key).startswith("-") else "--" + str(key).replace("_", "-")
        if val is None:
            continue
        if isinstance(val, bool):
            if val:
                argv.append(flag)
        elif isinstance(val, (list, tuple, np.ndarray)):
            argv.append(flag)
            argv.extend(str(v) for v in val)
        else:
            argv.extend([flag, str(val)])
    return argv


_KNOWN_MACHINES = (
    ("diii-d", "d3d"), ("diiid", "d3d"), ("d3d", "d3d"),
    ("nstx-u", "nstx"), ("nstxu", "nstx"), ("nstx", "nstx"),
    ("mast-u", "mast"), ("mastu", "mast"), ("mast", "mast"),
    ("iter", "iter"), ("kstar", "kstar"), ("east", "east"),
    ("jet", "jet"), ("asdex", "aug"), ("aug", "aug"),
    ("sparc", "sparc"), ("tcv", "tcv"), ("west", "west"),
    ("c-mod", "cmod"), ("cmod", "cmod"), ("jt-60", "jt60sa"),
    ("jt60", "jt60sa"),
)


def _infer_machine_and_shot(path, levels=4):
    """
    Heuristically infers an IMAS database (machine) name and a pulse/shot number
    from the directory path. Returns (machine_or_None, shot_or_None).
    """
    p = Path(path)
    names = [p.name]
    parent = p.parent
    for _ in range(max(0, int(levels) - 1)):
        names.append(parent.name)
        if parent.parent == parent:
            break
        parent = parent.parent

    text = "/".join(str(n).lower() for n in names)

    machine = None
    for key, norm in _KNOWN_MACHINES:
        if key in text:
            machine = norm
            break

    shot = None
    for name in names:
        m = re.search(r'(\d{5,8})', str(name))
        if m:
            try:
                shot = int(m.group(1))
            except ValueError:
                shot = None
            if shot is not None:
                break

    return machine, shot


def _imas_dd_version_dir(dd_version):
    """
    Returns the major-version directory component used by the nimrod2imas tools.

    The tools compute this as args.dd_version[0], i.e. the first character of
    the version string.  For '4.1.1' that is '4'; for '3.42.0' it is '3'.
    This function replicates that logic so the crawler and the tools always
    agree on the entry directory path.
    """
    s = str(dd_version or "4").strip()
    return s[0] if s else "4"


def _imas_entry_dir(dbpath, dd, dd_version, pulse, run):
    """
    Reproduces the entry directory layout used by the nimrod2imas tools:
        <dbpath>/<dd>/<dd_version[0]>/<pulse>/<run>

    Uses _imas_dd_version_dir() to ensure the version component always matches
    what the tools compute from args.dd_version[0].
    """
    ver_dir = _imas_dd_version_dir(dd_version)
    return Path(dbpath) / str(dd) / ver_dir / str(int(pulse)) / str(int(run))


# Runs handed out during this process, so that two model directories processed
# in the same batch never collide (the entry directory is created immediately
# after allocation and would otherwise still look "empty" to the next caller).
_ALLOCATED_IMAS_RUNS = set()


def _allocate_imas_run(dbpath, dd, dd_version, pulse, run=None, max_tries=10000):
    """
    Returns the run number to use. If 'run' is given explicitly it is honoured
    verbatim; otherwise the first run index that is neither already allocated in
    this process nor backed by a non-empty entry directory is selected, so that
    existing entries are never silently overwritten.
    """
    try:
        base = (str(Path(dbpath).resolve()), str(dd), _imas_dd_version_dir(dd_version), int(pulse))
    except Exception:
        base = (str(dbpath), str(dd), _imas_dd_version_dir(dd_version), int(pulse))

    if run is not None:
        r = int(run)
        _ALLOCATED_IMAS_RUNS.add(base + (r,))
        return r

    candidate = 1
    while candidate <= max_tries:
        if base + (candidate,) not in _ALLOCATED_IMAS_RUNS:
            ed = _imas_entry_dir(dbpath, dd, dd_version, pulse, candidate)
            try:
                free = (not ed.exists()) or (not any(ed.iterdir()))
            except OSError:
                free = True
            if free:
                _ALLOCATED_IMAS_RUNS.add(base + (candidate,))
                return candidate
        candidate += 1

    _ALLOCATED_IMAS_RUNS.add(base + (candidate,))
    return candidate


_GEQDSK_PATTERNS = (
    re.compile(r'^g\d{3,8}\.\d{2,6}'),
    re.compile(r'^geqdsk', re.IGNORECASE),
    re.compile(r'\.geqdsk$', re.IGNORECASE),
    re.compile(r'^g_?eqdsk', re.IGNORECASE),
)

_PEQDSK_PATTERNS = (
    re.compile(r'^p\d{3,8}\.\d{2,6}'),
    re.compile(r'^peqdsk', re.IGNORECASE),
    re.compile(r'\.peqdsk$', re.IGNORECASE),
    re.compile(r'^p_?eqdsk', re.IGNORECASE),
)


def _is_geqdsk_name(name):
    return any(pat.search(name) for pat in _GEQDSK_PATTERNS)


def _is_peqdsk_name(name):
    return any(pat.search(name) for pat in _PEQDSK_PATTERNS)


def _record_imas_manifest(output_directory, info):
    """
    Appends/updates a small JSON manifest mapping simulation directories onto
    the IMAS entries (or files) that were produced for them.
    """
    if not info:
        return
    manifest = Path(output_directory) / "imas_entries.json"
    data = {}
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    key = info.get("model_directory", str(len(data)))
    data[key] = info
    try:
        manifest.write_text(json.dumps(data, indent=2, sort_keys=True))
    except Exception as exc:
        printwarn(f"Could not update IMAS manifest {manifest}: {exc}")


def _subtract_field_results(val_finite, val_eq):
    """
    Subtracts two field evaluation results to compute the eigenfunction:
        eigenfunction = field(finite_time) - field(equilibrium)
    Handles both scalar numpy arrays and dicts of vector components.
    Returns the difference in the same format, or an empty array if either
    input is empty/None.
    """
    if val_finite is None or val_eq is None:
        return np.array([])

    if isinstance(val_finite, dict) and isinstance(val_eq, dict):
        result = {}
        for key in val_finite:
            if key in val_eq:
                f = np.asarray(val_finite[key])
                e = np.asarray(val_eq[key])
                if f.size == 0 or e.size == 0:
                    result[key] = np.array([])
                else:
                    result[key] = f - e
            else:
                result[key] = val_finite[key]
        return result
    elif isinstance(val_finite, np.ndarray) and isinstance(val_eq, np.ndarray):
        if val_finite.size == 0 or val_eq.size == 0:
            return np.array([])
        return val_finite - val_eq
    else:
        try:
            return np.asarray(val_finite) - np.asarray(val_eq)
        except Exception:
            return np.array([])


# ===========================================================================
# ADAPTER BASE CLASS
# Defines the blueprint that all underlying simulation codes must follow.
# ===========================================================================

class SimulationAdapter:

    CODE_NAME = "UNKNOWN"

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.shared_inner_wall_points = None
        self._warned_methods = set()

    def _warn_unimplemented(self, method_name):
        if method_name not in self._warned_methods:
            printwarn(f"Warning: {method_name}() is not yet implemented for {self.CODE_NAME}. Skipping.")
            self._warned_methods.add(method_name)

    # --- Abstract Methods (Safely stubbed to skip when not implemented by subclass) ---
    def get_metadata(self, mode_dirs):
        self._warn_unimplemented('get_metadata')
        return {
            "code": self.CODE_NAME,
            "source_directory": str(self.model_dir.resolve()),
            "release_version": "Unknown",
            "build_date": "Unknown"
        }

    def get_all_input_parameters(self):
        self._warn_unimplemented('get_all_input_parameters')
        return {}

    def get_global_parameters(self):
        self._warn_unimplemented('get_global_parameters')
        return {}

    def get_reproducibility_data(self, mode_dir):
        self._warn_unimplemented('get_reproducibility_data')
        return {}

    def get_1d_mesh(self, resolution, units, fcoords):
        self._warn_unimplemented('get_1d_mesh')
        return np.array([])

    def get_2d_mesh(self, grid_spec):
        self._warn_unimplemented('get_2d_mesh')
        return np.array([])

    def get_3d_mesh(self, grid_spec):
        self._warn_unimplemented('get_3d_mesh')
        return np.array([])

    def get_2d_field(self, name, grid_spec, units, time=-1):
        self._warn_unimplemented('get_2d_field')
        return np.array([])

    def get_3d_field(self, name, grid_spec, units, time=-1):
        self._warn_unimplemented('get_3d_field')
        return np.array([])

    def get_flux_average(self, name, resolution, units, fcoords, time=-1, use_eq_fs=True):
        self._warn_unimplemented('get_flux_average')
        return np.array([]), np.array([])

    def get_time_trace(self, name, units):
        self._warn_unimplemented('get_time_trace')
        return np.array([]), np.array([])

    def get_mode_metadata(self, mode_dir):
        self._warn_unimplemented('get_mode_metadata')
        return {"growth_rate": 0.0, "frequency": 0.0, "mode_type": -100}

    def get_mode_1d_profile(self, mode_dir, name, resolution, units):
        self._warn_unimplemented('get_mode_1d_profile')
        return np.array([])

    def get_mode_2d_mesh(self, mode_dir, grid_spec):
        self._warn_unimplemented('get_mode_2d_mesh')
        return np.array([])

    def get_mode_3d_mesh(self, mode_dir, grid_spec):
        self._warn_unimplemented('get_mode_3d_mesh')
        return np.array([])

    def get_mode_2d_field(self, mode_dir, name, grid_spec, units):
        self._warn_unimplemented('get_mode_2d_field')
        return np.array([])

    def get_mode_3d_field(self, mode_dir, name, grid_spec, units):
        self._warn_unimplemented('get_mode_3d_field')
        return np.array([])

    def get_time_metadata(self, time_slices):
        self._warn_unimplemented('get_time_metadata')
        return {
            "time_slice": np.array(time_slices, dtype=int),
            "simulation_time_step": np.zeros(len(time_slices), dtype=int),
            "simulation_time": np.zeros(len(time_slices), dtype=float)
        }

    def convert_to_imas(self, mode_dirs, output_directory, config, run=None, out_file=None):
        """
        Produces IMAS-compatible output for this model directory.

        Returns a dictionary describing the produced artefact (used for the
        output manifest), or None if nothing could be written.
        """
        self._warn_unimplemented('convert_to_imas')
        return None

    def get_all_mode_entries(self, mode_dir, config):
        """
        Returns a list of (group_name, mode_metadata) tuples to be written under
        the 'perturbations' group for a given mode directory.

        The default implementation returns a single entry using the directory name
        as the group name and get_mode_metadata() for the metadata. Subclasses may
        override this to expand one directory into multiple per-n entries (e.g. NIMROD).

        Returns: list of (str, dict) tuples, where each dict contains at minimum
                 'growth_rate', 'frequency', and 'mode_type'.
        """
        return [(mode_dir.name, self.get_mode_metadata(mode_dir))]

    def close(self): pass

    # --- Core Extraction Logic ---
    def extract_grids(self, config):
        """
        Extracts all static grids and the time coordinate mapping into one structure.
        """
        grids = {}
        res_1d = config.get("resolutions", {}).get("1d", 200)
        units = config.get("units", "codeunits")
        fcoords = config.get("fcoords", "pest")
        time_slices = config.get("time_slices", [])

        if time_slices:
            grids["time"] = self.get_time_metadata(time_slices)

        # Retrieve 1D mesh from the first successful flux average call
        fluxes_to_try = expand_request(config.get("flux_averages"), AVAILABLE_FLUX_AVERAGES)
        if not fluxes_to_try:
            fluxes_to_try = AVAILABLE_FLUX_AVERAGES

        grids["1d_grid"] = np.array([])
        for field in fluxes_to_try:
            f_mesh, _ = self.get_flux_average(field, res_1d, units, fcoords, time=-1)
            if len(f_mesh) > 0:
                grids["1d_grid"] = f_mesh
                break

        grid_2d = config.get("2d_grid", GridSpec("rectangular", (128, 128)))
        grids["2d_grid"] = self.get_2d_mesh(grid_2d)

        grid_3d = config.get("3d_grid", GridSpec("rectangular", (64, 16, 64)))
        grids["3d_grid"] = self.get_3d_mesh(grid_3d)

        return grids

    def extract_equilibrium(self, config):
        """
        Parses the config dictionary and extracts all the shared, 
        time-independent or base-state equilibrium quantities.
        Mesh grids are no longer saved here, only the field values.
        """
        data = {
            "global_parameters": self.get_global_parameters(),
            "2d_fields": {}, "3d_fields": {},
            "flux_averages": {}, "time_traces": {}
        }

        res_1d = config.get("resolutions", {}).get("1d", 200)
        units = config.get("units", "codeunits")
        fcoords = config.get("fcoords", "pest")

        # Process 2D fields
        fields_2d = expand_request(config.get("2d_fields"), AVAILABLE_2D_FIELDS)
        grid_2d = config.get("2d_grid", GridSpec("rectangular", (128, 128)))
        for field in fields_2d:
            val = self.get_2d_field(field, grid_2d, units, time=-1)
            if isinstance(val, dict):
                data["2d_fields"].update(val)
            else:
                data["2d_fields"][field] = val

        # Process 3D fields
        fields_3d = expand_request(config.get("3d_fields"), AVAILABLE_3D_FIELDS)
        grid_3d = config.get("3d_grid", GridSpec("rectangular", (64, 16, 64)))
        for field in fields_3d:
            val = self.get_3d_field(field, grid_3d, units, time=-1)
            if isinstance(val, dict):
                data["3d_fields"].update(val)
            else:
                data["3d_fields"][field] = val

        # Process Fluxes decoupled (only values, grid is saved in /grids)
        fluxes = expand_request(config.get("flux_averages"), AVAILABLE_FLUX_AVERAGES)
        for name in fluxes:
            _, f_vals = self.get_flux_average(name, res_1d, units, fcoords, time=-1)
            if isinstance(f_vals, dict):
                data["flux_averages"].update(f_vals)
            elif len(f_vals) > 0:
                data["flux_averages"][name] = f_vals

        # Process Time Traces
        traces = expand_request(config.get("time_traces"), AVAILABLE_TIME_TRACES)
        time_saved = False
        for name in traces:
            t_arr, v_arr = self.get_time_trace(name, units)
            
            # Save the common time array only once
            if not time_saved and len(t_arr) > 0:
                data["time_traces"]["time"] = t_arr
                time_saved = True
                
            if len(v_arr) > 0:
                data["time_traces"][name] = v_arr

        return data

    def extract_total_fields(self, config):
        """
        Parses the config dictionary and extracts total fields across selected time slices.
        Produces arrays of shape [N_t, ...]
        """
        time_slices = config.get("time_slices", [])
        if not time_slices:
            return None

        data = {
            "flux_averages": {},
            "2d_fields": {},
            "3d_fields": {}
        }

        res_1d = config.get("resolutions", {}).get("1d", 200)
        units = config.get("units", "codeunits")
        fcoords = config.get("fcoords", "pest")

        fluxes = expand_request(config.get("total_flux_averages"), AVAILABLE_FLUX_AVERAGES)
        for field in fluxes:
            results = []
            for ts in time_slices:
                _, f_vals = self.get_flux_average(field, res_1d, units, fcoords, time=ts, use_eq_fs=True)
                if not isinstance(f_vals, dict) and len(f_vals) == 0:
                    f_vals = np.zeros(res_1d) # fallback if extraction fails
                results.append(f_vals)
            if results and isinstance(results[0], dict):
                for comp in results[0].keys():
                    data["flux_averages"][comp] = np.stack([r[comp] for r in results], axis=0)
            else:
                data["flux_averages"][field] = np.stack(results, axis=0) if results else np.array([])

        fields_2d = expand_request(config.get("total_2d_fields"), AVAILABLE_2D_FIELDS)
        grid_2d = config.get("2d_grid", GridSpec("rectangular", (128, 128)))
        for field in fields_2d:
            results = []
            for ts in time_slices:
                val = self.get_2d_field(field, grid_2d, units, time=ts)
                results.append(val)
            if results and isinstance(results[0], dict):
                for comp in results[0].keys():
                    data["2d_fields"][comp] = np.stack([r[comp] for r in results], axis=0)
            else:
                data["2d_fields"][field] = np.stack(results, axis=0) if results else np.array([])

        fields_3d = expand_request(config.get("total_3d_fields"), AVAILABLE_3D_FIELDS)
        grid_3d = config.get("3d_grid", GridSpec("rectangular", (64, 16, 64)))
        for field in fields_3d:
            results = []
            for ts in time_slices:
                val = self.get_3d_field(field, grid_3d, units, time=ts)
                results.append(val)
            if results and isinstance(results[0], dict):
                for comp in results[0].keys():
                    data["3d_fields"][comp] = np.stack([r[comp] for r in results], axis=0)
            else:
                data["3d_fields"][field] = np.stack(results, axis=0) if results else np.array([])

        return data

    def extract_mode(self, mode_dir, config):
        """
        Extracts the mode-specific perturbation data (eigenfunction).
        The eigenfunction is computed as:
            eigenfunction = field(finite_time, mode sim) - field(equilibrium, time=-1)
        This matches the approach used in eigenfunction.py:
            ef = p1 - p0
        where p1 is evaluated at the finite time sim and p0 at the equilibrium sim.
        """
        res_1d = config.get("resolutions", {}).get("1d", 200)
        units = config.get("units", "codeunits")
        data = {
            "mode_information": self.get_mode_metadata(mode_dir),
            "1d_profiles": {}, "2d_fields": {}, "3d_fields": {}
        }

        fields_1d = expand_request(config.get("mode_1d_profiles"), AVAILABLE_1D_PROFILES)
        for field in fields_1d:
            data["1d_profiles"][field] = self.get_mode_1d_profile(mode_dir, field, res_1d, units)

        grid_2d = config.get("perturbation_2d_grid", config.get("2d_grid", GridSpec("rectangular", (128, 128))))
        fields_2d = expand_request(config.get("mode_2d_fields"), AVAILABLE_2D_FIELDS)
        for field in fields_2d:
            # Evaluate field at finite time (mode sim) and at equilibrium, then subtract
            val_finite = self.get_mode_2d_field(mode_dir, field, grid_2d, units)
            val_eq = self.get_2d_field(field, grid_2d, units, time=-1)
            val = _subtract_field_results(val_finite, val_eq)
            if isinstance(val, dict):
                data["2d_fields"].update(val)
            else:
                data["2d_fields"][field] = val

        grid_3d = config.get("perturbation_3d_grid", config.get("3d_grid", GridSpec("rectangular", (64, 16, 64))))
        fields_3d = expand_request(config.get("mode_3d_fields"), AVAILABLE_3D_FIELDS)
        for field in fields_3d:
            # Evaluate field at finite time (mode sim) and at equilibrium, then subtract
            val_finite = self.get_mode_3d_field(mode_dir, field, grid_3d, units)
            val_eq = self.get_3d_field(field, grid_3d, units, time=-1)
            val = _subtract_field_results(val_finite, val_eq)
            if isinstance(val, dict):
                data["3d_fields"].update(val)
            else:
                data["3d_fields"][field] = val

        return data


# ===========================================================================
# M3D-C1 ADAPTER IMPLEMENTATION
# Contains the specific logic to interface with M3D-C1 HDF5 files and meshes.
# ===========================================================================

class M3DC1Adapter(SimulationAdapter):

    CODE_NAME = "M3D-C1"

    def __init__(self, model_dir):
        super().__init__(model_dir)
        # Cache to store the (R, Z) points of the inner wall 
        self._inner_wall_points = None
        
        # Sim objects caches to ensure time=-1 and time='last' are loaded optimally
        self._eq_sim = None
        self._mode_sims = {}
        self._time_sims = {}
        
        # Internal state for Slurm file metadata
        self._slurm_parsed = False
        self._slurm_rel_ver = "Unknown"
        self._slurm_bld_date = "Unknown"
        self._slurm_inputs = {}

        # Look for the Gamma growth rate file inside the parent model directory
        self.gamma_data = None
        for f in self.model_dir.iterdir():
            if f.is_file() and f.suffix in ['.txt', '.dat', '.out', '']:
                try:
                    with open(f, 'r') as tmp:
                        first_lines = "".join([next(tmp) for _ in range(5)])
                    # Verify it has the expected headers
                    if 'gamma' in first_lines and 'sig_gamma' in first_lines:
                        self.gamma_data = Gamma_file(str(f))
                        break
                except Exception:
                    pass

    def _get_eq_sim(self):
        """
        Lazily loads the shared equilibrium simulation object (time=-1).
        Searches the model directory for any valid C1.h5 file to initialize it.
        """
        if self._eq_sim is None:
            # Find any C1.h5 within the modes to initialize the equilibrium object
            c1_paths = list(self.model_dir.rglob("C1.h5"))
            if not c1_paths:
                raise FileNotFoundError(f"No C1.h5 files found in {self.model_dir}")
            
            try:
                # time=-1 indicates the equilibrium time slice
                self._eq_sim = fpy.sim_data(filename=str(c1_paths[0]), time=-1)
            except NameError:
                pass # Triggered if fpy is not imported successfully
        return self._eq_sim

    def _get_mode_sim(self, mode_dir):
        """
        Lazily loads the simulation object for a mode directory at time='last'.
        This gives the total field at the last available time step, from which
        the equilibrium will be subtracted in extract_mode() to get the eigenfunction.
        """
        if mode_dir not in self._mode_sims:
            c1_path = mode_dir / 'C1.h5'
            try:
                self._mode_sims[mode_dir] = fpy.sim_data(filename=str(c1_path), time='last')
            except NameError:
                pass
        return self._mode_sims[mode_dir]

    def _get_time_sim(self, time_slice):
        """
        Lazily loads the specific simulation object for a selected time slice.
        """
        if time_slice not in self._time_sims:
            c1_paths = list(self.model_dir.rglob("C1.h5"))
            if not c1_paths:
                return None
            try:
                self._time_sims[time_slice] = fpy.sim_data(filename=str(c1_paths[0]), time=time_slice)
            except NameError:
                return None
        return self._time_sims[time_slice]

    def _map_units(self, units):
        """Translates generalized config units into M3D-C1 specific unit strings."""
        u = units.lower()
        if u == 'codeunits':
            return 'm3dc1'
        return u

    def _parse_slurm_data(self, mode_dirs):
        """
        Parses Slurm standard output files for the code release version, build date, 
        and the full list of physics input parameters. Handles array job naming conventions and restart logic.
        """
        release_version = "Unknown"
        build_date = "Unknown"
        inputs = {}
        candidates = []

        # Gather all potential slurm/out files
        for mdir in mode_dirs:
            for out_file in mdir.glob("slurm*.out"):
                candidates.append(out_file)
            for out_file in mdir.glob("*.out"):
                candidates.append(out_file)

            try:
                n_str = mdir.name.replace('n', '') # e.g. '01'
                n_val = int(n_str)                 # e.g. 1
                for out_file in self.model_dir.glob(f"slurm-*_{n_str}.out"):
                    candidates.append(out_file)
                for out_file in self.model_dir.glob(f"slurm-*_{n_val}.out"):
                    candidates.append(out_file)
            except ValueError:
                pass

        # Fallback check for generic slurm output in the parent directory
        for out_file in self.model_dir.glob("slurm-*.out"):
            if "_" not in out_file.name:
                candidates.append(out_file)

        # De-duplicate and sort by modification time (newest first for restarts)
        candidates = list(set(candidates))
        candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        for filepath in candidates:
            try:
                with open(filepath, 'r', errors='ignore') as f:
                    rel_found = False
                    bld_found = False
                    found_input_anchor = False
                    in_params = False
                    params_done = False
                    
                    # Process lines safely, halting dynamically when parameter block completes
                    for line_count, line in enumerate(f):
                        if line_count > 10000: # Broad threshold to catch headers and inputs
                            break

                        line_str = line.strip()
                        
                        # Catch Metadata Headers
                        if "RELEASE VERSION:" in line_str:
                            release_version = line_str.split("RELEASE VERSION:")[1].strip()
                            rel_found = True
                        if "BUILD DATE:" in line_str:
                            build_date = line_str.split("BUILD DATE:")[1].strip()
                            bld_found = True
                            
                        # Catch and loop through Input Parameters
                        if not params_done:
                            if not found_input_anchor:
                                if "Reading input file C1input" in line_str:
                                    found_input_anchor = True
                            elif not in_params:
                                # Lock onto the first valid parameter assignment (ignoring 'WARNING:' blocks)
                                if '=' in line_str and not line_str.startswith('='):
                                    parts = line_str.split('=', 1)
                                    key = parts[0].strip()
                                    if ' ' not in key and len(key) > 0 and not key.startswith("WARNING"):
                                        in_params = True
                                        inputs[key] = auto_cast(parts[1].strip())
                            else:
                                # We are inside the list, consume lines until the equal signs stop
                                if '=' in line_str and not line_str.startswith('='):
                                    parts = line_str.split('=', 1)
                                    inputs[parts[0].strip()] = auto_cast(parts[1].strip())
                                else:
                                    params_done = True # Block cleanly terminated
                                    
                    # If we grabbed anything meaningful from this file, accept it and break the candidate loop
                    if rel_found or bld_found or len(inputs) > 0:
                        return release_version, build_date, inputs
            except Exception:
                continue

        return release_version, build_date, inputs

    def get_metadata(self, mode_dirs):
        # Trigger Slurm parsing lazily on first request
        if not self._slurm_parsed:
            rel_ver, bld_date, inputs = self._parse_slurm_data(mode_dirs)
            self._slurm_rel_ver = rel_ver
            self._slurm_bld_date = bld_date
            self._slurm_inputs = inputs
            self._slurm_parsed = True

        return {
            "code": self.CODE_NAME,
            "source_directory": str(self.model_dir.resolve()),
            "release_version": self._slurm_rel_ver,
            "build_date": self._slurm_bld_date
        }

    def get_reproducibility_data(self, mode_dir):
        """Extracts hashes for crucial data files inside the mode directory."""
        hashes = {}

        c1_path = mode_dir / 'C1.h5'
        if c1_path.exists():
            hashes['C1.h5'] = compute_file_hash(c1_path)

        eq_path = mode_dir / 'equilibrium.h5'
        if eq_path.exists():
            hashes['equilibrium.h5'] = compute_file_hash(eq_path)

        for time_file in mode_dir.glob('time_*.h5'):
            hashes[time_file.name] = compute_file_hash(time_file)

        return hashes

    def get_all_input_parameters(self):
        """Retrieve input parameters parsed from the Slurm output file."""
        # Returned from internal cache populated during get_metadata
        return getattr(self, '_slurm_inputs', {})

    def get_global_parameters(self):
        """Retrieve globally shared parameters automatically from the gamma file and shape calculations."""
        globals_dict = {}
        if self.gamma_data is not None:
            globals_dict["vpnum"]       = getattr(self.gamma_data, "vpnum", -1)
            globals_dict["eta"]         = getattr(self.gamma_data, "eta", -1.0)
            globals_dict["bscale"]      = getattr(self.gamma_data, "bscale", -1.0)
            globals_dict["rotation"]    = getattr(self.gamma_data, "rotation", -1)
            globals_dict["fluidmodel"]  = getattr(self.gamma_data, "fluidmodel", "unknown")
            globals_dict["ipres"]       = getattr(self.gamma_data, "ipres", 0)
            globals_dict["pped"]        = getattr(self.gamma_data, "pped", -1.0)
            globals_dict["jped"]        = getattr(self.gamma_data, "jped", -1.0)
            globals_dict["jeliteped"]   = getattr(self.gamma_data, "jeliteped", -1.0)
            globals_dict["omegsti_max"] = getattr(self.gamma_data, "omegsti_max", -1.0)

        # Automatically calculate and append the plasma shape parameters
        try:
            sim = self._get_eq_sim()
            if sim is not None:
                shape_dict = get_shape(sim, quiet=True)
                globals_dict.update(shape_dict)
        except Exception as e:
            printwarn(f"Warning: Could not calculate shaping parameters: {e}")

        return globals_dict

    def get_1d_mesh(self, resolution, units, fcoords):
        """Extracts the 1D radial grid used by flux_average."""
        sim = self._get_eq_sim()
        if sim is None:
            return np.array([])
        m3dc1_units = self._map_units(units)
        try:
            # We call flux_average on a default field just to obtain the grid coordinates
            nflux, _ = flux_average(field='p', sim=sim, filename=sim.filename, fcoords=fcoords, use_eq_fs=True, points=resolution, time=-1, units=m3dc1_units)
            return np.asarray(nflux)
        except Exception:
            return np.array([])

    def get_2d_mesh(self, grid_spec):
        """Calculates and returns the (R, Z) coordinates for 2D field extractions."""
        if grid_spec.type == "rectangular":
            points = self._get_inner_wall_points()
            if len(points) > 0:
                rmin, rmax = np.min(points[:, 0]), np.max(points[:, 0])
                zmin, zmax = np.min(points[:, 1]), np.max(points[:, 1])
            else:
                rmin, rmax, zmin, zmax = 1.0, 2.0, -1.0, 1.0
            nR, nZ = grid_spec.resolution
            R_lin = np.linspace(rmin, rmax, nR)
            Z_lin = np.linspace(zmin, zmax, nZ)
            R, Z = np.meshgrid(R_lin, Z_lin, indexing='ij')
            return np.column_stack((R.ravel(), Z.ravel()))
        elif grid_spec.type == "inside_wall":
            points = self._get_inner_wall_points()
            if len(points) == 0:
                return np.array([])
            return points
        elif grid_spec.type == "native":
            return np.zeros((100, 2))
        raise ValueError(f"Unknown grid type: {grid_spec.type}")

    def get_3d_mesh(self, grid_spec):
        """Calculates and returns the (R, phi, Z) coordinates for 3D field extractions."""
        if grid_spec.type == "rectangular":
            points = self._get_inner_wall_points()
            if len(points) > 0:
                rmin, rmax = np.min(points[:, 0]), np.max(points[:, 0])
                zmin, zmax = np.min(points[:, 1]), np.max(points[:, 1])
            else:
                rmin, rmax, zmin, zmax = 1.0, 2.0, -1.0, 1.0
            nR, nPhi, nZ = grid_spec.resolution
            R_lin = np.linspace(rmin, rmax, nR)
            Phi_lin = np.linspace(0, 2*np.pi, nPhi, endpoint=False)
            Z_lin = np.linspace(zmin, zmax, nZ)
            R, phi, Z = np.meshgrid(R_lin, Phi_lin, Z_lin, indexing='ij')
            return np.column_stack((R.ravel(), phi.ravel(), Z.ravel()))
        elif grid_spec.type == "inside_wall":
            points = self._get_inner_wall_points()
            if len(points) == 0:
                return np.array([])
            R = points[:, 0]
            Z = points[:, 1]
            phi = np.zeros_like(R)
            return np.column_stack((R, phi, Z))
        elif grid_spec.type == "native":
            return np.zeros((100, 3))
        raise ValueError(f"Unknown grid type: {grid_spec.type}")

    def get_2d_field(self, name, grid_spec, units, time=-1):
        # Allow specific time slice logic; fallback to equilibrium if time=-1
        sim = self._get_eq_sim() if time == -1 else self._get_time_sim(time)
        m3dc1_units = self._map_units(units)
        
        if grid_spec.type == "rectangular":
            return self._evaluate_rectangular_grid(name, grid_spec, sim, m3dc1_units, is_3d=False)
        elif grid_spec.type == "native":
            return self._evaluate_native_grid(name, sim, m3dc1_units)
        elif grid_spec.type == "inside_wall":
            return self._evaluate_inside_wall(name, sim, m3dc1_units)
        raise ValueError(f"Unknown grid type: {grid_spec.type}")

    def get_3d_field(self, name, grid_spec, units, time=-1):
        # Allow specific time slice logic; fallback to equilibrium if time=-1
        sim = self._get_eq_sim() if time == -1 else self._get_time_sim(time)
        m3dc1_units = self._map_units(units)
        
        if grid_spec.type == "rectangular":
            return self._evaluate_rectangular_grid(name, grid_spec, sim, m3dc1_units, is_3d=True)
        elif grid_spec.type == "native":
            return self._evaluate_native_grid(name, sim, m3dc1_units)
        elif grid_spec.type == "inside_wall":
            return self._evaluate_inside_wall(name, sim, m3dc1_units)
        raise ValueError(f"Unknown grid type: {grid_spec.type}")

    def get_flux_average(self, name, resolution, units, fcoords, time=-1, use_eq_fs=True):
        """
        Utilizes the imported flux_average function on the equilibrium sim.
        Returns a tuple of (normalized_flux_array, values_array) as 1D numpy arrays.
        """
        sim = self._get_eq_sim() if time == -1 else self._get_time_sim(time)
        if sim is None:
            return np.array([]), np.array([])

        m3dc1_units = self._map_units(units)
        
        # Check field type to handle vectors vs scalars
        field_info = getattr(sim, 'typedict', getattr(sim, 'available_fields', {})).get(name, None)
        field_type = field_info[1] if field_info else 'scalar'

        try:
            if field_type == 'vector':
                nflux, fR = flux_average(field=name, sim=sim, filename=sim.filename, fcoords=fcoords,
                                         use_eq_fs=use_eq_fs, points=resolution, time=time,
                                         units=m3dc1_units, coord='R')
                _, fPhi = flux_average(field=name, sim=sim, filename=sim.filename, fcoords=fcoords,
                                       use_eq_fs=use_eq_fs, points=resolution, time=time,
                                       units=m3dc1_units, coord='phi')
                _, fZ = flux_average(field=name, sim=sim, filename=sim.filename, fcoords=fcoords,
                                     use_eq_fs=use_eq_fs, points=resolution, time=time,
                                     units=m3dc1_units, coord='Z')
                return np.asarray(nflux), {
                    f"{name}_R": np.asarray(fR),
                    f"{name}_phi": np.asarray(fPhi),
                    f"{name}_Z": np.asarray(fZ)
                }
            else:
                nflux, fa = flux_average(field=name, sim=sim, filename=sim.filename, fcoords=fcoords,
                                         use_eq_fs=use_eq_fs, points=resolution, time=time,
                                         units=m3dc1_units, coord='scalar')
                return np.asarray(nflux), np.asarray(fa)
        except Exception as e:
            printerr(f"Error calculating flux average for {name}: {e}")
            return np.array([]), np.array([])

    def get_time_trace(self, name, units):
        """
        Extracts the requested scalar time trace using the native get_timetrace function.
        Returns a tuple of (time_array, values_array) as 1D numpy arrays.
        """
        sim = self._get_eq_sim()
        if sim is None:
            return np.array([]), np.array([])

        m3dc1_units = self._map_units(units)

        try:
            # Call the native time trace function
            time_arr, values, label, unitlabel = get_timetrace(
                trace=name, sim=sim, filename=sim.filename,
                units=m3dc1_units, quiet=True, returnas='tuple'
            )
            return np.asarray(time_arr), np.asarray(values)
        except Exception as e:
            printerr(f"Error extracting time trace {name}: {e}")
            return np.array([]), np.array([])

    def get_mode_metadata(self, mode_dir):
        """Extracts growth rates and mode type specifically matching this directory's n."""
        meta = {
            "growth_rate": 0.0,
            "frequency": 0.0,
            "mode_type": -100 # Using -100 as default for unknown
        }

        if self.gamma_data is not None:
            try:
                # Mode directories follow the 'nXX' convention
                n_val = int(mode_dir.name.replace('n', ''))
                
                # Locate the index of this toroidal harmonic in the parsed text data
                idx_array = np.where(self.gamma_data.n_list == n_val)[0]
                if len(idx_array) > 0:
                    idx = idx_array[0]
                    meta["growth_rate"] = float(self.gamma_data.gamma_list[idx])
                    # pblist holds the mode type
                    meta["mode_type"] = int(self.gamma_data.pblist[idx])
            except ValueError:
                pass

        return meta

    def get_mode_1d_profile(self, mode_dir, name, resolution, units): return np.zeros(resolution)

    def get_mode_2d_mesh(self, mode_dir, grid_spec):
        return self.get_2d_mesh(grid_spec)

    def get_mode_3d_mesh(self, mode_dir, grid_spec):
        return self.get_3d_mesh(grid_spec)

    def get_mode_2d_field(self, mode_dir, name, grid_spec, units):
        """
        Evaluates the field at the mode's last time step (finite time).
        The eigenfunction is computed in extract_mode() by subtracting
        the equilibrium field (time=-1) from this result.
        """
        sim = self._get_mode_sim(mode_dir)
        m3dc1_units = self._map_units(units)
        if grid_spec.type == "rectangular":
            return self._evaluate_rectangular_grid(name, grid_spec, sim, m3dc1_units, is_3d=False)
        elif grid_spec.type == "native":
            return self._evaluate_native_grid(name, sim, m3dc1_units)
        elif grid_spec.type == "inside_wall":
            return self._evaluate_inside_wall(name, sim, m3dc1_units)
        raise ValueError(f"Unknown grid type: {grid_spec.type}")

    def get_mode_3d_field(self, mode_dir, name, grid_spec, units):
        """
        Evaluates the field at the mode's last time step (finite time).
        The eigenfunction is computed in extract_mode() by subtracting
        the equilibrium field (time=-1) from this result.
        """
        sim = self._get_mode_sim(mode_dir)
        m3dc1_units = self._map_units(units)
        if grid_spec.type == "rectangular":
            return self._evaluate_rectangular_grid(name, grid_spec, sim, m3dc1_units, is_3d=True)
        elif grid_spec.type == "native":
            return self._evaluate_native_grid(name, sim, m3dc1_units)
        elif grid_spec.type == "inside_wall":
            return self._evaluate_inside_wall(name, sim, m3dc1_units)
        raise ValueError(f"Unknown grid type: {grid_spec.type}")

    # --- Total Fields / Multiple Time Slices Logic ---

    def get_time_metadata(self, time_slices):
        meta = {
            "time_slice":            np.array(time_slices, dtype=int),
            "simulation_time_step":  np.zeros(len(time_slices), dtype=int),
            "simulation_time":       np.zeros(len(time_slices), dtype=float)
        }

        c1_paths = list(self.model_dir.rglob("C1.h5"))
        c1_file  = str(c1_paths[0]) if c1_paths else "C1.h5"
        file_dir = c1_paths[0].parent if c1_paths else self.model_dir

        for i, ts in enumerate(time_slices):
            sim = self._get_eq_sim() if ts == -1 else self._get_time_sim(ts)
            h5file = sim._all_attrs
            if sim is not None:
                if ts == -1:
                    fname = "equilibrium.h5"
                else:
                    fname = f"time_{ts:03d}.h5"
                    
                file_path = str(file_dir / fname)
                
                try:
                    meta["simulation_time"][i] = get_time_of_slice(ts, sim=sim, filename=c1_file, units='m3dc1')
                except Exception as e:
                    printerr(f"Error extracting simulation_time for slice {ts}: {e}")

                try:
                    meta["simulation_time_step"][i] = int(readParameter('ntimestep', h5file=h5file))
                except Exception as e:
                    printerr(f"Error extracting ntimestep for slice {ts} from {fname}: {e}")

        return meta

    def convert_to_imas(self, mode_dirs, output_directory, config, run=None, out_file=None):
        """
        Writes IMAS-compatible output for this M3D-C1 model directory.

        The conversion backend is selected by config['m3dc1_imas_conversion_lib']:
            'standalone'  (default) – uses m3dc1.convert2imas, which writes a
                                       self-contained HDF5 file without requiring
                                       imas-python.
            'imas-python'           – uses m3dc1.convert2imas_imaspy, which writes
                                       a proper IMAS data entry via the imas-python
                                       library.  imas-python must be installed.

        Returns a dictionary summarising the produced artefact, or None on failure.
        """
        conversion_lib = str(
            config.get("m3dc1_imas_conversion_lib", "standalone")
        ).strip().lower()

        target_filename = (
            str(mode_dirs[0] / 'C1.h5') if mode_dirs
            else str(self.model_dir / 'C1.h5')
        )

        res = config.get("resolutions", {}).get("1d", 200)

        # ------------------------------------------------------------------
        # STANDALONE (default): plain HDF5, no imas-python required
        # ------------------------------------------------------------------
        if conversion_lib == "standalone":
            if out_file is None:
                out_file = Path(output_directory) / generate_unique_filename(
                    self.model_dir, 'm3dc1'
                )
            out_file = Path(out_file)

            try:
                convert2imas(
                    filename=target_filename,
                    out_file=str(out_file),
                    nr=res,
                    nz=res,
                )
            except Exception as exc:
                printerr(f"convert2imas (standalone) failed for {self.model_dir}: {exc}")
                return None

            return {
                "code":             self.CODE_NAME,
                "conversion_lib":   "standalone",
                "model_directory":  str(self.model_dir.resolve()),
                "source_file":      target_filename,
                "output_file":      str(out_file),
            }

        # ------------------------------------------------------------------
        # IMAS-PYTHON: full IMAS data entry via imas-python
        # ------------------------------------------------------------------
        elif conversion_lib == "imas-python":
            imas_cfg = dict(config.get("imas", {}) or {})

            machine, shot = _infer_machine_and_shot(self.model_dir)
            db_name    = str(imas_cfg.get("dd")      or machine or "m3dc1")
            pulse      = int(imas_cfg.get("pulse")   or shot    or 1)
            db_version = int(imas_cfg.get("db_version", 1))
            backend    = str(imas_cfg.get("backend", "hdf5"))
            db_path    = imas_cfg.get("dbpath") or str(output_directory)

            if run is None:
                run = imas_cfg.get("run", None)
            run = _allocate_imas_run(db_path, db_name, db_version, pulse, run)

            try:
                convert2imas_imaspy(
                    filename=target_filename,
                    nr=res,
                    nz=res,
                    backend=backend,
                    db_name=db_name,
                    db_version=db_version,
                    pulse=pulse,
                    run=run,
                    db_path=db_path,
                )
            except Exception as exc:
                printerr(f"convert2imas_imaspy failed for {self.model_dir}: {exc}")
                return None

            entry = _imas_entry_dir(db_path, db_name, db_version, pulse, run)
            return {
                "code":             self.CODE_NAME,
                "conversion_lib":   "imas-python",
                "model_directory":  str(self.model_dir.resolve()),
                "source_file":      target_filename,
                "entry_directory":  str(entry),
                "db_name":          db_name,
                "pulse":            pulse,
                "run":              run,
                "backend":          backend,
            }

        else:
            printerr(
                f"Unknown m3dc1_imas_conversion_lib value '{conversion_lib}'. "
                "Valid choices are 'standalone' and 'imas-python'."
            )
            return None

    # -----------------------------------------------------------------------
    # M3D-C1 Specific Inner Wall / Mesh Logic
    # -----------------------------------------------------------------------

    def _get_inner_wall_points(self):
        """
        Uses the imported get_points_inside_inner_wall function to extract 
        the mesh points inside the wall. Caches the result to avoid redundant calculations.
        """
        # Override with shared grid if provided
        if getattr(self, 'shared_inner_wall_points', None) is not None:
            return self.shared_inner_wall_points

        if self._inner_wall_points is None:
            sim = self._get_eq_sim()
            if sim is None:
                self._inner_wall_points = np.array([])
            else:
                try:
                    self._inner_wall_points = get_points_inside_inner_wall(sim=sim, quiet=True)
                except Exception as e:
                    printerr(f"Error obtaining inner wall points: {e}")
                    self._inner_wall_points = np.array([])

        return self._inner_wall_points

    def _evaluate_inside_wall(self, field_name, sim, units):
        """
        Retrieves the (R, Z) coordinates inside the inner wall and evaluates 
        the requested field at those specific points using the provided sim object.
        Returns the field value perfectly aligned with the mesh.
        """
        points = self._get_inner_wall_points()
        if len(points) == 0: return np.array([])
        R = points[:, 0]
        Z = points[:, 1]
        phi = np.zeros_like(R)  # Assuming a single 2D poloidal plane cross-section 
        
        # Check field type to handle vectors vs scalars
        field_info = getattr(sim, 'typedict', getattr(sim, 'available_fields', {})).get(field_name, None)
        field_type = field_info[1] if field_info else 'scalar'

        if field_type == 'vector':
            # Returns 3 components (R, phi, Z) when coord='vector'
            vR, vPhi, vZ = eval_field(field_name, R, phi, Z, coord='vector', sim=sim, filename=sim.filename, quiet=False)
            return {
                f"{field_name}_R":   vR,
                f"{field_name}_phi": vPhi,
                f"{field_name}_Z":   vZ
            }
        else:
            # Scalar evaluation
            values = eval_field(field_name, R, phi, Z, coord='scalar', sim=sim, filename=sim.filename, quiet=False)
            return values

    def _evaluate_rectangular_grid(self, field_name, grid_spec, sim, units, is_3d=False):
        """
        Evaluates the field on a regular rectangular grid.
        Infers bounding box from inner wall points to keep the grid localized.
        """
        points = self._get_inner_wall_points()
        if len(points) > 0:
            rmin, rmax = np.min(points[:, 0]), np.max(points[:, 0])
            zmin, zmax = np.min(points[:, 1]), np.max(points[:, 1])
        else:
            # Fallback placeholder if no inner wall boundary is found
            rmin, rmax, zmin, zmax = 1.0, 2.0, -1.0, 1.0

        if is_3d:
            nR, nPhi, nZ = grid_spec.resolution
            R_lin   = np.linspace(rmin, rmax, nR)
            Phi_lin = np.linspace(0, 2*np.pi, nPhi, endpoint=False)
            Z_lin   = np.linspace(zmin, zmax, nZ)
            R, phi, Z = np.meshgrid(R_lin, Phi_lin, Z_lin, indexing='ij')
            R = R.ravel()
            phi = phi.ravel()
            Z = Z.ravel()
        else:
            nR, nZ = grid_spec.resolution
            R_lin = np.linspace(rmin, rmax, nR)
            Z_lin = np.linspace(zmin, zmax, nZ)
            R, Z = np.meshgrid(R_lin, Z_lin, indexing='ij')
            R = R.ravel()
            Z = Z.ravel()
            phi = np.zeros_like(R)

        # Check field type to handle vectors vs scalars
        field_info = getattr(sim, 'typedict', getattr(sim, 'available_fields', {})).get(field_name, None)
        field_type = field_info[1] if field_info else 'scalar'

        if field_type == 'vector':
            vR, vPhi, vZ = eval_field(field_name, R, phi, Z, coord='vector', sim=sim, filename=sim.filename, quiet=True)
            return {
                f"{field_name}_R":   vR,
                f"{field_name}_phi": vPhi,
                f"{field_name}_Z":   vZ
            }
        else:
            values = eval_field(field_name, R, phi, Z, coord='scalar', sim=sim, filename=sim.filename, quiet=True)
            return values

    def _evaluate_native_grid(self, field_name, sim, units):
        return np.zeros((100,))


# ===========================================================================
# NIMROD ADAPTER
# ===========================================================================

class NIMRODAdapter(SimulationAdapter):
    CODE_NAME = "NIMROD"

    # Sign flip for the toroidal (phi) component to match M3D-C1 conventions.
    # NIMROD eval returns (R, Z, phi). We unpack it and apply this flip to the phi array.
    PHI_SIGN_FLIP = -1.0

    def __init__(self, model_dir):
        super().__init__(model_dir)
        self._dumps_converted = False
        # Cache for nimgrowth results, keyed by mode directory path
        self._nimgrowth_cache = {}
        # Per-directory cache for indexed dump lists, keyed by directory path
        self._indexed_dumps_per_dir = {}
        # Cache for the common maximum index available across all known nXX directories
        self._common_max_index_cache = None
        # The set of known nXX mode directories, populated by set_mode_dirs()
        self._mode_dirs = []

    def set_mode_dirs(self, mode_dirs):
        """
        Registers the list of nXX mode directories for this model.
        Must be called before get_time_metadata() or _get_dump_file() so that
        the common-index fallback logic can consider all directories.
        """
        self._mode_dirs = [Path(d) for d in mode_dirs]
        # Invalidate the common index cache whenever mode dirs change
        self._common_max_index_cache = None

    def _ensure_all_h5_dumps(self):
        """Scans for binary dumps missing an .h5 counterpart and converts them."""
        if self._dumps_converted:
            return

        dump_dirs = set()
        for f in self.model_dir.rglob("dumpgll.*"):
            parts = f.name.split('.')
            # Only consider files whose suffix is purely digits (binary dumps),
            # explicitly excluding .h5 files to avoid treating them as binary dumps.
            if len(parts) == 2 and parts[1].isdigit():
                if (f.parent / "nimrod.in").exists():
                    dump_dirs.add(f.parent)

        for d in dump_dirs:
            self._convert_binary_dumps_to_h5(d)

        self._dumps_converted = True

    def _convert_binary_dumps_to_h5(self, target_dir):
        """Executes the 4-step workflow to convert binary dump files to HDF5."""
        # Find binary dumps lacking .h5 equivalents.
        # Only consider files whose suffix is purely digits (binary dumps),
        # explicitly excluding .h5 files.
        bin_dumps = []
        for f in target_dir.glob("dumpgll.*"):
            parts = f.name.split('.')
            if len(parts) == 2 and parts[1].isdigit():
                h5_path = target_dir / f"{f.name}.h5"
                if not h5_path.exists():
                    bin_dumps.append(f)

        if not bin_dumps:
            return

        printnote(f"Found {len(bin_dumps)} binary dump files in {target_dir} without .h5 counterparts. Starting conversion...")

        # 1. Copy dump2h5 tool
        tool_source = "/global/common/software/nimrod/spack/perlmutter/2026-4/linux-zen3/nimdevel-trunk-nq6numz6t3p7cat6jj3iro5gnajysvkb/bin/dump2h5"
        tool_dest = target_dir / "dump2h5"
        if not tool_dest.exists():
            try:
                shutil.copy(tool_source, tool_dest)
                os.chmod(tool_dest, 0o755) # Ensure executable
            except Exception as e:
                printwarn(f"Failed to copy dump2h5 tool to {target_dir}: {e}")
                return

        # 2 & 4. Modify nimrod.in
        nimrod_in = target_dir / "nimrod.in"
        if nimrod_in.exists():
            try:
                with open(nimrod_in, 'r') as f:
                    lines = f.readlines()

                in_output_input = False
                found_compat = False
                found_h5dump = False

                for i, line in enumerate(lines):
                    if re.match(r'^\s*&output_input', line, re.IGNORECASE):
                        in_output_input = True
                        continue

                    if in_output_input:
                        if re.match(r'^\s*/', line):
                            # End of namelist, insert missing keys
                            inserts = []
                            if not found_compat:
                                inserts.append(" nimuw_dump_compat=T\n")
                            if not found_h5dump:
                                inserts.append(" h5dump=T\n")
                            if inserts:
                                lines.insert(i, "".join(inserts))
                            in_output_input = False
                            continue

                        clean_line = line.split('!')[0]
                        if re.search(r'nimuw_dump_compat\s*=', clean_line, re.IGNORECASE):
                            lines[i] = re.sub(r'(nimuw_dump_compat\s*=\s*)[\w\.]+', r'\g<1>T', lines[i], flags=re.IGNORECASE)
                            found_compat = True
                        if re.search(r'h5dump\s*=', clean_line, re.IGNORECASE):
                            lines[i] = re.sub(r'(h5dump\s*=\s*)[\w\.]+', r'\g<1>T', lines[i], flags=re.IGNORECASE)
                            found_h5dump = True

                with open(nimrod_in, 'w') as f:
                    f.writelines(lines)
            except Exception as e:
                printwarn(f"Failed to modify nimrod.in in {target_dir}: {e}")
        else:
            printwarn(f"nimrod.in not found in {target_dir}. Proceeding with conversion anyway.")

        # 3. Run the conversion tool
        original_cwd = os.getcwd()
        try:
            os.chdir(target_dir)
            for bd in bin_dumps:
                printnote(f"Converting {bd.name} to HDF5...")
                subprocess.run(["./dump2h5", bd.name], check=True, capture_output=True)
        except Exception as e:
            printwarn(f"Error executing dump2h5 conversion in {target_dir}: {e}")
        finally:
            os.chdir(original_cwd)

    def _parse_nimgrowth(self, mode_dir):
        """
        Runs 'nimgrowth energy.bin' in the given mode directory and parses the
        printed output to extract growth rates for each toroidal mode number (keff).

        The output format is expected to be:
            Growth rates given in units of s^-1
            Var           Growth (Re gamma)           Npts
              keff = <n>
            E magnetic :  <value> +/- <uncertainty> ( <npts> )
            E kinetic  :  <value> +/- <uncertainty> ( <npts> )
              keff = <n>
            ...

        Only the 'E kinetic' lines are used. The first number on each such line
        is the growth rate; the uncertainty (after +/-) is ignored.

        Returns a dict mapping integer toroidal mode number n -> float growth rate (s^-1).
        Returns an empty dict if the command fails or produces no parseable output.
        """
        mode_dir = Path(mode_dir)
        energy_bin = mode_dir / "energy.bin"

        if not energy_bin.exists():
            printwarn(f"nimgrowth: energy.bin not found in {mode_dir}. Skipping growth rate extraction.")
            return {}

        original_cwd = os.getcwd()
        growth_rates = {}
        try:
            os.chdir(mode_dir)
            result = subprocess.run(["nimgrowth", "energy.bin"], capture_output=True, text=True)
            output = result.stdout

            current_keff = None
            for line in output.splitlines():
                line_stripped = line.strip()

                # Match lines like:   keff = 8.0
                keff_match = re.match(r'keff\s*=\s*([\d.]+)', line_stripped)
                if keff_match:
                    current_keff = int(float(keff_match.group(1)))
                    continue

                # Match lines like:  E kinetic  :  2.4998e+04 +/- 3.1207e+01 (   49 )
                # We only want the first number (the growth rate), ignoring the uncertainty.
                if line_stripped.startswith("E kinetic") and current_keff is not None:
                    kinetic_match = re.search(r':\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)', line_stripped)
                    if kinetic_match:
                        growth_rates[current_keff] = float(kinetic_match.group(1))

        except FileNotFoundError:
            printwarn(f"nimgrowth executable not found. Cannot extract NIMROD growth rates from {mode_dir}.")
        except Exception as e:
            printerr(f"Error running nimgrowth in {mode_dir}: {e}")
        finally:
            os.chdir(original_cwd)

        return growth_rates

    def _get_nimgrowth_for_dir(self, mode_dir):
        """
        Returns the cached nimgrowth results for a given mode directory,
        running the parser on first access.
        """
        mode_dir = Path(mode_dir)
        if mode_dir not in self._nimgrowth_cache:
            self._nimgrowth_cache[mode_dir] = self._parse_nimgrowth(mode_dir)
        return self._nimgrowth_cache[mode_dir]

    def _is_zero_dump(self, filepath):
        """
        Returns True if the HDF5 dump file contains only zero-valued data fields,
        which identifies it as the initial (t=0) state equivalent to M3D-C1's time_000.h5.
        Checks a representative scalar field dataset for non-zero values.
        """
        try:
            with h5py.File(filepath, 'r') as h5:
                # Look inside 'rblocks' for any numeric dataset and check for non-zeros
                rblocks = h5.get('rblocks')
                if rblocks:
                    for block_name in rblocks.keys():
                        for dset_name in rblocks[block_name].keys():
                            dset = rblocks[block_name][dset_name]
                            if hasattr(dset, 'shape') and dset.size > 0:
                                arr = dset[()]
                                if np.any(arr != 0):
                                    return False
                # If we found no non-zero values, treat it as a zero dump
                return True
        except Exception:
            # If we cannot read the file, assume it is not a zero dump
            return False

    def _get_indexed_dumps_for_dir(self, target_dir):
        """
        Builds and caches the indexed list of NIMROD HDF5 dump files for a single
        directory (e.g. one nXX subdirectory).

        Only HDF5 dump files (dumpgll.<digits>.h5) directly inside target_dir are
        considered; binary dump files are ignored entirely.

        Indexing convention (matching M3D-C1 semantics):
          - Index 0: the dump file whose numeric step in the filename is 0, OR
                     the first dump file found to contain only zero-valued data.
                     This is the equivalent of M3D-C1's 'time_000.h5'.
          - Index 1, 2, ...: the remaining dump files sorted by ascending step number.

        Returns a list of (index, step_number, filepath) tuples sorted by index.
        """
        target_dir = Path(target_dir)
        if target_dir in self._indexed_dumps_per_dir:
            return self._indexed_dumps_per_dir[target_dir]

        self._ensure_all_h5_dumps()

        # Collect only HDF5 dump files directly inside this directory
        raw = []
        for f in target_dir.glob("dumpgll.*.h5"):
            parts = f.name.split('.')
            # Expected format: dumpgll.<digits>.h5 -> parts = ['dumpgll', '<digits>', 'h5']
            if len(parts) == 3 and parts[1].isdigit() and parts[2] == 'h5':
                step = int(parts[1])
                raw.append((step, f))

        if not raw:
            self._indexed_dumps_per_dir[target_dir] = []
            return []

        # Sort by step number ascending
        sorted_entries = sorted(raw)

        printnote(
            f"NIMROD: Found {len(sorted_entries)} HDF5 dump(s) in {target_dir}: "
            + ", ".join(str(s) for s, _ in sorted_entries)
        )

        # Identify the zero dump:
        # Prefer a file whose step number is literally 0; otherwise check file contents.
        zero_entry = None
        nonzero_entries = []

        if sorted_entries[0][0] == 0:
            # The file with step 0 in its name is the zero dump
            zero_entry = sorted_entries[0]
            nonzero_entries = sorted_entries[1:]
        else:
            # No step-0 file exists; scan for a file containing only zeros
            for entry in sorted_entries:
                if zero_entry is None and self._is_zero_dump(str(entry[1])):
                    zero_entry = entry
                else:
                    nonzero_entries.append(entry)

        # Assign indices: 0 for the zero dump, then 1, 2, ... for the rest
        indexed = []
        if zero_entry is not None:
            indexed.append((0, zero_entry[0], zero_entry[1]))

        for sequential_idx, (step, path) in enumerate(nonzero_entries, start=1):
            indexed.append((sequential_idx, step, path))

        self._indexed_dumps_per_dir[target_dir] = indexed
        return indexed

    def _get_common_max_index(self, requested_idx):
        """
        Determines the best time slice index to use when the requested index does
        not exist in all nXX directories.

        For each registered nXX mode directory, the set of available sequential
        indices is computed. The method then finds the largest index that is:
          (a) <= requested_idx, and
          (b) present in every nXX directory.

        If no such common index exists, the globally smallest available index
        across all directories is returned as a last resort.

        Returns the resolved integer index and prints a warning if a fallback
        was necessary.
        """
        if not self._mode_dirs:
            # No mode dirs registered; cannot determine common index
            return requested_idx

        # Collect the set of available indices for each mode directory
        per_dir_index_sets = []
        for d in self._mode_dirs:
            indexed = self._get_indexed_dumps_for_dir(d)
            indices = set(idx for idx, _, _ in indexed)
            if indices:
                per_dir_index_sets.append(indices)

        if not per_dir_index_sets:
            return requested_idx

        # Find indices present in ALL directories (intersection)
        common_indices = per_dir_index_sets[0]
        for s in per_dir_index_sets[1:]:
            common_indices = common_indices & s

        if not common_indices:
            # No index is common to all directories; fall back to smallest overall
            all_indices = sorted(set().union(*per_dir_index_sets))
            fallback = all_indices[0]
            printwarn(
                f"Warning: No time slice index is common to all nXX directories. "
                f"Using the smallest available index {fallback} as fallback."
            )
            return fallback

        common_indices_sorted = sorted(common_indices)

        if requested_idx in common_indices:
            return requested_idx

        # Find the largest common index <= requested_idx
        smaller = [i for i in common_indices_sorted if i <= requested_idx]
        if smaller:
            fallback = max(smaller)
            printwarn(
                f"Warning: Requested time slice index {requested_idx} does not exist in all "
                f"nXX directories (common indices: {common_indices_sorted}). "
                f"Using the next smaller common index {fallback} instead."
            )
            return fallback

        # No common index <= requested_idx; use the smallest common index
        fallback = common_indices_sorted[0]
        printwarn(
            f"Warning: Requested time slice index {requested_idx} is smaller than all common "
            f"indices across nXX directories (common indices: {common_indices_sorted}). "
            f"Using the smallest common index {fallback} instead."
        )
        return fallback

    def _get_dump_file(self, time=-1):
        """
        Selects a NIMROD HDF5 dump file by its sequential index for the first
        registered nXX mode directory (used for equilibrium/mesh operations that
        are not per-mode-dir).

        Index 0 corresponds to the zero-state dump (equivalent to M3D-C1's time_000.h5).
        Indices 1, 2, ... correspond to subsequent dump files sorted by ascending step number.

        If time=-1 (equilibrium request), index 0 is used.
        If the requested index does not exist, the common fallback index across all
        nXX directories is used (see _get_common_max_index).

        Returns the file path as a string, or None if no dump files are found.
        """
        # Use the first registered mode dir, or fall back to searching model_dir directly
        if self._mode_dirs:
            search_dir = self._mode_dirs[0]
        else:
            # Fallback: find the first nXX-like subdirectory containing nimrod.in
            search_dir = None
            for d in sorted(self.model_dir.iterdir()):
                if d.is_dir() and re.match(r'^n(\d+|ln)', d.name) and (d / "nimrod.in").exists():
                    search_dir = d
                    break
            if search_dir is None:
                printwarn(f"Warning: Could not find any nXX directory with nimrod.in in {self.model_dir}")
                return None

        indexed = self._get_indexed_dumps_for_dir(search_dir)

        if not indexed:
            printwarn(f"Warning: Could not find any dumpgll.*.h5 files in {search_dir}")
            return None

        # Map time=-1 (equilibrium) to index 0
        requested_idx = 0 if time == -1 else time

        index_map = {idx: (step, path) for idx, step, path in indexed}

        if requested_idx in index_map:
            return str(index_map[requested_idx][1])

        # Requested index not in this directory; use the common fallback
        resolved_idx = self._get_common_max_index(requested_idx)
        if resolved_idx in index_map:
            return str(index_map[resolved_idx][1])

        # Last resort: use the largest available index in this directory
        fallback_idx = max(index_map.keys())
        return str(index_map[fallback_idx][1])

    def _get_dump_file_for_dir(self, target_dir, resolved_idx):
        """
        Returns the HDF5 dump file path for a specific nXX directory and a
        pre-resolved sequential index.

        This is used after _get_common_max_index() has already determined the
        correct index to use across all directories.

        Returns the file path as a string, or None if the index is not found.
        """
        indexed   = self._get_indexed_dumps_for_dir(target_dir)
        index_map = {idx: (step, path) for idx, step, path in indexed}

        if resolved_idx in index_map:
            return str(index_map[resolved_idx][1])

        # If the resolved index is somehow missing in this dir, use the largest available
        if index_map:
            fallback_idx = max(index_map.keys())
            return str(index_map[fallback_idx][1])

        return None

    def _select_dump_files(self, mode_dir, selection="all"):
        """
        Selects which HDF5 dump files of a given nXX directory should be handed
        to dump2imas.

        'selection' accepts:
            "all" / None        -> every indexed dump, in time order
            "first"             -> the zero/initial dump only
            "last"              -> the most evolved dump only
            "first_last"        -> both endpoints
            [i0, i1, ...]       -> explicit sequential indices (-1 is mapped to 0);
                                   indices missing from this directory are resolved
                                   with _get_common_max_index()

        Returns a list of absolute file paths sorted by sequential index.
        """
        indexed = self._get_indexed_dumps_for_dir(mode_dir)
        if not indexed:
            return []

        index_map = {idx: path for idx, _, path in indexed}

        if selection is None:
            return [str(p) for _, _, p in indexed]

        if isinstance(selection, str):
            sel = selection.strip().lower()
            if sel in ("all", "*"):
                return [str(p) for _, _, p in indexed]
            if sel == "first":
                return [str(indexed[0][2])]
            if sel == "last":
                return [str(indexed[-1][2])]
            if sel in ("first_last", "firstlast", "endpoints"):
                out = [str(indexed[0][2])]
                if len(indexed) > 1:
                    out.append(str(indexed[-1][2]))
                return out
            printwarn(f"Unknown dump selection '{selection}'; using all dumps.")
            return [str(p) for _, _, p in indexed]

        if isinstance(selection, (list, tuple, set, np.ndarray)):
            resolved = set()
            for s in selection:
                try:
                    idx = int(s)
                except (TypeError, ValueError):
                    continue
                if idx == -1:
                    idx = 0
                if idx not in index_map:
                    idx = self._get_common_max_index(idx)
                if idx in index_map:
                    resolved.add(idx)
            return [str(index_map[i]) for i in sorted(resolved)]

        return [str(p) for _, _, p in indexed]

    def _imas_search_dirs(self, mode_dirs, parent_levels=1):
        """
        Builds the ordered list of directories searched for GEQDSK/PEQDSK files
        and NIMROD namelists: the model directory, then each nXX directory, then
        up to 'parent_levels' parent directories.
        """
        search_dirs = [self.model_dir]
        for m in (mode_dirs or []):
            m = Path(m)
            if m not in search_dirs:
                search_dirs.append(m)
        p = self.model_dir
        for _ in range(max(0, int(parent_levels))):
            if p.parent == p:
                break
            p = p.parent
            if p not in search_dirs:
                search_dirs.append(p)
        return search_dirs

    def _find_input_file(self, filename, mode_dirs, parent_levels=1):
        """Locates a named input file (e.g. 'nimrod.in') near this model directory."""
        for d in self._imas_search_dirs(mode_dirs, parent_levels):
            try:
                cand = d / filename
            except Exception:
                continue
            if cand.is_file():
                return cand
        return None

    def _find_eqdsk_inputs(self, mode_dirs, parent_levels=1):
        """
        Locates the GEQDSK and PEQDSK files needed by input2imas.
        Returns (geqdsk_path_or_None, peqdsk_path_or_None).
        """
        geqdsk = None
        peqdsk = None
        for d in self._imas_search_dirs(mode_dirs, parent_levels):
            if not d.is_dir():
                continue
            try:
                entries = sorted(d.iterdir())
            except OSError:
                continue
            for f in entries:
                if not f.is_file():
                    continue
                if geqdsk is None and _is_geqdsk_name(f.name):
                    geqdsk = f
                if peqdsk is None and _is_peqdsk_name(f.name):
                    peqdsk = f
            if geqdsk is not None and peqdsk is not None:
                break
        return geqdsk, peqdsk

    def convert_to_imas(self, mode_dirs, output_directory, config, run=None, out_file=None):
        """
        Converts this NIMROD model directory into a single IMAS data entry using
        the external 'nimrod2imas' tools.

        Workflow:
          1. input2imas  -> occurrence 0 (GEQDSK/PEQDSK equilibrium + namelists)
          2. dump2imas   -> occurrence occ_base + i*occ_stride, per nXX directory
          3. gamma2imas  -> growth rates/frequencies into the same occurrence

        The IMAS HDF5 backend stores data using its own internal naming convention
        where Array-of-Structures (AoS) paths use '[]&' separators, e.g.:
            /core_profiles_1/profiles_1d[]&electrons&pressure
        This is the correct and expected format for the IMAS HDF5 backend and
        should be read back through the IMAS Access Layer (imas-python), not
        directly via h5py.

        Returns a dictionary summarising the created entry, or None on failure.
        """
        imas_cfg = dict(config.get("imas", {}) or {})

        # Resolves script location, execution mode and interpreter, and reports
        # path problems and environment problems distinctly.
        rt = _prepare_nimrod2imas_runtime(imas_cfg)
        if rt is None:
            return None

        have_input2imas = _nimrod2imas_tool_available(rt, "input2imas")
        have_dump2imas = _nimrod2imas_tool_available(rt, "dump2imas")
        have_gamma2imas = _nimrod2imas_tool_available(rt, "gamma2imas")

        if not have_dump2imas and not have_input2imas:
            printerr("Neither dump2imas nor input2imas can be executed; aborting NIMROD IMAS conversion.")
            return None

        # --- entry coordinates -------------------------------------------------
        # dd_version must always be a full version string like "4.1.1" because
        # the nimrod2imas tools compute the entry directory version component as
        # args.dd_version[0] (the first character).  Never pass None here.
        dd_version = str(
            imas_cfg.get("dd_version")
            or os.environ.get("IMAS_VERSION")
            or "4.1.1"
        )
        backend    = str(imas_cfg.get("backend", "hdf5"))
        dbpath     = Path(imas_cfg.get("dbpath") or output_directory)
        dbpath.mkdir(parents=True, exist_ok=True)
        dbpath = dbpath.resolve()

        machine, shot = _infer_machine_and_shot(self.model_dir)
        dd    = str(imas_cfg.get("dd") or machine or "nimrod")
        pulse = int(imas_cfg.get("pulse") or shot or 1)

        if run is None:
            run = imas_cfg.get("run", None)
        run = _allocate_imas_run(dbpath, dd, dd_version, pulse, run)

        # _imas_entry_dir uses _imas_dd_version_dir() which replicates the tools'
        # args.dd_version[0] logic, so the path computed here always matches the
        # path the tools will write to.
        entry = _imas_entry_dir(dbpath, dd, dd_version, pulse, run)
        entry.mkdir(parents=True, exist_ok=True)

        printnote(
            f"NIMROD -> IMAS: {self.model_dir}\n"
            f"                dd={dd} pulse={pulse} run={run} dd_version={dd_version}\n"
            f"                entry={entry}"
        )

        # Make sure binary dumps have been converted before anything else.
        self._ensure_all_h5_dumps()

        # Common arguments shared by all three tools.
        # --mode a  : append mode so that input2imas, dump2imas and gamma2imas
        #             all write into the same entry without overwriting each other.
        common_args = [
            "--dd", dd,
            "--dd-version", dd_version,
            "--pulse", str(pulse),
            "--run", str(run),
            "--dbpath", str(dbpath),
            "--backend", backend,
            "--mode", "a",
        ]

        parent_levels = int(imas_cfg.get("search_parents", 1))

        summary = {
            "code":            self.CODE_NAME,
            "model_directory": str(self.model_dir.resolve()),
            "entry_directory": str(entry),
            "dd":              dd,
            "dd_version":      dd_version,
            "pulse":           pulse,
            "run":             run,
            "backend":         backend,
            "run_mode":        rt["mode"],
            "python":          rt["python"] if rt["mode"] == "subprocess" else sys.executable,
            "occurrences":     {},
            "steps":           [],
        }

        # --- 1) equilibrium + kinetic profiles (input2imas) -------------------
        if have_input2imas and imas_cfg.get("run_input2imas", True):
            geqdsk, peqdsk = self._find_eqdsk_inputs(mode_dirs, parent_levels)
            if geqdsk is not None and peqdsk is not None:
                argv  = ["input2imas.py", str(geqdsk.resolve()), str(peqdsk.resolve())]
                argv += common_args + ["--occ", "0"]

                for flag, fname in (("--nimeq", "nimeq.in"),
                                    ("--oculus", "oculus.in"),
                                    ("--fluxgrid", "fluxgrid.in"),
                                    ("--nimrod", "nimrod.in")):
                    found = self._find_input_file(fname, mode_dirs, parent_levels)
                    # Passing a non-existent path is safe: input2imas simply skips it.
                    argv += [flag, str(found.resolve()) if found else str(self.model_dir / fname)]

                yaml_meta = imas_cfg.get("metadata_yaml")
                if yaml_meta:
                    argv += ["--input", str(yaml_meta)]

                argv += _options_dict_to_cli(imas_cfg.get("input2imas_options"))
                argv += [str(a) for a in imas_cfg.get("input2imas_args", [])]

                printnote(f"  input2imas  occ=0  (g={geqdsk.name}, p={peqdsk.name})")
                ok = _run_nimrod2imas_tool(rt, "input2imas", argv, label="input2imas")
                summary["steps"].append({
                    "tool":       "input2imas",
                    "ok":         bool(ok),
                    "occurrence": 0,
                    "geqdsk":     str(geqdsk),
                    "peqdsk":     str(peqdsk),
                })
            else:
                printwarn(
                    f"Skipping input2imas for {self.model_dir}: "
                    f"GEQDSK={'found' if geqdsk else 'missing'}, "
                    f"PEQDSK={'found' if peqdsk else 'missing'}. "
                    "dump2imas will still run, but occurrence 0 (shared equilibrium) "
                    "will be absent."
                )

        # --- 2/3) per-mode dumps + growth rates -------------------------------
        occ_base_start = int(imas_cfg.get("occ_base", 1))
        occ_stride     = int(imas_cfg.get("occ_stride", 10))

        selection = imas_cfg.get("dump_selection", None)
        if selection is None:
            selection = config.get("time_slices", None)
        if selection is None or (isinstance(selection, (list, tuple)) and len(selection) == 0):
            selection = "all"

        gamma_endian  = str(imas_cfg.get("gamma_endian", ">"))
        gamma_nsteps  = int(imas_cfg.get("gamma_nsteps", 50))
        use_history   = bool(imas_cfg.get("gamma_use_history", True))

        # dump2imas installs a process-wide RLIMIT_AS (default 64 GB). That is
        # harmless in subprocess mode, but in-process it would permanently lower
        # the crawler's own hard limit, so it is disabled unless requested.
        mem_limit_gb = imas_cfg.get("mem_limit_gb", None)
        if mem_limit_gb is None:
            mem_limit_gb = 64 if rt["mode"] == "subprocess" else 0

        for i, mode_dir in enumerate(sorted([Path(m) for m in mode_dirs], key=lambda d: d.name)):
            occ = occ_base_start + i * occ_stride
            summary["occurrences"][mode_dir.name] = occ

            # ---- dump2imas ----
            dumps = self._select_dump_files(mode_dir, selection)
            if not have_dump2imas:
                printwarn("dump2imas is not available; skipping NIMROD dump conversion.")
            elif not dumps:
                printwarn(f"No dumpgll.*.h5 files selected in {mode_dir}; skipping dump2imas.")
            else:
                argv  = ["dump2imas.py"] + [str(Path(d).resolve()) for d in dumps]
                argv += common_args + ["--occ-base", str(occ)]
                argv += ["--mem-limit-gb", str(mem_limit_gb)]
                argv += _options_dict_to_cli(imas_cfg.get("dump2imas_options"))
                argv += [str(a) for a in imas_cfg.get("dump2imas_args", [])]

                printnote(f"  dump2imas   [{mode_dir.name}] occ={occ}, {len(dumps)} dump(s)")
                ok = _run_nimrod2imas_tool(rt, "dump2imas", argv, label=f"dump2imas[{mode_dir.name}]")
                summary["steps"].append({
                    "tool":            "dump2imas",
                    "mode_directory":  str(mode_dir),
                    "occurrence":      occ,
                    "n_dumps":         len(dumps),
                    "dumps":           [Path(d).name for d in dumps],
                    "ok":              bool(ok),
                })

            # ---- gamma2imas ----
            if (not have_gamma2imas) or (not imas_cfg.get("run_gamma2imas", True)):
                continue

            energy_file = None
            for fname in ("energy.bin", "logen.bin"):
                cand = mode_dir / fname
                if cand.is_file():
                    energy_file = cand
                    break

            if energy_file is None:
                printwarn(f"No energy.bin/logen.bin in {mode_dir}; skipping gamma2imas.")
                continue

            # gamma2imas shares the same common_args (including --mode a) so it
            # appends into the existing entry rather than creating a new one.
            gargv = [
                "gamma2imas.py",
                "--dd", dd,
                "--dd-version", dd_version,
                "--pulse", str(pulse),
                "--run", str(run),
                "--dbpath", str(dbpath),
                "--occ", str(occ),
                "--endian", gamma_endian,
                "-n", str(gamma_nsteps),
            ]
            # gamma2imas only accepts the hdf5 backend.
            if backend.lower() == "hdf5":
                gargv += ["--backend", backend]
            gargv += _options_dict_to_cli(imas_cfg.get("gamma2imas_options"))
            gargv += [str(a) for a in imas_cfg.get("gamma2imas_args", [])]
            gargv.append(str(energy_file.resolve()))

            history = mode_dir / "nimhist.bin"
            printnote(f"  gamma2imas  [{mode_dir.name}] occ={occ} ({energy_file.name})")

            ok = False
            if use_history and history.is_file():
                # Frequency extraction needs a nimhist-layout record; retry without
                # it if the file cannot be parsed.
                ok = _run_nimrod2imas_tool(
                    rt,
                    "gamma2imas",
                    gargv + [str(history.resolve())],
                    label=f"gamma2imas[{mode_dir.name}]",
                    quiet=True,
                )
                if not ok:
                    printwarn(
                        f"gamma2imas could not use {history.name} in {mode_dir}; "
                        "retrying with growth rates only."
                    )
            if not ok:
                ok = _run_nimrod2imas_tool(rt, "gamma2imas", gargv, label=f"gamma2imas[{mode_dir.name}]")

            summary["steps"].append({
                "tool":            "gamma2imas",
                "mode_directory":  str(mode_dir),
                "occurrence":      occ,
                "energy_file":     str(energy_file),
                "ok":              bool(ok),
            })

        n_ok  = sum(1 for s in summary["steps"] if s.get("ok"))
        n_all = len(summary["steps"])
        if n_all == 0:
            printwarn(f"No nimrod2imas steps were executed for {self.model_dir}.")
            return None
        if n_ok == n_all:
            printnote(f"  -> {n_ok}/{n_all} nimrod2imas steps succeeded.")
        else:
            printwarn(f"  -> {n_ok}/{n_all} nimrod2imas steps succeeded.")

        return summary

    def get_time_metadata(self, time_slices):
        """
        Builds the time metadata dict for the requested time slice indices.

        For NIMROD, each nXX directory has its own set of dump files and therefore
        its own set of available sequential indices. The index count is per-directory,
        not global. When a requested index does not exist in all directories, the
        largest index that is common to all nXX directories and still <= the requested
        index is used instead (with a warning).

        The resolved index (not the raw user-requested value) is written to
        'time_slice' in the output, so it always corresponds to a real dump file.
        The 'simulation_time_step' is the step number from the dump filename of the
        first registered nXX directory at the resolved index.
        """
        # Resolve each requested time slice to the best common index across all nXX dirs
        resolved_indices = []
        for ts in time_slices:
            requested_idx = 0 if ts == -1 else ts
            resolved_idx  = self._get_common_max_index(requested_idx)
            resolved_indices.append(resolved_idx)

        meta = {
            # Store the resolved indices, not the raw user-requested values,
            # so that time_slice always points to an existing dump file.
            "time_slice":           np.array(resolved_indices, dtype=int),
            "simulation_time_step": np.zeros(len(time_slices), dtype=int),
            "simulation_time":      np.zeros(len(time_slices), dtype=float)
        }

        # Populate simulation_time_step from the first registered nXX directory
        if self._mode_dirs:
            ref_dir = self._mode_dirs[0]
        else:
            ref_dir = None

        if ref_dir is not None:
            indexed   = self._get_indexed_dumps_for_dir(ref_dir)
            index_map = {idx: (step, path) for idx, step, path in indexed}
            for i, idx in enumerate(resolved_indices):
                if idx in index_map:
                    step, _ = index_map[idx]
                    # The step number from the filename is the NIMROD simulation time step
                    meta["simulation_time_step"][i] = step

        return meta

    def get_reproducibility_data(self, mode_dir):
        hashes = {}
        
        # Explicitly requested files
        target_files = ['nimhist.bin', 'energy.bin', 'energy.txt', 'discharge.bin',
                        'growth.bin', 'growth.txt', 'timestat.bin']
        for fname in target_files:
            fpath = mode_dir / fname
            if fpath.exists():
                hashes[fname] = compute_file_hash(fpath)
                
        # Only hash HDF5 dump files; binary dump files are excluded
        for h5_file in mode_dir.glob("dumpgll.*.h5"):
            hashes[h5_file.name] = compute_file_hash(h5_file)
                
        # The equilibrium dump file might live in the parent model_dir. Ensure it gets hashed.
        eq_dump = self._get_dump_file(time=-1)
        if eq_dump:
            eq_path = Path(eq_dump)
            if eq_path.name not in hashes:
                hashes[eq_path.name] = compute_file_hash(eq_path)

        return hashes

    def _map_nimrod_field(self, name):
        """Maps user config field names to the specific 'nvptb' fields expected by nimpy."""
        mapping = {
            'ne': 'n', 'ni': 'n', 'n': 'n',
            'v': 'v',
            'p': 'p', 'pe': 'p', 'pi': 'p',
            'te': 't', 'ti': 't', 't': 't',
            'b': 'b', 'B': 'b'
        }
        return mapping.get(name.lower(), None)

    def _get_sorted_dump_files(self):
        """
        Returns a step-number-sorted list of valid HDF5 dump file paths from the
        first registered nXX mode directory. This is a convenience wrapper for use
        by code that does not need the index/step metadata.
        """
        if self._mode_dirs:
            indexed = self._get_indexed_dumps_for_dir(self._mode_dirs[0])
        else:
            indexed = []
        return [path for _, _, path in indexed]

    def _get_bounding_box(self, time=-1):
        """
        Reads the underlying NIMROD HDF5 dump file to dynamically find the 
        global minimum and maximum R and Z coordinates.
        Returns: (rmin, rmax, zmin, zmax)
        """
        dump_file = self._get_dump_file(time)
        if not dump_file:
            # Fallback if no dump file is found
            return 1.0, 2.0, -1.0, 1.0

        rmin, rmax = float('inf'), float('-inf')
        zmin, zmax = float('inf'), float('-inf')

        try:
            with h5py.File(dump_file, 'r') as h5:
                rblocks = h5.get('rblocks')
                if rblocks:
                    # Loop through all blocks (0001, 0002, etc.)
                    for block_name in rblocks.keys():
                        rz_name = f"rz{block_name}"
                        if rz_name in rblocks[block_name]:
                            # Load the R, Z coordinates array for this block
                            rzdat = rblocks[block_name][rz_name][()]
                            
                            # rzdat[..., 0] is R, rzdat[..., 1] is Z
                            r_vals = rzdat[..., 0]
                            z_vals = rzdat[..., 1]
                            
                            # Update global mins and maxes
                            rmin = min(rmin, np.min(r_vals))
                            rmax = max(rmax, np.max(r_vals))
                            zmin = min(zmin, np.min(z_vals))
                            zmax = max(zmax, np.max(z_vals))

            if rmin == float('inf'): # Failed to read any blocks
                return 1.0, 2.0, -1.0, 1.0

            return rmin, rmax, zmin, zmax

        except Exception as e:
            printwarn(f"Failed to read bounding box from {dump_file}: {e}")
            return 1.0, 2.0, -1.0, 1.0

    def get_2d_mesh(self, grid_spec):
        if grid_spec.type == "rectangular":
            rmin, rmax, zmin, zmax = self._get_bounding_box(time=-1)
            nR, nZ = grid_spec.resolution
            R_lin = np.linspace(rmin, rmax, nR)
            Z_lin = np.linspace(zmin, zmax, nZ)
            R, Z  = np.meshgrid(R_lin, Z_lin, indexing='ij')
            return np.column_stack((R.ravel(), Z.ravel()))
        else:
            printwarn(f"Grid type '{grid_spec.type}' not yet implemented for NIMROD.")
            return np.array([])

    def get_3d_mesh(self, grid_spec):
        if grid_spec.type == "rectangular":
            rmin, rmax, zmin, zmax = self._get_bounding_box(time=-1)
            nR, nPhi, nZ = grid_spec.resolution
            R_lin   = np.linspace(rmin, rmax, nR)
            Phi_lin = np.linspace(0, 2*np.pi, nPhi, endpoint=False)
            Z_lin   = np.linspace(zmin, zmax, nZ)
            R, phi, Z = np.meshgrid(R_lin, Phi_lin, Z_lin, indexing='ij')
            return np.column_stack((R.ravel(), phi.ravel(), Z.ravel()))
        else:
            printwarn(f"Grid type '{grid_spec.type}' not yet implemented for NIMROD.")
            return np.array([])

    def get_2d_field(self, name, grid_spec, units, time=-1):
        nim_field = self._map_nimrod_field(name)
        if not nim_field:
            printwarn(f"Field '{name}' is not currently mapped for NIMROD extraction.")
            return np.array([])

        dump_file = self._get_dump_file(time)
        if not dump_file:
            return np.array([])

        dump_path = Path(dump_file).resolve()
        target_dir = dump_path.parent

        original_cwd = os.getcwd()
        try:
            os.chdir(target_dir)
            eval_nimrod = EvalNimrod(str(dump_path), fieldlist='nvptb', path='./')

            if grid_spec.type == "rectangular":
                rmin, rmax, zmin, zmax = self._get_bounding_box(time)
                nR, nZ = grid_spec.resolution
                R_lin = np.linspace(rmin, rmax, nR)
                Z_lin = np.linspace(zmin, zmax, nZ)
                R, Z  = np.meshgrid(R_lin, Z_lin, indexing='ij')
                phi   = np.zeros_like(R)

                # Format expected by NIMROD (3, n_pts)
                rzp   = np.array([R.ravel(), Z.ravel(), phi.ravel()])

                res   = eval_nimrod.eval_field(nim_field, rzp=rzp, dmode=0, eq=2)

                # Check if scalar or vector returned
                if res.shape[0] == 1:
                    return res[0]
                elif res.shape[0] == 3:
                    # NIMROD returns (R, Z, phi) natively
                    return {
                        f"{name}_R":   res[0],
                        f"{name}_Z":   res[1],
                        f"{name}_phi": self.PHI_SIGN_FLIP * res[2]
                    }
            else:
                printwarn(f"Grid type '{grid_spec.type}' not yet implemented for NIMROD.")
                return np.array([])
        except Exception as e:
            printerr(f"Error evaluating 2D field {name} in NIMROD: {e}")
            return np.array([])
        finally:
            os.chdir(original_cwd)

    def get_3d_field(self, name, grid_spec, units, time=-1):
        nim_field = self._map_nimrod_field(name)
        if not nim_field:
            printwarn(f"Field '{name}' is not currently mapped for NIMROD extraction.")
            return np.array([])

        dump_file = self._get_dump_file(time)
        if not dump_file:
            return np.array([])

        dump_path  = Path(dump_file).resolve()
        target_dir = dump_path.parent

        original_cwd = os.getcwd()
        try:
            os.chdir(target_dir)
            eval_nimrod = EvalNimrod(str(dump_path), fieldlist='nvptb', path='./')

            if grid_spec.type == "rectangular":
                rmin, rmax, zmin, zmax = self._get_bounding_box(time)
                nR, nPhi, nZ = grid_spec.resolution
                R_lin   = np.linspace(rmin, rmax, nR)
                Phi_lin = np.linspace(0, 2*np.pi, nPhi, endpoint=False)
                Z_lin   = np.linspace(zmin, zmax, nZ)
                R, Phi, Z = np.meshgrid(R_lin, Phi_lin, Z_lin, indexing='ij')
                
                # Format expected by NIMROD (3, n_pts)
                rzp = np.array([R.ravel(), Z.ravel(), Phi.ravel()])
                
                res = eval_nimrod.eval_field(nim_field, rzp=rzp, dmode=0, eq=2)
                
                # Check if scalar or vector returned
                if res.shape[0] == 1:
                    return res[0]
                elif res.shape[0] == 3:
                    # NIMROD returns (R, Z, phi) natively
                    return {
                        f"{name}_R":   res[0],
                        f"{name}_Z":   res[1],
                        f"{name}_phi": self.PHI_SIGN_FLIP * res[2]
                    }
            else:
                printwarn(f"Grid type '{grid_spec.type}' not yet implemented for NIMROD.")
                return np.array([])
        except Exception as e:
            printerr(f"Error evaluating 3D field {name} in NIMROD: {e}")
            return np.array([])
        finally:
            os.chdir(original_cwd)

    def get_mode_metadata(self, mode_dir):
        """
        Extracts the growth rate for the toroidal mode number corresponding to this
        mode directory by running 'nimgrowth energy.bin' in the directory and
        parsing its output.

        In NIMROD, each subdirectory (e.g. 'n08') may contain results for multiple
        toroidal mode numbers. The toroidal mode number n is inferred from the
        directory name (e.g. 'n08' -> n=8), and the matching growth rate is looked
        up from the nimgrowth output (keyed by keff).

        Returns a dict with keys:
            growth_rate  - kinetic energy growth rate in s^-1 (0.0 if not found)
            frequency    - oscillation frequency (not available from nimgrowth; always 0.0)
            mode_type    - not available for NIMROD; always -100
        """
        meta = {
            "growth_rate": 0.0,
            "frequency":   0.0,
            "mode_type":   -100  # Using -100 as default for unknown
        }

        # Parse the directory name to determine the primary toroidal mode number.
        # Mode directories follow the 'nXX' convention (e.g. 'n08' -> n=8).
        primary_n = None
        try:
            primary_n = int(mode_dir.name.lstrip('n'))
        except ValueError:
            printwarn(f"Could not parse toroidal mode number from directory name '{mode_dir.name}'.")

        # Run nimgrowth and retrieve all growth rates found in this directory
        growth_rates = self._get_nimgrowth_for_dir(mode_dir)

        # Set the primary growth_rate to the value matching the directory's n
        if primary_n is not None and primary_n in growth_rates:
            meta["growth_rate"] = growth_rates[primary_n]
        elif primary_n is not None and growth_rates:
            printwarn(
                f"Toroidal mode n={primary_n} (from directory '{mode_dir.name}') "
                f"not found in nimgrowth output. Available n values: {list(growth_rates.keys())}."
            )

        return meta

    def get_all_mode_entries(self, mode_dir, config):
        """
        Overrides the base class method to expand a single NIMROD mode directory
        into one perturbation group per toroidal mode number found in the nimgrowth
        output.

        For each keff value reported by 'nimgrowth energy.bin' in the given directory,
        a separate group named 'nXX' (zero-padded to two digits) is created under
        'perturbations', each containing 'growth_rate', 'frequency', and 'mode_type'
        datasets matching the M3D-C1 structure.

        If nimgrowth produces no output, falls back to a single entry using the
        directory name and a growth_rate of 0.0.

        Returns: list of (group_name, mode_metadata) tuples.
        """
        growth_rates = self._get_nimgrowth_for_dir(mode_dir)

        if not growth_rates:
            # Fallback: single entry using the directory name with default metadata
            return [(mode_dir.name, {
                "growth_rate": 0.0,
                "frequency":   0.0,
                "mode_type":   -100  # Using -100 as default for unknown
            })]

        entries = []
        for n_val in sorted(growth_rates.keys()):
            # Format the group name as 'nXX' with zero-padding to two digits,
            # matching the M3D-C1 convention (e.g. n=8 -> 'n08', n=14 -> 'n14')
            group_name = f"n{n_val:02d}"
            mode_meta  = {
                "growth_rate": growth_rates[n_val],
                "frequency":   0.0,
                "mode_type":   -100  # Using -100 as default for unknown
            }
            entries.append((group_name, mode_meta))

        return entries

    def get_mode_2d_field(self, mode_dir, name, grid_spec, units):
        """
        Evaluates the field at the last dump file in the given mode directory (finite time).
        The eigenfunction is computed in extract_mode() by subtracting the equilibrium
        field (index 0 dump, time=-1) from this result.
        """
        nim_field = self._map_nimrod_field(name)
        if not nim_field:
            printwarn(f"Field '{name}' is not currently mapped for NIMROD extraction.")
            return np.array([])

        indexed = self._get_indexed_dumps_for_dir(mode_dir)
        if not indexed:
            printwarn(f"No dump files found in mode directory {mode_dir}.")
            return np.array([])
        # Use the last dump (highest sequential index = most evolved time)
        _, _, dump_path_obj = indexed[-1]
        dump_path  = Path(dump_path_obj).resolve()
        target_dir = dump_path.parent

        original_cwd = os.getcwd()
        try:
            os.chdir(target_dir)
            eval_nimrod = EvalNimrod(str(dump_path), fieldlist='nvptb', path='./')
            if grid_spec.type == "rectangular":
                rmin, rmax, zmin, zmax = self._get_bounding_box(time=-1)
                nR, nZ = grid_spec.resolution
                R_lin = np.linspace(rmin, rmax, nR)
                Z_lin = np.linspace(zmin, zmax, nZ)
                R, Z  = np.meshgrid(R_lin, Z_lin, indexing='ij')
                phi   = np.zeros_like(R)
                rzp   = np.array([R.ravel(), Z.ravel(), phi.ravel()])
                res   = eval_nimrod.eval_field(nim_field, rzp=rzp, dmode=0, eq=2)
                if res.shape[0] == 1:
                    return res[0]
                elif res.shape[0] == 3:
                    return {
                        f"{name}_R":   res[0],
                        f"{name}_Z":   res[1],
                        f"{name}_phi": self.PHI_SIGN_FLIP * res[2]
                    }
            else:
                printwarn(f"Grid type '{grid_spec.type}' not yet implemented for NIMROD.")
                return np.array([])
        except Exception as e:
            printerr(f"Error evaluating mode 2D field {name} in NIMROD: {e}")
            return np.array([])
        finally:
            os.chdir(original_cwd)

    def get_mode_3d_field(self, mode_dir, name, grid_spec, units):
        """
        Evaluates the field at the last dump file in the given mode directory (finite time).
        The eigenfunction is computed in extract_mode() by subtracting the equilibrium
        field (index 0 dump, time=-1) from this result.
        """
        nim_field = self._map_nimrod_field(name)
        if not nim_field:
            printwarn(f"Field '{name}' is not currently mapped for NIMROD extraction.")
            return np.array([])

        indexed = self._get_indexed_dumps_for_dir(mode_dir)
        if not indexed:
            printwarn(f"No dump files found in mode directory {mode_dir}.")
            return np.array([])
        _, _, dump_path_obj = indexed[-1]
        dump_path = Path(dump_path_obj).resolve()
        target_dir = dump_path.parent

        original_cwd = os.getcwd()
        try:
            os.chdir(target_dir)
            eval_nimrod = EvalNimrod(str(dump_path), fieldlist='nvptb', path='./')
            if grid_spec.type == "rectangular":
                rmin, rmax, zmin, zmax = self._get_bounding_box(time=-1)
                nR, nPhi, nZ = grid_spec.resolution
                R_lin   = np.linspace(rmin, rmax, nR)
                Phi_lin = np.linspace(0, 2*np.pi, nPhi, endpoint=False)
                Z_lin   = np.linspace(zmin, zmax, nZ)
                R, Phi, Z = np.meshgrid(R_lin, Phi_lin, Z_lin, indexing='ij')
                rzp = np.array([R.ravel(), Z.ravel(), Phi.ravel()])
                res = eval_nimrod.eval_field(nim_field, rzp=rzp, dmode=0, eq=2)
                if res.shape[0] == 1:
                    return res[0]
                elif res.shape[0] == 3:
                    return {
                        f"{name}_R":   res[0],
                        f"{name}_Z":   res[1],
                        f"{name}_phi": self.PHI_SIGN_FLIP * res[2]
                    }
            else:
                printwarn(f"Grid type '{grid_spec.type}' not yet implemented for NIMROD.")
                return np.array([])
        except Exception as e:
            printerr(f"Error evaluating mode 3D field {name} in NIMROD: {e}")
            return np.array([])
        finally:
            os.chdir(original_cwd)

    def get_all_input_parameters(self):
        inputs = {}
        target_files = ['nimrod.in', 'nimeq.in', 'oculus.in', 'fluxgrid.in']

        for filename in target_files:
            found_files = list(self.model_dir.rglob(filename))
            if not found_files:
                continue
            
            # Parse the first found file
            with open(found_files[0], 'r', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if '!' in line:
                        line = line.split('!')[0].strip()
                    if not line or line.startswith('&') or line == '/':
                        continue

                    if '=' in line:
                        parts = line.split('=', 1)
                        key   = parts[0].strip()
                        val   = parts[1].strip()

                        if val.startswith("'") and val.endswith("'"):
                            val = val[1:-1]
                        elif val.startswith('"') and val.endswith('"'):
                            val = val[1:-1]

                        inputs[key] = auto_cast(val)
        return inputs


# ===========================================================================
# SERIALIZATION & DISCOVERY
# Handles directory traversal and writing outputs to HDF5.
# ===========================================================================

def write_dict_as_attributes(group, data):
    """Saves flat python dictionaries as HDF5 group attributes."""
    for k, v in data.items():
        if isinstance(v, (str, int, float, np.integer, np.floating)):
            group.attrs[k] = v

def write_dict_as_datasets(group, data):
    """Saves flat python dictionaries as individual HDF5 datasets."""
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, str):
            # np.bytes_ forces it into a fixed-length bytes type (|S) instead of Object (|O)
            group.create_dataset(k, data=np.bytes_(v))
        else:
            group.create_dataset(k, data=v)

def write_array_group(parent, name, data_dict, unit_mapping=None, desc_mapping=None):
    """Creates an HDF5 group and saves all arrays inside a dictionary as datasets."""
    grp = parent.create_group(name)
    for key, value in data_dict.items():
        if value is None:
            continue

        if isinstance(value, str):
            dset = grp.create_dataset(key, data=np.bytes_(value))
        else:
            dset = grp.create_dataset(key, data=value)

        if unit_mapping or desc_mapping:
            # Strip vector components (_R, _phi, _Z) to match the base name in the dictionary
            base_key = key
            if base_key.endswith('_R') or base_key.endswith('_phi') or base_key.endswith('_Z'):
                base_key = base_key.rsplit('_', 1)[0]

            if unit_mapping and base_key in unit_mapping:
                dset.attrs['unit'] = unit_mapping[base_key]
            if desc_mapping and base_key in desc_mapping:
                dset.attrs['description'] = desc_mapping[base_key]

    return grp

def group_simulation_directories(sources):
    """
    Crawls the file system based on the defined sources to find valid data.
    Groups the discovered toroidal modes under their parent model directories.
    Returns: { (model_dir, code): [mode_dir1, mode_dir2, ...] }
    """
    grouped = defaultdict(list)

    for source in sources:
        code = source.get("code", "unknown").lower()
        
        # Extract target models if the user provided specific ones to filter by
        target_models = source.get("model", None)
        if isinstance(target_models, str):
            target_models = [target_models]

        for root in source.get("directories", []):
            if code == "m3dc1":
                # Validated strictly by the presence of C1.h5
                for c1_file in Path(root).rglob("C1.h5"):
                    # Ignore any paths that contain a directory starting with "base_"
                    if any(part.startswith("base_") for part in c1_file.parts):
                        continue

                    mode_dir  = c1_file.parent
                    model_dir = mode_dir.parent

                    # Filter by model if requested
                    if target_models and model_dir.name not in target_models:
                        continue

                    grouped[(model_dir, code)].append(mode_dir)

            elif code == "nimrod":
                # Validated based on the presence of BOTH nimhist.bin and nimrod.in
                for hist_file in Path(root).rglob("nimhist.bin"):
                    # Ignore any paths that contain a directory starting with "base_"
                    if any(part.startswith("base_") for part in hist_file.parts):
                        continue

                    mode_dir = hist_file.parent
                    
                    # Must have nimrod.in to prevent nimpy segfaults
                    if not (mode_dir / "nimrod.in").exists():
                        printwarn(f"Skipping {mode_dir}: 'nimhist.bin' found but 'nimrod.in' is missing (prevents segfault).")
                        continue
                    
                    # Verify it's an nXX directory using strict regex 
                    if re.match(r'^n(\d+|ln)', mode_dir.name):
                        model_dir = mode_dir.parent
                        
                        # Filter by model if requested
                        if target_models and model_dir.name not in target_models:
                            continue

                        if mode_dir not in grouped[(model_dir, code)]:
                            grouped[(model_dir, code)].append(mode_dir)

    return grouped


# ===========================================================================
# MAIN WORKFLOW
# The primary entry point for batch processing.
# ===========================================================================

def build_dataset(sources, output_directory, config):

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    #print(sources)
    
    include_inputs = config.get("include_input_namelist", False)
    output_format  = config.get("output_format", "reduced_h5")

    # Define unit and description mappings dynamically based on requested system
    req_units = config.get("units", "codeunits").lower()
    master_unit_mapping = {}
    master_desc_mapping = {}
    
    # 1. Descriptions are invariant of unit choice
    for k, v in TIME_TRACE_LABELS.items():
        master_desc_mapping[k] = v[0]
    for k, v in FIELD_LABELS.items():
        master_desc_mapping[k] = v[0]
    for k, v in FLUX_AVERAGE_ONLY_LABELS.items():
        master_desc_mapping[k] = v[0]
    master_desc_mapping["time"] = "Time"

    # 2. Units depend on user choice
    if req_units == 'codeunits':
        master_unit_mapping.update(TIME_TRACE_UNITS_CODE)
        master_unit_mapping.update(FIELD_UNITS_CODE)
        master_unit_mapping.update(FLUX_AVERAGE_ONLY_UNITS_CODE)
        master_unit_mapping["time"] = "dummy_conversion_formula"
    else:
        for k, v in TIME_TRACE_LABELS.items():
            master_unit_mapping[k] = v[1]
        for k, v in FIELD_LABELS.items():
            master_unit_mapping[k] = v[1]
        for k, v in FLUX_AVERAGE_ONLY_LABELS.items():
            master_unit_mapping[k] = v[1]
        master_unit_mapping["time"] = "s"

    # 1. Detect all valid directories
    grouped = group_simulation_directories(sources)

    if not grouped:
        printwarn("Warning: No valid simulation directories were found. Please check your source paths!")
        return

    # Determine shared grid if specified
    shared_grid_source      = config.get("shared_grid_source", None)
    shared_inner_wall_points = None

    if shared_grid_source and grouped:
        if shared_grid_source == "first":
            # Find the first m3dc1 model to extract the grid
            for (m_dir, m_code), m_dirs in grouped.items():
                if m_code == "m3dc1":
                    tmp_adapter = M3DC1Adapter(m_dir)
                    shared_inner_wall_points = tmp_adapter._get_inner_wall_points()
                    tmp_adapter.close()
                    break
        elif os.path.exists(shared_grid_source):
            tmp_adapter = M3DC1Adapter(shared_grid_source)
            shared_inner_wall_points = tmp_adapter._get_inner_wall_points()
            tmp_adapter.close()
        else:
            printwarn(f"Shared grid source '{shared_grid_source}' not found or invalid. Defaulting to local grids.")

    for (model_dir, code), mode_dirs in grouped.items():
        print(model_dir)

        out_name = generate_unique_filename(model_dir, code)
        out_file = output_directory / out_name

        if output_format == "reduced_bp":
            out_file = out_file.with_suffix('.bp')
            
        # Safeguard to prevent overwriting existing files
        base_stem = out_file.stem
        ext       = out_file.suffix
        counter   = 1
        while out_file.exists():
            out_file = output_directory / f"{base_stem}_{counter}{ext}"
            counter += 1

        if output_format == "reduced_h5":
            if code == "m3dc1":
                adapter = M3DC1Adapter(model_dir)
            elif code == "nimrod":
                adapter = NIMRODAdapter(model_dir)
                # Register the nXX mode directories so that per-directory index logic works
                adapter.set_mode_dirs(mode_dirs)
            else:
                print(f"Skipping {model_dir}: Unsupported code '{code}'")
                continue

            # Inject shared grid if available
            if shared_inner_wall_points is not None:
                adapter.shared_inner_wall_points = shared_inner_wall_points

            # Core Extractions
            metadata = adapter.get_metadata(mode_dirs)
            # Add ML config specifications directly to metadata group for traceability
            metadata["units"]         = config.get("units", "codeunits")
            metadata["fcoords"]       = config.get("fcoords", "pest")
            metadata["output_format"] = output_format

            grids        = adapter.extract_grids(config)
            equilibrium  = adapter.extract_equilibrium(config)
            total_fields = adapter.extract_total_fields(config)
            if include_inputs:
                inputs = adapter.get_all_input_parameters()

            # Write HDF5 File according to diagram schema
            with h5py.File(out_file, "w") as h5:

                # METADATA & HASHES
                grp_meta   = h5.create_group("metadata")
                write_dict_as_datasets(grp_meta, metadata)
                grp_hashes = grp_meta.create_group("file_hashes")

                # INPUTS
                if include_inputs:
                    grp_inputs = h5.create_group("inputs")
                    write_dict_as_datasets(grp_inputs, inputs)

                # GLOBAL PARAMETERS
                grp_globals = h5.create_group("global_parameters")
                write_dict_as_datasets(grp_globals, equilibrium["global_parameters"])

                # TIME TRACES
                write_array_group(h5, "time_traces", equilibrium["time_traces"], unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)

                # GRIDS
                grp_grids = h5.create_group("grids")
                if "time" in grids:
                    grp_time = grp_grids.create_group("time")
                    write_dict_as_datasets(grp_time, grids["time"])
                if "1d_grid" in grids:
                    grp_grids.create_dataset("1d_grid", data=grids["1d_grid"])
                if "2d_grid" in grids:
                    grp_grids.create_dataset("2d_grid", data=grids["2d_grid"])
                if "3d_grid" in grids:
                    grp_grids.create_dataset("3d_grid", data=grids["3d_grid"])

                # EQUILIBRIUM
                grp_eq = h5.create_group("equilibrium")
                write_array_group(grp_eq, "flux_averages", equilibrium["flux_averages"], unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                write_array_group(grp_eq, "2d_fields",     equilibrium["2d_fields"],     unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                write_array_group(grp_eq, "3d_fields",     equilibrium["3d_fields"],     unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)

                # TOTAL FIELDS (over time)
                if total_fields is not None:
                    grp_tot = h5.create_group("total_fields")
                    write_array_group(grp_tot, "flux_averages", total_fields["flux_averages"], unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                    write_array_group(grp_tot, "2d_fields",     total_fields["2d_fields"],     unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                    write_array_group(grp_tot, "3d_fields",     total_fields["3d_fields"],     unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)

                # PERTURBATIONS
                # extract_mode() computes eigenfunction = field(finite_time) - field(equilibrium)
                # matching the approach in eigenfunction.py: ef = p1 - p0
                grp_modes = h5.create_group("perturbations")
                for mode_dir in mode_dirs:
                    mode_name = mode_dir.name

                    repro_hashes   = adapter.get_reproducibility_data(mode_dir)
                    grp_hash_mode  = grp_hashes.create_group(mode_name)
                    write_dict_as_datasets(grp_hash_mode, repro_hashes)

                    # Extract the actual mode perturbation data by calling extract_mode()
                    mode_data    = adapter.extract_mode(mode_dir, config)

                    # Expand this directory into one or more (group_name, mode_metadata) entries
                    mode_entries = adapter.get_all_mode_entries(mode_dir, config)
                    for entry_name, entry_meta in mode_entries:
                        # Avoid duplicate group names (e.g. if two mode_dirs map to the same nXX)
                        if entry_name in grp_modes:
                            printwarn(f"Perturbation group '{entry_name}' already exists; skipping duplicate from {mode_dir}.")
                            continue
                        grp_mode      = grp_modes.create_group(entry_name)
                        grp_mode_meta = grp_mode.create_group("mode_information")
                        write_dict_as_datasets(grp_mode_meta, entry_meta)
                        # Write the extracted field data (1d_profiles, 2d_fields, 3d_fields)
                        write_array_group(grp_mode, "1d_profiles", mode_data["1d_profiles"], unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                        write_array_group(grp_mode, "2d_fields",   mode_data["2d_fields"],   unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                        write_array_group(grp_mode, "3d_fields",   mode_data["3d_fields"],   unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)

        elif output_format == "reduced_bp":
            try:
                import adios2
            except ImportError:
                printwarn("ADIOS2 is not installed. Cannot write .bp files.")
                continue

            if code == "m3dc1":
                adapter = M3DC1Adapter(model_dir)
            elif code == "nimrod":
                adapter = NIMRODAdapter(model_dir)
                # Register the nXX mode directories so that per-directory index logic works
                adapter.set_mode_dirs(mode_dirs)
            else:
                print(f"Skipping {model_dir}: Unsupported code '{code}'")
                continue

            # Inject shared grid if available
            if shared_inner_wall_points is not None:
                adapter.shared_inner_wall_points = shared_inner_wall_points

            # Core Extractions
            metadata = adapter.get_metadata(mode_dirs)
            # Add ML config specifications directly to metadata group for traceability
            metadata["units"]         = config.get("units", "codeunits")
            metadata["fcoords"]       = config.get("fcoords", "pest")
            metadata["output_format"] = output_format

            grids        = adapter.extract_grids(config)
            equilibrium  = adapter.extract_equilibrium(config)
            total_fields = adapter.extract_total_fields(config)
            if include_inputs:
                inputs = adapter.get_all_input_parameters()

            def write_adios_recursive(fh, data_dict, base_path="", unit_mapping=None, desc_mapping=None):
                for key, val in data_dict.items():
                    if val is None:
                        continue
                    path = f"{base_path}/{key}" if base_path else key
                    if isinstance(val, dict):
                        write_adios_recursive(fh, val, path, unit_mapping, desc_mapping)
                    else:
                        if isinstance(val, str):
                            fh.write(path, val)
                        elif np.isscalar(val):
                            fh.write(path, np.array([val]))
                        else:
                            fh.write(path, np.asarray(val))

                        if unit_mapping or desc_mapping:
                            # Strip vector components to match base names
                            base_key = key
                            if base_key.endswith('_R') or base_key.endswith('_phi') or base_key.endswith('_Z'):
                                base_key = base_key.rsplit('_', 1)[0]

                            if unit_mapping and base_key in unit_mapping:
                                fh.write_attribute('unit', str(unit_mapping[base_key]), variable_name=path)
                            if desc_mapping and base_key in desc_mapping:
                                fh.write_attribute('description', str(desc_mapping[base_key]), variable_name=path)

            with adios2.Stream(str(out_file), "w") as fh:

                # METADATA & HASHES
                write_adios_recursive(fh, metadata, "metadata")

                # INPUTS
                if include_inputs:
                    write_adios_recursive(fh, inputs, "inputs")
                
                # GLOBAL PARAMETERS
                write_adios_recursive(fh, equilibrium["global_parameters"], "global_parameters")

                # TIME TRACES
                write_adios_recursive(fh, equilibrium["time_traces"], "time_traces", unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)

                # GRIDS
                if "time" in grids:
                    write_adios_recursive(fh, grids["time"], "grids/time")
                if "1d_grid" in grids:
                    fh.write("grids/1d_grid", np.asarray(grids["1d_grid"]))
                if "2d_grid" in grids:
                    fh.write("grids/2d_grid", np.asarray(grids["2d_grid"]))
                if "3d_grid" in grids:
                    fh.write("grids/3d_grid", np.asarray(grids["3d_grid"]))

                # EQUILIBRIUM
                write_adios_recursive(fh, equilibrium["flux_averages"], "equilibrium/flux_averages", unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                write_adios_recursive(fh, equilibrium["2d_fields"],     "equilibrium/2d_fields",     unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                write_adios_recursive(fh, equilibrium["3d_fields"],     "equilibrium/3d_fields",     unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)

                # TOTAL FIELDS (over time)
                if total_fields is not None:
                    write_adios_recursive(fh, total_fields["flux_averages"], "total_fields/flux_averages", unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                    write_adios_recursive(fh, total_fields["2d_fields"],     "total_fields/2d_fields",     unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                    write_adios_recursive(fh, total_fields["3d_fields"],     "total_fields/3d_fields",     unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)

                # PERTURBATIONS
                # For codes like NIMROD where one directory may contain results for multiple
                # toroidal mode numbers, get_all_mode_entries() expands each directory into
                # one group per n value. For M3D-C1, it returns a single entry per directory.
                seen_bp_entries = set()
                for mode_dir in mode_dirs:
                    mode_name = mode_dir.name

                    repro_hashes = adapter.get_reproducibility_data(mode_dir)
                    write_adios_recursive(fh, repro_hashes, f"metadata/file_hashes/{mode_name}")

                    # Extract the actual mode perturbation data by calling extract_mode()
                    mode_data    = adapter.extract_mode(mode_dir, config)

                    # Expand this directory into one or more (group_name, mode_metadata) entries
                    mode_entries = adapter.get_all_mode_entries(mode_dir, config)
                    for entry_name, entry_meta in mode_entries:
                        if entry_name in seen_bp_entries:
                            printwarn(f"Perturbation entry '{entry_name}' already written; skipping duplicate from {mode_dir}.")
                            continue
                        seen_bp_entries.add(entry_name)
                        write_adios_recursive(fh, entry_meta,              f"perturbations/{entry_name}/mode_information")
                        # Write the extracted field data (1d_profiles, 2d_fields, 3d_fields)
                        write_adios_recursive(fh, mode_data["1d_profiles"], f"perturbations/{entry_name}/1d_profiles", unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                        write_adios_recursive(fh, mode_data["2d_fields"],   f"perturbations/{entry_name}/2d_fields",   unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                        write_adios_recursive(fh, mode_data["3d_fields"],   f"perturbations/{entry_name}/3d_fields",   unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)

        elif output_format == "imas_h5":
            # M3D-C1 produces a single self-contained IMAS HDF5 file (standalone) or a
            # full IMAS data entry (imas-python), while NIMROD (via the nimrod2imas tools)
            # always produces a full IMAS data entry directory.
            # Both paths are funnelled through adapter.convert_to_imas() and recorded
            # in a shared JSON manifest (<output_directory>/imas_entries.json).
            if code == "m3dc1":
                adapter = M3DC1Adapter(model_dir)
                if shared_inner_wall_points is not None:
                    adapter.shared_inner_wall_points = shared_inner_wall_points
                info = adapter.convert_to_imas(mode_dirs, output_directory, config, out_file=out_file)
                adapter.close()
                if info:
                    _record_imas_manifest(output_directory, info)

            elif code == "nimrod":
                adapter = NIMRODAdapter(model_dir)
                # Register the nXX mode directories so that per-directory index logic works
                adapter.set_mode_dirs(mode_dirs)
                info = adapter.convert_to_imas(mode_dirs, output_directory, config)
                adapter.close()
                if info:
                    _record_imas_manifest(output_directory, info)
                    printnote(f"IMAS entry written: {info.get('entry_directory')}")
                else:
                    printwarn(f"No IMAS output produced for {model_dir}.")

            else:
                printwarn(f"IMAS format not implemented for {code}")

