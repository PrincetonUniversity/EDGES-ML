"""
Shared utilities, constants, label registries, and IMAS helpers for edges-ml.
"""

import os
import re
import sys
import json
import hashlib
import importlib
import subprocess
import shlex
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Union
from termcolor import colored


# ===========================================================================
# LOGGING HELPERS
# ===========================================================================

def printwarn(string):
    print(colored(string, 'yellow'))


def printerr(string):
    print(colored(string, 'red'))


def printnote(string):
    print(colored(string, 'blue'))


# ===========================================================================
# QUANTITY REGISTRIES & LABELS
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

FLUX_AVERAGE_LABELS = {k: v for k, v in FIELD_LABELS.items() if k not in exclude_for_flux_average}

FLUX_AVERAGE_ONLY_LABELS = {
    'q': ('safety factor', ''),
    'rho': ('normalized toroidal flux', ''),
    'flux_t': ('toroidal flux', 'Wb'),
    'flux_p': ('poloidal flux', 'Wb'),
    'Ip': ('plasma current', 'A'),
    'V': ('plasma volume', r'm$^3$'),
    'f': ('f (RB_phi)', r'T$\cdot$m'),
    'ffprime': ("ff'", r'T$^2\cdot$m'),
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
# UNIT DICTIONARIES
# ===========================================================================

FIELD_UNITS_CODE = {k: 'dummy_conversion_formula' for k in FIELD_LABELS.keys()}
FLUX_AVERAGE_ONLY_UNITS_CODE = {k: 'dummy_conversion_formula' for k in FLUX_AVERAGE_ONLY_LABELS.keys()}
TIME_TRACE_UNITS_CODE = {k: 'dummy_conversion_formula' for k in TIME_TRACE_LABELS.keys()}

AVAILABLE_1D_PROFILES   = ["q_profile"]
AVAILABLE_2D_FIELDS     = list(FIELD_LABELS.keys())
AVAILABLE_3D_FIELDS     = list(FIELD_LABELS.keys())
AVAILABLE_FLUX_AVERAGES = list(FLUX_AVERAGE_LABELS.keys())
AVAILABLE_TIME_TRACES   = list(TIME_TRACE_LABELS.keys())
AVAILABLE_INPUTS        = ["Ip", "Bt"]


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
    - type: 'rectangular', 'native', or 'inside_wall'
    - resolution: Tuple defining the dimensions of the grid (if applicable).
    """
    type: str
    resolution: Union[tuple, int, None] = None


def generate_unique_filename(model_dir, code):
    """
    Generates a unique filename by combining the shortened parent directory
    and the full model directory name.
    """
    parts = [p for p in model_dir.parts if p not in ('.', '..', '/', '\\')]
    parts = [p for p in parts if p and not p.endswith(':')]

    full_name = "_".join(parts)
    filename  = f"{full_name}_{code}.h5"

    if len(filename) > 240:
        path_hash    = hashlib.md5(str(model_dir.resolve()).encode()).hexdigest()[:8]
        suffix       = f"_{path_hash}_{code}.h5"
        allowed_len  = 240 - len(suffix)
        truncated_name = full_name[-allowed_len:].lstrip('_')
        filename     = f"{truncated_name}{suffix}"

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


def _subtract_field_results(val_finite, val_eq):
    """
    Subtracts two field evaluation results to compute the eigenfunction:
        eigenfunction = field(finite_time) - field(equilibrium)
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
# IMAS HELPERS
# ===========================================================================

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
    p     = Path(path)
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
    """
    s = str(dd_version or "4").strip()
    return s[0] if s else "4"


def _imas_entry_dir(dbpath, dd, dd_version, pulse, run):
    """
    Reproduces the entry directory layout used by the nimrod2imas tools:
        <dbpath>/<dd>/<dd_version[0]>/<pulse>/<run>
    """
    ver_dir = _imas_dd_version_dir(dd_version)
    return Path(dbpath) / str(dd) / ver_dir / str(int(pulse)) / str(int(run))


_ALLOCATED_IMAS_RUNS = set()


def _allocate_imas_run(dbpath, dd, dd_version, pulse, run=None, max_tries=10000):
    """
    Returns the run number to use. If 'run' is given explicitly it is honoured
    verbatim; otherwise the first free run index is selected.
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


# ===========================================================================
# NIMROD2IMAS DISCOVERY & EXECUTION HELPERS
# ===========================================================================

_NIMROD2IMAS_TOOLS = ("dump2imas", "input2imas", "gamma2imas")

_NIMROD2IMAS_DIR        = None
_NIMROD2IMAS_SCRIPTS    = {}
_NIMROD2IMAS_SEARCHED   = []
_NIMROD2IMAS_DISCOVERED = False

_NIMROD2IMAS_MODULES      = None
_NIMROD2IMAS_INJECTED     = []
_NIMROD2IMAS_IMPORT_ERRORS = {}

_IMAS_PROBE_CACHE = {}


def _nimrod2imas_candidate_dirs(extra_path=None):
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
    Returns (tools_dir_or_None, {tool_name: script_path}).
    """
    global _NIMROD2IMAS_DIR, _NIMROD2IMAS_SCRIPTS, _NIMROD2IMAS_SEARCHED, _NIMROD2IMAS_DISCOVERED

    if _NIMROD2IMAS_DISCOVERED and not force:
        return _NIMROD2IMAS_DIR, dict(_NIMROD2IMAS_SCRIPTS)

    candidates = _nimrod2imas_candidate_dirs(extra_path)
    _NIMROD2IMAS_SEARCHED = [str(c) for c in candidates]

    best_dir    = None
    best_scripts = {}
    best_count  = 0

    for cand in candidates:
        scripts = {}
        for tool in _NIMROD2IMAS_TOOLS:
            script = cand / f"{tool}.py"
            if script.is_file():
                scripts[tool] = script
        if len(scripts) > best_count:
            best_dir    = cand
            best_scripts = scripts
            best_count  = len(scripts)
        if best_count == len(_NIMROD2IMAS_TOOLS):
            break

    _NIMROD2IMAS_DIR        = best_dir
    _NIMROD2IMAS_SCRIPTS    = best_scripts
    _NIMROD2IMAS_DISCOVERED = True

    return _NIMROD2IMAS_DIR, dict(_NIMROD2IMAS_SCRIPTS)


def _missing_module_from_exc(exc):
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

    mods   = {}
    errors = {}
    for tool in _NIMROD2IMAS_TOOLS:
        mod         = None
        tool_errors = []
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

    _NIMROD2IMAS_MODULES      = mods
    _NIMROD2IMAS_IMPORT_ERRORS = errors
    return mods


def _classify_nimrod2imas_import_errors():
    missing = set()
    others  = []
    for tool, errs in (_NIMROD2IMAS_IMPORT_ERRORS or {}).items():
        for modname, exc in errs:
            if modname.startswith("nimrod2imas."):
                continue
            dep = _missing_module_from_exc(exc)
            if dep and dep.split('.')[0] not in _NIMROD2IMAS_TOOLS:
                missing.add(dep.split('.')[0])
            else:
                others.append(f"{tool}: {type(exc).__name__}: {exc}")
    return missing, others


def _current_python_has_imas():
    try:
        importlib.import_module("imas")
        return True
    except Exception:
        return False


def _run_module_main(module, argv, label="", quiet=False):
    """
    Runs the argparse-based main() of an external CLI module by temporarily
    patching sys.argv. Returns True on success.
    """
    if module is None:
        return False

    label   = label or (argv[0] if argv else "tool")
    old_argv = sys.argv
    old_cwd  = os.getcwd()
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
        prev  = env.get("PYTHONPATH", "")
        parts = [str(tools_dir)] + [p for p in prev.split(os.pathsep) if p]
        seen  = set()
        uniq  = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        env["PYTHONPATH"] = os.pathsep.join(uniq)
    for k, v in (extra_env or {}).items():
        env[str(k)] = str(v)
    return env


def _compose_subprocess_cmd(python_exe, script, args, setup_cmd=None):
    base = [str(python_exe), str(script)] + [str(a) for a in args]
    if setup_cmd:
        inner = " ".join(shlex.quote(x) for x in base)
        return ["bash", "-lc", f"{setup_cmd} && exec {inner}"]
    return base


def _python_has_imas(python_exe, env=None, setup_cmd=None):
    key = (str(python_exe), str(setup_cmd or ""))
    if key in _IMAS_PROBE_CACHE:
        return _IMAS_PROBE_CACHE[key]

    probe = "import imas, sys; sys.stdout.write(getattr(imas,'__version__','?'))"
    base  = [str(python_exe), "-c", probe]
    if setup_cmd:
        inner = " ".join(shlex.quote(x) for x in base)
        cmd   = ["bash", "-lc", f"{setup_cmd} && exec {inner}"]
    else:
        cmd = base

    ok = False
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        ok   = (proc.returncode == 0)
    except Exception:
        ok = False

    _IMAS_PROBE_CACHE[key] = ok
    return ok


def _prepare_nimrod2imas_runtime(imas_cfg):
    """
    Resolves how the nimrod2imas tools will be executed for this run.
    Returns a runtime dict, or None if nothing can be executed.
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

    setup_cmd  = imas_cfg.get("setup_cmd") or os.environ.get("NIMROD2IMAS_SETUP_CMD") or None
    python_exe = (
        imas_cfg.get("python")
        or os.environ.get("NIMROD2IMAS_PYTHON")
        or sys.executable
    )
    env = _nimrod2imas_env(tools_dir, imas_cfg.get("env"))

    modules = {}
    mode    = requested_mode

    if requested_mode in ("auto", "inprocess"):
        modules    = load_nimrod2imas(imas_cfg.get("nimrod2imas_path"))
        inproc_ok  = _current_python_has_imas() and all(
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
                    "    * run this crawler inside your IMAS environment;\n"
                    "    * config['imas']['python']    = '/path/to/env/bin/python';\n"
                    "    * config['imas']['setup_cmd'] = 'module load imas';\n"
                    "    * config['imas']['ignore_imas_probe'] = True to attempt anyway."
                )
                return None

    printnote(
        f"nimrod2imas: dir={tools_dir}; mode={mode}"
        + (f"; python={python_exe}" if mode == "subprocess" else "")
        + (f"; setup_cmd={setup_cmd!r}" if (mode == "subprocess" and setup_cmd) else "")
    )

    return {
        "dir":      tools_dir,
        "scripts":  scripts,
        "modules":  modules,
        "mode":     mode,
        "python":   str(python_exe),
        "setup_cmd": setup_cmd,
        "env":      env,
        "capture":  bool(imas_cfg.get("capture_output", False)),
    }


def _nimrod2imas_tool_available(rt, tool):
    if rt is None:
        return False
    if rt["mode"] == "inprocess":
        return rt["modules"].get(tool) is not None
    return tool in rt["scripts"]


def _run_subprocess_tool(rt, tool, argv, label="", quiet=False):
    script = rt["scripts"].get(tool)
    if script is None:
        if not quiet:
            printwarn(f"{label or tool}: script not found; skipping.")
        return False

    cmd     = _compose_subprocess_cmd(rt["python"], script, argv[1:], setup_cmd=rt["setup_cmd"])
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
    if rt is None:
        return False
    label = label or tool
    if rt["mode"] == "inprocess":
        return _run_module_main(rt["modules"].get(tool), argv, label=label, quiet=quiet)
    return _run_subprocess_tool(rt, tool, argv, label=label, quiet=quiet)


def _options_dict_to_cli(options):
    """
    Converts a {option_name: value} dictionary into a list of CLI tokens.
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


# ===========================================================================
# GEQDSK / PEQDSK FILENAME PATTERNS
# ===========================================================================

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
