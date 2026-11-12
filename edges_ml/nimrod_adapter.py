"""
NIMROD adapter for edges-ml.

Contains the NIMRODAdapter class, which implements the SimulationAdapter
interface for NIMROD HDF5 dump files.

External NIMROD tooling used here:
    nimpy.eval_nimrod.EvalNimrod / EvalGrid   -> point-wise field evaluation
                                                 (eval_field: real space,
                                                  eval_comp_field: per toroidal mode)
    nimpy.fsa.FSA / find_pf_null / FSAInput   -> flux surface averages, q, fluxes
    nimpy.read_bin.readBin                    -> discharge.bin / energy.bin traces
    nimrod2imas (dump2imas/input2imas/gamma2imas) -> IMAS conversion

Mode perturbations
------------------
NIMROD stores the perturbation as complex Fourier coefficients c_n(R, Z) for
each toroidal mode in 'keff'.  The physical perturbation is
    f(R, phi, Z) = c_0 + sum_{n>0} 2 Re[c_n e^{i n phi}].
Each toroidal mode in the dump becomes its own 'perturbations/nXX' entry
(extract_mode_entries).  The coefficients are evaluated with
EvalNimrod.eval_comp_field(), so single-n and multi-n runs are treated
identically and no subtraction of two full fields is needed.

For every quantity listed under config['perturbations'] the adapter writes
    * flux averages:  <field>(last dump, eq + n=0 perturbation) - <field>(equilibrium).
                      The flux-surface average of an n>0 component vanishes,
                      so this n=0 profile modification is the same for every
                      toroidal-mode entry of a directory.
    * 2D / 3D fields: 2 Re[c_n e^{i n phi}] on '2d_grid' / '3d_grid'
                      (the same grids as all other fields).
    * poloidal spectra of the scalar fields, computed on flux-aligned
      (PEST / Boozer / Hamada) coordinates of the equilibrium, built by
      edges_ml.nimrod_fluxcoords (a port of M3D-C1's flux_coordinates()).
      The safety factor is not part of the spectrum output; it is available
      as a flux average.

Current density
---------------
The flux-surface averages themselves only need the magnetic field (field-line
tracing).  The enclosed plasma current I(psi) (flux average 'Ip' and the
global parameters Ip / beta_n) is computed from B with Ampere's law,
    I(psi) = (1/mu0) * closed_integral( B_pol . dl )
along the FSA contours, so it does not need the current density either.

Only quantities that evaluate 'j' directly (the 'j' field and its flux
average) need the current density in the dump (dump_ja=T in nimrod.in).
The adapter reads dump_ja from nimrod.in (override with
config['nimrod_dump_ja']) and skips those quantities when it is not set.

Crash-safe probing
------------------
nimpy's Fortran library terminates the whole Python process when a field
cannot be evaluated (NIM_STOP or a segmentation fault); neither can be caught.
Therefore every EvalNimrod initialisation and every nimpy field is first tried
once in a forked child process (config['nimrod_safe_probe'], default True).
Field lists / fields whose probe crashes or fails are not used, and the
quantities that need them are skipped with a warning.  The probe results are
cached, so each field is probed only once per dump.

Set config['nimrod_verbose'] = True to print a marker before every nimpy
stage (useful to locate a remaining crash).
"""

import os
import re
import sys
import time
import shutil
import pickle
import select
import signal
import hashlib
import contextlib
import subprocess
import numpy as np
import h5py
from pathlib import Path

# ---------------------------------------------------------------------------
# EXTERNAL DEPENDENCIES
# ---------------------------------------------------------------------------
try:
    from nimpy.eval_nimrod import EvalNimrod, EvalGrid
    _NIMPY_EVAL_AVAILABLE = True
except ImportError:
    _NIMPY_EVAL_AVAILABLE = False
    print("Warning: nimpy.eval_nimrod not found. NIMROD field extraction will be skipped.")

try:
    from nimpy.fsa import FSA, find_pf_null, FSAInput
    _NIMPY_FSA_AVAILABLE = True
except ImportError:
    _NIMPY_FSA_AVAILABLE = False
    print("Warning: nimpy.fsa not found. NIMROD flux averages / global parameters will be skipped.")

try:
    from nimpy.read_bin import readBin
    _NIMPY_READBIN_AVAILABLE = True
except ImportError:
    _NIMPY_READBIN_AVAILABLE = False
    print("Warning: nimpy.read_bin not found. NIMROD time traces will be skipped.")

# Kept for backwards compatibility with earlier revisions of this module.
_NIMPY_AVAILABLE = _NIMPY_EVAL_AVAILABLE

from .base import SimulationAdapter
from .nimrod_fluxcoords import build_flux_coordinates
from .utils import (
    printwarn, printerr, printnote,
    compute_file_hash, auto_cast,
    brief_error,
    export_selection,
    spectrum_settings, poloidal_spectrum, parse_toroidal_mode_number,
    GridSpec,
    AVAILABLE_FLUX_AVERAGES,
    _infer_machine_and_shot, _allocate_imas_run, _imas_entry_dir,
    _prepare_nimrod2imas_runtime, _nimrod2imas_tool_available,
    _run_nimrod2imas_tool, _options_dict_to_cli,
)


# ===========================================================================
# SMALL HELPERS
# ===========================================================================

@contextlib.contextmanager
def _pushd(path):
    """Temporarily change the working directory (nimpy reads nimrod.in from cwd)."""
    prev = os.getcwd()
    try:
        os.chdir(str(path))
        yield
    finally:
        try:
            os.chdir(prev)
        except Exception:
            pass


def _finite(value, default=np.nan):
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _trapz_profile(y, x):
    """Cumulative trapezoidal integral of y over a (possibly reversed/negative) x."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2 or x.size != y.size:
        return np.zeros_like(y)
    xx = x - x[0]
    if xx[-1] < 0:
        xx = -xx
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(xx))
    return out


def _points_signature(*arrays):
    """Hash of one or more point arrays (used as a cache key)."""
    h = hashlib.md5()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=float).tobytes())
    return h.hexdigest()


def _run_isolated(func, timeout=600.0, quiet=True):
    """
    Runs func() in a forked child process, so that a crash inside nimpy's
    Fortran library (segmentation fault, NIM_STOP / Fortran STOP) cannot take
    down the main process.

    Returns (ok, result_or_message):
        (True,  result)   func() returned normally (result must be picklable)
        (False, message)  func() raised, or the child crashed / exited
        (None,  message)  the child did not finish within *timeout* seconds
    Without os.fork (non-POSIX systems) func() is run directly.
    """
    if not hasattr(os, "fork"):
        try:
            return True, func()
        except Exception as exc:
            return False, brief_error(exc)

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                                   # ---- child ----
        try:
            os.close(r)
            if quiet:
                try:
                    devnull = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull, 1)
                    os.dup2(devnull, 2)
                except Exception:
                    pass
            try:
                payload = pickle.dumps((True, func()))
            except BaseException as exc:
                payload = pickle.dumps((False, f"{type(exc).__name__}: {exc}"))
            with os.fdopen(w, "wb") as fh:
                fh.write(payload)
        finally:
            os._exit(0)

    # ---- parent ----
    os.close(w)
    chunks = []
    deadline = time.monotonic() + float(timeout)
    timed_out = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            ready, _, _ = select.select([r], [], [], remaining)
            if not ready:
                continue
            data = os.read(r, 65536)
            if not data:
                break
            chunks.append(data)
    finally:
        os.close(r)

    if timed_out:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            os.waitpid(pid, 0)
        except Exception:
            pass
        return None, f"timeout after {timeout:.0f} s"

    try:
        _, status = os.waitpid(pid, 0)
    except Exception as exc:
        return False, f"waitpid failed: {exc}"

    if os.WIFSIGNALED(status):
        sig = os.WTERMSIG(status)
        try:
            name = signal.Signals(sig).name
        except Exception:
            name = f"signal {sig}"
        return False, f"crashed ({name})"

    if chunks:
        try:
            ok, res = pickle.loads(b"".join(chunks))
            return bool(ok), res
        except Exception:
            pass

    code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
    return False, f"terminated (exit code {code}, e.g. NIM_STOP)"


# ===========================================================================
# ADAPTER
# ===========================================================================

class NIMRODAdapter(SimulationAdapter):
    CODE_NAME = "NIMROD"

    # Sign flip for the toroidal (phi) component to match M3D-C1 conventions.
    PHI_SIGN_FLIP = -1.0

    MU0 = 4.0e-7 * np.pi

    # --- field registries -------------------------------------------------
    # edges-ml name -> (nimpy field name, quantity index / species selector)
    _SCALAR_FIELD_MAP = {
        'p':   ('p',  0),
        'pe':  ('pe', 0),
        'te':  ('te', 0),
        'ti':  ('ti', 0),
        't':   ('t',  0),
        'ne':  ('n',  'electron'),
        'ni':  ('n',  'ion'),
        'n':   ('n',  'electron'),
        'psi': ('psi', 0),
    }

    # edges-ml name -> nimpy vector field name
    _VECTOR_FIELD_MAP = {
        'b': 'b', 'B': 'b',
        'v': 'v',
        'j': 'j', 'J': 'j',
        'e': 'e', 'E': 'e',
    }

    # Composite quantities built from other fields.
    _DERIVED_FIELDS = {
        'pi': ('p', 'pe'),   # ion pressure = total - electron
    }

    # Base nimpy 'fieldlist' (same as the standard nimpy FSA scripts) and the
    # fallbacks tried if it cannot be initialised.  'j' is added only for runs
    # with dump_ja=T, 'e' only if the electric field is requested.
    _BASE_FIELDLIST = 'nvptb'
    _FALLBACK_FIELDLISTS = ('nvpt', 'nvp', 'n')

    # Kept for backwards compatibility (no longer used directly).
    _FIELDLIST_CANDIDATES = ('nvptbje', 'nvptbj', 'nvptb', 'nvpt', 'nvp', 'n')

    # nimpy fields that need the current density in the dump (dump_ja=T).
    _CURRENT_FIELDS = frozenset({'j'})

    # FSA integrand keys that evaluate the current density.
    _CURRENT_FSA_KEYS = frozenset({'j', 'j_R', 'j_Z', 'j_phi', 'jphi_over_R'})

    # FSA integrand key -> nimpy fields evaluated by it (used for probing).
    _FSA_KEY_FIELDS = {
        'p': ('p',), 'pe': ('pe',), 'te': ('te',), 'ti': ('ti',),
        'psi': ('psi',), 'ne': ('n',), 'ni': ('n',),
        'B': ('b',), 'B_R': ('b',), 'B_Z': ('b',), 'B_phi': ('b',),
        'Bp': ('b',), 'Bp2': ('b',), 'f': ('b',),
        'v': ('v',), 'v_R': ('v',), 'v_Z': ('v',), 'v_phi': ('v',),
        'j': ('j',), 'j_R': ('j',), 'j_Z': ('j',), 'j_phi': ('j',),
        'jphi_over_R': ('j',),
    }

    # Flux-average names that require no FSA integrand (taken from dvar).
    _FSA_FROM_DVAR = {
        'psi_norm': 0,
        'rho':      1,
        'flux_p':   2,
        'flux_t':   3,
        'q':        7,
    }

    # NIMROD binary-file traces mapped onto the shared edges-ml registry.
    #   key -> (file kind, variable name in nimpy's readBin registry)
    _TIME_TRACE_MAP = {
        'time':             ('discharge', 't'),
        'E_P':              ('discharge', 'Total Int E'),
        'E_PE':             ('discharge', 'Ele Int E'),
        'toroidal_current': ('discharge', 'I total'),
        'loop_voltage':     ('discharge', 'Volt'),
        'toroidal_flux':    ('discharge', 'Total Flux'),
    }
    # Traces obtained by summing energy.bin over all toroidal modes.
    _TIME_TRACE_ENERGY_SUM = {
        'W_M': 'E magnetic',
    }

    def __init__(self, model_dir):
        super().__init__(model_dir)
        self._dumps_converted = False
        self._nimgrowth_cache = {}
        self._indexed_dumps_per_dir = {}
        self._common_max_index_cache = None
        self._mode_dirs = []

        # Configuration captured from build_dataset (see extract_* overrides).
        self._config = {}

        # nimpy EvalNimrod handling (the shared library can only hold one file).
        self._active_eval = None
        self._active_eval_key = None
        self._fieldlist = None

        # Caches
        self._eval_grid_cache = {}
        self._fsa_cache = {}
        self._fsa_failed = set()
        self._axis_cache = {}
        self._bin_cache = {}
        self._native_points_cache = {}
        self._wall_polygon = None
        self._wall_polygon_searched = False
        self._metadata_cache = None
        self._global_params_cache = None
        self._warned_fields = set()

        # dump_ja (current density in the dump) per run directory
        self._dump_ja_cache = {}

        # Crash-safe probing of nimpy (see _run_isolated)
        self._init_probe_cache = {}
        self._field_probe_cache = {}
        self._probe_point_cache = {}
        self._probe_disabled = False

        # Per-mode evaluation / spectrum caches
        self._geometry_cache = {}
        self._mode_table_cache = {}
        self._comp_cache = {}
        self._fluxcoord_cache = {}
        self._fluxcoord_failed = set()

    # ======================================================================
    # CONFIG CAPTURE (so the adapter can size/limit the expensive FSA calls)
    # ======================================================================

    def extract_grids(self, config):
        self._config = config or {}
        return super().extract_grids(config)

    def extract_equilibrium(self, config):
        self._config = config or {}
        return super().extract_equilibrium(config)

    def extract_total_fields(self, config):
        self._config = config or {}
        return super().extract_total_fields(config)

    def extract_mode(self, mode_dir, config):
        self._config = config or {}
        return super().extract_mode(mode_dir, config)

    def close(self):
        """Release cached nimpy handles and derived data."""
        self._active_eval = None
        self._active_eval_key = None
        self._eval_grid_cache.clear()
        self._fsa_cache.clear()
        self._axis_cache.clear()
        self._bin_cache.clear()
        self._comp_cache.clear()
        self._mode_table_cache.clear()
        self._fluxcoord_cache.clear()
        self._fluxcoord_failed.clear()
        self._field_probe_cache.clear()
        self._init_probe_cache.clear()
        self._probe_point_cache.clear()

    def _stage(self, message):
        """Prints a stage marker if config['nimrod_verbose'] is set."""
        if (self._config or {}).get("nimrod_verbose", False):
            printnote(f"NIMROD [stage] {message}")

    # ======================================================================
    # CRASH-SAFE PROBING OF NIMPY
    # ======================================================================

    def _probing_enabled(self):
        return (bool((self._config or {}).get("nimrod_safe_probe", True))
                and not self._probe_disabled and hasattr(os, "fork"))

    def _probe_timeout(self):
        try:
            return float((self._config or {}).get("nimrod_probe_timeout", 600.0))
        except Exception:
            return 600.0

    def _disable_probing(self, message):
        if not self._probe_disabled:
            printwarn(f"NIMROD: the crash-safe subprocess probe did not finish ({message}); "
                      "probing is disabled for the rest of this run.")
        self._probe_disabled = True

    def _probe_eval_init(self, dump_path, fieldlist):
        """
        Tries EvalNimrod(dump, fieldlist) once in a child process.
        Returns (ok, message); ok=True if the initialisation is safe.
        """
        if not self._probing_enabled():
            return True, ""
        key = (str(dump_path), fieldlist)
        if key in self._init_probe_cache:
            return self._init_probe_cache[key]

        self._stage(f"probing EvalNimrod({dump_path.name}, fieldlist='{fieldlist}')")

        def job():
            with _pushd(dump_path.parent):
                EvalNimrod(str(dump_path), fieldlist=fieldlist, path='./')
            return True

        ok, msg = _run_isolated(job, self._probe_timeout())
        if ok is None:
            self._disable_probing(msg)
            result = (True, "")
        elif ok:
            result = (True, "")
        else:
            printwarn(f"NIMROD: EvalNimrod with fieldlist '{fieldlist}' fails for "
                      f"{dump_path.parent.name}/{dump_path.name} ({msg}); trying the next "
                      "field list.")
            result = (False, str(msg))
        self._init_probe_cache[key] = result
        return result

    def _native_points_for_file(self, dump_file):
        """(R, Z) coordinates of the native finite element nodes of *dump_file*."""
        key = str(dump_file)
        if key in self._native_points_cache:
            return self._native_points_cache[key]

        chunks = []
        try:
            with h5py.File(dump_file, 'r') as h5:
                rblocks = h5.get('rblocks')
                if rblocks:
                    for block_name in rblocks.keys():
                        rz_name = f"rz{block_name}"
                        if rz_name in rblocks[block_name]:
                            rzdat = np.asarray(rblocks[block_name][rz_name][()])
                            chunks.append(rzdat.reshape(-1, rzdat.shape[-1])[:, :2])
        except Exception as exc:
            printwarn(f"Failed to read native mesh from {dump_file}: {exc}")

        if chunks:
            pts = np.concatenate(chunks, axis=0)
            pts = np.unique(np.round(pts, 12), axis=0)
        else:
            pts = np.empty((0, 2))

        self._native_points_cache[key] = pts
        return pts

    def _probe_point(self, dump_file):
        """A point inside the domain of *dump_file* (mesh node nearest the centre)."""
        key = str(Path(dump_file).resolve())
        if key in self._probe_point_cache:
            return self._probe_point_cache[key]

        point = None
        pts = self._native_points_for_file(dump_file)
        if len(pts) > 0:
            rc = 0.5 * (pts[:, 0].min() + pts[:, 0].max())
            zc = 0.5 * (pts[:, 1].min() + pts[:, 1].max())
            i = int(np.argmin((pts[:, 0] - rc) ** 2 + (pts[:, 1] - zc) ** 2))
            point = np.array([pts[i, 0], pts[i, 1], 0.0], dtype=float)

        self._probe_point_cache[key] = point
        return point

    def _field_ok(self, dump_file, nim_field, eq=1):
        """
        True if the nimpy field *nim_field* can be evaluated for *dump_file*
        without crashing the process.  The first evaluation is done once in a
        forked child process; the result is cached.
        """
        if dump_file is None:
            return False
        if nim_field in self._CURRENT_FIELDS and not self._current_available(dump_file):
            return False
        if not self._probing_enabled():
            return True

        ev = self._get_eval(dump_file)
        if ev is None:
            return False

        dump_path = Path(dump_file).resolve()
        key = (str(dump_path), self._fieldlist, str(nim_field), int(eq))
        if key in self._field_probe_cache:
            return self._field_probe_cache[key]

        point = self._probe_point(dump_file)
        if point is None:
            self._field_probe_cache[key] = True
            return True

        self._stage(f"probing field '{nim_field}' (eq={eq}) on {dump_path.name}")

        def job():
            with _pushd(dump_path.parent):
                v = ev.eval_field(nim_field, point, dmode=0, eq=eq)
            return np.asarray(v, dtype=float).ravel().tolist()

        ok, msg = _run_isolated(job, self._probe_timeout())
        if ok is None:
            self._disable_probing(msg)
            ok = True
        elif not ok:
            printwarn(f"NIMROD: evaluating field '{nim_field}' (eq={eq}) fails for "
                      f"{dump_path.parent.name}/{dump_path.name} ({msg}); quantities "
                      "that need it are skipped.")

        self._field_probe_cache[key] = bool(ok)
        return bool(ok)

    def _nim_fields(self, kind, payload):
        """nimpy field names needed by a resolved edges-ml field."""
        if kind == 'scalar':
            return [payload[0]]
        if kind == 'vector':
            return [payload]
        if kind == 'derived':
            out = []
            for name in payload:
                for f in self._nim_fields(*self._resolve_field(name)):
                    if f not in out:
                        out.append(f)
            return out
        return []

    def _skip_unsafe_field(self, name, kind, payload, dump_file, eq):
        """Warns once and returns True if a nimpy field needed by *name* fails."""
        for f in self._nim_fields(kind, payload):
            if not self._field_ok(dump_file, f, eq):
                self._warn_field_once(
                    f"unsafe::{name}",
                    f"Field '{name}' cannot be evaluated from this NIMROD dump "
                    f"(nimpy field '{f}' fails); skipping."
                )
                return True
        return False

    # ======================================================================
    # CURRENT DENSITY AVAILABILITY (dump_ja)
    # ======================================================================

    @staticmethod
    def _read_dump_ja(run_dir):
        """
        Reads dump_ja from <run_dir>/nimrod.in.

        Returns True / False, or None if nimrod.in cannot be read.  A missing
        dump_ja entry means NIMROD's default (F).
        """
        nml_file = Path(run_dir) / "nimrod.in"
        if not nml_file.is_file():
            return None

        try:
            import f90nml
            nml = f90nml.read(str(nml_file))
            for grp in nml.values():
                groups = grp if isinstance(grp, list) else [grp]
                for g in groups:
                    if isinstance(g, dict) and 'dump_ja' in g:
                        return bool(g['dump_ja'])
            return False
        except Exception:
            pass

        try:
            rx = re.compile(r'\bdump_ja\s*=\s*([^\s,/!]+)', re.IGNORECASE)
            with open(nml_file, 'r', errors='ignore') as f:
                for line in f:
                    m = rx.search(line.split('!')[0])
                    if m:
                        return m.group(1).strip().strip('.').lower().startswith('t')
            return False
        except Exception:
            return None

    def _current_available(self, dump_file):
        """
        True if the current density 'j' can be evaluated for *dump_file*.

        nimpy aborts the process (NIM_STOP, not catchable) if 'j' is evaluated
        for a run with dump_ja=F, so this must be checked before every request.
        config['nimrod_dump_ja'] (True/False) overrides the nimrod.in value.
        """
        override = (self._config or {}).get("nimrod_dump_ja", None)
        if override is not None:
            return bool(override)
        if dump_file is None:
            return False

        run_dir = Path(dump_file).resolve().parent
        if run_dir in self._dump_ja_cache:
            return self._dump_ja_cache[run_dir]

        value = self._read_dump_ja(run_dir)
        if value is None:
            printwarn(f"NIMROD: could not read dump_ja from {run_dir / 'nimrod.in'}; "
                      "assuming dump_ja=F (current density not available).")
            value = False
        if not value:
            printnote(f"NIMROD: {run_dir.name}/nimrod.in has dump_ja=F. The current "
                      "density 'j' is not in the dumps; quantities that evaluate it "
                      "directly (the 'j' field and its flux average) are skipped. "
                      "Ip is computed from B (Ampere's law) and is not affected. "
                      "Set dump_ja=T in &output_input to write j "
                      "(or config['nimrod_dump_ja'] to override).")

        self._dump_ja_cache[run_dir] = value
        return value

    def _needs_current(self, kind, payload):
        """True if the resolved field requires the current density."""
        if kind == 'vector':
            return payload in self._CURRENT_FIELDS
        if kind == 'scalar':
            return payload[0] in self._CURRENT_FIELDS
        if kind == 'derived':
            return any(self._needs_current(*self._resolve_field(n)) for n in payload)
        return False

    def _skip_current_field(self, name, kind, payload, dump_file):
        """Warns once and returns True if *name* needs 'j' but dump_ja=F."""
        if self._needs_current(kind, payload) and not self._current_available(dump_file):
            self._warn_field_once(
                f"nocurrent::{name}",
                f"Field '{name}' needs the current density, which is not in the "
                "NIMROD dumps (dump_ja=F); skipping."
            )
            return True
        return False

    # ======================================================================
    # MODE DIRECTORY / DUMP DISCOVERY  (unchanged behaviour)
    # ======================================================================

    def set_mode_dirs(self, mode_dirs):
        """
        Registers the list of nXX mode directories for this model.
        Must be called before get_time_metadata() or _get_dump_file().
        """
        self._mode_dirs = [Path(d) for d in mode_dirs]
        self._common_max_index_cache = None

    def _ensure_all_h5_dumps(self):
        """Scans for binary dumps missing an .h5 counterpart and converts them."""
        if self._dumps_converted:
            return

        dump_dirs = set()
        for f in self.model_dir.rglob("dumpgll.*"):
            parts = f.name.split('.')
            if len(parts) == 2 and parts[1].isdigit():
                if (f.parent / "nimrod.in").exists():
                    dump_dirs.add(f.parent)

        for d in dump_dirs:
            self._convert_binary_dumps_to_h5(d)

        self._dumps_converted = True

    def _convert_binary_dumps_to_h5(self, target_dir):
        """Executes the 4-step workflow to convert binary dump files to HDF5."""
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

        tool_source = "/global/common/software/nimrod/spack/perlmutter/2026-4/linux-zen3/nimdevel-trunk-nq6numz6t3p7cat6jj3iro5gnajysvkb/bin/dump2h5"
        tool_dest = target_dir / "dump2h5"
        if not tool_dest.exists():
            try:
                shutil.copy(tool_source, tool_dest)
                os.chmod(tool_dest, 0o755)
            except Exception as e:
                printwarn(f"Failed to copy dump2h5 tool to {target_dir}: {e}")
                return

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

        Returns a dict mapping integer toroidal mode number n -> float growth rate (s^-1).
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

                keff_match = re.match(r'keff\s*=\s*([\d.]+)', line_stripped)
                if keff_match:
                    current_keff = int(float(keff_match.group(1)))
                    continue

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
        """Returns the cached nimgrowth results for a given mode directory."""
        mode_dir = Path(mode_dir)
        if mode_dir not in self._nimgrowth_cache:
            self._nimgrowth_cache[mode_dir] = self._parse_nimgrowth(mode_dir)
        return self._nimgrowth_cache[mode_dir]

    def _is_zero_dump(self, filepath):
        """
        Returns True if the HDF5 dump file contains only zero-valued data fields.
        """
        try:
            with h5py.File(filepath, 'r') as h5:
                rblocks = h5.get('rblocks')
                if rblocks:
                    for block_name in rblocks.keys():
                        for dset_name in rblocks[block_name].keys():
                            dset = rblocks[block_name][dset_name]
                            if hasattr(dset, 'shape') and dset.size > 0:
                                arr = dset[()]
                                if np.any(arr != 0):
                                    return False
                return True
        except Exception:
            return False

    def _get_indexed_dumps_for_dir(self, target_dir):
        """
        Builds and caches the indexed list of NIMROD HDF5 dump files for a single directory.

        Indexing convention:
          - Index 0: the dump file whose numeric step in the filename is 0, OR
                     the first dump file found to contain only zero-valued data.
          - Index 1, 2, ...: the remaining dump files sorted by ascending step number.

        Returns a list of (index, step_number, filepath) tuples sorted by index.
        """
        target_dir = Path(target_dir)
        if target_dir in self._indexed_dumps_per_dir:
            return self._indexed_dumps_per_dir[target_dir]

        self._ensure_all_h5_dumps()

        raw = []
        for f in target_dir.glob("dumpgll.*.h5"):
            parts = f.name.split('.')
            if len(parts) == 3 and parts[1].isdigit() and parts[2] == 'h5':
                step = int(parts[1])
                raw.append((step, f))

        if not raw:
            self._indexed_dumps_per_dir[target_dir] = []
            return []

        sorted_entries = sorted(raw)

        printnote(
            f"NIMROD: Found {len(sorted_entries)} HDF5 dump(s) in {target_dir}: "
            + ", ".join(str(s) for s, _ in sorted_entries)
        )

        zero_entry = None
        nonzero_entries = []

        if sorted_entries[0][0] == 0:
            zero_entry = sorted_entries[0]
            nonzero_entries = sorted_entries[1:]
        else:
            for entry in sorted_entries:
                if zero_entry is None and self._is_zero_dump(str(entry[1])):
                    zero_entry = entry
                else:
                    nonzero_entries.append(entry)

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
        """
        if not self._mode_dirs:
            return requested_idx

        per_dir_index_sets = []
        for d in self._mode_dirs:
            indexed = self._get_indexed_dumps_for_dir(d)
            indices = set(idx for idx, _, _ in indexed)
            if indices:
                per_dir_index_sets.append(indices)

        if not per_dir_index_sets:
            return requested_idx

        common_indices = per_dir_index_sets[0]
        for s in per_dir_index_sets[1:]:
            common_indices = common_indices & s

        if not common_indices:
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

        smaller = [i for i in common_indices_sorted if i <= requested_idx]
        if smaller:
            fallback = max(smaller)
            printwarn(
                f"Warning: Requested time slice index {requested_idx} does not exist in all "
                f"nXX directories (common indices: {common_indices_sorted}). "
                f"Using the next smaller common index {fallback} instead."
            )
            return fallback

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
        registered nXX mode directory.
        """
        if self._mode_dirs:
            search_dir = self._mode_dirs[0]
        else:
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

        requested_idx = 0 if time == -1 else time
        index_map = {idx: (step, path) for idx, step, path in indexed}

        if requested_idx in index_map:
            return str(index_map[requested_idx][1])

        resolved_idx = self._get_common_max_index(requested_idx)
        if resolved_idx in index_map:
            return str(index_map[resolved_idx][1])

        fallback_idx = max(index_map.keys())
        return str(index_map[fallback_idx][1])

    def _get_dump_file_for_dir(self, target_dir, resolved_idx):
        """
        Returns the HDF5 dump file path for a specific nXX directory and a
        pre-resolved sequential index.
        """
        indexed   = self._get_indexed_dumps_for_dir(target_dir)
        index_map = {idx: (step, path) for idx, step, path in indexed}

        if resolved_idx in index_map:
            return str(index_map[resolved_idx][1])

        if index_map:
            fallback_idx = max(index_map.keys())
            return str(index_map[fallback_idx][1])

        return None

    def _last_dump_in_dir(self, mode_dir):
        """Returns the path of the final (largest index) dump in a mode directory."""
        indexed = self._get_indexed_dumps_for_dir(Path(mode_dir))
        if not indexed:
            return None
        return str(indexed[-1][2])

    def _select_dump_files(self, mode_dir, selection="all"):
        """
        Selects which HDF5 dump files of a given nXX directory should be handed
        to dump2imas.
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

    # ======================================================================
    # EVALNIMROD HANDLING
    # ======================================================================

    def _fieldlist_candidates(self, dump_file):
        """
        Ordered nimpy field lists to try for *dump_file*.

        The base list 'nvptb' is the one used by the standard nimpy FSA
        scripts.  'j' is only added for runs with dump_ja=T, 'e' only if the
        electric field is requested (or config['nimrod_eval_e'] is set).
        """
        cfg = self._config or {}
        allow_j = self._current_available(dump_file)

        want_e = bool(cfg.get("nimrod_eval_e", False))
        try:
            sel = export_selection(cfg)
            for section in ("equilibrium", "total_fields", "perturbations"):
                for kind in ("flux_averages", "2d_fields", "3d_fields"):
                    if any(str(n) in ('E', 'e') for n in sel[section][kind]):
                        want_e = True
        except Exception:
            pass

        base = self._BASE_FIELDLIST
        with_j = base + ('j' if allow_j else '')
        first = with_j + ('e' if want_e else '')

        out = []
        for fl in (first, with_j, base) + self._FALLBACK_FIELDLISTS:
            if fl and fl not in out:
                out.append(fl)
        return out

    def _get_eval(self, dump_file, fieldlist=None):
        """
        Returns a cached EvalNimrod instance for the given dump file.

        The nimpy shared library can only keep one dump file open at a time, so
        a single active handle is tracked and re-initialised on demand.  Every
        field list is first tried in a child process (crash-safe probing); if
        the run has dump_ja=F, 'j' is removed from the field list.
        """
        if not _NIMPY_EVAL_AVAILABLE or dump_file is None:
            return None

        dump_path = Path(dump_file).resolve()
        allow_j = self._current_available(dump_file)

        def _strip(fl):
            return fl if allow_j else fl.replace('j', '')

        requested = fieldlist or self._fieldlist
        if requested is not None:
            requested = _strip(requested)
            key = (str(dump_path), requested)
            if self._active_eval_key == key and self._active_eval is not None:
                return self._active_eval

        candidates = []
        for fl in ([requested] if requested else []) + self._fieldlist_candidates(dump_file):
            fl = _strip(fl)
            if fl and fl not in candidates:
                candidates.append(fl)

        errors = []
        for fl in candidates:
            ok, msg = self._probe_eval_init(dump_path, fl)
            if not ok:
                errors.append(f"{fl}: {msg}")
                continue
            try:
                self._stage(f"EvalNimrod({dump_path.name}, fieldlist='{fl}')")
                with _pushd(dump_path.parent):
                    ev = EvalNimrod(str(dump_path), fieldlist=fl, path='./')
                self._active_eval = ev
                self._active_eval_key = (str(dump_path), fl)
                self._fieldlist = fl
                return ev
            except Exception as exc:
                errors.append(f"{fl}: {type(exc).__name__}: {exc}")

        printerr(
            f"Could not initialise EvalNimrod for {dump_path.name}. Attempts:\n  "
            + "\n  ".join(errors)
        )
        self._active_eval = None
        self._active_eval_key = None
        return None

    def _get_eval_grid(self, ev, dump_file, rzp, field):
        """
        Returns a cached EvalGrid for the given point set (logical mapping reuse).
        Returns None if EvalGrid is unavailable or disabled.
        """
        if not _NIMPY_EVAL_AVAILABLE:
            return None
        if not self._config.get("nimrod_use_eval_grid", True):
            return None

        try:
            sig = hashlib.md5(np.ascontiguousarray(rzp, dtype=float).tobytes()).hexdigest()
        except Exception:
            return None

        key = (str(Path(dump_file).resolve().parent), sig)
        grid = self._eval_grid_cache.get(key)
        if grid is not None:
            return grid

        try:
            grid = EvalGrid(np.asarray(rzp, dtype=float))
            with _pushd(Path(dump_file).resolve().parent):
                grid.set_logical_grid(field, ev)
            self._eval_grid_cache[key] = grid
            return grid
        except Exception as exc:
            printwarn(f"EvalGrid construction failed ({type(exc).__name__}: {exc}); using direct evaluation.")
            return None

    def _eval_raw(self, ev, dump_file, nim_field, R, Z, phi, eq=2, use_grid=True):
        """
        Evaluates a nimpy field at the supplied points; returns (nqty, N) array.

        use_grid=False skips the EvalGrid logical-mapping cache; this is
        cheaper for point sets that are evaluated only once.
        """
        rzp = np.array([np.asarray(R, dtype=float).ravel(),
                        np.asarray(Z, dtype=float).ravel(),
                        np.asarray(phi, dtype=float).ravel()])

        grid = self._get_eval_grid(ev, dump_file, rzp, nim_field) if use_grid else None

        with _pushd(Path(dump_file).resolve().parent):
            if grid is not None:
                try:
                    return ev.eval_field(nim_field, grid, dmode=0, eq=eq)
                except Exception:
                    pass
            return ev.eval_field(nim_field, rzp, dmode=0, eq=eq)

    def _eval_comp_raw(self, ev, dump_file, nim_field, R, Z):
        """
        Per-toroidal-mode evaluation with EvalNimrod.eval_comp_field.

        Returns a complex array of shape (nqty, nmodes, N).  Results are cached
        per (dump, field, point set), so all toroidal modes (and repeated
        requests) reuse a single evaluation pass.
        """
        R = np.asarray(R, dtype=float).ravel()
        Z = np.asarray(Z, dtype=float).ravel()
        key = (str(Path(dump_file).resolve()), nim_field, _points_signature(R, Z))
        if key in self._comp_cache:
            return self._comp_cache[key]

        # Always pass a (3, N) array; eval_comp_field ignores phi.
        rzp = np.array([R, Z, np.zeros_like(R)])
        grid = self._get_eval_grid(ev, dump_file, rzp, nim_field)

        res = None
        with _pushd(Path(dump_file).resolve().parent):
            if grid is not None:
                try:
                    res = ev.eval_comp_field(nim_field, grid, dmode=0)
                except Exception:
                    res = None
            if res is None:
                res = ev.eval_comp_field(nim_field, rzp, dmode=0)

        res = np.asarray(res)
        # (nqty, nmodes, nderiv, N) -> (nqty, nmodes, N)
        res = res[:, :, 0, ...].reshape(res.shape[0], res.shape[1], -1)
        self._comp_cache[key] = res
        return res

    def _warn_field_once(self, name, message):
        if name not in self._warned_fields:
            printwarn(message)
            self._warned_fields.add(name)

    def _ion_index(self, ev):
        """Index of the first ion species inside the nimpy 'n' field."""
        try:
            return 1 if int(getattr(ev, 'ndnq', 1)) > 1 else 0
        except Exception:
            return 0

    def _resolve_field(self, name):
        """
        Maps an edges-ml field name onto a NIMROD field specification.
        Returns (kind, payload) with kind in {'scalar', 'vector', 'derived'} or
        (None, None) if the field is not available for NIMROD.
        """
        raw = str(name)
        low = raw.lower()

        if low in self._DERIVED_FIELDS:
            return 'derived', self._DERIVED_FIELDS[low]
        if raw in self._VECTOR_FIELD_MAP:
            return 'vector', self._VECTOR_FIELD_MAP[raw]
        if low in self._VECTOR_FIELD_MAP:
            return 'vector', self._VECTOR_FIELD_MAP[low]
        if low in self._SCALAR_FIELD_MAP:
            return 'scalar', self._SCALAR_FIELD_MAP[low]
        return None, None

    def _map_nimrod_field(self, name):
        """Backwards-compatible helper: returns the bare nimpy field name."""
        kind, payload = self._resolve_field(name)
        if kind == 'scalar':
            return payload[0]
        if kind == 'vector':
            return payload
        return None

    def _species_index(self, ev, idx):
        if idx == 'electron':
            return 0
        if idx == 'ion':
            return self._ion_index(ev)
        return int(idx)

    def _evaluate_field_at_points(self, name, R, Z, phi, dump_file, eq=2):
        """
        Evaluates an edges-ml field name at arbitrary (R, Z, phi) points.
        Returns an ndarray for scalars, or a {component: ndarray} dict for vectors.
        """
        if dump_file is None:
            return np.array([])

        kind, payload = self._resolve_field(name)
        if kind is None:
            self._warn_field_once(
                name, f"Field '{name}' is not available from NIMROD dumps; skipping."
            )
            return np.array([])

        if self._skip_current_field(name, kind, payload, dump_file):
            return np.array([])
        if self._skip_unsafe_field(name, kind, payload, dump_file, eq):
            return np.array([])

        ev = self._get_eval(dump_file)
        if ev is None:
            return np.array([])

        try:
            if kind == 'derived':
                total_name, sub_name = payload
                total = self._evaluate_field_at_points(total_name, R, Z, phi, dump_file, eq)
                sub   = self._evaluate_field_at_points(sub_name,   R, Z, phi, dump_file, eq)
                if np.size(total) == 0 or np.size(sub) == 0:
                    return np.array([])
                return np.asarray(total) - np.asarray(sub)

            if kind == 'scalar':
                nim_field, idx = payload
                res = self._eval_raw(ev, dump_file, nim_field, R, Z, phi, eq=eq)
                res = np.atleast_2d(np.asarray(res))
                k = min(self._species_index(ev, idx), res.shape[0] - 1)
                return res[k]

            # vector
            nim_field = payload
            res = np.atleast_2d(np.asarray(self._eval_raw(ev, dump_file, nim_field, R, Z, phi, eq=eq)))
            if res.shape[0] < 3:
                self._warn_field_once(
                    name, f"Field '{name}' did not return 3 components from NIMROD; skipping."
                )
                return np.array([])
            return {
                f"{name}_R":   res[0],
                f"{name}_Z":   res[1],
                f"{name}_phi": self.PHI_SIGN_FLIP * res[2],
            }

        except Exception as exc:
            self._warn_field_once(
                name, f"Error evaluating NIMROD field '{name}': {type(exc).__name__}: {exc}"
            )
            return np.array([])

    # ======================================================================
    # PER-MODE (COMPLEX) EVALUATION
    # ======================================================================

    def _geometry(self, dump_file):
        """
        Returns (torgeom, per_length) for the run that produced *dump_file*,
        read from the nimrod.in next to it.
        """
        d = Path(dump_file).resolve().parent
        if d in self._geometry_cache:
            return self._geometry_cache[d]

        geom, per_length = "tor", 1.0
        found = False
        try:
            import f90nml
            nml = f90nml.read(str(d / "nimrod.in"))
            gi = nml.get('grid_input', {}) or {}
            geom = str(gi.get('geom', 'tor')).strip().lower()
            per_length = float(gi.get('per_length', 1.0))
            found = True
        except Exception:
            pass
        if not found:
            try:
                import nimpy.util.nimrodin as nimrodin_mod
                with _pushd(d):
                    nin = nimrodin_mod.NimrodIn()
                    geom = str(nin['geom']).strip().lower()
                    per_length = float(nin['per_length'])
            except Exception:
                printwarn(f"Could not read geometry from {d / 'nimrod.in'}; "
                          "assuming toroidal geometry.")

        result = (geom.startswith('tor'), per_length)
        self._geometry_cache[d] = result
        return result

    def _mode_table(self, dump_file):
        """
        List of (mode_index, n) for the toroidal modes stored in *dump_file*.

        The mode index addresses the mode axis returned by eval_comp_field,
        which is assumed to follow the order of 'keff' in the dump.  For 'lin'
        geometry keff is converted to a mode index (keff * per_length / 2 pi).
        """
        if dump_file is None:
            return []
        key = str(Path(dump_file).resolve())
        if key in self._mode_table_cache:
            return self._mode_table_cache[key]

        table = []
        try:
            with h5py.File(dump_file, 'r') as h5:
                keff = np.asarray(h5['keff'][()], dtype=float).ravel()
        except Exception as exc:
            printwarn(f"Could not read keff from {Path(dump_file).name}: {exc}")
            self._mode_table_cache[key] = table
            return table

        torgeom, per_length = self._geometry(dump_file)
        if not torgeom:
            keff = keff * per_length / (2.0 * np.pi)

        n_vals = [int(round(k)) for k in keff]

        ev = self._get_eval(dump_file)
        nmodes_peq = None
        if ev is not None:
            try:
                nmodes_peq = int(getattr(ev, 'nmodes_peq'))
            except Exception:
                nmodes_peq = None
        if nmodes_peq is not None and nmodes_peq != len(n_vals):
            nuse = min(nmodes_peq, len(n_vals))
            printwarn(
                f"NIMROD: dump keff has {len(n_vals)} modes but nimpy reports "
                f"nmodes_peq={nmodes_peq} for {Path(dump_file).name}; using the "
                f"first {nuse} modes."
            )
            n_vals = n_vals[:nuse]

        table = list(enumerate(n_vals))
        self._mode_table_cache[key] = table
        return table

    def _mode_coefficients(self, name, R, Z, dump_file):
        """
        Complex Fourier coefficients of an edges-ml field at (R, Z) points.

        Returns {key: complex array [nmodes, N]} with key = name for scalars
        and name_R / name_Z / name_phi for vectors (phi component multiplied
        by PHI_SIGN_FLIP).  The coefficients follow NIMROD's convention:
            field(phi_nimrod) = 2 Re[c e^{i n phi_nimrod}]   (n > 0).
        """
        kind, payload = self._resolve_field(name)
        if kind is None:
            self._warn_field_once(
                name, f"Field '{name}' is not available from NIMROD dumps; skipping."
            )
            return {}

        if self._skip_current_field(name, kind, payload, dump_file):
            return {}
        if self._skip_unsafe_field(name, kind, payload, dump_file, 2):
            return {}

        ev = self._get_eval(dump_file)
        if ev is None:
            return {}

        try:
            if kind == 'derived':
                total_name, sub_name = payload
                a = self._mode_coefficients(total_name, R, Z, dump_file)
                b = self._mode_coefficients(sub_name, R, Z, dump_file)
                if total_name not in a or sub_name not in b:
                    return {}
                return {name: a[total_name] - b[sub_name]}

            if kind == 'scalar':
                nim_field, idx = payload
                res = self._eval_comp_raw(ev, dump_file, nim_field, R, Z)
                k = min(self._species_index(ev, idx), res.shape[0] - 1)
                return {name: res[k]}

            nim_field = payload
            res = self._eval_comp_raw(ev, dump_file, nim_field, R, Z)
            if res.shape[0] < 3:
                self._warn_field_once(
                    f"comp::{name}",
                    f"Field '{name}' did not return 3 complex components from NIMROD; skipping."
                )
                return {}
            return {
                f"{name}_R":   res[0],
                f"{name}_Z":   res[1],
                f"{name}_phi": self.PHI_SIGN_FLIP * res[2],
            }
        except Exception as exc:
            self._warn_field_once(
                f"comp::{name}",
                f"Error evaluating NIMROD per-mode field '{name}': {brief_error(exc)}"
            )
            return {}

    def _mode_fields_on_grid(self, name, grid_spec, dump_file, imode, n, is_3d,
                             store_complex):
        """
        Physical perturbation of toroidal mode n on a 2D/3D output grid:
            2 Re[c_n e^{i n phi}]  (n > 0),   Re[c_0]  (n = 0),
        with phi the grid's toroidal angle (NIMROD convention, as used for the
        equilibrium 3D fields; phi = 0 for 2D grids).  Each unique (R, Z) point
        is evaluated once; the phi dependence is reconstructed analytically.
        """
        if is_3d:
            R, phi, Z = self._grid_points_3d(grid_spec, time=-1)
        else:
            R, Z = self._grid_points_2d(grid_spec, time=-1)
            phi = np.zeros_like(R)
        if R.size == 0:
            return np.array([])

        rz = np.column_stack((R, Z))
        uniq, inv = np.unique(rz, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()

        coeffs = self._mode_coefficients(name, uniq[:, 0], uniq[:, 1], dump_file)
        if not coeffs:
            return np.array([])

        out = {}
        phase = np.exp(1j * n * phi) if n != 0 else None
        for key, arr in coeffs.items():
            arr = np.asarray(arr)
            if imode >= arr.shape[0]:
                return np.array([])
            c = arr[imode][inv].astype(np.complex128)
            if n == 0:
                out[key] = np.real(c)
            else:
                out[key] = 2.0 * np.real(c * phase)
            if store_complex:
                out[f"{key}_re"] = np.real(c)
                out[f"{key}_im"] = np.imag(c)

        if list(out.keys()) == [name]:
            return out[name]
        return out

    # ======================================================================
    # PER-MODE ENTRIES (one perturbation group per toroidal mode)
    # ======================================================================

    def extract_mode_entries(self, mode_dir, config):
        """
        One perturbation entry per toroidal mode stored in the final dump of
        the mode directory.  Works identically for single-n and multi-n runs.

        Falls back to the legacy behaviour (summed perturbation from
        extract_mode() + get_all_mode_entries()) if the mode table cannot be
        determined.
        """
        self._config = config or {}
        mode_dir = Path(mode_dir)
        dump_file = self._last_dump_in_dir(mode_dir)
        table = self._mode_table(dump_file) if (dump_file and _NIMPY_EVAL_AVAILABLE) else []

        if not table:
            printwarn(f"NIMROD: could not determine the toroidal modes of {mode_dir}; "
                      "falling back to the legacy (summed) perturbation output.")
            return super().extract_mode_entries(mode_dir, config)

        include_n0 = bool(self._config.get("nimrod_include_n0_mode", False))
        growth = self._get_nimgrowth_for_dir(mode_dir)

        # Perturbation flux averages describe the n=0 profile modification of
        # the directory (n>0 components average to zero); compute once.
        flux_avgs = self._extract_mode_flux_averages(mode_dir, self._config)

        entries = []
        for imode, n in table:
            if n == 0 and not include_n0 and len(table) > 1:
                continue
            if growth and n not in growth:
                printwarn(f"Toroidal mode n={n} of {mode_dir.name} not found in nimgrowth output.")
            meta = {
                "growth_rate": float(growth.get(n, 0.0)),
                "frequency":   0.0,
                "mode_type":   -100,
            }
            data = self._extract_single_mode(mode_dir, dump_file, imode, n, meta, flux_avgs)
            entries.append((f"n{n:02d}", meta, data))

        if not entries:
            printwarn(f"NIMROD: no toroidal modes selected in {mode_dir} "
                      "(only n=0 present; set nimrod_include_n0_mode to export it).")
        return entries

    def _extract_single_mode(self, mode_dir, dump_file, imode, n, meta, flux_avgs):
        """
        Perturbation data of a single toroidal mode for every quantity listed
        under config['perturbations']: flux averages, 2D / 3D fields (on the
        same grids as all other fields) and the poloidal spectra.
        """
        cfg = self._config or {}
        pert = export_selection(cfg)["perturbations"]
        data = {
            "mode_information": dict(meta),
            "flux_averages": dict(flux_avgs),
            "2d_fields": {},
            "3d_fields": {},
        }
        store_complex = bool(cfg.get("nimrod_store_complex", False))

        grid_2d = self._grid_spec(cfg, 2)
        for field in pert["2d_fields"]:
            val = self._mode_fields_on_grid(field, grid_2d, dump_file, imode, n, False, store_complex)
            if isinstance(val, dict):
                data["2d_fields"].update(val)
            elif np.size(val) > 0:
                data["2d_fields"][field] = val

        grid_3d = self._grid_spec(cfg, 3)
        for field in pert["3d_fields"]:
            val = self._mode_fields_on_grid(field, grid_3d, dump_file, imode, n, True, store_complex)
            if isinstance(val, dict):
                data["3d_fields"].update(val)
            elif np.size(val) > 0:
                data["3d_fields"][field] = val

        settings = spectrum_settings(cfg)
        spectra = {}
        for field in settings["fields"]:
            try:
                res = self.get_mode_spectrum(mode_dir, field, settings, n=n, imode=imode)
            except Exception as exc:
                printerr(f"Poloidal spectrum of '{field}' (n={n}) failed for {mode_dir}: "
                         f"{brief_error(exc)}")
                res = {}
            if res:
                spectra[field] = res
        if spectra:
            data["spectra"] = spectra

        return data

    # ======================================================================
    # POLOIDAL SPECTRA
    # ======================================================================

    def _get_flux_coordinates(self, dump_file, settings):
        """
        Flux-aligned coordinates of the equilibrium (eq=1 field), cached per
        model (all mode directories of a model share the equilibrium).  The
        dump that is evaluated anyway is used, so the single-file nimpy
        library does not have to switch files.
        """
        cfg = self._config or {}
        points = int(settings["points"])
        fcoords = settings["fcoords"]
        oversample = int(settings["theta_oversample"])
        ray_samples = int(cfg.get("nimrod_psi_ray_samples", 4 * points))
        include_omega = bool(settings["include_omega"])
        key = (points, fcoords, oversample, ray_samples, include_omega)

        if key in self._fluxcoord_cache:
            return self._fluxcoord_cache[key]
        if key in self._fluxcoord_failed:
            return None

        ev = self._get_eval(dump_file)
        if ev is None or not self._field_ok(dump_file, 'b', 1):
            printerr(f"NIMROD: the magnetic field cannot be evaluated for "
                     f"{Path(dump_file).name}; no flux coordinates / spectra.")
            self._fluxcoord_failed.add(key)
            return None

        try:
            self._stage(f"flux coordinates ({fcoords}, {points} points) on {Path(dump_file).name}")
            axis = self._find_magnetic_axis(ev, dump_file)
            bbox = self._get_bounding_box(-1)
            torgeom, per_length = self._geometry(dump_file)
            period = 2.0 * np.pi if torgeom else per_length

            def eval_b(R, Z):
                R = np.asarray(R, dtype=float).ravel()
                Z = np.asarray(Z, dtype=float).ravel()
                res = np.atleast_2d(np.asarray(
                    self._eval_raw(ev, dump_file, 'b', R, Z, np.zeros_like(R),
                                   eq=1, use_grid=False)))
                return res[0], res[1], self.PHI_SIGN_FLIP * res[2]

            find_null = None
            if _NIMPY_FSA_AVAILABLE:
                def find_null(guess):
                    try:
                        with _pushd(Path(dump_file).resolve().parent):
                            return np.asarray(find_pf_null(
                                ev, [float(guess[0]), float(guess[1]), 0.0], addpert=False))
                    except Exception:
                        return None

            guesses = []
            raw_guess = cfg.get("nimrod_xpoint_guess", None)
            if raw_guess is not None:
                arr = np.atleast_2d(np.asarray(raw_guess, dtype=float))
                if arr.shape[1] >= 2:
                    guesses = [list(row[:2]) for row in arr]

            fc = build_flux_coordinates(
                eval_b, axis, bbox, points,
                fcoords=fcoords, torgeom=torgeom, period=period,
                find_null=find_null, xpoint_guesses=guesses,
                psi_lcfs=cfg.get("nimrod_psi_lcfs", None),
                ray_samples=ray_samples, theta_oversample=oversample,
                include_omega=include_omega, verbose=True,
            )
        except Exception as exc:
            printerr(f"NIMROD flux coordinates failed for {Path(dump_file).name}: "
                     f"{brief_error(exc)}")
            self._fluxcoord_failed.add(key)
            return None

        self._fluxcoord_cache[key] = fc
        return fc

    def _select_mode(self, table, mode_dir, n, imode):
        """Resolves (mode index, n) from the mode table."""
        if imode is not None and n is not None:
            return imode, n
        if n is not None:
            for i, nn in table:
                if nn == n:
                    return i, nn
            return None, n
        if len(table) == 1:
            return table[0]
        guess = parse_toroidal_mode_number(Path(mode_dir).name)
        if guess is not None:
            for i, nn in table:
                if nn == guess:
                    return i, nn
        return None, None

    def get_mode_spectrum(self, mode_dir, field, settings, n=None, imode=None):
        """
        Poloidal Fourier spectrum of toroidal mode n of *field* on flux-aligned
        coordinates of the equilibrium.

        The real eigenfunction in the M3D-C1 toroidal-angle convention is
            g(theta, psi) = 2 Re[conj(c_n) e^{i n phi}],  phi = zeta_t (- omega)
        where conj() converts NIMROD's toroidal angle direction to M3D-C1's and
        omega is only applied for Boozer / Hamada if include_omega is set.
        The spectrum is |rfft(g, axis=0, norm='forward')|, as in M3D-C1.

        Returned datasets: psi_norm, m, amplitude[m, psi_norm] (plus the flux
        grid if 'store_fluxgrid' is enabled).  q is not included; it is
        available as a flux average.
        """
        if not _NIMPY_EVAL_AVAILABLE:
            return {}

        dump_file = self._last_dump_in_dir(mode_dir)
        if dump_file is None:
            return {}

        table = self._mode_table(dump_file)
        if not table:
            return {}
        imode, n = self._select_mode(table, mode_dir, n, imode)
        if imode is None:
            self._warn_field_once(
                f"spec_mode::{mode_dir}::{n}",
                f"Toroidal mode n={n} not found in {Path(dump_file).name}; no spectrum."
            )
            return {}

        kind, payload = self._resolve_field(field)
        if kind not in ('scalar', 'derived'):
            self._warn_field_once(
                f"spec_vec::{field}",
                f"Poloidal spectra of non-scalar field '{field}' are not supported yet; skipping."
            )
            return {}
        if self._skip_current_field(field, kind, payload, dump_file):
            return {}
        if self._skip_unsafe_field(field, kind, payload, dump_file, 2):
            return {}

        fc = self._get_flux_coordinates(dump_file, settings)
        if fc is None:
            return {}

        shape = fc.rpath.shape
        coeffs = self._mode_coefficients(field, fc.rpath.ravel(), fc.zpath.ravel(), dump_file)
        if field not in coeffs:
            return {}
        arr = np.asarray(coeffs[field])
        if imode >= arr.shape[0]:
            return {}
        c = arr[imode].reshape(shape).astype(np.complex128)

        n_bad = int(np.count_nonzero(~np.isfinite(c)))
        if n_bad:
            printwarn(f"NIMROD spectrum '{field}' n={n}: {n_bad} flux-grid points could "
                      "not be evaluated; affected surfaces will contain NaN.")

        F = np.conj(c)
        phit = float(settings["phi"])
        apply_omega = bool(settings["include_omega"]) and fc.omega is not None
        if n == 0:
            ef = np.real(F)
        else:
            phi = np.full(shape, phit)
            if apply_omega:
                phi = phi - fc.omega
            ef = 2.0 * np.real(F * np.exp(1j * n * phi))

        amplitude, m = poloidal_spectrum(ef)

        out = {
            "psi_norm":  np.asarray(fc.psi_norm, dtype=float),
            "m":         m,
            "amplitude": amplitude,
            "_attrs": {
                "code":             self.CODE_NAME,
                "field":            str(field),
                "n":                int(n),
                "fcoords":          fc.fcoords,
                "phi":              phit,
                "toroidal_angle":   "zeta" if apply_omega else "geometric_phi",
                "points":           int(settings["points"]),
                "normalization":    "rfft_forward_abs",
                "theta_convention": "theta=0 outboard midplane, clockwise in (R,Z)",
                "psi_lcfs_source":  str(fc.lcfs_source),
                "psi_lcfs":         float(fc.psi_lcfs),
                "n_clipped":        int(fc.n_clipped),
                "n_unreached":      int(fc.n_unreached),
            },
        }
        if settings.get("store_fluxgrid"):
            out["eigenfunction"] = ef
            out["R"] = np.asarray(fc.rpath, dtype=float)
            out["Z"] = np.asarray(fc.zpath, dtype=float)
            out["theta"] = np.asarray(fc.theta, dtype=float)
            if fc.omega is not None:
                out["omega"] = np.asarray(fc.omega, dtype=float)
        return out

    # ======================================================================
    # GEOMETRY / GRIDS
    # ======================================================================

    def _get_bounding_box(self, time=-1):
        """
        Reads the underlying NIMROD HDF5 dump file to dynamically find the
        global minimum and maximum R and Z coordinates.
        Returns: (rmin, rmax, zmin, zmax)
        """
        pts = self._get_native_mesh_points(time)
        if len(pts) > 0:
            return (float(np.min(pts[:, 0])), float(np.max(pts[:, 0])),
                    float(np.min(pts[:, 1])), float(np.max(pts[:, 1])))
        return 1.0, 2.0, -1.0, 1.0

    def _get_native_mesh_points(self, time=-1):
        """Returns the (R, Z) coordinates of the native NIMROD finite element nodes."""
        dump_file = self._get_dump_file(time)
        if not dump_file:
            return np.empty((0, 2))
        return self._native_points_for_file(dump_file)

    def _get_wall_polygon(self):
        """
        Returns the (R, Z) wall polygon from contours.h5 or sol.grn, or None.
        """
        if self._wall_polygon_searched:
            return self._wall_polygon
        self._wall_polygon_searched = True

        search_dirs = list(self._mode_dirs) + [self.model_dir, self.model_dir.parent]
        for d in search_dirs:
            cfile = Path(d) / "contours.h5"
            if cfile.is_file():
                try:
                    with h5py.File(cfile, 'r') as h5:
                        if '/wall/points' in h5:
                            pts = np.asarray(h5['/wall/points'][()])
                            self._wall_polygon = np.vstack([pts, pts[0, :]])
                            return self._wall_polygon
                except Exception:
                    pass

            sfile = Path(d) / "sol.grn"
            if sfile.is_file():
                try:
                    with open(sfile, 'r') as f:
                        lines = f.readlines()
                    nsep = int(lines[8].split()[1])
                    rz = np.zeros((nsep, 2))
                    for idx in range(nsep):
                        rz[idx, 0] = float(lines[9].split()[idx])
                        rz[idx, 1] = float(lines[11].split()[idx])
                    self._wall_polygon = np.vstack([rz, rz[0, :]])
                    return self._wall_polygon
                except Exception:
                    pass

        return self._wall_polygon

    def _get_inside_wall_points(self, time=-1):
        """
        Native mesh nodes restricted to the interior of the wall contour (when one
        is available).  Falls back to the full native mesh.
        """
        if getattr(self, 'shared_inner_wall_points', None) is not None:
            return np.asarray(self.shared_inner_wall_points)

        pts = self._get_native_mesh_points(time)
        if len(pts) == 0:
            return pts

        poly = self._get_wall_polygon()
        if poly is None:
            return pts

        try:
            from matplotlib.path import Path as MplPath
            mask = MplPath(np.asarray(poly)[:, :2]).contains_points(pts)
            inside = pts[mask]
            if len(inside) > 0:
                return inside
        except Exception as exc:
            printwarn(f"Could not filter points by the NIMROD wall contour: {exc}")

        return pts

    def _grid_points_2d(self, grid_spec, time=-1):
        """Returns (R, Z) 1D arrays for the requested 2D grid specification."""
        if grid_spec.type == "rectangular":
            rmin, rmax, zmin, zmax = self._get_bounding_box(time)
            nR, nZ = grid_spec.resolution
            R_lin = np.linspace(rmin, rmax, nR)
            Z_lin = np.linspace(zmin, zmax, nZ)
            R, Z = np.meshgrid(R_lin, Z_lin, indexing='ij')
            return R.ravel(), Z.ravel()

        if grid_spec.type == "native":
            pts = self._get_native_mesh_points(time)
            if len(pts) == 0:
                return np.array([]), np.array([])
            return pts[:, 0], pts[:, 1]

        if grid_spec.type == "inside_wall":
            pts = self._get_inside_wall_points(time)
            if len(pts) == 0:
                return np.array([]), np.array([])
            return pts[:, 0], pts[:, 1]

        printwarn(f"Grid type '{grid_spec.type}' is not supported for NIMROD.")
        return np.array([]), np.array([])

    def _grid_points_3d(self, grid_spec, time=-1):
        """Returns (R, phi, Z) 1D arrays for the requested 3D grid specification."""
        if grid_spec.type == "rectangular":
            rmin, rmax, zmin, zmax = self._get_bounding_box(time)
            nR, nPhi, nZ = grid_spec.resolution
            R_lin   = np.linspace(rmin, rmax, nR)
            Phi_lin = np.linspace(0, 2 * np.pi, nPhi, endpoint=False)
            Z_lin   = np.linspace(zmin, zmax, nZ)
            R, Phi, Z = np.meshgrid(R_lin, Phi_lin, Z_lin, indexing='ij')
            return R.ravel(), Phi.ravel(), Z.ravel()

        if grid_spec.type in ("native", "inside_wall"):
            R, Z = self._grid_points_2d(grid_spec, time)
            return R, np.zeros_like(R), Z

        printwarn(f"Grid type '{grid_spec.type}' is not supported for NIMROD.")
        return np.array([]), np.array([]), np.array([])

    def get_2d_mesh(self, grid_spec):
        R, Z = self._grid_points_2d(grid_spec, time=-1)
        if R.size == 0:
            return np.array([])
        return np.column_stack((R, Z))

    def get_3d_mesh(self, grid_spec):
        R, phi, Z = self._grid_points_3d(grid_spec, time=-1)
        if R.size == 0:
            return np.array([])
        return np.column_stack((R, phi, Z))

    def get_2d_field(self, name, grid_spec, units, time=-1):
        dump_file = self._get_dump_file(time)
        if not dump_file:
            return np.array([])
        R, Z = self._grid_points_2d(grid_spec, time)
        if R.size == 0:
            return np.array([])
        phi = np.zeros_like(R)
        return self._evaluate_field_at_points(name, R, Z, phi, dump_file, eq=2)

    def get_3d_field(self, name, grid_spec, units, time=-1):
        dump_file = self._get_dump_file(time)
        if not dump_file:
            return np.array([])
        R, phi, Z = self._grid_points_3d(grid_spec, time)
        if R.size == 0:
            return np.array([])
        return self._evaluate_field_at_points(name, R, Z, phi, dump_file, eq=2)

    def get_mode_2d_mesh(self, mode_dir, grid_spec):
        return self.get_2d_mesh(grid_spec)

    def get_mode_3d_mesh(self, mode_dir, grid_spec):
        return self.get_3d_mesh(grid_spec)

    def get_mode_2d_field(self, mode_dir, name, grid_spec, units):
        """
        Evaluates the field at the final dump of the given mode directory.
        Only used by the legacy fallback path of extract_mode_entries().
        """
        dump_file = self._last_dump_in_dir(mode_dir)
        if dump_file is None:
            printwarn(f"No dump files found in mode directory {mode_dir}.")
            return np.array([])
        R, Z = self._grid_points_2d(grid_spec, time=-1)
        if R.size == 0:
            return np.array([])
        phi = np.zeros_like(R)
        return self._evaluate_field_at_points(name, R, Z, phi, dump_file, eq=2)

    def get_mode_3d_field(self, mode_dir, name, grid_spec, units):
        """
        Evaluates the field at the final dump of the given mode directory.
        Only used by the legacy fallback path of extract_mode_entries().
        """
        dump_file = self._last_dump_in_dir(mode_dir)
        if dump_file is None:
            printwarn(f"No dump files found in mode directory {mode_dir}.")
            return np.array([])
        R, phi, Z = self._grid_points_3d(grid_spec, time=-1)
        if R.size == 0:
            return np.array([])
        return self._evaluate_field_at_points(name, R, Z, phi, dump_file, eq=2)

    # ======================================================================
    # FLUX SURFACE AVERAGES (nimpy.fsa)
    # ======================================================================

    # Integrand evaluators: key -> callable(ev, rzc, eq, ion_index) -> float
    @staticmethod
    def _fsa_evaluators():
        def scalar(field, idx=0):
            def _f(ev, rzc, eq, ion):
                v = ev.eval_field(field, rzc, dmode=0, eq=eq)
                k = ion if idx == 'ion' else int(idx)
                return float(np.atleast_1d(v)[min(k, np.size(v) - 1)])
            return _f

        def vec_comp(field, comp):
            def _f(ev, rzc, eq, ion):
                v = ev.eval_field(field, rzc, dmode=0, eq=eq)
                return float(v[comp])
            return _f

        def vec_mag(field):
            def _f(ev, rzc, eq, ion):
                v = ev.eval_field(field, rzc, dmode=0, eq=eq)
                return float(np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2))
            return _f

        def bpol(power):
            def _f(ev, rzc, eq, ion):
                b = ev.eval_field('b', rzc, dmode=0, eq=eq)
                bp2 = b[0] ** 2 + b[1] ** 2
                return float(bp2 if power == 2 else np.sqrt(bp2))
            return _f

        def rbphi(ev, rzc, eq, ion):
            b = ev.eval_field('b', rzc, dmode=0, eq=eq)
            return float(rzc[0] * b[2])

        def jphi_over_r(ev, rzc, eq, ion):
            j = ev.eval_field('j', rzc, dmode=0, eq=eq)
            return float(j[2] / rzc[0])

        return {
            'p':    scalar('p'),
            'pe':   scalar('pe'),
            'te':   scalar('te'),
            'ti':   scalar('ti'),
            'psi':  scalar('psi'),
            'ne':   scalar('n', 0),
            'ni':   scalar('n', 'ion'),
            'B':    vec_mag('b'),
            'v':    vec_mag('v'),
            'j':    vec_mag('j'),
            'B_R':   vec_comp('b', 0),
            'B_Z':   vec_comp('b', 1),
            'B_phi': vec_comp('b', 2),
            'v_R':   vec_comp('v', 0),
            'v_Z':   vec_comp('v', 1),
            'v_phi': vec_comp('v', 2),
            'j_R':   vec_comp('j', 0),
            'j_Z':   vec_comp('j', 1),
            'j_phi': vec_comp('j', 2),
            'Bp':    bpol(1),
            'Bp2':   bpol(2),
            'f':     rbphi,
            'jphi_over_R': jphi_over_r,
        }

    # edges-ml flux average name -> integrand keys that must be computed.
    # 'Ip' needs no integrand: it is computed from B along the FSA contours
    # (Ampere's law, see _enclosed_current).
    _FSA_NAME_TO_KEYS = {
        'p':    ('p',),
        'pe':   ('pe',),
        'pi':   ('p', 'pe'),
        'ne':   ('ne',),
        'ni':   ('ni',),
        'te':   ('te',),
        'ti':   ('ti',),
        'psi':  ('psi',),
        'B':    ('B_R', 'B_Z', 'B_phi'),
        'v':    ('v_R', 'v_Z', 'v_phi'),
        'j':    ('j_R', 'j_Z', 'j_phi'),
        'f':    ('f',),
        'ffprime': ('f',),
        'Ip':   (),
        'V':    (),
        'q':    (),
        'rho':  (),
        'flux_p': (),
        'flux_t': (),
    }

    # Integrands needed by get_global_parameters() (Ip comes from the contours).
    _GLOBAL_FSA_KEYS = ('p', 'Bp', 'Bp2', 'f')

    def _requested_fsa_keys(self, dump_file=None, eq=1):
        """
        Integrand keys required by the flux averages of all three sections
        (equilibrium, total_fields, perturbations).

        If *dump_file* is given, keys that evaluate the current density are
        dropped for runs with dump_ja=F, and keys whose nimpy fields fail the
        crash-safe probe are dropped as well.
        """
        cfg = self._config or {}
        sel = export_selection(cfg)
        names = set()
        for section in ("equilibrium", "total_fields", "perturbations"):
            names.update(sel[section]["flux_averages"])
        if not names:
            names = set(AVAILABLE_FLUX_AVERAGES)

        keys = set()
        for n in names:
            keys.update(self._FSA_NAME_TO_KEYS.get(n, ()))

        if cfg.get("nimrod_compute_globals", True):
            keys.update(self._GLOBAL_FSA_KEYS)

        if dump_file is not None and not self._current_available(dump_file):
            dropped = keys & self._CURRENT_FSA_KEYS
            if dropped:
                keys -= self._CURRENT_FSA_KEYS
                affected = sorted(n for n in names
                                  if set(self._FSA_NAME_TO_KEYS.get(n, ())) & self._CURRENT_FSA_KEYS)
                self._warn_field_once(
                    "nocurrent::fsa",
                    "NIMROD: the current density is not in the dumps (dump_ja=F); "
                    f"flux averages that evaluate it are skipped: {', '.join(affected)}. "
                    "(Ip is computed from B and is not affected.)"
                )

        if dump_file is not None:
            bad = []
            for k in sorted(keys):
                fields = self._FSA_KEY_FIELDS.get(k, ())
                if any(not self._field_ok(dump_file, f, eq) for f in fields):
                    bad.append(k)
            if bad:
                keys -= set(bad)
                affected = sorted(n for n in names
                                  if set(self._FSA_NAME_TO_KEYS.get(n, ())) & set(bad))
                self._warn_field_once(
                    f"unsafe::fsa::{eq}",
                    f"NIMROD: flux-surface-average integrands {bad} cannot be evaluated "
                    f"(eq={eq}); the flux averages {affected} are skipped."
                )

        return tuple(sorted(keys))

    def _fsa_nsurf(self, resolution):
        cfg = self._config or {}
        return int(cfg.get("nimrod_fsa_nsurf", min(int(resolution), 60)))

    def _find_magnetic_axis(self, ev, dump_file, time=-1):
        """Coarse search for |Bpol| minimum, refined with nimpy's find_pf_null."""
        key = str(Path(dump_file).resolve())
        if key in self._axis_cache:
            return self._axis_cache[key]

        self._stage(f"magnetic-axis search on {Path(dump_file).name}")

        rmin, rmax, zmin, zmax = self._get_bounding_box(time)
        dr = 0.08 * (rmax - rmin)
        dz = 0.08 * (zmax - zmin)
        R_lin = np.linspace(rmin + dr, rmax - dr, 25)
        Z_lin = np.linspace(zmin + dz, zmax - dz, 25)
        Rg, Zg = np.meshgrid(R_lin, Z_lin, indexing='ij')
        phi = np.zeros_like(Rg)

        axis = np.array([0.5 * (rmin + rmax), 0.5 * (zmin + zmax), 0.0])
        try:
            res = self._eval_raw(ev, dump_file, 'b', Rg.ravel(), Zg.ravel(), phi.ravel(), eq=1)
            bp = np.sqrt(np.asarray(res[0]) ** 2 + np.asarray(res[1]) ** 2)
            if np.any(np.isfinite(bp)):
                i = int(np.nanargmin(bp))
                axis = np.array([Rg.ravel()[i], Zg.ravel()[i], 0.0])
        except Exception as exc:
            printwarn(f"Coarse magnetic-axis search failed: {exc}")

        if _NIMPY_FSA_AVAILABLE:
            self._stage("find_pf_null (magnetic axis refinement)")
            try:
                with _pushd(Path(dump_file).resolve().parent):
                    axis = np.asarray(find_pf_null(ev, axis, addpert=False))
            except Exception as exc:
                printwarn(f"find_pf_null could not refine the magnetic axis: {exc}")

        self._axis_cache[key] = axis
        return axis

    def _get_fsa_record(self, dump_file, nsurf, eq=1):
        """
        Runs (and caches) a single FSA pass for the given dump, computing every
        integrand key required by the configuration.

        Returns a dict with 'dvar', 'contours', 'axis', 'values' (key -> array),
        'dump_file' and 'eq', or None if the FSA could not be performed.
        All surface-indexed arrays (dvar, values, contours) are sorted by
        ascending psi_norm.
        """
        if dump_file is None or not (_NIMPY_FSA_AVAILABLE and _NIMPY_EVAL_AVAILABLE):
            return None
        if not (self._config or {}).get("nimrod_enable_fsa", True):
            return None

        # Field-line tracing needs the magnetic field.
        if not self._field_ok(dump_file, 'b', eq):
            self._warn_field_once(
                f"fsa::nob::{dump_file}::{eq}",
                f"NIMROD: the magnetic field cannot be evaluated for "
                f"{Path(dump_file).name} (eq={eq}); flux averages are skipped."
            )
            return None

        keys = self._requested_fsa_keys(dump_file, eq)
        cache_key = (str(Path(dump_file).resolve()), int(nsurf), int(eq), keys)
        if cache_key in self._fsa_cache:
            return self._fsa_cache[cache_key]
        if cache_key in self._fsa_failed:
            return None

        ev = self._get_eval(dump_file)
        if ev is None:
            self._fsa_failed.add(cache_key)
            return None

        axis = self._find_magnetic_axis(ev, dump_file)
        ion = self._ion_index(ev)
        evaluators = self._fsa_evaluators()
        funcs = [evaluators[k] for k in keys]
        neq = max(1, len(funcs))

        def integrand(rzc, y, dy, eval_nimrod, fdict):
            for i, fn in enumerate(funcs):
                try:
                    dy[4 + i] = fn(eval_nimrod, rzc, eq, ion) * dy[2]
                except Exception:
                    dy[4 + i] = np.nan
            return dy

        printnote(f"NIMROD: running flux surface average on {Path(dump_file).name} "
                  f"({nsurf} surfaces, {len(keys)} quantities, eq={eq}).")
        self._stage(f"FSA integrands: {list(keys)}")

        try:
            # FSAInput.d is a class-level global in nimpy; reset it so that a
            # previous model's axis/x-point are not reused for this one.
            FSAInput.d['rzo'] = [float(axis[0]), float(axis[1]), 0.0]
            FSAInput.d['rzx'] = None
            FSAInput.d['rzs'] = None
            FSAInput.d['psis'] = None
            FSAInput.d['phis'] = None

            with _pushd(Path(dump_file).resolve().parent):
                dvar, yvals, contours = FSA(
                    ev, list(axis), integrand, neq,
                    nsurf=int(nsurf),
                    depvar='eta',
                    plot_contours=False,
                    normalize=True,
                    addpert=(eq != 1),
                )
        except Exception as exc:
            printerr(f"FSA failed for {Path(dump_file).name}: {type(exc).__name__}: {exc}")
            self._fsa_failed.add(cache_key)
            return None
        finally:
            FSAInput.d['rzo'] = None
            FSAInput.d['rzx'] = None

        self._stage("FSA finished")

        dvar = np.asarray(dvar)
        yvals = np.asarray(yvals)
        contours = np.asarray(contours)
        if dvar.size == 0:
            self._fsa_failed.add(cache_key)
            return None

        nsurf_out = dvar.shape[1]
        order = np.argsort(dvar[0, :])
        dvar = dvar[:, order]
        yvals = yvals[:, order] if yvals.size else yvals
        # Keep the contours aligned with dvar (surface index = last axis).
        if contours.ndim == 3 and contours.shape[-1] == nsurf_out:
            contours = contours[..., order]

        values = {}
        for i, k in enumerate(keys):
            if i < yvals.shape[0]:
                values[k] = np.asarray(yvals[i, :], dtype=float)

        record = {
            'dvar':      dvar,
            'values':    values,
            'contours':  contours,
            'axis':      axis,
            'dump_file': str(dump_file),
            'eq':        int(eq),
        }
        self._fsa_cache[cache_key] = record
        return record

    def _enclosed_current(self, record):
        """
        Toroidal plasma current enclosed by each FSA surface, from Ampere's law

            I(psi) = (1/mu0) * closed_integral( B_pol . dl )

        evaluated along the FSA contours with the same field (eq) as the
        averages.  Only the magnetic field is needed (no dump_ja).

        Each contour is oriented counter-clockwise in (R, Z), which in NIMROD's
        right-handed (R, Z, phi) system gives the current along +phi_NIMROD;
        the result is converted to the edges-ml / M3D-C1 convention with
        PHI_SIGN_FLIP.  Returns an array [nsurf] (NaN where a surface could not
        be integrated), sorted like record['dvar'].
        """
        if 'Ip_enclosed' in record:
            return record['Ip_enclosed']

        dvar = record['dvar']
        nsurf = dvar.shape[1]
        ip = np.full(nsurf, np.nan)
        record['Ip_enclosed'] = ip

        contours = np.asarray(record.get('contours'))
        dump_file = record.get('dump_file')
        if (dump_file is None or contours.ndim != 3 or contours.shape[0] < 2
                or contours.shape[-1] != nsurf):
            self._warn_field_once(
                "ip::contours",
                "NIMROD: FSA contours are not available in the expected layout; "
                "the plasma current Ip cannot be computed."
            )
            return ip

        ev = self._get_eval(dump_file)
        if ev is None:
            return ip

        segments = []          # (surface index, n_segments, orientation, dR, dZ)
        mid_R, mid_Z = [], []
        for i in range(nsurf):
            r = np.asarray(contours[0, :, i], dtype=float)
            z = np.asarray(contours[1, :, i], dtype=float)
            good = np.isfinite(r) & np.isfinite(z)
            r, z = r[good], z[good]
            if r.size < 3:
                continue
            r1 = np.append(r, r[0])
            z1 = np.append(z, z[0])
            area = 0.5 * np.sum(r1[:-1] * z1[1:] - r1[1:] * z1[:-1])
            if not np.isfinite(area) or area == 0.0:
                continue
            orient = 1.0 if area > 0 else -1.0
            dR = np.diff(r1)
            dZ = np.diff(z1)
            mid_R.append(0.5 * (r1[:-1] + r1[1:]))
            mid_Z.append(0.5 * (z1[:-1] + z1[1:]))
            segments.append((i, dR.size, orient, dR, dZ))

        if not segments:
            return ip

        R = np.concatenate(mid_R)
        Z = np.concatenate(mid_Z)
        try:
            self._stage("B on FSA contours (Ampere's law)")
            b = np.atleast_2d(np.asarray(self._eval_raw(
                ev, dump_file, 'b', R, Z, np.zeros_like(R),
                eq=record.get('eq', 1), use_grid=False)))
        except Exception as exc:
            printwarn(f"NIMROD: evaluating B on the FSA contours failed "
                      f"({brief_error(exc)}); Ip is not available.")
            return ip
        if b.shape[0] < 2 or b.shape[-1] != R.size:
            return ip

        start = 0
        for i, nseg, orient, dR, dZ in segments:
            br = b[0, start:start + nseg]
            bz = b[1, start:start + nseg]
            start += nseg
            circ = float(np.sum(br * dR + bz * dZ))
            if np.isfinite(circ):
                ip[i] = self.PHI_SIGN_FLIP * orient * circ / self.MU0

        record['Ip_enclosed'] = ip
        return ip

    def _enclosed_current_profile(self, record, resolution, fcoords):
        """Enclosed current I(psi) interpolated onto the 1D output grid."""
        grid, _ = self._fsa_grid(record, resolution, fcoords)
        if grid.size == 0:
            return np.array([])
        row = 1 if 'rho' in str(fcoords).lower() else 0
        x = np.asarray(record['dvar'][row, :], dtype=float)
        y = self._enclosed_current(record)
        good = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(good) < 2:
            return np.array([])
        x, y = x[good], y[good]
        # No current is enclosed at the magnetic axis.
        if x[0] > 0.0:
            x = np.insert(x, 0, 0.0)
            y = np.insert(y, 0, 0.0)
        return np.interp(grid, x, y)

    def _fsa_grid(self, record, resolution, fcoords='pest'):
        """Returns the 1D radial coordinate grid used for all NIMROD 1D output."""
        dvar = record['dvar']
        row = 1 if 'rho' in str(fcoords).lower() else 0
        src = np.asarray(dvar[row, :], dtype=float)
        good = np.isfinite(src)
        if np.count_nonzero(good) < 2:
            return np.array([]), np.array([])
        src = src[good]
        return np.linspace(float(src[0]), float(src[-1]), int(resolution)), src

    def _interp_to_grid(self, record, values, resolution, fcoords='pest'):
        grid, src = self._fsa_grid(record, resolution, fcoords)
        if grid.size == 0:
            return np.array([])
        vals = np.asarray(values, dtype=float)
        n = min(vals.size, src.size)
        if n < 2:
            return np.array([])
        return np.interp(grid, src[:n], vals[:n])

    def get_1d_mesh(self, resolution, units, fcoords):
        record = self._get_fsa_record(self._get_dump_file(-1), self._fsa_nsurf(resolution), eq=1)
        if record is None:
            return np.array([])
        grid, _ = self._fsa_grid(record, resolution, fcoords)
        return grid

    def get_flux_average(self, name, resolution, units, fcoords, time=-1, use_eq_fs=True):
        """
        Flux surface average of the requested quantity, computed with nimpy.fsa.

        NIMROD data are already in SI units (temperatures in eV); the `units`
        argument therefore only affects the labelling performed by the writer.
        `use_eq_fs` is ignored: NIMROD surfaces always follow the axisymmetric
        (n = 0) field of the requested dump.
        """
        dump_file = self._get_dump_file(time)
        if dump_file is None:
            return np.array([]), np.array([])

        eq = 1 if time == -1 else 3
        record = self._get_fsa_record(dump_file, self._fsa_nsurf(resolution), eq=eq)
        if record is None:
            return np.array([]), np.array([])

        return self._flux_average_from_record(record, name, resolution, fcoords)

    def _flux_average_from_record(self, record, name, resolution, fcoords):
        """
        Extracts flux average *name* from an FSA record.
        Returns (grid, values) with values an array or a {component: array} dict.
        """
        grid, src = self._fsa_grid(record, resolution, fcoords)
        if grid.size == 0:
            return np.array([]), np.array([])

        dvar = record['dvar']
        vals = record['values']

        def interp(arr):
            return self._interp_to_grid(record, arr, resolution, fcoords)

        low = str(name).lower()

        # --- quantities that come straight out of dvar -----------------
        if low == 'q':
            return grid, interp(dvar[7, :])
        if low == 'rho':
            return grid, interp(dvar[1, :])
        if low == 'flux_p':
            return grid, interp(2.0 * np.pi * dvar[2, :])
        if low == 'flux_t':
            return grid, interp(dvar[3, :])

        # --- radially integrated quantities ----------------------------
        # Careful: 'v' is the velocity field, 'V' is the volume.
        if name == 'V':
            vol = 2.0 * np.pi * _trapz_profile(dvar[6, :], dvar[2, :])
            return grid, interp(vol)

        # Enclosed plasma current from Ampere's law (B only).
        if name == 'Ip' or low == 'ip':
            prof = self._enclosed_current_profile(record, resolution, fcoords)
            if prof.size == 0:
                return grid, np.array([])
            return grid, prof

        if low == 'f':
            if 'f' not in vals:
                return grid, np.array([])
            return grid, self.PHI_SIGN_FLIP * interp(vals['f'])

        if low == 'ffprime':
            if 'f' not in vals:
                return grid, np.array([])
            f = self.PHI_SIGN_FLIP * np.asarray(vals['f'], dtype=float)
            psi = np.asarray(dvar[2, :], dtype=float)
            n = min(f.size, psi.size)
            if n < 3:
                return grid, np.array([])
            dfdpsi = np.gradient(f[:n], psi[:n])
            return grid, interp(f[:n] * dfdpsi)

        # --- vector quantities -----------------------------------------
        if name in ('B', 'v', 'j'):
            comps = {}
            for suffix in ('R', 'Z', 'phi'):
                key = f"{name}_{suffix}"
                if key not in vals:
                    continue
                arr = interp(vals[key])
                if suffix == 'phi':
                    arr = self.PHI_SIGN_FLIP * arr
                comps[f"{name}_{suffix}"] = arr
            if not comps:
                return grid, np.array([])
            return grid, comps

        # --- derived scalars -------------------------------------------
        if low == 'pi':
            if 'p' in vals and 'pe' in vals:
                return grid, interp(np.asarray(vals['p']) - np.asarray(vals['pe']))
            return grid, np.array([])

        # --- plain scalars ----------------------------------------------
        if low in vals:
            return grid, interp(vals[low])
        if name in vals:
            return grid, interp(vals[name])

        self._warn_field_once(
            f"fsa::{name}",
            f"Flux average '{name}' is not available for NIMROD; skipping."
        )
        return grid, np.array([])

    def get_mode_flux_average(self, mode_dir, name, resolution, units, fcoords):
        """
        Flux average of the perturbation of *name* for a mode directory:
            <field>(last dump, eq + n=0 perturbation) - <field>(equilibrium)
        interpolated onto the equilibrium 1D grid.

        The flux-surface average of an n>0 component vanishes, so this is the
        n=0 profile modification of the run; it is shared by all toroidal-mode
        entries of the directory.
        """
        nsurf = self._fsa_nsurf(resolution)
        eq_dump = self._get_dump_file(-1)
        mode_dump = self._last_dump_in_dir(mode_dir)
        if eq_dump is None or mode_dump is None:
            return np.array([])

        rec_eq = self._get_fsa_record(eq_dump, nsurf, eq=1)
        rec_md = self._get_fsa_record(mode_dump, nsurf, eq=3)
        if rec_eq is None or rec_md is None:
            return np.array([])

        g_eq, v_eq = self._flux_average_from_record(rec_eq, name, resolution, fcoords)
        g_md, v_md = self._flux_average_from_record(rec_md, name, resolution, fcoords)
        if np.size(g_eq) == 0 or np.size(g_md) == 0:
            return np.array([])

        def diff(a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            if a.size < 2 or a.size != np.size(g_md) or b.size != np.size(g_eq):
                return np.array([])
            return np.interp(g_eq, g_md, a) - b

        if isinstance(v_md, dict) and isinstance(v_eq, dict):
            return {k: diff(v_md[k], v_eq[k]) for k in v_md if k in v_eq}
        if isinstance(v_md, dict) or isinstance(v_eq, dict):
            return np.array([])
        return diff(v_md, v_eq)

    # ======================================================================
    # GLOBAL PARAMETERS (mirrors nimpy.globaleq without its yaml side effects)
    # ======================================================================

    def get_global_parameters(self):
        if self._global_params_cache is not None:
            return self._global_params_cache

        out = {}
        cfg = self._config or {}
        if not cfg.get("nimrod_compute_globals", True):
            self._global_params_cache = out
            return out

        dump_file = self._get_dump_file(-1)
        resolution = int(cfg.get("resolutions", {}).get("1d", 200))
        record = self._get_fsa_record(dump_file, self._fsa_nsurf(resolution), eq=1)
        if record is None:
            self._global_params_cache = out
            return out

        try:
            dvar = record['dvar']
            vals = record['values']
            contours = record['contours']
            axis = np.asarray(record['axis'], dtype=float)

            psi = np.asarray(dvar[2, :], dtype=float)
            vprime = np.asarray(dvar[6, :], dtype=float)
            q = np.asarray(dvar[7, :], dtype=float)
            psin = np.asarray(dvar[0, :], dtype=float)

            out['R_axis'] = _finite(axis[0])
            out['Z_axis'] = _finite(axis[1])

            # --- LCFS shaping (outermost surface; contours sorted by psi_N) --
            if contours.size and contours.ndim == 3:
                lcfs = np.asarray(contours[:, :, -1], dtype=float)
                r_l = lcfs[0, :]
                z_l = lcfs[1, :]
                good = np.isfinite(r_l) & np.isfinite(z_l)
                r_l = r_l[good]
                z_l = z_l[good]
                if r_l.size > 3:
                    rmax_l, rmin_l = float(np.max(r_l)), float(np.min(r_l))
                    ztop, zbot = float(np.max(z_l)), float(np.min(z_l))
                    rgeo = 0.5 * (rmax_l + rmin_l)
                    a = 0.5 * (rmax_l - rmin_l)
                    out['R_geo'] = rgeo
                    out['a_minor'] = a
                    out['aspect_ratio'] = rgeo / a if a > 0 else np.nan
                    out['kappa'] = (ztop - zbot) / (2.0 * a) if a > 0 else np.nan
                    r_top = float(r_l[int(np.argmax(z_l))])
                    r_bot = float(r_l[int(np.argmin(z_l))])
                    out['delta_upper'] = (rgeo - r_top) / a if a > 0 else np.nan
                    out['delta_lower'] = (rgeo - r_bot) / a if a > 0 else np.nan
                    out['delta'] = 0.5 * (out['delta_upper'] + out['delta_lower'])

            a_minor = out.get('a_minor', np.nan)

            # --- safety factor ------------------------------------------
            if q.size:
                out['q0'] = _finite(q[0])
                out['qa'] = _finite(q[-1])
                out['q_min'] = _finite(np.nanmin(q))
                out['q_max'] = _finite(np.nanmax(q))
                if psin.size == q.size and psin.size > 1:
                    out['q95'] = _finite(np.interp(0.95, psin, q))

            # --- fluxes ---------------------------------------------------
            if psi.size:
                out['psi_axis'] = _finite(2.0 * np.pi * psi[0])
                out['psi_lcfs'] = _finite(2.0 * np.pi * psi[-1])
                out['poloidal_flux'] = _finite(2.0 * np.pi * (psi[-1] - psi[0]))
            if dvar.shape[0] > 3:
                out['toroidal_flux'] = _finite(dvar[3, -1])

            # --- volume ---------------------------------------------------
            vol_prof = 2.0 * np.pi * _trapz_profile(vprime, psi)
            volume = float(vol_prof[-1]) if vol_prof.size else np.nan
            out['volume'] = _finite(volume)

            # --- plasma current (Ampere's law on the outermost surface) ---
            ip_prof = self._enclosed_current(record)
            ip_fin = ip_prof[np.isfinite(ip_prof)]
            ip = float(ip_fin[-1]) if ip_fin.size else np.nan
            out['Ip'] = _finite(ip)

            # --- on-axis values -------------------------------------------
            ev = self._get_eval(dump_file)
            if ev is not None:
                rzo = np.array([axis[0], axis[1], 0.0])

                def _axis_scalar(field, idx=0):
                    if not self._field_ok(dump_file, field, 1):
                        return np.nan
                    try:
                        with _pushd(Path(dump_file).resolve().parent):
                            v = ev.eval_field(field, rzo, dmode=0, eq=1)
                        return float(np.atleast_1d(v)[idx])
                    except Exception:
                        return np.nan

                b0 = np.nan
                if self._field_ok(dump_file, 'b', 1):
                    try:
                        with _pushd(Path(dump_file).resolve().parent):
                            bvec = ev.eval_field('b', rzo, dmode=0, eq=1)
                        b0 = float(bvec[2])
                    except Exception:
                        pass

                out['B0'] = _finite(self.PHI_SIGN_FLIP * b0)
                out['p0'] = _finite(_axis_scalar('p'))
                out['ne0'] = _finite(_axis_scalar('n', 0))
                out['ni0'] = _finite(_axis_scalar('n', self._ion_index(ev)))
                out['Te0'] = _finite(_axis_scalar('te'))
                out['Ti0'] = _finite(_axis_scalar('ti'))

            b0_abs = abs(_finite(out.get('B0', np.nan), 0.0))

            # --- averaged pressure / betas --------------------------------
            if 'p' in vals and vprime.size:
                p_raw = np.asarray(vals['p']) * vprime
                p_int = _trapz_profile(p_raw, psi)
                v_int = _trapz_profile(vprime, psi)
                if v_int.size and v_int[-1] != 0:
                    pave = float(p_int[-1] / v_int[-1])
                    out['p_average'] = _finite(pave)
                    if b0_abs > 0:
                        betat = 2.0 * self.MU0 * pave / (b0_abs ** 2)
                        out['beta_t'] = _finite(betat)
                        if np.isfinite(a_minor) and np.isfinite(ip) and ip != 0:
                            out['beta_n'] = _finite(1.0e8 * a_minor * b0_abs * betat / abs(ip))

                    # poloidal beta / internal inductance
                    if 'Bp' in vals and 'Bp2' in vals:
                        bp_a = float(np.asarray(vals['Bp'])[-1])
                        bp2_a = float(np.asarray(vals['Bp2'])[-1])
                        if bp_a != 0:
                            bp0 = bp2_a / bp_a
                            if bp0 != 0:
                                out['beta_p'] = _finite(2.0 * self.MU0 * pave / (bp0 ** 2))
                                bp2_raw = np.asarray(vals['Bp2']) * vprime
                                bp2_int = _trapz_profile(bp2_raw, psi)
                                if v_int[-1] != 0:
                                    out['li'] = _finite((bp2_int[-1] / v_int[-1]) / (bp0 ** 2))

        except Exception as exc:
            printwarn(f"Could not compute NIMROD global parameters: {type(exc).__name__}: {exc}")

        self._global_params_cache = out
        return out

    # ======================================================================
    # TIME TRACES (nimpy.read_bin)
    # ======================================================================

    def _find_bin_file(self, candidates):
        dirs = list(self._mode_dirs) + [self.model_dir]
        for d in dirs:
            for nm in candidates:
                p = Path(d) / nm
                if p.is_file():
                    return p
        return None

    def _read_nimrod_bin(self, filepath):
        """
        Reads a NIMROD binary/text diagnostic file with nimpy's readBin.
        Returns (var_names, data[nvar, nrps, nsteps]) or (None, None).
        """
        if not _NIMPY_READBIN_AVAILABLE or filepath is None:
            return None, None

        key = str(Path(filepath).resolve())
        if key in self._bin_cache:
            return self._bin_cache[key]

        try:
            with _pushd(Path(filepath).parent):
                rb = readBin()
                ftype = rb.get_file_type(Path(filepath).name)
                var_names, depvar, plot_list, mode_file = rb.data_dict[ftype]
                nvar = len(var_names)
                nrps = rb.nimrodin['nmodes'] if mode_file else 1
                data = rb.read_file(Path(filepath).name, True, nrps, nvar)
            result = (list(var_names), np.asarray(data))
        except Exception as exc:
            printwarn(f"Could not read NIMROD diagnostic file {filepath}: {type(exc).__name__}: {exc}")
            result = (None, None)

        self._bin_cache[key] = result
        return result

    def get_time_trace(self, name, units):
        """
        Extracts a scalar time trace from NIMROD's discharge.bin / energy.bin.

        Only quantities with an unambiguous counterpart in the shared registry
        are mapped; everything else returns empty arrays and is skipped by the
        writer.
        """
        if not _NIMPY_READBIN_AVAILABLE:
            return np.array([]), np.array([])

        discharge = self._find_bin_file(("discharge.bin", "discharge.txt"))
        energy    = self._find_bin_file(("energy.bin", "energy.txt", "logen.bin"))

        # --- discharge.bin based traces --------------------------------
        if name in self._TIME_TRACE_MAP:
            kind, var = self._TIME_TRACE_MAP[name]
            if kind == 'discharge' and discharge is not None:
                var_names, data = self._read_nimrod_bin(discharge)
                if var_names and var in var_names:
                    t = np.asarray(data[var_names.index('t'), 0, :], dtype=float)
                    v = np.asarray(data[var_names.index(var), 0, :], dtype=float)
                    return t, v

        # --- energy.bin (summed over toroidal modes) ---------------------
        if name in self._TIME_TRACE_ENERGY_SUM and energy is not None:
            var = self._TIME_TRACE_ENERGY_SUM[name]
            var_names, data = self._read_nimrod_bin(energy)
            if var_names and var in var_names:
                t = np.asarray(data[var_names.index('t'), 0, :], dtype=float)
                v = np.asarray(data[var_names.index(var), :, :], dtype=float).sum(axis=0)
                return t, v

        # --- fall back for the bare time axis ----------------------------
        if name == 'time' and energy is not None:
            var_names, data = self._read_nimrod_bin(energy)
            if var_names and 't' in var_names:
                t = np.asarray(data[var_names.index('t'), 0, :], dtype=float)
                return t, t

        self._warn_field_once(
            f"trace::{name}",
            f"Time trace '{name}' has no NIMROD counterpart; skipping."
        )
        return np.array([]), np.array([])

    # ======================================================================
    # METADATA / INPUTS / REPRODUCIBILITY
    # ======================================================================

    def _parse_nimrod_version(self, mode_dirs):
        """Best-effort extraction of the NIMROD release version and build date."""
        release_version = "Unknown"
        build_date = "Unknown"

        # 1) HDF5 dump attributes
        dump_file = self._get_dump_file(-1)
        if dump_file:
            try:
                with h5py.File(dump_file, 'r') as h5:
                    for key, val in h5.attrs.items():
                        k = str(key).lower()
                        try:
                            sval = val.decode() if isinstance(val, bytes) else str(np.ravel(val)[0])
                        except Exception:
                            sval = str(val)
                        if 'version' in k and release_version == "Unknown":
                            release_version = sval.strip()
                        if ('date' in k or 'build' in k) and build_date == "Unknown":
                            build_date = sval.strip()
            except Exception:
                pass

        # 2) stdout / log files
        candidates = []
        for d in list(mode_dirs or []) + [self.model_dir]:
            d = Path(d)
            for pattern in ("slurm*.out", "*.out", "nimrod.log", "*.log"):
                candidates.extend(sorted(d.glob(pattern)))
        candidates = list(dict.fromkeys(candidates))
        candidates.sort(key=lambda f: f.stat().st_mtime if f.exists() else 0, reverse=True)

        rx_ver = re.compile(r'(?:NIMROD|RELEASE|CODE)\s*(?:VERSION|RELEASE)?\s*[:=]\s*(\S.*)', re.IGNORECASE)
        rx_bld = re.compile(r'BUILD\s*DATE\s*[:=]\s*(\S.*)', re.IGNORECASE)

        for f in candidates[:20]:
            if release_version != "Unknown" and build_date != "Unknown":
                break
            try:
                with open(f, 'r', errors='ignore') as fh:
                    for i, line in enumerate(fh):
                        if i > 500:
                            break
                        s = line.strip()
                        if release_version == "Unknown":
                            m = rx_ver.search(s)
                            if m:
                                release_version = m.group(1).strip()
                        if build_date == "Unknown":
                            m = rx_bld.search(s)
                            if m:
                                build_date = m.group(1).strip()
            except Exception:
                continue

        return release_version, build_date

    def get_metadata(self, mode_dirs):
        if self._metadata_cache is not None:
            return self._metadata_cache

        rel, bld = self._parse_nimrod_version(mode_dirs)
        self._metadata_cache = {
            "code": self.CODE_NAME,
            "source_directory": str(self.model_dir.resolve()),
            "release_version": rel,
            "build_date": bld,
        }
        return self._metadata_cache

    def get_reproducibility_data(self, mode_dir):
        hashes = {}

        target_files = ['nimrod.in', 'nimeq.in', 'oculus.in', 'fluxgrid.in',
                        'nimhist.bin', 'energy.bin', 'energy.txt', 'discharge.bin',
                        'growth.bin', 'growth.txt', 'timestat.bin',
                        'contours.h5', 'sol.grn']
        for fname in target_files:
            fpath = mode_dir / fname
            if fpath.exists():
                hashes[fname] = compute_file_hash(fpath)

        for h5_file in sorted(mode_dir.glob("dumpgll.*.h5")):
            hashes[h5_file.name] = compute_file_hash(h5_file)

        eq_dump = self._get_dump_file(time=-1)
        if eq_dump:
            eq_path = Path(eq_dump)
            if eq_path.name not in hashes:
                hashes[eq_path.name] = compute_file_hash(eq_path)

        return hashes

    def get_all_input_parameters(self):
        inputs = {}
        target_files = ['nimrod.in', 'nimeq.in', 'oculus.in', 'fluxgrid.in']

        for filename in target_files:
            found_files = list(self.model_dir.rglob(filename))
            if not found_files:
                continue

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

    # ======================================================================
    # MODE METADATA
    # ======================================================================

    def get_mode_metadata(self, mode_dir):
        """
        Extracts the growth rate for the toroidal mode number corresponding to this
        mode directory by running 'nimgrowth energy.bin'.
        Used by the legacy fallback path of extract_mode_entries().
        """
        meta = {
            "growth_rate": 0.0,
            "frequency":   0.0,
            "mode_type":   -100
        }

        primary_n = None
        try:
            primary_n = int(mode_dir.name.lstrip('n'))
        except ValueError:
            printwarn(f"Could not parse toroidal mode number from directory name '{mode_dir.name}'.")

        growth_rates = self._get_nimgrowth_for_dir(mode_dir)

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
        Expands a single NIMROD mode directory into one perturbation group per
        toroidal mode number found in the nimgrowth output.
        Used by the legacy fallback path of extract_mode_entries().
        """
        growth_rates = self._get_nimgrowth_for_dir(mode_dir)

        if not growth_rates:
            return [(mode_dir.name, {
                "growth_rate": 0.0,
                "frequency":   0.0,
                "mode_type":   -100
            })]

        entries = []
        for n_val in sorted(growth_rates.keys()):
            group_name = f"n{n_val:02d}"
            mode_meta  = {
                "growth_rate": growth_rates[n_val],
                "frequency":   0.0,
                "mode_type":   -100
            }
            entries.append((group_name, mode_meta))

        return entries

    def get_time_metadata(self, time_slices):
        """
        Builds the time metadata dict for the requested time slice indices.
        """
        resolved_indices = []
        for ts in time_slices:
            requested_idx = 0 if ts == -1 else ts
            resolved_idx  = self._get_common_max_index(requested_idx)
            resolved_indices.append(resolved_idx)

        meta = {
            "time_slice":           np.array(resolved_indices, dtype=int),
            "simulation_time_step": np.zeros(len(time_slices), dtype=int),
            "simulation_time":      np.zeros(len(time_slices), dtype=float)
        }

        ref_dir = self._mode_dirs[0] if self._mode_dirs else None

        if ref_dir is not None:
            indexed   = self._get_indexed_dumps_for_dir(ref_dir)
            index_map = {idx: (step, path) for idx, step, path in indexed}
            for i, idx in enumerate(resolved_indices):
                if idx in index_map:
                    step, path = index_map[idx]
                    meta["simulation_time_step"][i] = step
                    meta["simulation_time"][i] = self._read_dump_time(path)

        return meta

    @staticmethod
    def _read_dump_time(dump_path):
        """Reads the physical time stored in a NIMROD HDF5 dump (dumpTime/vsTime)."""
        try:
            with h5py.File(str(dump_path), 'r') as h5:
                if 'dumpTime' in h5:
                    g = h5['dumpTime']
                    for key in ('vsTime', 'time', 't'):
                        if key in g.attrs:
                            return float(np.ravel(np.asarray(g.attrs[key], dtype=float))[0])
                if 'time' in h5:
                    return float(np.ravel(np.asarray(h5['time'][()], dtype=float))[0])
        except Exception:
            pass
        return 0.0

    # ======================================================================
    # IMAS CONVERSION (nimrod2imas)
    # ======================================================================

    def _imas_search_dirs(self, mode_dirs, parent_levels=1):
        """
        Builds the ordered list of directories searched for GEQDSK/PEQDSK files
        and NIMROD namelists.
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
        from .utils import _is_geqdsk_name, _is_peqdsk_name

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

        Returns a dictionary summarising the created entry, or None on failure.
        """
        self._config = config or {}
        imas_cfg = dict(config.get("imas", {}) or {})

        rt = _prepare_nimrod2imas_runtime(imas_cfg)
        if rt is None:
            return None

        have_input2imas = _nimrod2imas_tool_available(rt, "input2imas")
        have_dump2imas  = _nimrod2imas_tool_available(rt, "dump2imas")
        have_gamma2imas = _nimrod2imas_tool_available(rt, "gamma2imas")

        if not have_dump2imas and not have_input2imas:
            printerr("Neither dump2imas nor input2imas can be executed; aborting NIMROD IMAS conversion.")
            return None

        dd_version = str(
            imas_cfg.get("dd_version")
            or os.environ.get("IMAS_VERSION")
            or "4.1.1"
        )
        backend = str(imas_cfg.get("backend", "hdf5"))
        dbpath  = Path(imas_cfg.get("dbpath") or output_directory)
        dbpath.mkdir(parents=True, exist_ok=True)
        dbpath = dbpath.resolve()

        machine, shot = _infer_machine_and_shot(self.model_dir)
        dd    = str(imas_cfg.get("dd") or machine or "nimrod")
        pulse = int(imas_cfg.get("pulse") or shot or 1)

        if run is None:
            run = imas_cfg.get("run", None)
        run = _allocate_imas_run(dbpath, dd, dd_version, pulse, run)

        entry = _imas_entry_dir(dbpath, dd, dd_version, pulse, run)
        entry.mkdir(parents=True, exist_ok=True)

        printnote(
            f"NIMROD -> IMAS: {self.model_dir}\n"
            f"                dd={dd} pulse={pulse} run={run} dd_version={dd_version}\n"
            f"                entry={entry}"
        )

        self._ensure_all_h5_dumps()

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

        # --- 1) equilibrium + kinetic profiles (input2imas) ---
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

        # --- 2/3) per-mode dumps + growth rates ---
        occ_base_start = int(imas_cfg.get("occ_base", 1))
        occ_stride     = int(imas_cfg.get("occ_stride", 10))

        selection = imas_cfg.get("dump_selection", None)
        if selection is None:
            selection = config.get("time_slices", None)
        if selection is None or (isinstance(selection, (list, tuple)) and len(selection) == 0):
            selection = "all"

        gamma_endian = str(imas_cfg.get("gamma_endian", ">"))
        gamma_nsteps = int(imas_cfg.get("gamma_nsteps", 50))
        use_history  = bool(imas_cfg.get("gamma_use_history", True))

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
                    "tool":           "dump2imas",
                    "mode_directory": str(mode_dir),
                    "occurrence":     occ,
                    "n_dumps":        len(dumps),
                    "dumps":          [Path(d).name for d in dumps],
                    "ok":             bool(ok),
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
            if backend.lower() == "hdf5":
                gargv += ["--backend", backend]
            gargv += _options_dict_to_cli(imas_cfg.get("gamma2imas_options"))
            gargv += [str(a) for a in imas_cfg.get("gamma2imas_args", [])]
            gargv.append(str(energy_file.resolve()))

            history = mode_dir / "nimhist.bin"
            printnote(f"  gamma2imas  [{mode_dir.name}] occ={occ} ({energy_file.name})")

            ok = False
            if use_history and history.is_file():
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
                "tool":           "gamma2imas",
                "mode_directory": str(mode_dir),
                "occurrence":     occ,
                "energy_file":    str(energy_file),
                "ok":             bool(ok),
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
