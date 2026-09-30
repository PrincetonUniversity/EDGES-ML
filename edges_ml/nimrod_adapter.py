"""
NIMROD adapter for edges-ml.

Contains the NIMRODAdapter class, which implements the SimulationAdapter
interface for NIMROD HDF5 dump files.

External NIMROD tooling used here:
    nimpy.eval_nimrod.EvalNimrod / EvalGrid   -> point-wise field evaluation
    nimpy.fsa.FSA / find_pf_null / FSAInput   -> flux surface averages, q, fluxes
    nimpy.read_bin.readBin                    -> discharge.bin / energy.bin traces
    nimrod2imas (dump2imas/input2imas/gamma2imas) -> IMAS conversion
"""

import os
import re
import sys
import shutil
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
from .utils import (
    printwarn, printerr, printnote,
    compute_file_hash, auto_cast,
    expand_request,
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

    # Ordered list of nimpy 'fieldlist' strings to attempt when initialising
    # EvalNimrod.  The first one that initialises successfully is kept.
    _FIELDLIST_CANDIDATES = ('nvptbje', 'nvptbj', 'nvptb', 'nvpt', 'nvp', 'n')

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

    def _get_eval(self, dump_file, fieldlist=None):
        """
        Returns a cached EvalNimrod instance for the given dump file.

        The nimpy shared library can only keep one dump file open at a time, so
        a single active handle is tracked and re-initialised on demand.
        """
        if not _NIMPY_EVAL_AVAILABLE or dump_file is None:
            return None

        dump_path = Path(dump_file).resolve()
        requested = fieldlist or self._fieldlist

        if requested is not None:
            key = (str(dump_path), requested)
            if self._active_eval_key == key and self._active_eval is not None:
                return self._active_eval

        if requested is not None:
            candidates = [requested] + [f for f in self._FIELDLIST_CANDIDATES if f != requested]
        else:
            candidates = list(self._FIELDLIST_CANDIDATES)

        errors = []
        for fl in candidates:
            try:
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

    def _eval_raw(self, ev, dump_file, nim_field, R, Z, phi, eq=2):
        """Evaluates a nimpy field at the supplied points; returns (nqty, N) array."""
        rzp = np.array([np.asarray(R, dtype=float).ravel(),
                        np.asarray(Z, dtype=float).ravel(),
                        np.asarray(phi, dtype=float).ravel()])

        grid = self._get_eval_grid(ev, dump_file, rzp, nim_field)

        with _pushd(Path(dump_file).resolve().parent):
            if grid is not None:
                try:
                    return ev.eval_field(nim_field, grid, dmode=0, eq=eq)
                except Exception:
                    pass
            return ev.eval_field(nim_field, rzp, dmode=0, eq=eq)

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
                if idx == 'electron':
                    k = 0
                elif idx == 'ion':
                    k = self._ion_index(ev)
                else:
                    k = int(idx)
                k = min(k, res.shape[0] - 1)
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

        if dump_file in self._native_points_cache:
            return self._native_points_cache[dump_file]

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

        self._native_points_cache[dump_file] = pts
        return pts

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
        """Evaluates the field at the final dump of the given mode directory."""
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
        """Evaluates the field at the final dump of the given mode directory."""
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

    # edges-ml flux average name -> integrand keys that must be computed
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
        'Ip':   ('jphi_over_R',),
        'V':    (),
        'q':    (),
        'rho':  (),
        'flux_p': (),
        'flux_t': (),
    }

    _GLOBAL_FSA_KEYS = ('p', 'Bp', 'Bp2', 'jphi_over_R', 'f')

    def _requested_fsa_keys(self):
        """Integrand keys required by the current configuration."""
        cfg = self._config or {}
        names = set()
        for opt in ("flux_averages", "total_flux_averages"):
            sel = cfg.get(opt, None)
            names.update(expand_request(sel, AVAILABLE_FLUX_AVERAGES) or [])
        if not names:
            names = set(AVAILABLE_FLUX_AVERAGES)

        keys = set()
        for n in names:
            keys.update(self._FSA_NAME_TO_KEYS.get(n, ()))

        if cfg.get("nimrod_compute_globals", True):
            keys.update(self._GLOBAL_FSA_KEYS)

        # q_profile in the mode section also needs FSA (dvar only).
        return tuple(sorted(keys))

    def _fsa_nsurf(self, resolution):
        cfg = self._config or {}
        return int(cfg.get("nimrod_fsa_nsurf", min(int(resolution), 60)))

    def _find_magnetic_axis(self, ev, dump_file, time=-1):
        """Coarse search for |Bpol| minimum, refined with nimpy's find_pf_null."""
        key = str(Path(dump_file).resolve())
        if key in self._axis_cache:
            return self._axis_cache[key]

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

        Returns a dict with 'dvar', 'contours', 'axis' and 'values' (key -> array),
        or None if the FSA could not be performed.
        """
        if dump_file is None or not (_NIMPY_FSA_AVAILABLE and _NIMPY_EVAL_AVAILABLE):
            return None
        if not (self._config or {}).get("nimrod_enable_fsa", True):
            return None

        keys = self._requested_fsa_keys()
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
                  f"({nsurf} surfaces, {len(keys)} quantities).")

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

        dvar = np.asarray(dvar)
        yvals = np.asarray(yvals)
        if dvar.size == 0:
            self._fsa_failed.add(cache_key)
            return None

        order = np.argsort(dvar[0, :])
        dvar = dvar[:, order]
        yvals = yvals[:, order] if yvals.size else yvals

        values = {}
        for i, k in enumerate(keys):
            if i < yvals.shape[0]:
                values[k] = np.asarray(yvals[i, :], dtype=float)

        record = {
            'dvar':     dvar,
            'values':   values,
            'contours': np.asarray(contours),
            'axis':     axis,
        }
        self._fsa_cache[cache_key] = record
        return record

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
        if low == 'v':
            # Careful: 'v' is the velocity field, 'V' is the volume.
            if name == 'V':
                vol = 2.0 * np.pi * _trapz_profile(dvar[6, :], dvar[2, :])
                return grid, interp(vol)
        if name == 'V':
            vol = 2.0 * np.pi * _trapz_profile(dvar[6, :], dvar[2, :])
            return grid, interp(vol)

        if name == 'Ip' or low == 'ip':
            if 'jphi_over_R' not in vals:
                return grid, np.array([])
            raw = np.asarray(vals['jphi_over_R']) * np.asarray(dvar[6, :])
            cur = self.PHI_SIGN_FLIP * _trapz_profile(raw, dvar[2, :])
            return grid, interp(cur)

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

    def get_mode_1d_profile(self, mode_dir, name, resolution, units):
        """1D radial profiles for a mode directory (currently the q profile)."""
        if str(name).lower() not in ('q_profile', 'q'):
            self._warn_field_once(
                f"mode1d::{name}",
                f"1D profile '{name}' is not available for NIMROD; skipping."
            )
            return np.array([])

        dump_file = self._last_dump_in_dir(mode_dir)
        if dump_file is None:
            return np.array([])

        record = self._get_fsa_record(dump_file, self._fsa_nsurf(resolution), eq=3)
        if record is None:
            return np.array([])

        fcoords = (self._config or {}).get("fcoords", "pest")
        return self._interp_to_grid(record, record['dvar'][7, :], resolution, fcoords)

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

            # --- LCFS shaping ------------------------------------------
            if contours.size:
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

            # --- plasma current -------------------------------------------
            ip = np.nan
            if 'jphi_over_R' in vals:
                raw = np.asarray(vals['jphi_over_R']) * vprime
                ip_prof = self.PHI_SIGN_FLIP * _trapz_profile(raw, psi)
                ip = float(ip_prof[-1]) if ip_prof.size else np.nan
            out['Ip'] = _finite(ip)

            # --- on-axis values -------------------------------------------
            ev = self._get_eval(dump_file)
            if ev is not None:
                rzo = np.array([axis[0], axis[1], 0.0])
                with _pushd(Path(dump_file).resolve().parent):
                    def _axis_scalar(field, idx=0):
                        try:
                            v = ev.eval_field(field, rzo, dmode=0, eq=1)
                            return float(np.atleast_1d(v)[idx])
                        except Exception:
                            return np.nan

                    b0 = np.nan
                    try:
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
        Overrides the base class method to expand a single NIMROD mode directory
        into one perturbation group per toroidal mode number found in the nimgrowth output.
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
