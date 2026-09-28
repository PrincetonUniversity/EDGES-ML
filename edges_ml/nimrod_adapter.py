"""
NIMROD adapter for edges-ml.

Contains the NIMRODAdapter class, which implements the SimulationAdapter
interface for NIMROD HDF5 dump files.
"""

import os
import re
import sys
import shlex
import shutil
import importlib
import subprocess
import numpy as np
import h5py
from pathlib import Path

from termcolor import colored

# ---------------------------------------------------------------------------
# EXTERNAL DEPENDENCIES
# ---------------------------------------------------------------------------
try:
    from nimpy.eval_nimrod import EvalNimrod
    _NIMPY_AVAILABLE = True
except ImportError:
    _NIMPY_AVAILABLE = False
    print("Warning: nimpy module not found. NIMROD extraction will fail if called.")

from .base import SimulationAdapter
from .utils import (
    printwarn, printerr, printnote,
    compute_file_hash, auto_cast,
    GridSpec,
    _infer_machine_and_shot, _allocate_imas_run, _imas_entry_dir,
    _imas_dd_version_dir,
    _prepare_nimrod2imas_runtime, _nimrod2imas_tool_available,
    _run_nimrod2imas_tool, _options_dict_to_cli,
    _record_imas_manifest,
    _GEQDSK_PATTERNS, _PEQDSK_PATTERNS,
    _is_geqdsk_name, _is_peqdsk_name,
)


class NIMRODAdapter(SimulationAdapter):
    CODE_NAME = "NIMROD"

    # Sign flip for the toroidal (phi) component to match M3D-C1 conventions.
    PHI_SIGN_FLIP = -1.0

    def __init__(self, model_dir):
        super().__init__(model_dir)
        self._dumps_converted = False
        self._nimgrowth_cache = {}
        self._indexed_dumps_per_dir = {}
        self._common_max_index_cache = None
        self._mode_dirs = []

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
                    meta["simulation_time_step"][i] = step

        return meta

    def get_reproducibility_data(self, mode_dir):
        hashes = {}

        target_files = ['nimhist.bin', 'energy.bin', 'energy.txt', 'discharge.bin',
                        'growth.bin', 'growth.txt', 'timestat.bin']
        for fname in target_files:
            fpath = mode_dir / fname
            if fpath.exists():
                hashes[fname] = compute_file_hash(fpath)

        for h5_file in mode_dir.glob("dumpgll.*.h5"):
            hashes[h5_file.name] = compute_file_hash(h5_file)

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
        first registered nXX mode directory.
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
            return 1.0, 2.0, -1.0, 1.0

        rmin, rmax = float('inf'), float('-inf')
        zmin, zmax = float('inf'), float('-inf')

        try:
            with h5py.File(dump_file, 'r') as h5:
                rblocks = h5.get('rblocks')
                if rblocks:
                    for block_name in rblocks.keys():
                        rz_name = f"rz{block_name}"
                        if rz_name in rblocks[block_name]:
                            rzdat = rblocks[block_name][rz_name][()]
                            r_vals = rzdat[..., 0]
                            z_vals = rzdat[..., 1]
                            rmin = min(rmin, np.min(r_vals))
                            rmax = max(rmax, np.max(r_vals))
                            zmin = min(zmin, np.min(z_vals))
                            zmax = max(zmax, np.max(z_vals))

            if rmin == float('inf'):
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
        if not _NIMPY_AVAILABLE:
            printwarn("nimpy is not available; cannot evaluate NIMROD fields.")
            return np.array([])

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
            printerr(f"Error evaluating 2D field {name} in NIMROD: {e}")
            return np.array([])
        finally:
            os.chdir(original_cwd)

    def get_3d_field(self, name, grid_spec, units, time=-1):
        if not _NIMPY_AVAILABLE:
            printwarn("nimpy is not available; cannot evaluate NIMROD fields.")
            return np.array([])

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
            printerr(f"Error evaluating 3D field {name} in NIMROD: {e}")
            return np.array([])
        finally:
            os.chdir(original_cwd)

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

    def get_mode_2d_field(self, mode_dir, name, grid_spec, units):
        """
        Evaluates the field at the last dump file in the given mode directory (finite time).
        """
        if not _NIMPY_AVAILABLE:
            printwarn("nimpy is not available; cannot evaluate NIMROD fields.")
            return np.array([])

        nim_field = self._map_nimrod_field(name)
        if not nim_field:
            printwarn(f"Field '{name}' is not currently mapped for NIMROD extraction.")
            return np.array([])

        indexed = self._get_indexed_dumps_for_dir(mode_dir)
        if not indexed:
            printwarn(f"No dump files found in mode directory {mode_dir}.")
            return np.array([])
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
        """
        if not _NIMPY_AVAILABLE:
            printwarn("nimpy is not available; cannot evaluate NIMROD fields.")
            return np.array([])

        nim_field = self._map_nimrod_field(name)
        if not nim_field:
            printwarn(f"Field '{name}' is not currently mapped for NIMROD extraction.")
            return np.array([])

        indexed = self._get_indexed_dumps_for_dir(mode_dir)
        if not indexed:
            printwarn(f"No dump files found in mode directory {mode_dir}.")
            return np.array([])
        _, _, dump_path_obj = indexed[-1]
        dump_path  = Path(dump_path_obj).resolve()
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
