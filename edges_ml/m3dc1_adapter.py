"""
M3D-C1 adapter for edges-ml.

Contains the M3DC1Adapter class, which implements the SimulationAdapter
interface for M3D-C1 HDF5 output files.

The adapter works in two modes:

*local*
    'model_dir' is a directory on this machine containing nXX/C1.h5 files.

*campaign (ADIOS2 .aca streaming)*
    'campaign' is a CampaignContext describing a remote simulation registered
    in an HPC campaign archive.  Every fpy.sim_data object is then opened with
    filetype='adios2' and the data is streamed over SSH on demand.

Three pieces of shared state make the M3D-C1 / fusion-io stack fragile; all of
them are handled here:

1.  fio_py.set_int_option(FIO_TIMESLICE, ...) is GLOBAL, not per source.
    Creating any other sim_data object (for instance the short-lived
    equilibrium object that m3dc1's flux_average() opens internally) silently
    changes the time slice that every subsequent field evaluation uses.  In
    campaign mode this surfaces as
        'm3dc1_numpy_source::get_field: fields ... not injected for timeslice -1'
    followed by 'SystemError: get_field() returned NULL ...'.
    _sync_timeslice() re-asserts the time slice of the sim object that is
    about to be used.

2.  Several m3dc1 helpers open 'filename' with h5py as a fallback.  In
    campaign mode sim.filename is the .aca archive (an SQLite database), which
    fails with 'Unable to synchronously open file (file signature not found)'.
    _h5_filename() hands out the HDF5 skeletons that fusion-io extracts from
    the archive instead.

3.  Not every run writes every diagnostic (pellet / KPRAD / wall-force traces
    are typically absent).  Asking for one of those costs a full SSH round trip
    and then raises an exception whose text contains the complete remote stderr
    (Lmod chatter plus a remote traceback).  The requested traces are therefore
    pre-filtered against what the simulation really contains, and any remaining
    failure is reported with utils.brief_error() as a single short line.

Perturbations
-------------
For every quantity listed under config['perturbations'] the adapter provides
    * flux averages:  <field>(mode sim, time='last') - <field>(equilibrium),
                      both on the equilibrium flux surfaces
                      (get_mode_flux_average),
    * 2D / 3D fields: field(mode sim) - field(equilibrium) (base.extract_mode),
    * poloidal spectra (get_mode_spectrum) for scalar fields.

get_mode_spectrum() is a port of the numerical core of
m3dc1.eigenfunction.eigenfunction(fourier=True) that is safe with respect to
the shared state listed above:
    * flux_coordinates() is called with an explicit flux coordinate system,
      and the equilibrium's 'fc' attribute (shared with flux_average()) is
      saved and restored around the call;
    * the fusion-io time slice is re-asserted before every evaluation;
    * the toroidal mode number is taken from the directory name, never from a
      parameter file (which is not available in campaign mode);
    * the unit conversion uses the actual field name instead of 'p'.
The safety factor is not part of the spectrum output; it is available as a
flux average.
"""

import os
import re
import hashlib
import subprocess
import numpy as np
from pathlib import Path

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
    _M3DC1_AVAILABLE = True
except ImportError:
    _M3DC1_AVAILABLE = False
    print("Warning: fpy or m3dc1 modules not found. M3D-C1 extraction will fail if called.")

# Only needed for poloidal mode spectra; imported separately so that a missing
# module does not affect any other M3D-C1 functionality.
try:
    from m3dc1.flux_coordinates import flux_coordinates
    import m3dc1.fpylib as fpyl
    _M3DC1_SPECTRUM_AVAILABLE = True
except ImportError:
    _M3DC1_SPECTRUM_AVAILABLE = False

from .base import SimulationAdapter
from .campaign import CampaignArchive, CampaignContext
from .utils import (
    printwarn, printerr, printnote,
    compute_file_hash, auto_cast, generate_unique_filename,
    brief_error, is_missing_error, summarize_names,
    poloidal_spectrum,
    GridSpec,
    _subtract_field_results,
    _infer_machine_and_shot, _allocate_imas_run, _imas_entry_dir,
)


class M3DC1Adapter(SimulationAdapter):

    CODE_NAME = "M3D-C1"

    # Traces that fpy exposes even though they are not datasets of 'scalars'.
    _SYNTHETIC_TRACES = ("time", "bharmonics", "keharmonics")

    def __init__(self, model_dir, campaign=None):
        super().__init__(model_dir)

        # Campaign archive streaming context (None for local data).
        self.campaign = campaign

        # Cache to store the (R, Z) points of the inner wall
        self._inner_wall_points = None

        # Sim objects caches to ensure time=-1 and time='last' are loaded optimally
        self._eq_sim = None
        self._mode_sims = {}
        self._time_sims = {}

        # Cached h5py-openable filename for campaign (skeleton) mode.
        self._campaign_h5_file = None

        # Time-trace bookkeeping (see _available_traces / get_time_trace).
        self._trace_names = None
        self._skipped_traces = set()
        self._failed_traces = set()

        # Poloidal spectrum bookkeeping (see get_mode_spectrum).
        self._spectrum_fc_cache = {}
        self._spectrum_fc_failed = set()
        self._spectrum_warned = set()

        # Mode directories (local paths) or campaign simulation paths.
        if campaign is not None and getattr(campaign, "modes", None):
            self._mode_dirs = [Path(m) for m in campaign.modes]
        else:
            self._mode_dirs = []

        # Internal state for Slurm file metadata
        self._slurm_parsed = False
        self._slurm_rel_ver = "Unknown"
        self._slurm_bld_date = "Unknown"
        self._slurm_inputs = {}

        # Look for the Gamma growth rate file inside the parent model directory.
        # This is only possible for locally available data.
        self.gamma_data = None
        if _M3DC1_AVAILABLE and self.campaign is None and self.model_dir.is_dir():
            try:
                entries = list(self.model_dir.iterdir())
            except OSError:
                entries = []
            for f in entries:
                if f.is_file() and f.suffix in ['.txt', '.dat', '.out', '']:
                    try:
                        with open(f, 'r') as tmp:
                            first_lines = "".join([next(tmp) for _ in range(5)])
                        if 'gamma' in first_lines and 'sig_gamma' in first_lines:
                            self.gamma_data = Gamma_file(str(f))
                            break
                    except Exception:
                        pass

    # -----------------------------------------------------------------------
    # Campaign / local plumbing
    # -----------------------------------------------------------------------

    @property
    def is_campaign(self):
        return self.campaign is not None

    def set_mode_dirs(self, mode_dirs):
        """
        Registers the mode (nXX) directories — or, in campaign mode, the
        simulation paths inside the archive — that belong to this model.
        """
        self._mode_dirs = [Path(d) for d in (mode_dirs or [])]

    def _primary_target(self):
        """
        Returns the 'thing' that identifies this model for fpy:
        a local C1.h5 path, or a campaign simulation path.
        """
        if self.is_campaign:
            if self._mode_dirs:
                return str(self._mode_dirs[0]).strip('/')
            if self.campaign.simulation:
                return str(self.campaign.simulation).strip('/')
            return None

        c1_paths = list(self.model_dir.rglob("C1.h5"))
        if not c1_paths:
            return None
        return str(c1_paths[0])

    def _sim_filename(self):
        """Filename that is handed to the m3dc1 helper routines."""
        if self.is_campaign:
            return str(self.campaign.archive)
        target = self._primary_target()
        return target if target else str(self.model_dir / "C1.h5")

    def _open_sim(self, target, time):
        """
        Opens an fpy.sim_data object.

        target : str
            Local path to a C1.h5 file, or (campaign mode) the simulation path
            inside the campaign archive.
        """
        if not _M3DC1_AVAILABLE:
            raise NameError("fpy is not available")

        if self.is_campaign:
            kwargs = self.campaign.sim_data_kwargs(simulation=target, time=time)
            if self.campaign.verbose:
                printnote(
                    f"Streaming {kwargs['campaign_simulation']} "
                    f"(time={time}) from {self.campaign.describe()}"
                )
            return fpy.sim_data(**kwargs)

        return fpy.sim_data(filename=str(target), time=time)

    def _get_eq_sim(self):
        """
        Lazily loads the shared equilibrium simulation object (time=-1).
        """
        if self._eq_sim is None:
            target = self._primary_target()
            if target is None:
                if self.is_campaign:
                    printerr(
                        f"No campaign simulation registered for {self.model_dir}.")
                    return None
                raise FileNotFoundError(f"No C1.h5 files found in {self.model_dir}")
            try:
                self._eq_sim = self._open_sim(target, -1)
            except NameError:
                pass
            except Exception as exc:
                printerr(f"Could not open equilibrium simulation ({target}): "
                         f"{brief_error(exc)}")
                self._eq_sim = None
        return self._eq_sim

    def _get_mode_sim(self, mode_dir):
        """
        Lazily loads the simulation object for a mode directory at time='last'.
        """
        key = Path(mode_dir)
        if key not in self._mode_sims:
            if self.is_campaign:
                target = str(key).strip('/')
            else:
                target = str(key / 'C1.h5')
            try:
                self._mode_sims[key] = self._open_sim(target, 'last')
            except NameError:
                self._mode_sims[key] = None
            except Exception as exc:
                printerr(f"Could not open mode simulation ({target}): "
                         f"{brief_error(exc)}")
                self._mode_sims[key] = None
        return self._mode_sims.get(key)

    def _get_time_sim(self, time_slice):
        """
        Lazily loads the specific simulation object for a selected time slice.
        """
        if time_slice not in self._time_sims:
            target = self._primary_target()
            if target is None:
                return None
            try:
                self._time_sims[time_slice] = self._open_sim(target, time_slice)
            except NameError:
                return None
            except Exception as exc:
                printerr(f"Could not open simulation for time slice {time_slice}: "
                         f"{brief_error(exc)}")
                self._time_sims[time_slice] = None
        return self._time_sims.get(time_slice)

    def close(self):
        """Drops the cached fpy objects (closing SSH connections in campaign mode)."""
        if self._skipped_traces:
            printnote(
                f"Skipped {len(self._skipped_traces)} time trace(s) that this "
                f"simulation does not write: "
                f"{summarize_names(self._skipped_traces)}"
            )
        self._skipped_traces = set()
        self._failed_traces = set()
        self._trace_names = None

        self._spectrum_fc_cache = {}
        self._spectrum_fc_failed = set()

        self._eq_sim = None
        self._mode_sims = {}
        self._time_sims = {}

    def _map_units(self, units):
        """Translates generalized config units into M3D-C1 specific unit strings."""
        u = units.lower()
        if u == 'codeunits':
            return 'm3dc1'
        return u

    # -----------------------------------------------------------------------
    # Robustness helpers for the shared fusion-io / m3dc1 global state
    # -----------------------------------------------------------------------

    def _sync_timeslice(self, sim):
        """
        Re-asserts the time slice of *sim* on fusion-io's GLOBAL option stack.

        fio_py.set_int_option(FIO_TIMESLICE, ...) is process-wide state shared
        by every open source.  fpy only sets it when a sim_data object is
        created (or when set_timeslice() is called explicitly), so creating any
        other sim_data object in between — which m3dc1's flux_average() does
        internally whenever it needs the equilibrium flux surfaces — silently
        re-points every subsequent field evaluation at the wrong time slice.

        With the numpy (campaign) source this is fatal:
            m3dc1_numpy_source::get_field: fields 'psi' or 'I' not injected
            for timeslice -1
            SystemError: <built-in function get_field> returned NULL ...
        """
        if sim is None:
            return
        ts = getattr(sim, 'timeslice', None)
        if ts is None:
            return
        try:
            sim.set_timeslice(int(ts))
        except Exception as exc:
            printwarn(f"Could not re-assert M3D-C1 time slice {ts}: {brief_error(exc)}")

    def _sim_time(self, sim, default=-1):
        """Returns the time slice an evaluation should be performed at."""
        ts = getattr(sim, 'timeslice', None)
        if ts is None:
            return default
        try:
            return int(ts)
        except (TypeError, ValueError):
            return default

    def _h5_filename(self, sim=None):
        """
        Returns a path that the m3dc1 helpers can open with h5py.

        Several helpers (flux_average, get_timetrace, readParameter, ...) fall
        back to reading the simulation file directly.  In campaign mode
        sim.filename is the .aca archive, i.e. an SQLite database, so h5py
        aborts with
            'Unable to synchronously open file (file signature not found)'.

        fusion-io extracts the embedded HDF5 skeletons of the archive and links
        them into the current working directory (C1.h5, equilibrium.h5,
        time_NNN.h5), so those are handed out instead.
        """
        if not self.is_campaign:
            if sim is not None and getattr(sim, 'filename', None):
                return sim.filename
            return self._sim_filename()

        if self._campaign_h5_file is not None:
            return self._campaign_h5_file

        backend = getattr(sim, '_aca', None)
        if backend is None:
            backend = getattr(self._eq_sim, '_aca', None)

        if backend is not None:
            try:
                # Creates (and links into the CWD) the C1.h5 skeleton ...
                backend.get_skeleton_path('C1.h5')
                # ... and the equilibrium.h5 / time_NNN.h5 siblings.
                backend.ensure_all_skeletons_in_cwd()
            except Exception as exc:
                printwarn(f"Could not extract the campaign HDF5 skeletons: "
                          f"{brief_error(exc)}")

        candidate = Path.cwd() / 'C1.h5'
        if candidate.exists():
            self._campaign_h5_file = str(candidate)
            return self._campaign_h5_file

        if backend is not None:
            try:
                self._campaign_h5_file = str(backend.get_skeleton_path('C1.h5'))
                return self._campaign_h5_file
            except Exception:
                pass

        return getattr(sim, 'filename', None) or self._sim_filename()

    def _attach_equilibrium_fc(self, sim, resolution, units, fcoords):
        """
        Hands the cached equilibrium flux-coordinate system to a finite-time
        sim object (campaign mode only).

        With use_eq_fs=True, m3dc1's flux_average() needs the equilibrium flux
        coordinates.  If the sim object it is given does not carry them it
        opens the simulation file itself to build a fresh equilibrium object —
        which cannot work for a campaign archive.  Since 'use_eq_fs' means
        'use the equilibrium flux surfaces' anyway, we simply reuse the object
        that was already computed on the cached equilibrium sim.
        """
        if sim is None or getattr(sim, 'fc', None) is not None:
            return

        eq = self._get_eq_sim()
        if eq is None or eq is sim:
            return

        fc = getattr(eq, 'fc', None)
        if fc is None:
            # Force the equilibrium flux coordinates to be computed once.
            try:
                self._sync_timeslice(eq)
                flux_average(field='psi', sim=eq, filename=self._h5_filename(eq),
                             fcoords=fcoords, use_eq_fs=True, points=resolution,
                             time=-1, units=self._map_units(units), coord='scalar')
                fc = getattr(eq, 'fc', None)
            except Exception as exc:
                printwarn(f"Could not pre-compute the equilibrium flux coordinates: "
                          f"{brief_error(exc)}")
                fc = None

        if fc is not None:
            try:
                sim.fc = fc
            except Exception as exc:
                printwarn(f"Could not reuse the equilibrium flux coordinates: "
                          f"{brief_error(exc)}")

    # -----------------------------------------------------------------------
    # Metadata / inputs
    # -----------------------------------------------------------------------

    def _parse_slurm_data(self, mode_dirs):
        """
        Parses Slurm standard output files for the code release version, build date,
        and the full list of physics input parameters.
        """
        release_version = "Unknown"
        build_date = "Unknown"
        inputs = {}

        # Slurm logs are not part of the campaign archive.
        if self.is_campaign:
            return release_version, build_date, inputs

        candidates = []

        for mdir in mode_dirs:
            mdir = Path(mdir)
            for out_file in mdir.glob("slurm*.out"):
                candidates.append(out_file)
            for out_file in mdir.glob("*.out"):
                candidates.append(out_file)

            try:
                n_str = mdir.name.replace('n', '')
                n_val = int(n_str)
                for out_file in self.model_dir.glob(f"slurm-*_{n_str}.out"):
                    candidates.append(out_file)
                for out_file in self.model_dir.glob(f"slurm-*_{n_val}.out"):
                    candidates.append(out_file)
            except ValueError:
                pass

        for out_file in self.model_dir.glob("slurm-*.out"):
            if "_" not in out_file.name:
                candidates.append(out_file)

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

                    for line_count, line in enumerate(f):
                        if line_count > 10000:
                            break

                        line_str = line.strip()

                        if "RELEASE VERSION:" in line_str:
                            release_version = line_str.split("RELEASE VERSION:")[1].strip()
                            rel_found = True
                        if "BUILD DATE:" in line_str:
                            build_date = line_str.split("BUILD DATE:")[1].strip()
                            bld_found = True

                        if not params_done:
                            if not found_input_anchor:
                                if "Reading input file C1input" in line_str:
                                    found_input_anchor = True
                            elif not in_params:
                                if '=' in line_str and not line_str.startswith('='):
                                    parts = line_str.split('=', 1)
                                    key = parts[0].strip()
                                    if ' ' not in key and len(key) > 0 and not key.startswith("WARNING"):
                                        in_params = True
                                        inputs[key] = auto_cast(parts[1].strip())
                            else:
                                if '=' in line_str and not line_str.startswith('='):
                                    parts = line_str.split('=', 1)
                                    inputs[parts[0].strip()] = auto_cast(parts[1].strip())
                                else:
                                    params_done = True

                    if rel_found or bld_found or len(inputs) > 0:
                        return release_version, build_date, inputs
            except Exception:
                continue

        return release_version, build_date, inputs

    def _campaign_metadata(self):
        """Metadata for a simulation streamed from a campaign archive."""
        output_version = "Unknown"
        try:
            sim = self._get_eq_sim()
            if sim is not None:
                output_version = str(sim.get_constants().version)
        except Exception:
            pass

        return {
            "code":                self.CODE_NAME,
            "source_directory":    str(self.campaign.simulation),
            "data_access":         "campaign_archive",
            "campaign_archive":    str(self.campaign.archive),
            "campaign_simulation": str(self.campaign.simulation),
            "campaign_login":      str(self.campaign.login or "local"),
            "output_version":      output_version,
            "release_version":     "Unknown",
            "build_date":          "Unknown",
        }

    def get_metadata(self, mode_dirs):
        if self.is_campaign:
            return self._campaign_metadata()

        if not self._slurm_parsed:
            rel_ver, bld_date, inputs = self._parse_slurm_data(mode_dirs)
            self._slurm_rel_ver = rel_ver
            self._slurm_bld_date = bld_date
            self._slurm_inputs = inputs
            self._slurm_parsed = True

        return {
            "code": self.CODE_NAME,
            "source_directory": str(self.model_dir.resolve()),
            "data_access": "local",
            "release_version": self._slurm_rel_ver,
            "build_date": self._slurm_bld_date
        }

    def get_reproducibility_data(self, mode_dir):
        """
        Extracts hashes for crucial data files inside the mode directory.

        In campaign mode the files are remote, so the archive's dataset UUIDs
        are recorded instead of local checksums.
        """
        if self.is_campaign:
            info = {"campaign_simulation": str(mode_dir).strip('/')}
            try:
                archive = CampaignArchive.open(self.campaign.archive)
                for basename, meta in sorted(archive.files_for(mode_dir).items()):
                    info[basename] = meta.get("uuid", "")
            except Exception as exc:
                printwarn(f"Could not read dataset UUIDs for {mode_dir}: "
                          f"{brief_error(exc)}")
            return info

        hashes = {}
        mode_dir = Path(mode_dir)

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
        if self.is_campaign:
            return {}
        return getattr(self, '_slurm_inputs', {})

    def get_global_parameters(self):
        """Retrieve globally shared parameters from the gamma file and shape calculations."""
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

        try:
            sim = self._get_eq_sim()
            if sim is not None:
                self._sync_timeslice(sim)
                shape_dict = get_shape(sim, quiet=True)
                globals_dict.update(shape_dict)
        except Exception as e:
            printwarn(f"Warning: Could not calculate shaping parameters: {brief_error(e)}")

        return globals_dict

    # -----------------------------------------------------------------------
    # Meshes and fields
    # -----------------------------------------------------------------------

    def get_1d_mesh(self, resolution, units, fcoords):
        """Extracts the 1D radial grid used by flux_average."""
        sim = self._get_eq_sim()
        if sim is None:
            return np.array([])
        m3dc1_units = self._map_units(units)
        self._sync_timeslice(sim)
        try:
            nflux, _ = flux_average(field='p', sim=sim, filename=self._h5_filename(sim),
                                    fcoords=fcoords, use_eq_fs=True, points=resolution,
                                    time=-1, units=m3dc1_units)
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
        sim = self._get_eq_sim() if time == -1 else self._get_time_sim(time)
        if sim is None:
            return np.array([])
        m3dc1_units = self._map_units(units)

        if grid_spec.type == "rectangular":
            return self._evaluate_rectangular_grid(name, grid_spec, sim, m3dc1_units, is_3d=False)
        elif grid_spec.type == "native":
            return self._evaluate_native_grid(name, sim, m3dc1_units)
        elif grid_spec.type == "inside_wall":
            return self._evaluate_inside_wall(name, sim, m3dc1_units)
        raise ValueError(f"Unknown grid type: {grid_spec.type}")

    def get_3d_field(self, name, grid_spec, units, time=-1):
        sim = self._get_eq_sim() if time == -1 else self._get_time_sim(time)
        if sim is None:
            return np.array([])
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
        Utilizes the imported flux_average function on the equilibrium sim
        (time=-1) or on the sim of the requested time slice.
        Returns a tuple of (normalized_flux_array, values_array) as 1D numpy arrays.
        """
        sim = self._get_eq_sim() if time == -1 else self._get_time_sim(time)
        if sim is None:
            return np.array([]), np.array([])
        return self._flux_average_on_sim(sim, name, resolution, units, fcoords,
                                         time, use_eq_fs=use_eq_fs)

    def _flux_average_on_sim(self, sim, name, resolution, units, fcoords, time,
                             use_eq_fs=True, label=None):
        """
        Runs m3dc1's flux_average() for *name* on an arbitrary sim object
        (equilibrium, time slice or mode simulation).
        """
        m3dc1_units = self._map_units(units)
        h5_file     = self._h5_filename(sim)

        # fusion-io's time slice option is global; make sure it matches 'sim'.
        self._sync_timeslice(sim)

        # Remote data: reuse the equilibrium flux coordinates instead of
        # letting flux_average() rebuild them from the (unreadable) archive.
        if self.is_campaign and use_eq_fs and sim is not self._eq_sim:
            self._attach_equilibrium_fc(sim, resolution, units, fcoords)
            self._sync_timeslice(sim)

        field_info = getattr(sim, 'typedict', getattr(sim, 'available_fields', {})).get(name, None)
        field_type = field_info[1] if field_info else 'scalar'

        try:
            if field_type == 'vector':
                nflux, fR = flux_average(field=name, sim=sim, filename=h5_file, fcoords=fcoords,
                                         use_eq_fs=use_eq_fs, points=resolution, time=time,
                                         units=m3dc1_units, coord='R')
                self._sync_timeslice(sim)
                _, fPhi = flux_average(field=name, sim=sim, filename=h5_file, fcoords=fcoords,
                                       use_eq_fs=use_eq_fs, points=resolution, time=time,
                                       units=m3dc1_units, coord='phi')
                self._sync_timeslice(sim)
                _, fZ = flux_average(field=name, sim=sim, filename=h5_file, fcoords=fcoords,
                                     use_eq_fs=use_eq_fs, points=resolution, time=time,
                                     units=m3dc1_units, coord='Z')
                return np.asarray(nflux), {
                    f"{name}_R":   np.asarray(fR),
                    f"{name}_phi": np.asarray(fPhi),
                    f"{name}_Z":   np.asarray(fZ)
                }
            else:
                nflux, fa = flux_average(field=name, sim=sim, filename=h5_file, fcoords=fcoords,
                                         use_eq_fs=use_eq_fs, points=resolution, time=time,
                                         units=m3dc1_units, coord='scalar')
                return np.asarray(nflux), np.asarray(fa)
        except Exception as e:
            where = label or f"time={time}"
            msg = f"Flux average '{name}' ({where}) failed: {brief_error(e)}"
            if is_missing_error(e):
                printwarn(msg)
            else:
                printerr(msg)
            return np.array([]), np.array([])
        finally:
            # flux_average() may have created (and destroyed) its own sim
            # object, which resets fusion-io's global time slice.
            self._sync_timeslice(sim)

    # -----------------------------------------------------------------------
    # Time traces
    # -----------------------------------------------------------------------

    def _available_traces(self, sim):
        """
        Returns the set of time-trace names this simulation actually contains.

        fpy exposes the datasets of the 'scalars' group through
        sim.available_traces; the pellet diagnostics live in a separate
        'pellet' group and are added here.  An empty set means 'could not
        determine', in which case no filtering is applied.
        """
        if self._trace_names is not None:
            return self._trace_names

        names = set(self._SYNTHETIC_TRACES)

        try:
            names.update(getattr(sim, 'available_traces', None) or [])
        except Exception:
            pass

        # Pellet diagnostics ('pellet/<name>').
        try:
            if self.is_campaign:
                backend = getattr(sim, '_aca', None)
                if backend is not None and backend.group_exists('pellet'):
                    names.update(backend.list_group('pellet'))
            else:
                h5 = getattr(sim, '_all_attrs', None)
                if h5 is not None and 'pellet' in h5:
                    names.update(h5['pellet'].keys())
        except Exception:
            pass

        # If only the synthetic names are known, discovery evidently failed;
        # fall back to "try everything" rather than skipping all traces.
        if len(names) <= len(self._SYNTHETIC_TRACES):
            names = set()

        self._trace_names = names
        return names

    def get_time_trace(self, name, units):
        """
        Extracts the requested scalar time trace using the native get_timetrace function.
        Returns a tuple of (time_array, values_array) as 1D numpy arrays.

        Quantities that this run simply does not write (pellet / KPRAD / wall
        force diagnostics, ...) are skipped without contacting the remote host;
        they are reported as a single summary line by close().
        """
        sim = self._get_eq_sim()
        if sim is None:
            return np.array([]), np.array([])

        known = self._available_traces(sim)
        if known and name not in known:
            self._skipped_traces.add(name)
            return np.array([]), np.array([])

        m3dc1_units = self._map_units(units)
        self._sync_timeslice(sim)

        try:
            time_arr, values, label, unitlabel = get_timetrace(
                trace=name, sim=sim, filename=self._h5_filename(sim),
                units=m3dc1_units, quiet=True, returnas='tuple'
            )
            return np.asarray(time_arr), np.asarray(values)
        except Exception as e:
            if is_missing_error(e):
                # Not present in this run - record it and stay quiet.
                self._skipped_traces.add(name)
            elif name not in self._failed_traces:
                self._failed_traces.add(name)
                printerr(f"Time trace '{name}' failed: {brief_error(e)}")
            return np.array([]), np.array([])

    # -----------------------------------------------------------------------
    # Mode data
    # -----------------------------------------------------------------------

    def get_mode_metadata(self, mode_dir):
        """Extracts growth rates and mode type specifically matching this directory's n."""
        meta = {
            "growth_rate": 0.0,
            "frequency":   0.0,
            "mode_type":   -100
        }

        if self.gamma_data is not None:
            try:
                n_val = int(Path(mode_dir).name.replace('n', ''))
                idx_array = np.where(self.gamma_data.n_list == n_val)[0]
                if len(idx_array) > 0:
                    idx = idx_array[0]
                    meta["growth_rate"] = float(self.gamma_data.gamma_list[idx])
                    meta["mode_type"]   = int(self.gamma_data.pblist[idx])
            except ValueError:
                pass

        return meta

    def get_mode_flux_average(self, mode_dir, name, resolution, units, fcoords):
        """
        Flux average of the perturbation of *name*:
            <field>(mode sim, time='last') - <field>(equilibrium, time=-1)
        Both averages are taken on the equilibrium flux surfaces, so the
        result lives on the equilibrium 1D grid ('grids/1d_grid').
        """
        mode = self._get_mode_sim(mode_dir)
        if mode is None:
            return np.array([])

        _, f_mode = self._flux_average_on_sim(
            mode, name, resolution, units, fcoords,
            time=self._sim_time(mode), use_eq_fs=True,
            label=f"{Path(mode_dir).name}, time='last'",
        )
        _, f_eq = self.get_flux_average(name, resolution, units, fcoords, time=-1)
        return _subtract_field_results(f_mode, f_eq)

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
        if sim is None:
            return np.array([])
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
        if sim is None:
            return np.array([])
        m3dc1_units = self._map_units(units)
        if grid_spec.type == "rectangular":
            return self._evaluate_rectangular_grid(name, grid_spec, sim, m3dc1_units, is_3d=True)
        elif grid_spec.type == "native":
            return self._evaluate_native_grid(name, sim, m3dc1_units)
        elif grid_spec.type == "inside_wall":
            return self._evaluate_inside_wall(name, sim, m3dc1_units)
        raise ValueError(f"Unknown grid type: {grid_spec.type}")

    # -----------------------------------------------------------------------
    # Poloidal mode spectra (port of m3dc1.eigenfunction.eigenfunction)
    # -----------------------------------------------------------------------

    def _warn_spectrum_once(self, key, message):
        if key not in self._spectrum_warned:
            printwarn(message)
            self._spectrum_warned.add(key)

    def _get_spectrum_fc(self, settings):
        """
        Builds (and caches) the flux coordinates used for poloidal spectra on
        the equilibrium sim.

        flux_coordinates() stores its result in eq.fc, which is shared with
        flux_average() and _attach_equilibrium_fc().  The previous object is
        therefore saved and restored so that the flux averages are unaffected
        by the spectrum resolution / coordinate choice.

        flux_coordinates() reports failure by returning None or 0 (instead of
        raising); in that case eq.fc is not replaced, which is what is checked
        here.  Note that for the equilibrium sim (time slice -1) it reads
        xmag / zmag / psi_lcfs from the LAST entry of the time traces; this is
        existing M3D-C1 behaviour and is kept.
        """
        fcoords = settings["fcoords"]
        m3dc1_fcoords = "" if fcoords == "geometric" else fcoords
        points = int(settings["points"])
        phit = float(settings["phi"])
        key = (points, m3dc1_fcoords, phit)

        if key in self._spectrum_fc_cache:
            return self._spectrum_fc_cache[key]
        if key in self._spectrum_fc_failed:
            return None

        eq = self._get_eq_sim()
        if eq is None:
            self._spectrum_fc_failed.add(key)
            return None

        if fcoords == "geometric":
            self._warn_spectrum_once(
                "geometric",
                "Poloidal spectra requested with the GEOMETRIC poloidal angle. "
                "Spectra depend on the angle definition; use 'pest' or 'boozer' "
                "unless this is intended."
            )

        saved_fc = getattr(eq, 'fc', None)
        fc = None
        try:
            self._sync_timeslice(eq)
            res = flux_coordinates(sim=eq, filename=self._h5_filename(eq),
                                   time=self._sim_time(eq), fcoords=m3dc1_fcoords,
                                   phit=phit, points=points, makeplot=False,
                                   quiet=True)
            new_fc = getattr(eq, 'fc', None)
            if res is not None and not (isinstance(res, (int, float)) and res == 0) \
                    and new_fc is not None and new_fc is not saved_fc:
                fc = new_fc
            else:
                printerr(f"flux_coordinates() failed ({fcoords or 'geometric'}, "
                         f"points={points}) for {self.model_dir}; no poloidal spectra.")
        except Exception as exc:
            printerr(f"flux_coordinates() failed for {self.model_dir}: {brief_error(exc)}")
        finally:
            try:
                eq.fc = saved_fc
            except Exception:
                pass
            self._sync_timeslice(eq)

        if fc is None:
            self._spectrum_fc_failed.add(key)
            return None

        self._spectrum_fc_cache[key] = fc
        return fc

    def get_mode_spectrum(self, mode_dir, field, settings, n=None):
        """
        Poloidal Fourier spectrum of the eigenfunction
            ef = field(mode sim, time='last') - field(equilibrium, time=-1)
        evaluated on the flux-coordinate grid of the equilibrium at constant
        geometric toroidal angle phi = settings['phi'], exactly as in
        m3dc1.eigenfunction.eigenfunction(fourier=True):
            amplitude = |rfft(ef, axis=0, norm='forward')|

        Returned datasets: psi_norm, m, amplitude[m, psi_norm] (plus the flux
        grid if 'store_fluxgrid' is enabled).  q is not included; it is
        available as a flux average.
        """
        if not (_M3DC1_AVAILABLE and _M3DC1_SPECTRUM_AVAILABLE):
            self._warn_spectrum_once(
                "unavailable",
                "m3dc1.flux_coordinates / m3dc1.fpylib not available; "
                "M3D-C1 poloidal spectra are skipped."
            )
            return {}

        if settings.get("include_omega"):
            self._warn_spectrum_once(
                "omega",
                "spectrum include_omega is not implemented for M3D-C1; spectra are "
                "evaluated at constant geometric toroidal angle (as in eigenfunction())."
            )

        eq = self._get_eq_sim()
        mode = self._get_mode_sim(mode_dir)
        if eq is None or mode is None:
            return {}

        field_info = getattr(mode, 'typedict', getattr(mode, 'available_fields', {})).get(field, None)
        if field_info and field_info[1] != 'scalar':
            self._warn_spectrum_once(
                f"vector::{field}",
                f"Poloidal spectra of non-scalar field '{field}' are not supported yet; skipping."
            )
            return {}

        fc = self._get_spectrum_fc(settings)
        if fc is None:
            return {}

        phit = float(settings["phi"])
        rpath = np.asarray(fc.rpath, dtype=float)
        zpath = np.asarray(fc.zpath, dtype=float)
        torphi = np.full_like(rpath, phit)

        try:
            self._sync_timeslice(mode)
            p1 = eval_field(field, rpath, torphi, zpath, coord='scalar', sim=mode,
                            filename=self._h5_filename(mode),
                            time=self._sim_time(mode), quiet=True)
            self._sync_timeslice(eq)
            p0 = eval_field(field, rpath, torphi, zpath, coord='scalar', sim=eq,
                            filename=self._h5_filename(eq),
                            time=self._sim_time(eq), quiet=True)
        except Exception as exc:
            printerr(f"Eigenfunction evaluation of '{field}' failed for {mode_dir}: "
                     f"{brief_error(exc)}")
            return {}
        finally:
            self._sync_timeslice(eq)

        ef = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
        if ef.shape != rpath.shape:
            try:
                ef = ef.reshape(rpath.shape)
            except Exception:
                printerr(f"Eigenfunction of '{field}' has unexpected shape {ef.shape} "
                         f"(expected {rpath.shape}) for {mode_dir}.")
                return {}

        # Same conversion rule as eigenfunction(), but with the real field name.
        m3dc1_units = self._map_units(settings.get("units", "codeunits"))
        if m3dc1_units == 'm3dc1':
            try:
                ef = np.asarray(fpyl.get_conv_field(m3dc1_units, field, ef, sim=eq),
                                dtype=float)
            except Exception as exc:
                self._warn_spectrum_once(
                    f"conv::{field}",
                    f"Unit conversion of '{field}' for the spectrum failed "
                    f"({brief_error(exc)}); raw values are used."
                )

        n_bad = int(np.count_nonzero(~np.isfinite(ef)))
        if n_bad:
            printwarn(f"M3D-C1 spectrum '{field}' ({Path(mode_dir).name}): {n_bad} "
                      "flux-grid points could not be evaluated; affected surfaces "
                      "will contain NaN.")

        amplitude, m = poloidal_spectrum(ef)

        theta = getattr(fc, 'theta', None)
        if theta is None:
            theta = 2.0 * np.pi * np.arange(rpath.shape[0]) / rpath.shape[0]

        out = {
            "psi_norm":  np.asarray(fc.psi_norm, dtype=float),
            "m":         m,
            "amplitude": amplitude,
            "_attrs": {
                "code":             self.CODE_NAME,
                "field":            str(field),
                "n":                int(n) if n is not None else -1,
                "fcoords":          settings["fcoords"],
                "phi":              phit,
                "toroidal_angle":   "geometric_phi",
                "points":           int(settings["points"]),
                "normalization":    "rfft_forward_abs",
                "theta_convention": "theta=0 outboard midplane, clockwise in (R,Z)",
            },
        }
        if settings.get("store_fluxgrid"):
            out["eigenfunction"] = ef
            out["R"] = rpath
            out["Z"] = zpath
            out["theta"] = np.asarray(theta, dtype=float)
        return out

    def get_time_metadata(self, time_slices):
        meta = {
            "time_slice":            np.array(time_slices, dtype=int),
            "simulation_time_step":  np.zeros(len(time_slices), dtype=int),
            "simulation_time":       np.zeros(len(time_slices), dtype=float)
        }

        for i, ts in enumerate(time_slices):
            sim = self._get_eq_sim() if ts == -1 else self._get_time_sim(ts)
            if sim is None:
                printwarn(f"Could not load simulation object for time slice {ts}; skipping.")
                continue

            self._sync_timeslice(sim)
            h5file  = sim._all_attrs
            c1_file = self._h5_filename(sim)

            fname = "equilibrium.h5" if ts == -1 else f"time_{ts:03d}.h5"

            try:
                meta["simulation_time"][i] = get_time_of_slice(ts, sim=sim, filename=c1_file, units='m3dc1')
            except Exception as e:
                printerr(f"Could not read simulation_time for slice {ts}: {brief_error(e)}")

            try:
                meta["simulation_time_step"][i] = int(readParameter('ntimestep', h5file=h5file))
            except Exception as e:
                printwarn(f"Could not read ntimestep for slice {ts} ({fname}): "
                          f"{brief_error(e)}")

            self._sync_timeslice(sim)

        return meta

    def convert_to_imas(self, mode_dirs, output_directory, config, run=None, out_file=None):
        """
        Writes IMAS-compatible output for this M3D-C1 model directory.

        The conversion backend is selected by config['m3dc1_imas_conversion_lib']:
            'standalone'  (default) – uses m3dc1.convert2imas
            'imas-python'           – uses m3dc1.convert2imas_imaspy

        Returns a dictionary summarising the produced artefact, or None on failure.
        """
        if self.is_campaign:
            printwarn(
                "IMAS conversion is not supported for campaign-archive (remote) "
                f"data: {self.campaign.describe()}. "
                "Use output_format='reduced_h5' / 'reduced_bp' instead, or run "
                "the conversion on the remote host."
            )
            return None

        conversion_lib = str(
            config.get("m3dc1_imas_conversion_lib", "standalone")
        ).strip().lower()

        target_filename = (
            str(Path(mode_dirs[0]) / 'C1.h5') if mode_dirs
            else str(self.model_dir / 'C1.h5')
        )

        res = config.get("resolutions", {}).get("1d", 200)

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
                printerr(f"convert2imas (standalone) failed for {self.model_dir}: "
                         f"{brief_error(exc)}")
                return None

            return {
                "code":            self.CODE_NAME,
                "conversion_lib":  "standalone",
                "model_directory": str(self.model_dir.resolve()),
                "source_file":     target_filename,
                "output_file":     str(out_file),
            }

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
                printerr(f"convert2imas_imaspy failed for {self.model_dir}: "
                         f"{brief_error(exc)}")
                return None

            entry = _imas_entry_dir(db_path, db_name, db_version, pulse, run)
            return {
                "code":            self.CODE_NAME,
                "conversion_lib":  "imas-python",
                "model_directory": str(self.model_dir.resolve()),
                "source_file":     target_filename,
                "entry_directory": str(entry),
                "db_name":         db_name,
                "pulse":           pulse,
                "run":             run,
                "backend":         backend,
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
        the mesh points inside the wall. Caches the result.
        """
        if getattr(self, 'shared_inner_wall_points', None) is not None:
            return self.shared_inner_wall_points

        if self._inner_wall_points is None:
            sim = self._get_eq_sim()
            if sim is None:
                self._inner_wall_points = np.array([])
            else:
                try:
                    self._sync_timeslice(sim)
                    self._inner_wall_points = get_points_inside_inner_wall(sim=sim, quiet=True)
                except Exception as e:
                    printerr(f"Error obtaining inner wall points: {brief_error(e)}")
                    self._inner_wall_points = np.array([])

        return self._inner_wall_points

    def _evaluate_inside_wall(self, field_name, sim, units):
        """
        Retrieves the (R, Z) coordinates inside the inner wall and evaluates
        the requested field at those specific points.
        """
        points = self._get_inner_wall_points()
        if len(points) == 0:
            return np.array([])
        R   = points[:, 0]
        Z   = points[:, 1]
        phi = np.zeros_like(R)

        # The inner-wall lookup above may have touched the equilibrium sim, so
        # re-assert the time slice of the object we are about to evaluate on.
        self._sync_timeslice(sim)
        time     = self._sim_time(sim)
        filename = self._h5_filename(sim)

        field_info = getattr(sim, 'typedict', getattr(sim, 'available_fields', {})).get(field_name, None)
        field_type = field_info[1] if field_info else 'scalar'

        if field_type == 'vector':
            vR, vPhi, vZ = eval_field(field_name, R, phi, Z, coord='vector', sim=sim,
                                      filename=filename, time=time, quiet=False)
            return {
                f"{field_name}_R":   vR,
                f"{field_name}_phi": vPhi,
                f"{field_name}_Z":   vZ
            }
        else:
            values = eval_field(field_name, R, phi, Z, coord='scalar', sim=sim,
                                filename=filename, time=time, quiet=False)
            return values

    def _evaluate_rectangular_grid(self, field_name, grid_spec, sim, units, is_3d=False):
        """
        Evaluates the field on a regular rectangular grid.
        """
        points = self._get_inner_wall_points()
        if len(points) > 0:
            rmin, rmax = np.min(points[:, 0]), np.max(points[:, 0])
            zmin, zmax = np.min(points[:, 1]), np.max(points[:, 1])
        else:
            rmin, rmax, zmin, zmax = 1.0, 2.0, -1.0, 1.0

        if is_3d:
            nR, nPhi, nZ = grid_spec.resolution
            R_lin   = np.linspace(rmin, rmax, nR)
            Phi_lin = np.linspace(0, 2*np.pi, nPhi, endpoint=False)
            Z_lin   = np.linspace(zmin, zmax, nZ)
            R, phi, Z = np.meshgrid(R_lin, Phi_lin, Z_lin, indexing='ij')
            R   = R.ravel()
            phi = phi.ravel()
            Z   = Z.ravel()
        else:
            nR, nZ = grid_spec.resolution
            R_lin = np.linspace(rmin, rmax, nR)
            Z_lin = np.linspace(zmin, zmax, nZ)
            R, Z  = np.meshgrid(R_lin, Z_lin, indexing='ij')
            R   = R.ravel()
            Z   = Z.ravel()
            phi = np.zeros_like(R)

        self._sync_timeslice(sim)
        time     = self._sim_time(sim)
        filename = self._h5_filename(sim)

        field_info = getattr(sim, 'typedict', getattr(sim, 'available_fields', {})).get(field_name, None)
        field_type = field_info[1] if field_info else 'scalar'

        if field_type == 'vector':
            vR, vPhi, vZ = eval_field(field_name, R, phi, Z, coord='vector', sim=sim,
                                      filename=filename, time=time, quiet=True)
            return {
                f"{field_name}_R":   vR,
                f"{field_name}_phi": vPhi,
                f"{field_name}_Z":   vZ
            }
        else:
            values = eval_field(field_name, R, phi, Z, coord='scalar', sim=sim,
                                filename=filename, time=time, quiet=True)
            return values

    def _evaluate_native_grid(self, field_name, sim, units):
        return np.zeros((100,))
