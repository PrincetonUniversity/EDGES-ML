"""
ADIOS2 campaign-archive (.aca) support for edges-ml.

This module lets edges-ml stream M3D-C1 output that physically lives on a
remote cluster instead of crawling a local file system.  The set of available
simulations is taken from an HPC campaign archive (a small SQLite file with the
'.aca' extension); the actual field data is pulled on demand by fusion-io's
fpy.sim_data(..., filetype='adios2') backend over a persistent SSH connection.

A "campaign source" is declared exactly like a normal source dictionary, except
that the local 'directories' are replaced by the archive and (optionally) by
path prefixes *inside* that archive:

    {
        "code": "m3dc1",
        "campaign_archive": "mastu_45272_vped477.aca",
        "login": "user@perlmutter.nersc.gov",

        # Optional: restrict the crawl to these prefixes inside the archive.
        "directories": ["99", "102"],

        # Optional: only keep model directories with this name.
        "model": "1f_eqrotnc-C_eta_x1",

        # Optional: bypass the discovery entirely and give explicit simulations.
        "simulations": ["99/1f_eqrotnc-C_eta_x1/n40"],

        # Optional fusion-io streaming options.
        "cache_dir":     ".cache/aca",
        "remote_python": "python3",
        "remote_setup":  "module load python",
        "verbose":       True,
    }

Directory convention
--------------------
The archive registers one dataset per HDF5 file, with a name that is the path
relative to the campaign root, e.g.

    99/1f_eqrotnc-C_eta_x1/n40/C1.h5
    99/1f_eqrotnc-C_eta_x1/n40/equilibrium.h5
    99/1f_eqrotnc-C_eta_x1/n40/time_002.h5

Mirroring the local crawler, the directory that contains 'C1.h5' is treated as
a *mode* directory and its parent as the *model* directory.
"""

import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from .utils import printwarn, printerr, printnote


# ===========================================================================
# DEFAULTS
# ===========================================================================

DEFAULT_CACHE_DIR     = ".cache/aca"
DEFAULT_REMOTE_PYTHON = "python3"
DEFAULT_REMOTE_SETUP  = "module load python >/dev/null 2>&1 || true"

# Keys accepted inside a source dictionary to point at a campaign archive.
_ARCHIVE_KEYS = ("campaign_archive", "archive", "aca", "aca_file")

# Keys accepted to list explicit simulations inside the archive.
_SIMULATION_KEYS = ("simulations", "campaign_simulations", "sims")

# Keys accepted to restrict the crawl to a subset of the archive.
_PREFIX_KEYS = ("directories", "prefixes", "campaign_prefixes")


# ===========================================================================
# CONTEXT
# ===========================================================================

@dataclass(frozen=True)
class CampaignContext:
    """
    Everything that is needed to open a remote simulation through fusion-io.

    Attributes
    ----------
    archive : str
        Absolute path to the .aca campaign archive on the local machine.
    login : str
        SSH login string ('user@host').  Empty means 'the data is reachable
        locally', in which case fusion-io runs the helper scripts in a shell.
    simulation : str
        Path of the *model* inside the archive (e.g. '99/1f_eqrotnc-C_eta_x1').
    modes : tuple of str
        Paths of the mode directories belonging to this model.
    """
    archive: str
    login: str = ""
    simulation: str = ""
    modes: tuple = ()
    cache_dir: str = DEFAULT_CACHE_DIR
    remote_python: str = DEFAULT_REMOTE_PYTHON
    remote_setup: str = DEFAULT_REMOTE_SETUP
    verbose: bool = False

    # -- convenience ----------------------------------------------------
    @property
    def archive_name(self):
        """Stem of the archive file; used to namespace output filenames."""
        return Path(self.archive).stem

    def with_simulation(self, simulation, modes=()):
        """Returns a copy bound to one model directory and its mode list."""
        return replace(
            self,
            simulation=str(simulation).strip("/"),
            modes=tuple(str(m).strip("/") for m in modes),
        )

    def sim_data_kwargs(self, simulation=None, time=-1):
        """
        Builds the keyword arguments for fpy.sim_data in adios2 mode.
        """
        sim = simulation if simulation is not None else self.simulation
        return {
            "filename":            self.archive,
            "filetype":            "adios2",
            "campaign_simulation": str(sim).strip("/"),
            "time":                time,
            "login":               self.login,
            "cache_dir":           self.cache_dir,
            "remote_python":       self.remote_python,
            "remote_setup":        self.remote_setup,
            "verbose":             self.verbose,
        }

    def describe(self):
        return (f"{self.archive_name}:{self.simulation}"
                + (f" @ {self.login}" if self.login else " (local)"))


# ===========================================================================
# ARCHIVE READER
# ===========================================================================

def _matches_any_prefix(path, prefixes):
    """
    True if *path* starts with (or contains) any of the given path prefixes.

    Prefixes are matched on whole path components, so '99' matches
    '99/1f_x/n40' and 'mastu/99/1f_x/n40' but not '991/...'.
    """
    if not prefixes:
        return True
    padded = "/" + str(path).strip("/") + "/"
    for pref in prefixes:
        pref = "/" + str(pref).strip("/") + "/"
        if pref == "//":
            return True
        if padded.startswith(pref) or pref in padded:
            return True
    return False


class CampaignArchive:
    """
    Read-only accessor for the dataset catalogue stored in an .aca file.

    Only the SQLite metadata is touched here; no remote connection is opened.
    Instances are cached per absolute filename.
    """

    _CACHE = {}

    def __init__(self, filename):
        self.filename = str(Path(str(filename)).expanduser().resolve())
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(
                f"Campaign archive not found: {self.filename}")
        self._datasets = None

    # -- construction ---------------------------------------------------
    @classmethod
    def open(cls, filename):
        key = str(Path(str(filename)).expanduser().resolve())
        if key not in cls._CACHE:
            cls._CACHE[key] = cls(key)
        return cls._CACHE[key]

    # -- catalogue ------------------------------------------------------
    @property
    def datasets(self):
        if self._datasets is None:
            self._datasets = self._load_datasets()
        return self._datasets

    def _load_datasets(self):
        out = []
        con = sqlite3.connect("file:{}?mode=ro".format(self.filename), uri=True)
        try:
            cur = con.cursor()

            try:
                rows = list(cur.execute(
                    "SELECT rowid, name, uuid, fileformat, deltime FROM dataset"))
            except sqlite3.Error:
                rows = [(r[0], r[1], None, None, None)
                        for r in cur.execute("SELECT rowid, name FROM dataset")]

            sizes = {}
            try:
                for dsid, size, deltime in cur.execute(
                        "SELECT datasetid, size, deltime FROM replica"):
                    if deltime:
                        continue
                    sizes[dsid] = int(size or 0)
            except sqlite3.Error:
                pass

            for rowid, name, uuid, fmt, deltime in rows:
                if deltime:
                    continue
                if not name:
                    continue
                out.append({
                    "name":       str(name).strip("/"),
                    "uuid":       str(uuid or ""),
                    "fileformat": str(fmt or ""),
                    "size":       sizes.get(rowid, 0),
                })
        finally:
            con.close()
        return out

    # -- queries --------------------------------------------------------
    def list_simulations(self, basename="C1.h5"):
        """All simulation paths (directories) that contain *basename*."""
        sims = set()
        for ds in self.datasets:
            p = PurePosixPath(ds["name"])
            if p.name == basename:
                sims.add(str(p.parent).strip("/"))
        return sorted(sims)

    def group_models(self, prefixes=None, models=None, basename="C1.h5"):
        """
        Groups the registered simulations into {model_path: [mode_path, ...]}.

        prefixes : iterable of str or None
            Restrict the crawl to these path prefixes inside the archive.
        models : iterable of str or None
            Keep only models whose directory name (or full path) is listed.
        """
        groups = defaultdict(list)

        for ds in self.datasets:
            p = PurePosixPath(ds["name"])
            if p.name != basename:
                continue

            mode_path  = str(p.parent).strip("/")
            model_path = str(p.parent.parent).strip("/")
            if mode_path in ("", "."):
                continue
            if model_path in ("", "."):
                # Flat layout: the simulation is its own model.
                model_path = mode_path

            if not _matches_any_prefix(mode_path, prefixes):
                continue

            if models:
                model_name = PurePosixPath(model_path).name
                if model_name not in models and model_path not in models:
                    continue

            if mode_path not in groups[model_path]:
                groups[model_path].append(mode_path)

        return {k: sorted(v) for k, v in groups.items()}

    def files_for(self, sim_path):
        """
        Returns {basename: {'uuid': ..., 'size': ...}} for all datasets
        registered directly inside *sim_path*.
        """
        sim_path = str(sim_path).strip("/")
        out = {}
        for ds in self.datasets:
            p = PurePosixPath(ds["name"])
            if str(p.parent).strip("/") != sim_path:
                continue
            out[p.name] = {"uuid": ds["uuid"], "size": ds["size"]}
        return out


# ===========================================================================
# SOURCE-DICTIONARY HELPERS
# ===========================================================================

def archive_path_from_source(source):
    """Returns the archive path declared by *source*, or None."""
    for key in _ARCHIVE_KEYS:
        val = source.get(key)
        if val:
            return str(Path(str(val)).expanduser())
    return None


def is_campaign_source(source):
    """True if the source dictionary points at a campaign archive."""
    return archive_path_from_source(source) is not None


def context_from_source(source):
    """Builds a CampaignContext from a source dictionary."""
    archive = archive_path_from_source(source)
    if archive is None:
        raise ValueError("Source does not define a campaign archive.")

    return CampaignContext(
        archive=str(Path(archive).expanduser().resolve()),
        login=str(source.get("login", "") or ""),
        cache_dir=str(source.get("cache_dir", DEFAULT_CACHE_DIR)),
        remote_python=str(source.get("remote_python", DEFAULT_REMOTE_PYTHON)),
        remote_setup=str(source.get("remote_setup", DEFAULT_REMOTE_SETUP)),
        verbose=bool(source.get("verbose", False)),
    )


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def group_campaign_source(source, context=None):
    """
    Discovers the simulations declared by a campaign source.

    Returns a sorted list of (model_path, [mode_path, ...]) tuples, with all
    paths given relative to the campaign root (i.e. ready to be handed to
    fpy.sim_data(campaign_simulation=...)).
    """
    context = context or context_from_source(source)
    archive = CampaignArchive.open(context.archive)

    explicit = None
    for key in _SIMULATION_KEYS:
        if source.get(key):
            explicit = _as_list(source.get(key))
            break

    if explicit:
        groups = defaultdict(list)
        for sim in explicit:
            sim = str(sim).strip("/")
            parent = str(PurePosixPath(sim).parent).strip("/")
            if parent in ("", "."):
                parent = sim
            if sim not in groups[parent]:
                groups[parent].append(sim)
        entries = {k: sorted(v) for k, v in groups.items()}
    else:
        prefixes = None
        for key in _PREFIX_KEYS:
            if source.get(key):
                prefixes = _as_list(source.get(key))
                break
        models = _as_list(source.get("model"))
        entries = archive.group_models(prefixes=prefixes, models=models)

    if not entries:
        printwarn(
            f"No simulations matched in campaign archive {context.archive}. "
            f"Registered simulations: {archive.list_simulations()[:20]}"
        )
    elif context.verbose:
        printnote(
            f"Campaign archive {Path(context.archive).name}: "
            f"{len(entries)} model(s), "
            f"{sum(len(v) for v in entries.values())} simulation(s)."
        )

    return sorted(entries.items())
