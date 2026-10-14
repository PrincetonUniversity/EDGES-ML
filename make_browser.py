#!/usr/bin/env python3
"""
make_browser.py  –  manual generator for the edges-ml simulation browser.

Run this script whenever new simulation data has been added to the directories
you want to expose.  It crawls the directory trees, reads the code input
namelists, works out which physical model each simulation uses, picks up the
M3D-C1 growth-rate tables, and writes a single self-contained HTML file that
can be opened directly in a browser (or copied to a web server).

The embedded data is packed and gzip-compressed, and the namelists / growth-rate
tables are only parsed in the browser when you actually open them, so the page
stays small and responsive even for large databases.

Tabs in the generated page
--------------------------
    Overview      metadata, physical model, contents, directory listing
    Inputs        C1input (M3D-C1), nimrod.in / nimeq.in (NIMROD), README...
    Growth rates  the M3D-C1 'gamma' output table (when present)
    Plots         figures (only with --scan-plots)

Examples
--------
    # Crawl one root directory for both M3D-C1 and NIMROD runs:
    python make_browser.py /scratch/mastu_runs -o simulations.html

    # Separate roots per code, custom title, open the result afterwards:
    python make_browser.py --m3dc1 /data/m3dc1 --nimrod /data/nimrod \
        --title "MAST-U pedestal database" --open

    # Also link figures that already exist next to the simulations:
    python make_browser.py /scratch/mastu_runs --scan-plots

    # Smaller page: no per-directory file listings, shorter input previews:
    python make_browser.py /scratch/mastu_runs --max-files 0 --max-doc-bytes 32768

    # Plain (uncompressed) payload for very old browsers:
    python make_browser.py /scratch/mastu_runs --no-compress

Run 'python make_browser.py --help' for the full list of options.
"""

import sys
from pathlib import Path

try:
    from edges_ml.browser import main
except ImportError:  # running from a source checkout without installation
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from edges_ml.browser import main


if __name__ == "__main__":
    raise SystemExit(main())
