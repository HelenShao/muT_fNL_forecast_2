#!/usr/bin/env python3
"""Build all ``contours.py`` figures using SPECTER ``w_mu_inv``; writes under ``cmbs4/results/contours_specter/``."""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))


import contours
from output_paths import contours_specter_dir, ensure_dir


def main():
    out = ensure_dir(contours_specter_dir())
    contours.main(["--specter", "--output-dir", str(out)])


if __name__ == "__main__":
    main()
