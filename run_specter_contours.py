#!/usr/bin/env python3
"""Build all ``contours.py`` figures using SPECTER ``w_mu_inv``; writes under ``SPECTER_results/contours/``."""

from __future__ import annotations

from pathlib import Path

import contours


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "SPECTER_results" / "contours"
    contours.main(["--specter", "--output-dir", str(out)])


if __name__ == "__main__":
    main()
