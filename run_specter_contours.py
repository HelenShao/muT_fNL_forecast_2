#!/usr/bin/env python3
"""Build all ``contours.py`` figures using SPECTER ``w_mu_inv``; writes under ``cmbs4/results/contours_specter/``."""

from __future__ import annotations

import contours
from output_paths import contours_specter_dir, ensure_dir


def main() -> None:
    out = ensure_dir(contours_specter_dir())
    contours.main(["--specter", "--output-dir", str(out)])


if __name__ == "__main__":
    main()
