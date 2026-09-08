"""Regenerate locations.csv, temporal_metadata.csv, dataset_statistics.csv."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from india_multitemporal_sentinel2.metadata_builder import regenerate_all  # noqa: E402


def main() -> int:
    summary = regenerate_all()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
