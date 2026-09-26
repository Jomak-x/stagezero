"""Review exact saved ARDY Core clips with NVIDIA's CoreSkin renderer.

No model is loaded and no inference occurs. The displayed four-frame prefix
and 80 generated frames are the original arrays from core_motion_probe.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core_motion_probe import serve_review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--port", type=int, default=2340)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    serve_review(args.directory, args.port)


if __name__ == "__main__":
    main()
