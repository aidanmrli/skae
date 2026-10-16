"""``python -m paper data``: build the Dysts trajectory caches.

Writes ``<data>/dysts/<System>_<split>.pt`` for every requested system and
split. Existing files are skipped, so rerunning the command resumes an
interrupted build.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from paper.systems import DYSTS_SYSTEMS

SPLITS = ("train", "val", "test")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m paper data", description="Build the Dysts trajectory caches.")
    parser.add_argument("--systems", nargs="+", choices=DYSTS_SYSTEMS, default=list(DYSTS_SYSTEMS), metavar="SYSTEM",
                        help="Dysts systems (default: all ten)")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS),
                        help="splits (default: train val test)")
    parser.add_argument("--data", type=Path, default=Path("data"), help="data directory (default: data)")
    parser.add_argument("--workers", type=int, default=1, help="parallel integration processes (default: 1)")
    args = parser.parse_args(argv)
    from paper.systems.dysts import NUM_TRAJECTORIES, cache_path, write_split  # after parsing: --help stays fast

    for name in args.systems:
        for split in args.splits:
            path = cache_path(args.data, name, split)
            if path.exists():
                print(f"{name:18s} {split:5s} exists: {path}", flush=True)
                continue
            start = time.perf_counter()
            streams = write_split(args.data, name, split, args.workers)
            rejected = max(streams) + 1 - NUM_TRAJECTORIES
            print(f"{name:18s} {split:5s} {time.perf_counter() - start:7.1f} s  {rejected} rejected draws  {path}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
