"""Write the immutable dependency-chain record after SLURM submission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.neurips_2026.local_edmd_reproduction.contract import (
    CARD_PATH,
    LOCK_PATH,
    PROTOCOL_ID,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue-job-id", required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--collect-job-id", required=True)
    parser.add_argument("--check-job-id", required=True)
    parser.add_argument("--task-tsv", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "queue_job_id": args.queue_job_id,
        "array_job_id": args.array_job_id,
        "collect_job_id": args.collect_job_id,
        "check_job_id": args.check_job_id,
        "dependencies": {
            "collect": f"afterok:{args.array_job_id}",
            "check": f"afterok:{args.collect_job_id}",
        },
        "array_spec": "0-74%32",
        "compute": {
            "partition": "long",
            "cpus_per_task": 4,
            "memory": "8G",
            "gpus": 0,
        },
        "task_tsv": str(args.task_tsv),
        "task_tsv_sha256": hashlib.sha256(args.task_tsv.read_bytes()).hexdigest(),
        "result_root": str(args.result_root),
        "card_sha256": hashlib.sha256(CARD_PATH.read_bytes()).hexdigest(),
        "source_lock_sha256": hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

