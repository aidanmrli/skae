"""Run the frozen smoke auditor with V5's 60-second telemetry policy."""

from experiments.neurips_2026.allen_cahn_periodic_reencoding import smoke_audit
from experiments.neurips_2026.allen_cahn_periodic_reencoding_v5.telemetry_policy import (
    install,
)


def main() -> None:
    install()
    smoke_audit.main()


if __name__ == "__main__":
    main()
