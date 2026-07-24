"""Run the frozen scientific auditor with V5's telemetry policy."""

from experiments.neurips_2026.allen_cahn_periodic_reencoding import telemetry
from experiments.neurips_2026.allen_cahn_periodic_reencoding_v5.telemetry_policy import (
    install,
)


def main() -> None:
    install()
    telemetry.main()


if __name__ == "__main__":
    main()
