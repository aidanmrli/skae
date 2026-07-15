"""Guard the boundary between submitted SLURM wrappers and task payloads."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

WORKER_PAIRS = (
    (
        "common/run_benchmark_array.sh",
        "common/run_benchmark_packed_array.sh",
        "common/run_benchmark_task.sh",
    ),
    (
        "neurips_2026/dysts/run_evaluation_array.sh",
        "neurips_2026/dysts/run_evaluation_packed_array.sh",
        "neurips_2026/dysts/run_evaluation_task.sh",
    ),
)


def test_packed_workers_call_allocation_free_payloads() -> None:
    for array_name, packed_name, payload_name in WORKER_PAIRS:
        array_text = (SCRIPTS_DIR / array_name).read_text()
        packed_text = (SCRIPTS_DIR / packed_name).read_text()
        payload_text = (SCRIPTS_DIR / payload_name).read_text()

        assert "#SBATCH" in array_text
        assert "#SBATCH" in packed_text
        assert not any(
            line.startswith("#SBATCH") for line in payload_text.splitlines()
        )
        assert f"bash scripts/{payload_name}" in packed_text
        assert f"bash scripts/{array_name}" not in packed_text


def test_controlled_alignment_queue_routes_the_exact_six_paper_rows() -> None:
    text = (
        SCRIPTS_DIR / "neurips_2026/controlled/queue_alignment.sh"
    ).read_text()
    assert 'RESULTS_ROOT="${RESULTS_ROOT:-${SKAE_SCRATCH_ROOT}/results}"' in text
    assert "source scripts/common/cluster_env.sh" in text
    assignments = {}
    for line in text.splitlines():
        if "_ROOTS_CSV=" not in line:
            continue
        name, raw = line.split("=", 1)
        assignments[name] = raw.strip('"').split(",")
    routed = [item for values in assignments.values() for item in values]
    assert routed == [
        "lista_blockdiag_signsplit_hardinit_basin_partition",
        "mlp_sparse_hardinit_basin_partition_control",
        "mlp_zero_sparse_hardinit_basin_partition_control",
        "mlp_sparse_blockdiag_hardinit_basin_partition_control",
        "lista_dense_softblock_signsplit_p256_hardinit_basin_partition",
        "lista_dense_signsplit_p256_hardinit_basin_partition",
    ]
    assert len(routed) == len(set(routed)) == 6
    assert "transition_rich_sparse_mlp_bd_repaired_table1_20260506" in text


def test_coordinate_intervention_gpu_job_uses_guard_and_telemetry() -> None:
    text = (SCRIPTS_DIR / "neurips_2026/interventions/run.sh").read_text()
    assert "#SBATCH --gres=gpu:1" in text
    assert "source scripts/common/gpu_guard.sh" in text
    assert "module load cuda/12.6.0" in text
    assert "gpu_guard_assert_cuda_visible" in text
    assert "gpu_guard_start_sampler" in text
    assert "gpu_guard_stop_sampler" in text
