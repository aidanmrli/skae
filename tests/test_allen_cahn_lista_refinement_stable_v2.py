import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_batch48_card_freezes_pairing_and_batches():
    card = json.loads(
        (
            ROOT
            / "experiments/neurips_2026/allen_cahn_lista_refinement_stable_v2/prediction_card.json"
        ).read_text()
    )
    assert card["protocol_id"] == "allen_cahn_lista_refinement_stable_v3_batch48"
    assert card["arms"] == {"refinements": [2, 3], "seeds": [64, 65], "fits": 4}
    assert card["optimization"]["shared_pretrain_batch"] == 32
    assert card["optimization"]["forecast_batch"] == 48
    assert card["compute"]["telemetry_seconds"] == 10
    assert "unchanged" in card["compute"]["batch48_rationale"]
    predecessor = "\n".join(card["invalid_predecessors"])
    assert "10173588" in predecessor
    assert "71.640%" in predecessor
    assert "no stability-gate or forecast outcome" in predecessor


def test_launchers_use_explicit_warm_start_and_no_branch_pretraining():
    for name in ("smoke.sh", "run_array.sh"):
        text = (
            ROOT
            / f"scripts/neurips_2026/allen_cahn_lista_refinement_stable_v2/{name}"
        ).read_text()
        assert "--warm_start_pretrain_checkpoint" in text
        assert "--warm_start_pretrain_steps 2000" in text
        assert "--num_steps 0 --pretrain_steps 2000 --batch_size 32" in text
        assert "--pretrain_steps 0 --batch_size 48" in text
        assert "--pretrain_steps 0 --batch_size 16" not in text
        assert 'PINNED_SOURCE_DIR="/network/scratch/l/lia/skae-rebuttal"' in text


def test_launchers_use_fresh_v3_batch48_outputs_and_job_names():
    script_dir = ROOT / "scripts/neurips_2026/allen_cahn_lista_refinement_stable_v2"
    smoke = (script_dir / "smoke.sh").read_text()
    run_array = (script_dir / "run_array.sh").read_text()
    select = (script_dir / "select.sh").read_text()
    queue = (script_dir / "queue.sh").read_text()
    assert "smoke-ac-lista-v3b48" in smoke
    assert "ac-lista-v3b48" in run_array
    assert "ac-lista-v3b48-select" in select
    assert "queue-ac-lista-v3b48" in queue
    assert "allen_cahn_lista_refinement_stable_smoke_20260722_v3_b48" in smoke
    assert "allen_cahn_lista_refinement_stable_smoke_20260722_v3_b48" in run_array
    assert "allen_cahn_lista_refinement_stable_20260722_v3_b48" in run_array
    assert "allen_cahn_lista_refinement_stable_20260722_v3_b48" in select


def test_production_launcher_refuses_existing_branch_outputs():
    text = (
        ROOT
        / "scripts/neurips_2026/allen_cahn_lista_refinement_stable_v2/run_array.sh"
    ).read_text()
    assert 'for DEPTH in 2 3; do' in text
    assert 'Refusing to overwrite ${BRANCH_ROOT}' in text


def test_pair_validators_require_final_generator_identity():
    for relative in (
        "experiments/neurips_2026/allen_cahn_lista_refinement_stable/validate_smoke.py",
        "experiments/neurips_2026/allen_cahn_lista_refinement_stable_v2/validate_pair.py",
    ):
        assert "final_training_generator_sha256" in (ROOT / relative).read_text()


def test_spatialized_latent_is_four_times_overcomplete():
    card = json.loads(
        (
            ROOT
            / "experiments/neurips_2026/allen_cahn_lista_refinement_stable_v2/prediction_card.json"
        ).read_text()
    )
    assert card["system"]["latent_dim"] >= 4 * card["system"]["state_dim"]
