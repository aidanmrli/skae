import json
from pathlib import Path

from experiments.neurips_2026.dysts_corrected_dt30_full_v2.adjudicate_smoke import (
    _validate_final_metrics,
    _validate_refinement_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_repaired_queue_passes_isolated_cache_to_both_training_stages():
    script = (
        ROOT
        / "scripts/neurips_2026/dysts_corrected_dt30_full_v2/queue.sh"
    ).read_text()
    assert script.count('DYSTS_CACHE_DIR="${CACHE_DIR}"') >= 4
    assert 'SPLITS="train val policy test"' in script
    assert 'PACK_CONCURRENCY=12' in script
    assert 'REQUIRE_COMPLETE_COVERAGE=1 EXPECTED_TASK_COUNT=900' in script


def test_sakarya_restart_supports_uniform_explicit_solver_policy():
    cache_script = (
        ROOT / "scripts/neurips_2026/dysts/prebuild_cache.sh"
    ).read_text()
    systems = (
        ROOT
        / "experiments/neurips_2026/dysts_corrected_dt30_full_v2/sakarya_systems.txt"
    ).read_text().splitlines()

    assert systems == ["Sakarya"]
    assert '--primary_method "${CACHE_PRIMARY_METHOD}"' in cache_script
    assert (
        '--trajectory_timeout_seconds "${CACHE_TRAJECTORY_TIMEOUT_SECONDS}"'
        in cache_script
    )
    assert '--timeout_fallback_method "${CACHE_TIMEOUT_FALLBACK_METHOD}"' in cache_script
    assert '--fallback_timeout_seconds "${CACHE_FALLBACK_TIMEOUT_SECONDS}"' in cache_script


def test_prediction_card_freezes_matched_refinement_and_direct_endpoint():
    card = json.loads(
        (
            ROOT
            / "experiments/neurips_2026/dysts_corrected_dt30_full_v2/prediction_card.json"
        ).read_text()
    )
    assert card["roster"]["expected_fits"] == 900
    assert card["training"]["lista_refinements"] == 1
    assert card["evaluation"]["primary"].startswith("direct repeated-K strict")
    assert card["training"]["dense_contract"].startswith("tanh hidden")
    assert card["cache_restart"]["primary_method"] == "DOP853"
    assert card["cache_restart"]["fallback_method"] == ""
    assert card["cache_restart"]["all_trajectories_use_one_solver"] is True
    assert card["cache_restart"]["numerical_gate"]["status"] == "passed"


def test_dense_generic_model_is_not_mistaken_for_lista_refinement_model():
    config = {
        "MODEL": {
            "MODEL_NAME": "GenericKM",
            "ENCODER": {
                # This legacy/default field is unused by GenericKM.  The smoke
                # gate must classify the instantiated model via MODEL_NAME.
                "ENCODER_TYPE": "lista",
                "LISTA": {"NUM_LOOPS": 5},
            },
        }
    }
    _validate_refinement_contract(config, Path("dense_mlp_tanh/run"))


def test_actual_lista_model_requires_exactly_one_refinement():
    config = {
        "MODEL": {
            "MODEL_NAME": "LISTAKM",
            "ENCODER": {
                "ENCODER_TYPE": "lista",
                "LISTA": {"NUM_LOOPS": 2},
            },
        }
    }
    try:
        _validate_refinement_contract(config, Path("lista/run"))
    except ValueError as error:
        assert "wrong LISTA refinement count" in str(error)
    else:
        raise AssertionError("LISTAKM with two refinements must fail closed")


def test_final_metrics_allow_text_metadata_but_reject_nonfinite_scalars():
    metrics = {
        "loss": 1.0,
        "alignment_loss": 0.5,
        "reconst_loss": 0.2,
        "prediction_loss": 0.1,
        "sparsity_loss": 0.3,
        "sparsity_ratio": 0.4,
        "obs_loss_dim_normalization": "sqrt_dim",
    }
    _validate_final_metrics(metrics, Path("run"))
    metrics["prediction_loss"] = float("inf")
    try:
        _validate_final_metrics(metrics, Path("run"))
    except ValueError as error:
        assert "nonfinite final metric prediction_loss" in str(error)
    else:
        raise AssertionError("nonfinite scalar metric must fail closed")
