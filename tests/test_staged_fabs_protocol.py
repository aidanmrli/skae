from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import re

import numpy as np
import pytest
import torch

from skae.config import get_config
from skae.data import VectorWrapper
from tools.train_staged_support_family_local_k import (
    FINAL_EVALUATION_BATCH_SIZE,
    FINAL_EVALUATION_SEED_OFFSET,
    FAMILY_JACCARD_THRESHOLD,
    FIT_CONFIGURED_ROWS,
    FIT_DUPLICATION_FACTOR,
    FIT_NUM_TRAJECTORIES,
    FIT_SEED_OFFSET,
    FIT_STATES,
    FIT_TRAJECTORY_LENGTH,
    FIT_TRANSITIONS,
    FIT_UNIQUE_TRAJECTORIES,
    LOCAL_MAP_PARAMETERIZATION,
    PAPER_REENCODE_PERIODS,
    ROUTE_PROTOCOL,
    STAGE1_TRAINING_STEPS,
    STAGE2_SELECTION_BATCH_SIZE,
    STAGE2_SELECTION_CANDIDATE_STEPS,
    STAGE2_SELECTION_HORIZONS,
    STAGE2_SELECTION_SEED_OFFSET,
    STAGE2_TRAINING_STEPS,
    SUPPORT_DEFINITION,
    TOTAL_TRAINING_STEPS,
    SourceTargetLocalMapBundle,
    _build_parser,
    _make_eval_settings,
    _make_stage2_selection_starts,
    _paper_route_metadata,
    _strictly_improves,
    _validate_frozen_fabs_artifact,
)
from tools.staged_fabs_protocol import _finite_prefix_start_mean
from tools.staged_fabs_io import (
    _restore_training_rng_states,
    _save_checkpoint,
)
from tools.train_support_family_local_maps import (
    FAMILY_CLUSTERING_RULE,
    FAMILY_REPRESENTATIVE_RULE,
    MIN_FAMILY_TRANSITIONS,
    _build_route_codebook,
    _generate_source_route_fit_batches,
    _route_indices_np,
)


ROOT = Path(__file__).resolve().parents[1]


def _two_family_latents() -> np.ndarray:
    return np.asarray(
        [
            [[0.01, 0.0, 0.0, 0.0]] * 3,
            [[0.0, 0.01, 0.0, 0.0]] * 3,
        ],
        dtype=np.float32,
    )


def _source_shape_codebook() -> dict[str, object]:
    codebook = _build_route_codebook(_two_family_latents())
    family_ids = codebook["fitted_family_ids"]
    codebook["support_mask"] = np.zeros((512, 193, 4), dtype=bool)
    codebook["family_counts"] = Counter(
        {family_ids[0]: 49_152, family_ids[1]: 49_152}
    )
    codebook["clustering_state_count"] = 98_816
    codebook["source_transition_count"] = 98_304
    return codebook


def test_paper_protocol_constants_are_frozen() -> None:
    assert SUPPORT_DEFINITION == "absolute:0.001"
    assert FAMILY_JACCARD_THRESHOLD == pytest.approx(0.40)
    assert FIT_NUM_TRAJECTORIES == 512
    assert FIT_TRAJECTORY_LENGTH == 192
    assert FIT_CONFIGURED_ROWS == 512
    assert FIT_UNIQUE_TRAJECTORIES == 256
    assert FIT_DUPLICATION_FACTOR == 2
    assert FIT_TRANSITIONS == 192
    assert FIT_STATES == 193
    assert FIT_SEED_OFFSET == 271_828
    assert MIN_FAMILY_TRANSITIONS == 1
    assert TOTAL_TRAINING_STEPS == 200_000
    assert STAGE1_TRAINING_STEPS == STAGE2_TRAINING_STEPS == 100_000
    assert LOCAL_MAP_PARAMETERIZATION == "source_target_affine_learned_intercept"
    assert PAPER_REENCODE_PERIODS == (1, 2, 5, 10, 20, 25, 50, 100)
    assert STAGE2_SELECTION_HORIZONS == (100, 500, 1000)
    assert STAGE2_SELECTION_BATCH_SIZE == 32
    assert STAGE2_SELECTION_SEED_OFFSET == 12_345
    assert STAGE2_SELECTION_CANDIDATE_STEPS == (
        *range(100_500, 200_000, 500),
        199_999,
    )
    assert len(STAGE2_SELECTION_CANDIDATE_STEPS) == 200
    assert FINAL_EVALUATION_BATCH_SIZE == 100
    assert FINAL_EVALUATION_SEED_OFFSET == 12_345


def test_trainer_cli_has_no_route_ablation_surface() -> None:
    parser = _build_parser()
    destinations = {action.dest for action in parser._actions}
    removed = {
        "routing_object",
        "support_definition",
        "family_jaccard_threshold",
        "support_family_fit_source",
        "stable_base_object",
        "stable_fit_trajectories",
        "baseline_latent_cluster_count",
        "latent_fate_max_clusters",
        "local_map_parameterization",
        "local_lr",
        "stage2_selection_metric",
        "stage2_selection_periods",
        "stage2_selection_horizons",
        "stage2_selection_batch_size",
        "stage2_selection_seed_offset",
    }
    assert destinations.isdisjoint(removed)
    args = parser.parse_args(
        ["--task_tsv", "tasks.tsv", "--base_out", "/tmp/staged"]
    )
    assert args.eval_profile == "full"


def test_fabs_codebook_routes_seen_masks_and_falls_back_for_disjoint_masks() -> None:
    codebook = _build_route_codebook(_two_family_latents())
    assert codebook["routing_object"] == "support_family"
    assert len(codebook["fitted_family_ids"]) == 2

    family_to_index = {
        str(family_id): index
        for index, family_id in enumerate(codebook["fitted_family_ids"])
    }
    query = np.asarray(
        [
            [0.01, 0.0, 0.0, 0.0],
            [0.0, 0.01, 0.0, 0.0],
            [0.0, 0.0, 0.01, 0.0],
        ],
        dtype=np.float32,
    )
    routes = _route_indices_np(
        query,
        support_key_to_family=codebook["support_key_to_family"],
        family_prototypes=codebook["family_prototypes"],
        family_to_index=family_to_index,
        family_cache={},
    )
    assert routes[0] >= 0
    assert routes[1] >= 0
    assert routes[0] != routes[1]
    assert routes[2] == -1


def test_route_fit_explicitly_duplicates_one_unique_batch() -> None:
    class FakeFitEnv:
        batch_size = 256

        def __init__(self) -> None:
            self.calls = 0

        def generate_sequence_batch(
            self, rng: torch.Generator, window_length: int
        ) -> torch.Tensor:
            self.calls += 1
            assert rng.initial_seed() == 271_828
            assert window_length == 192
            values = torch.arange(256 * 193 * 2, dtype=torch.float32)
            return values.reshape(256, 193, 2)

    env = FakeFitEnv()
    batches = _generate_source_route_fit_batches(env, seed=271_828)
    assert env.calls == 1
    assert [tuple(batch.shape) for batch in batches] == [(256, 193, 2)] * 2
    assert batches[0].data_ptr() != batches[1].data_ptr()
    assert torch.equal(batches[0].view(torch.uint8), batches[1].view(torch.uint8))


def test_terminal_supports_form_families_but_modal_source_mask_routes() -> None:
    # A opens the family because three terminal A masks make it most frequent;
    # B is the modal source mask and must replace A as runtime representative.
    a = [0.01, 0.01, 0.0]
    b = [0.01, 0.0, 0.0]
    latents = np.asarray(
        [[b, b, a], [b, b, a], [a, a, a]], dtype=np.float32
    )
    codebook = _build_route_codebook(latents)
    assert len(codebook["fitted_family_ids"]) == 1
    family_id = codebook["fitted_family_ids"][0]
    assert codebook["family_representative_rule"] == FAMILY_REPRESENTATIVE_RULE
    assert codebook["family_clustering_rule"] == FAMILY_CLUSTERING_RULE
    assert codebook["clustering_state_count"] == 9
    assert codebook["source_transition_count"] == 6
    np.testing.assert_array_equal(
        codebook["family_prototypes"][family_id],
        np.asarray([True, False, False]),
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"scheme": "topk", "value": 8.0}, "absolute support"),
        ({"family_jaccard_threshold": 0.5}, "Jaccard threshold"),
        ({"min_operator_transitions": 2}, "at least one transition"),
    ],
)
def test_route_builder_rejects_nonpaper_protocols(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _build_route_codebook(_two_family_latents(), **kwargs)


def test_affine_bundle_starts_at_global_map_and_learns_intercept() -> None:
    global_k = np.asarray([[1.0, 2.0], [0.5, -1.0]], dtype=np.float32)
    source = {0: np.asarray([0.25, -0.5], dtype=np.float32)}
    target = {0: source[0] @ global_k}
    bundle = SourceTargetLocalMapBundle(
        family_ids=[0],
        source_centers=source,
        target_centers=target,
        global_k=global_k,
        device="cpu",
        learn_target_centers=True,
    )
    assert isinstance(bundle.target_centers, torch.nn.Parameter)

    z = torch.tensor([[1.0, 3.0], [-2.0, 0.25]])
    routed, used = bundle(z, torch.tensor([0, -1]))
    torch.testing.assert_close(routed, z @ torch.from_numpy(global_k))
    assert used.tolist() == [True, False]


def test_learned_bundle_loads_legacy_fixed_target_state() -> None:
    global_k = np.eye(2, dtype=np.float32)
    source = {0: np.asarray([1.0, -1.0], dtype=np.float32)}
    target = {0: source[0] @ global_k}
    legacy = SourceTargetLocalMapBundle(
        family_ids=[0],
        source_centers=source,
        target_centers=target,
        global_k=global_k,
        device="cpu",
        learn_target_centers=False,
    )
    learned = SourceTargetLocalMapBundle(
        family_ids=[0],
        source_centers=source,
        target_centers=target,
        global_k=global_k,
        device="cpu",
        learn_target_centers=True,
    )
    learned.load_state_dict(legacy.state_dict())
    torch.testing.assert_close(learned.target_centers, legacy.target_centers)


def test_route_metadata_and_evaluation_cadence_match_paper() -> None:
    codebook = _source_shape_codebook()
    metadata = _paper_route_metadata(codebook, fit_seed=271_828)
    assert metadata["fit_num_trajectories"] == 512
    assert metadata["fit_trajectory_length"] == 192
    assert metadata["protocol"] == ROUTE_PROTOCOL
    assert metadata["route_schema_version"] == 3
    assert metadata["fit_configured_rows"] == 512
    assert metadata["fit_unique_trajectories"] == 256
    assert metadata["fit_duplication_factor"] == 2
    assert metadata["fit_states"] == 193
    assert metadata["fit_transitions"] == 192
    assert metadata["fit_supports_considered"] == 98_816
    assert metadata["fit_source_transitions"] == 98_304
    assert metadata["fit_unique_source_transitions"] == 49_152
    assert metadata["family_representative_rule"] == "modal_source_support"
    assert metadata["routing_cadence"] == "every_latent_transition_step"
    assert "refreshes_latent" in metadata["reencoding_role"]
    assert metadata["learn_target_centers"] is True
    assert metadata["periodic_reencode_periods"] == list(PAPER_REENCODE_PERIODS)
    assert metadata["checkpoint_selection_candidate_steps"] == list(
        STAGE2_SELECTION_CANDIDATE_STEPS
    )

    settings = _make_eval_settings("full", get_config("generic_sparse"))
    assert settings.periodic_reencode_periods == PAPER_REENCODE_PERIODS
    assert settings.batch_size == 100
    assert settings.seed_offset == 12_345


def test_selector_reducer_overlap_and_strict_tie_rule() -> None:
    squared_error = torch.tensor(
        [[1.0, torch.nan, 4.0], [3.0, torch.nan, torch.inf], [torch.nan, torch.nan, 0.0]]
    )
    assert _finite_prefix_start_mean(squared_error, 3) == pytest.approx(2.0)
    assert _strictly_improves(0.9, 1.0)
    assert not _strictly_improves(1.0, 1.0)

    class SeedEchoEnv:
        def reset(self, rng: torch.Generator) -> torch.Tensor:
            return torch.tensor([rng.initial_seed()], dtype=torch.float64)

    base_env = SeedEchoEnv()
    _, selector = _make_stage2_selection_starts(
        base_env, seed=7, device="cpu"
    )
    reported = VectorWrapper(base_env, 100).reset(
        torch.Generator().manual_seed(7 + FINAL_EVALUATION_SEED_OFFSET)
    )
    torch.testing.assert_close(selector, reported[:32])


def test_resume_validation_accepts_legacy_fabs_and_rejects_new_512_unique(
    tmp_path: Path,
) -> None:
    codebook = _build_route_codebook(_two_family_latents())
    legacy_codebook = dict(codebook)
    legacy_codebook.update(
        {
            "stable_fit_trajectories": 512,
            "stable_fit_trajectory_length": 192,
            "stable_fit_seed": 271_828,
            "learn_target_centers": True,
            "local_map_parameterization": LOCAL_MAP_PARAMETERIZATION,
        }
    )
    legacy_metadata = {
        "fitted_family_ids": [
            str(item) for item in legacy_codebook["fitted_family_ids"]
        ],
        "family_class_count_fit": len(legacy_codebook["fitted_family_ids"]),
    }
    legacy_payload = {
        "route_codebook": legacy_codebook,
        "route_metadata": legacy_metadata,
    }
    legacy_path = tmp_path / "legacy_checkpoint.pt"
    torch.save(legacy_payload, legacy_path)
    loaded = torch.load(legacy_path, weights_only=False)
    _validate_frozen_fabs_artifact(
        loaded["route_codebook"],
        loaded["route_metadata"],
        expected_fit_seed=271_828,
    )
    mismatched_legacy = dict(legacy_codebook)
    mismatched_legacy["stable_fit_trajectories"] = 256
    with pytest.raises(ValueError, match="fit_num_trajectories=512"):
        _validate_frozen_fabs_artifact(mismatched_legacy, {})

    source_shape_codebook = _source_shape_codebook()
    metadata = _paper_route_metadata(source_shape_codebook, fit_seed=271_828)
    _validate_frozen_fabs_artifact(
        source_shape_codebook, metadata, expected_fit_seed=271_828
    )
    future_unique = dict(metadata)
    future_unique["fit_unique_trajectories"] = 512
    future_unique["fit_duplication_factor"] = 1
    with pytest.raises(ValueError, match="fit_unique_trajectories"):
        _validate_frozen_fabs_artifact(source_shape_codebook, future_unique)
    bad = dict(codebook)
    bad["routing_object"] = "oracle_basin"
    with pytest.raises(ValueError, match="Only frozen F_abs"):
        _validate_frozen_fabs_artifact(bad, {})


def test_launchers_expose_only_the_frozen_route_contract() -> None:
    queue_text = (ROOT / "scripts/queue_staged_support_family_local_k_table1.sh").read_text()
    runner_text = (ROOT / "scripts/run_staged_support_family_local_k_array.sh").read_text()
    combined = queue_text + runner_text
    for required in (
        "absolute:0.001",
        "512",
        "256",
        "193",
        "271_828",
        "192",
        "1,2,5,10,20,25,50,100",
        "source_target_affine_learned_intercept",
        "staged_fabs_route_source_v3",
        "every_latent_transition_step",
    ):
        assert required in combined
    for removed in (
        "ROUTING_OBJECT",
        "STABLE_BASE_OBJECT",
        "ORACLE_BASIN",
        "LATENT_FATE_MAX_CLUSTERS",
        "BASELINE_LATENT_CLUSTER_COUNT",
        "staged_fabs_route_source_v2",
    ):
        assert removed not in combined
    assert "build_transition_rich_basin_partition_tasks.py" in queue_text
    assert "--paper_protocol" in queue_text
    assert "SOURCE_TSV" not in queue_text
    heredocs = re.findall(r"<<'PY'\n(.*?)\nPY", combined, flags=re.DOTALL)
    assert len(heredocs) == 3
    for index, source in enumerate(heredocs):
        ast.parse(source, filename=f"staged-launcher-heredoc-{index}.py")


def test_staged_modules_respect_active_text_line_cap() -> None:
    paths = [
        ROOT / "tools/train_staged_support_family_local_k.py",
        ROOT / "tools/train_support_family_local_maps.py",
        *sorted((ROOT / "tools").glob("staged_fabs_*.py")),
        ROOT / "tools/reevaluate_staged_vs_global_wide_periodic.py",
    ]
    assert all(len(path.read_text().splitlines()) <= 500 for path in paths)


def test_best_and_last_stage2_checkpoints_are_resumable() -> None:
    source = (ROOT / "tools/staged_fabs_training.py").read_text()
    tree = ast.parse(source)
    checkpoint_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_save_checkpoint"
        and any(
            keyword.arg == "local_optimizer"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "optimizer"
            for keyword in node.keywords
        )
    ]
    assert len(checkpoint_calls) == 2
    for call in checkpoint_calls:
        include = next(
            keyword.value
            for keyword in call.keywords
            if keyword.arg == "include_optimizer_state"
        )
        assert isinstance(include, ast.Constant) and include.value is True
        assert any(
            keyword.arg == "training_generators"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "rngs"
            for keyword in call.keywords
        )


def test_schema3_checkpoint_restores_training_and_torch_rng(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    generators = [
        torch.Generator().manual_seed(10),
        torch.Generator().manual_seed(20),
    ]
    torch.manual_seed(30)
    path = tmp_path / "checkpoint.pt"
    _save_checkpoint(
        path,
        stage="stage1_joint",
        next_step=5,
        model=model,
        optimizer=None,
        bundle=None,
        local_optimizer=None,
        best_eval_final_error=float("inf"),
        metrics={},
        cfg=get_config("generic_sparse"),
        training_generators=generators,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["checkpoint_schema_version"] == 3
    expected_global = torch.rand(4)
    expected_generators = [
        torch.rand(4, generator=generator) for generator in generators
    ]

    torch.manual_seed(999)
    for index, generator in enumerate(generators):
        generator.manual_seed(1_000 + index)
    assert _restore_training_rng_states(payload, generators)
    torch.testing.assert_close(torch.rand(4), expected_global)
    for generator, expected in zip(generators, expected_generators):
        torch.testing.assert_close(
            torch.rand(4, generator=generator),
            expected,
        )
