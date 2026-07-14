"""Contract tests for frozen local-map evidence and generated TeX tables."""

import hashlib

from tools.build_local_map_forecasting_tables import (
    DATA_PATH,
    PATH_COLUMNS,
    TABLE_PATH,
    build_table,
    load_rows,
    verify_provenance,
)


def test_sanitized_local_map_rows_have_complete_path_free_paper_matrix():
    provenance = verify_provenance()
    rows = load_rows()
    assert len(rows) == 225
    assert all(column not in rows[0] for column in PATH_COLUMNS)
    assert "/network/" not in DATA_PATH.read_text()
    assert "/home/" not in DATA_PATH.read_text()
    assert provenance["dataset"]["row_count"] == 225
    assert provenance["schema_version"] == 3
    route_fit = provenance["training_contract"]["staged_fabs"]["route_fit"]
    assert route_fit["configured_rows"] == 512
    assert route_fit["unique_trajectories"] == 256
    assert route_fit["duplication_factor"] == 2
    assert route_fit["states_per_trajectory"] == 193
    assert route_fit["transitions_per_trajectory"] == 192
    assert route_fit["supports_clustered"] == 98_816
    assert route_fit["map_fit_source_transitions"] == 98_304
    assert route_fit["unique_map_fit_source_transitions"] == 49_152
    assert route_fit["family_representative"] == "modal source-state support mask"

    selection = provenance["checkpoint_selection_contract"]
    assert selection["selector_is_asymmetric"] is True
    assert selection["staged_fabs"]["candidate_steps"]["count"] == 200
    assert selection["staged_fabs"]["starts"] == 32
    assert selection["staged_fabs"]["seed_offset"] == 12_345
    assert selection["global_k"]["starts"] == 16
    assert selection["global_k"]["seed_offset"] == 999_999
    overlap = provenance["evaluation_contract"]["staged_checkpoint_selector_overlap"]
    assert overlap["count"] == 32
    assert overlap["fraction"] == 0.32
    assert "every latent transition" in provenance["evaluation_contract"][
        "staged_routing_cadence"
    ]
    assert "optimistic" in provenance["comparison_contract"]["interpretation"]


def test_frozen_local_map_values_and_table_remain_byte_identical():
    data_sha = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    table_sha = hashlib.sha256(TABLE_PATH.read_bytes()).hexdigest()
    assert data_sha == "a18af75c30edfa0284fa27e6a76f6288150212aef84d11b919a88cc9e5111622"
    assert table_sha == "b2e80c535e59a3e7c95a60b033515088a1f44694da38abc47ae175b20b98c9cd"


def test_local_map_tex_fragments_are_deterministic():
    assert TABLE_PATH.read_text() == build_table()
