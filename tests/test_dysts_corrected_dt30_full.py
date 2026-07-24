import pytest

from experiments.neurips_2026.protocol import DYSTS_MODEL_ROW_IDS
from experiments.neurips_2026.dysts_corrected_dt30_full.prepare import _validate


def _rows():
    rows = []
    for variant in DYSTS_MODEL_ROW_IDS:
        row = {
            "model_variant": variant,
            "dt_multiplier": "30",
            "lista_num_loops": "1" if variant.startswith("lista") else "",
            "config_name": "generic_sparse",
            "sparsity_coeff": "0.006",
            "k_structure": "dense",
        }
        if variant == "dense_mlp_tanh":
            row.update({
                "config_name": "generic_no_shrink",
                "sparsity_coeff": "0.0",
                "k_structure": "dense",
            })
        rows.append(row)
    return rows


def test_corrected_roster_requires_one_refinement_for_all_lista_rows():
    _validate(_rows(), expected=6)
    rows = _rows()
    next(row for row in rows if row["model_variant"] == "lista_sb")[
        "lista_num_loops"
    ] = "2"
    with pytest.raises(ValueError, match="one refinement"):
        _validate(rows, expected=6)


def test_corrected_roster_rejects_sparse_dense_baseline():
    rows = _rows()
    next(row for row in rows if row["model_variant"] == "dense_mlp_tanh")[
        "sparsity_coeff"
    ] = "0.006"
    with pytest.raises(ValueError, match="dense baseline"):
        _validate(rows, expected=6)
