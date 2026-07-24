# Evidence rebuild audit — 2026-07-20

## Concrete result

An initial rebuild sequence associated with allocation `10163012` rebuilt and
checked the forecast-optimized Allen--Cahn child packet, its high-dimensional
parent packet, the Dysts displays, and the retained-15 ground-truth displays.
That sequence invoked `salloc` without an explicit `srun`; its shell therefore
remained on the login host. Its successful paper check was useful as an early
integration diagnostic, but it is not evidence of execution on `cn-m001` and is
superseded by the true compute-node checks recorded below.

The forecast child provenance remained
`00cee2ec300c4b7a70c0129a904bc3b2d0af396909b28c61b604d154fa1bbf07`.
The parent high-dimensional provenance changed from the stale
`7b42a95a08847edd74ee52e88dd654c2e0916dc6d008a8e2db136d18b05025bc`
to the derived
`0994552a578269cb0051f17f2da553e6c30ed4a1a7f068db5ebae1d220586972`.
The parent figures themselves were unchanged by the repair.

## Dysts generated tables

The original full-precision CSV serialization was not byte-reproducible across
preliminary rebuilds. Those rebuilds were associated with allocations on
`cn-m004` and `cn-f002`, but they also omitted `srun` and therefore ran on the
login host; the observed last-bit variation must not be attributed to those CPU
families. Fourteen significant digits still left two rounding-boundary
mismatches in that diagnostic sequence, so the two full-statistic CSVs now use
an explicit 13-significant-digit serialization contract. This is substantially
finer than any reported paper precision. A regression test contains the
representative observed drift pairs.

Only two of six generated Dysts artifacts change under the final contract.
Their displayed values, directions, decisions, and significance conclusions
do not change.

| Artifact | Historical SHA-256 | Canonical SHA-256 |
|---|---|---|
| `_tables/dysts_dt30_iqm_summary.csv` | `2b9b65e9a9dad73cc200085cbf522cd45500d5c874729b8d30a753e41315f322` | `e929eacfaaa2784cd76dc3555e1b45758c2ac58f38c7a0ad745a88c5cf0f0c04` |
| `_tables/dysts_dt30_aggregate_tests_vs_dense.csv` | `d9ed237ee34d406549d0dc6f55e423e750ded8d9da6fc8392533831a9b8bed5e` | `c6a2e3dc9f36ce23bb3971864b4babc1951beb5c32c1dca000f4f0f364a2a2bb` |

The remaining Dysts output hashes remain byte-identical:

- `dysts_dt30_iqm_over_iqm_summary.csv`: `a7408c1dbfaced1fe2e90780d12dde109349957832eb27b3e0f56a0b820d3b72`
- `table_dysts_dt30_iqm_over_iqm.tex`: `49e93b6a3f09f6dbaba94a3d2d2b04e51476fcd5ece50210315720a223490bca`
- `table_dysts_dt30_ratio_to_dense.tex`: `730c6128bb5799b1c36f98a01598c303757f06b565ea8c5b335fabd3f55d2f97`
- `fig_dysts_dt30_iqm_over_iqm_horizon.pdf`: `e2ae5c1bf8d5c31836396406c85a2ff546b3ccdfcf08807f813f4d518a64717b`

The commands associated with allocations `10163287` and `10163297` were the
historical failed attempt to establish cross-node reproducibility: because
they omitted `srun`, they were login-host checks rather than checks on
`cn-m004` and `cn-f002`. The serialization contract remains justified by its
precision, regression test, and the later true-node paper check below, but
those two allocation IDs are not cross-architecture evidence.

## Ground-truth generated displays: superseded intermediate stage

The following hashes record the intermediate metadata-normalization rebuild.
They are preserved as historical provenance, but they are not the final
artifacts and the associated `salloc`-only checks did not establish
cross-architecture reproducibility.

| Artifact | Before SHA-256 | After SHA-256 |
|---|---|---|
| `ground_truth_vector_field_arrested_spiral.pdf` | `faa0531bb9fa70a50ce9a9214d525e6fe133b5146bd3b4f75a22dc279f65d1f0` | `d69080d9c0840c05a7f3b36386f5d46a036097b281518dde1a8b9a6eb4106cef` |
| `ground_truth_vector_field_cal_asymmetric_3.pdf` | `e015825abaab22980931a1748444de78c6505abcfae90f117190fb3aff0922b2` | `46f06a51201fb07153554d9ab6dbae51e3da6b8b88dafb65a69926a70d6e6b8f` |
| `ground_truth_vector_field_cal_hexagon_6.pdf` | `3e8f54ea2e8059bb5e9a2c247d8abd7a4726abdf75867b5672ab53e3dbcf6b09` | `38cc54dd9dea7c5403e7a74a1a3c38a440c0aabaa4a577700952adc6fda83f9b` |
| `ground_truth_vector_field_cal_high_cross_3.pdf` | `ee201259273442654fdd5fc71dfbbea3732fceec2fffe05dd2237c656dc1aff2` | `caf5cf28681263c4466a7e1153a373d58f4ca29f0cb7a4dc42a5d25aa3bd4807` |
| `ground_truth_vector_field_cal_octagon_8.pdf` | `216c5095adeb0e03386ce2385c418a94220d507cbd1af8702f732b7188ae16f6` | `db576dc0b96ed666a975bb6a1f9c0baad56e6e986dcddd944871e3b594498cb5` |
| `ground_truth_vector_field_cal_pentagon_5.pdf` | `65fa984c441fca3c0468d99febf2797a5ffb617affeb3ed669a1abe7d828afcb` | `029b2de7a67f635000a6664be95c6e220e1451ff02e78ae9b26879314343ec96` |
| `ground_truth_vector_field_cal_square_4.pdf` | `e89b98763e4cdc3c8428706dd328f7ef976076f943f8c3911b3511a7c8b206d4` | `5bf574e66d90de2c0899916c78cfcde7ae92789932955c8709e21d0d6f6ee698` |
| `ground_truth_vector_field_duffing_triple_well.pdf` | `96514513efddace80e8ddffe2d7314dc5602a3c1617b2ad332397f95c17861fe` | `db206e3153a1ec7f0c423dde4951e6e4ed46fe4fb8c351d521a1e9b9ec290e43` |
| `ground_truth_vector_field_gated_local_linear.pdf` | `abbe02207a74e7d8d10b2a73e827cd21177955669e237cd573e62da024f115a5` | `3509175be4e7382201560652744af8902b95d1149f4f4c94d4a09d2011749bc9` |
| `ground_truth_vector_field_gated_transfer_linear.pdf` | `eead67e42f2ee90bb02d26db3d9dcb17c5966de4617ec2265b49e0af7f81d5ed` | `c959d0088e7cef89feb164a92ba47b28e79eb61490cd62eb178b565e640def40` |
| `ground_truth_vector_field_snic_multi.pdf` | `51c7e1e355ae3f3aea0924f5c36c09f42dee4097cd403bc576290809a9b9f856` | `6f5d0d70c0c95e8fab33aab660c491c9cc7406e04ea365e125ca0f164e297703` |
| `ground_truth_vector_field_transition_routes_4.pdf` | `f85e7b1e32f6828dcba7f36a97c4814cd52a4f21bffbf8012f3ec1d2d41efeeb` | `c9d2f27075a5d707e0dc11a3093936b42ce7f640ee909349555fffe9dce4fcd8` |
| `ground_truth_vector_field_var_depth_gradient_4.pdf` | `5032863e4f8554aef4849c906c32b8058b898d985e699d96b3c49dc5863d8e40` | `1d985c67897ee67590f0f9d4214e860eaf13fe3efb0f24038f11ce75b77e0a65` |
| `ground_truth_vector_field_var_diamond_4.pdf` | `7ae50c58369d0e7746bf85313b7ac4c51e165ab1e205043c0e1c7b8a0c0e14d7` | `f89230c91c34be5168c3a50b9cc948a78bb22499f7519119ee0237c3e864156e` |
| `ground_truth_vector_field_var_l_shape_5.pdf` | `f0d810ee2851b6fad8e3239e122e76ef3d61d9199940e2f4445a46d3a04f0e9b` | `9e4cf7ea234da47d86ed91b5ed050c9c1a4df05af013d2e8de453fc13707aa36` |
| `ground_truth_vector_fields_retained15.pdf` | `b4b4a70c6e216470dc5d1fd86595cd941c3721271b39a3634528e96b77190377` | `8a1867f848c1810d8019e8401611dbdfd8c3cb5295d9af248703c2441b863990` |
| `manifest.json` | `55450a223f2f0717b413ec56f683a0a5e5d924149e892967c7d6d0c37708309f` | `5399fad03a2bbdab9cbe80e1f90c98ce06d8981f8ba9ea9f7d656cfe300ae8a2` |

The retained-15 composite was rasterized after the rebuild and visually checked:
all panels, titles, streamlines, and attractor markers were legible, with no
obvious clipping or malformed panel.

## Final cross-architecture display repair

The intermediate ground-truth PDFs still differed at the last decimal of PDF
path and color commands when rebuilt on different CPU families. The final
renderer leaves environment outputs and all numerical evidence untouched, but
canonicalizes display-only quantities below visual resolution: field samples
to 4 decimal places, log-speed colors to 7, and normalized streamline
directions to 8. Manifest schema 3 records all three values. The final
ground-truth manifest SHA-256 is
`1b75acffd441cbbda943e8ea1434b39b68d2094fd6f4b66618722c0be48d8a06`.

The final ground-truth PDF hashes are:

| Artifact | Final SHA-256 |
|---|---|
| `ground_truth_vector_field_arrested_spiral.pdf` | `9b5bbab431b85b1d23261510aba1a96bca1a107d8670bf91d6941055a4e2208e` |
| `ground_truth_vector_field_cal_asymmetric_3.pdf` | `7a4dbdf2f21d8e2a3e02cbe17f8c200420f0cc474b4feb70b9bb35a1b8126d10` |
| `ground_truth_vector_field_cal_hexagon_6.pdf` | `19757a77c7dcc56b7fac611402b0beba1d3de5cab11bf032c00c10e3b224e613` |
| `ground_truth_vector_field_cal_high_cross_3.pdf` | `a8155c8e9878478199f7e928091ddd27dfba9751e19ed70e951bd02d2a2486ec` |
| `ground_truth_vector_field_cal_octagon_8.pdf` | `237a0a908720011fb5f3489140192128630fbf7d4a49e90c80c94291955e2045` |
| `ground_truth_vector_field_cal_pentagon_5.pdf` | `4b1665dd978108dfb93d39619857d1975e06537875dab5d765f6763761c08601` |
| `ground_truth_vector_field_cal_square_4.pdf` | `c8555fe9dc0e383bf77b4c20e5e71f889890c1e861e3fca21f9798d90cf45e29` |
| `ground_truth_vector_field_duffing_triple_well.pdf` | `c70d8a75790456d075e26c4af021078acb0d01c46ab4d971562c2da9f0606431` |
| `ground_truth_vector_field_gated_local_linear.pdf` | `139a5578d51319fa2397f290bdd505f1146c2ce6c868455b9ba680e11617816b` |
| `ground_truth_vector_field_gated_transfer_linear.pdf` | `dd8c4ae10632fa5e3cf87031bca898fea2bf1ee19ecccf732118bd4cbb89e4a0` |
| `ground_truth_vector_field_snic_multi.pdf` | `c8b52d338598121496a4d14389d3da83eaaf6220dab4dd844414a881d45a0261` |
| `ground_truth_vector_field_transition_routes_4.pdf` | `b7f8da63b57b9d23a09cef2ad96b92bdb1d91d1c2cfaab525ac9938147e3cefc` |
| `ground_truth_vector_field_var_depth_gradient_4.pdf` | `f8b0f42b66030d5322f546e7ea4ccac3996327c4736aedbaa042b70337a43ebf` |
| `ground_truth_vector_field_var_diamond_4.pdf` | `7434325eba13fcb872354de58e10facd2eef98ffa2c06bfac7d99df39c53ba3d` |
| `ground_truth_vector_field_var_l_shape_5.pdf` | `4a2add7e86eb7c23f02b57b361ff3a07959454eed4b7935471ae59691baf8425` |
| `ground_truth_vector_fields_retained15.pdf` | `d605b4f35f2657be5832992cced67d6376191c069a3517efcaad571032a8e0d0` |

The Global-K support-closure PNG had a separate ten-pixel title-rasterization
difference at 300 dpi. Rendering at 320 dpi moves the glyph off the unstable
half-pixel boundary while leaving its PDF and table unchanged. Final SHA-256
values are:

- PNG: `3c12219125dcff25b7591d23573c2a45c454f46c240260a1827542e64ef790a7`
- PDF: `ec1c8dd0ee72955279d3aef0323171af87bff5120825d4c2511798ac2117e7ec`
- table: `744a67a3a8962fb1b874eb2e5ac4491ee4f94329e1e13555ecff893cf928dd5d`
- protocol provenance: `ba831891924c3afd92f00aae5022b4f13428fb584dee1b1a6f408856029b1f89`

True compute-node verification used `salloc` followed by explicit `srun`:

- job `10164395`, Sapphire Rapids node `cn-m003`: both display checks and 9
  focused tests passed;
- job `10164404`, Milan node `cn-h001`: both display checks passed;
- job `10164403`, Rome node `cn-f003`: both display checks passed;
- job `10164439`, Sapphire Rapids node `cn-m003`: all six Dysts paper
  artifacts passed a clean byte-rebuild check;
- job `10164440`, Milan node `cn-h001`: all six Dysts paper artifacts passed
  the same clean byte-rebuild check;
- job `10164408`, Rome node `cn-f004`: the complete
  `uv run skae-paper check` gate passed, ending with
  `All frozen paper evidence checks passed.`

## Interpretation and project implication

The parent high-dimensional builder previously rendered figures without writing
its combined provenance, so child rebuilds could leave a stale parent digest.
The maintained builder now derives component and display hashes, writes the
parent provenance, and includes it in byte-rebuild checks. A focused test checks
that the active parent JSON equals a fresh derivation from current children.

These repairs do not add or change a scientific result. They make the existing
rebuttal evidence reproducible in the current environment and prevent a stale
child hash from silently surviving future rebuilds.

## Verification

- `uv run skae-paper build allen-cahn-forecast-optimized --check`: passed.
- `uv run skae-paper build high-dimensional --check`: passed.
- `uv run pytest tests/test_highdimensional_evidence.py -q`: 6 passed.
- `uv run skae-paper build dysts --check`: six artifacts verified.
- `uv run skae-paper build ground-truth --formats pdf --check`: 16 PDFs verified.
- `uv run skae-paper check`: all frozen paper evidence checks passed.

For the subsequent Dysts serialization repair, the focused test module passed
8/8 twice during the preliminary login-host sequence associated with
allocations `10163287` and `10163297`. That sequence also rebuilt and checked
all six Dysts artifacts, but, because it omitted `srun`, it is not evidence of
the named CPU families. The final true-node repository-wide check is job
`10164408` above; it verified the Dysts artifacts as part of the complete
frozen-paper gate after the concurrent Allen--Cahn integration stabilized.
True Sapphire Rapids and Milan Dysts byte-rebuild checks subsequently passed
as jobs `10164439` and `10164440`, respectively.
