# Spatial Alluvial Prototypes

Date: May 6, 2026

These files prototype ways to add state-space context to the current
basin-to-support-to-family alluvial display.

- `prototype_a_triptych_composite.svg`: spatial support-family map, current
  alluvial, and family blocks as a three-step reading order.
- `prototype_b_spatial_source_column.svg`: alluvial-shaped layout where the
  left basin blocks are replaced by state-space source thumbnails.
- `prototype_c_family_callouts.svg`: spatial family regions connected directly
  to prototype support barcodes.

The SVGs are layout prototypes built from existing paper assets. The
data-driven renderer is
`tools/make_spatial_alluvial_prototypes.py`; it is intended to generate PDF,
PNG, and JSON versions from the same selected grid points once a compute-node
slot is available.

Suggested compute-node command:

```bash
salloc --mem=8G -c 4 --partition=long --time=00:30:00 srun uv run python tools/make_spatial_alluvial_prototypes.py
```
