# Quality report: example-badge

**Verdict: print-ready**

Levels run: 1, 2

## Level 1

- [PASS] **export_3mf.readable** -- example-badge.3mf loaded cleanly.
- [PASS] **export_stl.readable** -- example-badge.stl loaded cleanly.
- [PASS] **base.watertight** -- base: mesh is watertight.
- [PASS] **base.winding_consistent** -- base: face winding is consistent.
- [PASS] **base.is_volume** -- base: mesh is a valid, closed volume.
- [PASS] **base.no_degenerate_faces** -- base: no degenerate faces.
- [PASS] **base.body_count** -- base: exactly one body, as expected.
- [PASS] **detail.watertight** -- detail: mesh is watertight.
- [PASS] **detail.winding_consistent** -- detail: face winding is consistent.
- [PASS] **detail.is_volume** -- detail: mesh is a valid, closed volume.
- [PASS] **detail.no_degenerate_faces** -- detail: no degenerate faces.
- [PASS] **detail.body_count** -- detail: 1 raised region(s), each printing fused to the base.
- [PASS] **combined_stl.watertight** -- combined_stl: mesh is watertight.
- [PASS] **combined_stl.winding_consistent** -- combined_stl: face winding is consistent.
- [PASS] **combined_stl.is_volume** -- combined_stl: mesh is a valid, closed volume.
- [PASS] **combined_stl.no_degenerate_faces** -- combined_stl: no degenerate faces.
- [PASS] **combined_stl.body_count** -- combined_stl: exactly one body, as expected.
- [PASS] **bed_fit** -- Model measures 50.0 x 38.6 x 3.8 mm, within the 180x180x180 mm bed.
- [PASS] **wall_thickness** -- No wall thinner than 1.2 mm found in the printed base.
- [PASS] **hole_diameter** -- Hole measures 5.19 mm across (spec: 5.2 mm).
- [PASS] **hole_margin** -- Hole has 2.01 mm of margin (spec: 2.0 mm).
- [PASS] **detail_depth** -- Detail depth is 0.8 mm, at or above the 0.6 mm minimum.
- [PASS] **detail_present** -- The raised detail layer is in the export (374.1 mm2 of it).
- [PASS] **detail_depth_measured** -- The raised detail measures 0.80 mm of relief in the export (spec: 0.8 mm).
- [PASS] **detail_stroke_width** -- No detail stroke thinner than 1.0 mm found.

## Level 2

- [PASS] **slice_success** -- Sliced successfully; ~2558s estimated print time.

## Slicer

- exe: `C:\Users\jconkle\AppData\Local\Programs\OrcaSlicer\orca-slicer.exe`
- elapsed: 0.82s
