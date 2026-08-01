# 3d — print3d

`print3d` is a free-software pipeline that turns a 2D image into a
print-ready 3D model for a Bambu Lab A1 Mini (0.4mm nozzle, PLA, AMS Lite
4-slot). v1 supports one object type: **keychains**. Every tool in the
pipeline is free/open-source software — no paid tools or APIs.

The pipeline traces the source image to vector geometry (silhouette +
optional detail layer), cleans it up, builds a 3D solid (base + raised or
recessed detail, keyring hole, chamfer), and exports a multi-object 3MF
(for AMS two-color printing) plus an STL.

## Quick start

```
uv sync
uv run make3d <image> --type keychain
```

`uv` provisions Python itself (see `.python-version`) — there is no system
Python on this machine, and none is required.

## Quality gates

A design is only called **print-ready** once it has passed both quality
levels:

- **Level 1 (geometry):** the exported mesh is a valid volume (trimesh
  `is_volume`), and no wall is thinner than the minimum (checked via a
  shapely `buffer(-0.6)`: any region under 1.2mm wall thickness fails).
- **Level 2 (slice):** the model slices successfully through the
  OrcaSlicer CLI in headless mode. Judged by the slicer's exit code, the
  existence and nonzero size of the output `.gcode.3mf`, and the embedded
  gcode header — never by stdout, which is unreliable on Windows.

Only a design that has passed level 2 may be described as print-ready
anywhere in this repo (docs, reports, commit messages).

## Design folder catalog

Each design lives under `designs/<slug>/` and is committed to the repo:
source image, traced SVG, exported 3MF, exported STL, and a quality report
(`report.md` + `report.json`) recording level 1 / level 2 results.

**Drive overflow rule:** any file over 10MB is never committed. Instead,
upload it to Google Drive and record the link in the design's
`report.md`.

`*.gcode.3mf` files (slicer output, not the design's exported 3MF) are
gitignored — they're regenerable and printer-run-specific.

## Repo layout

- `src/print3d/` — the pipeline package (`trace.py`, `build.py`, `qc.py`,
  `cli.py`, `spec.py`)
- `designs/` — the design catalog (see above)
- `profiles/` — slicer profiles
- `tests/` — test suite

See `CLAUDE.md` for session rules when working in this repo with an agent.
