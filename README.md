# 3d — print3d

`print3d` is a free-software pipeline: an AI-assisted workflow over a
deterministic core. It turns a 2D image into a print-ready 3D model for
a Bambu Lab A1 Mini (0.4mm nozzle, PLA, AMS Lite 4-slot). v1 supports
one object type: **keychains**. Every tool in the pipeline is
free/open-source software — no paid tools or APIs.

The pipeline traces the source image to vector geometry (silhouette +
optional detail layer), cleans it up, builds a 3D solid (base + raised or
recessed detail, keyring hole, chamfer), and exports a multi-object 3MF
(for AMS two-color printing) plus an STL.

The ordinary way to run it is the `/make-3d` skill: an assistant runs the
interview and drives the pipeline. The scripts underneath are the whole
pipeline and stay fully runnable on their own, with or without an
assistant — see Quick start below for the direct path.

**New here?** Start with [`START_HERE.md`](START_HERE.md) for a
step-by-step, kid-friendly walkthrough of building your first keychain.

## The seam — capability vs judgment

The boundary between the assistant and the scripts is fixed, and it runs
through judgment, not capability:

- **Scripts own every capability, end to end** — trace, build, quality
  gate, slice. All of it stays runnable headless, without an assistant,
  and is covered by tests. This is a hard constraint, not an aspiration.
- **The assistant owns judgment** — interview, interpretation,
  recommendation, and consent. It never holds behaviour the scripts lack.

Rule of thumb: *capability → scripts; judgment → the skill.*

## Quick start

The direct, scripted path — no assistant involved:

```
uv sync
uv run make3d <image> --type keychain
```

`uv` provisions Python itself (see `.python-version`), independent of
whatever Python (if any) is already installed on the machine — the
Python environment is the same everywhere, and a system install is
never touched.

## Quality gates

A design is only called **print-ready** once it has passed both quality
levels:

- **Level 1 (geometry):** the exported mesh is a valid volume (trimesh
  `is_volume`), and no wall is thinner than the minimum (checked via a
  shapely `buffer(-0.6)`: any region under 1.2mm wall thickness fails).
- **Level 2 (slice):** the model slices successfully through the
  OrcaSlicer CLI in headless mode, judged by two measured artifact gates
  and nothing else: the `.gcode.3mf` exists at the exact `--export-3mf`
  path and is over 1KB, and its embedded `Metadata/plate_N.gcode` header
  parses to a non-zero print time. OrcaSlicer's exit code and stdout
  carry no failure signal at all on this platform (see
  `profiles/README.md`), so neither is ever consulted.

Only a design that has passed level 2 may be described as print-ready
anywhere in this repo (docs, reports, commit messages).

## Design folder catalog

Each design lives under `designs/<slug>/` and is committed to the repo:
source image, traced SVG, exported 3MF, exported STL, and a quality report
(`report.md` + `report.json`) recording level 1 / level 2 results.

**Drive overflow rule:** any committable artifact over 10MB (source image
copy, `traced.svg`, 3MF, STL — never the report files) is never
committed. Instead it is moved into a gitignored `designs/<slug>/_oversize/`
folder, and `DRIVE_LINK.md` in the design folder names each moved file
for the Master Conductor to upload to Google Drive manually — the
pipeline never touches Drive itself.

`*.gcode.3mf` files (slicer output, not the design's exported 3MF) are
gitignored — they're regenerable and printer-run-specific.

## Repo layout

- `src/print3d/` — the pipeline package (`trace.py`, `build.py`, `qc.py`,
  `cli.py`, `spec.py`)
- `designs/` — the design catalog (see above)
- `profiles/` — slicer profiles
- `tests/` — test suite

See `CLAUDE.md` for session rules.
