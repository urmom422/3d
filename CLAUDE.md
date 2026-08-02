# print3d — session rules

This repo is `print3d`: a free-software pipeline converting 2D images into
print-ready 3D models for a Bambu Lab A1 Mini. Read `README.md` first for
what the repo is and the quality-gate concept.

## Toolchain

- **uv only.** uv brings and pins its own Python, so the repo's Python
  environment is the same on every machine regardless of what Python (if
  any) is already installed there — and an installed Python is never
  touched. Always run `uv sync` / `uv run ...` from the repo root. Never
  invoke a bare `python`/`pip`.
- Python version is pinned in `.python-version`; `uv` provisions it.
- Commit `uv.lock` — lockfiles are checked in, not gitignored.

## Storage rules

- `designs/<slug>/` (source image, traced SVG, 3MF, STL, `report.md` +
  `report.json`) is committed to the repo.
- Any file over 10MB is **never** committed. Upload it to Google Drive
  instead and record the link in the design's `report.md`.
- `*.gcode.3mf` (slicer run output, distinct from a design's exported
  3MF) is gitignored — regenerable, printer-run-specific.

## Quality vocabulary

A design may only be called **print-ready** once it has passed level 2
(the OrcaSlicer CLI slice check), not merely level 1 (geometry checks).
Don't use "print-ready" loosely in docs, reports, or commit messages.

## The seam — capability vs judgment

The pipeline is an AI-assisted workflow over a deterministic core. The
boundary between the two is fixed, and it runs through judgment, not
capability:

- **Scripts own every capability, end to end** — trace, build, quality
  gate, slice, and the triage, repair and prototype stages as they land.
  All of it stays runnable headless, without an assistant, and covered by
  tests. This is a hard constraint, not an aspiration.
- **The assistant owns judgment** — interview, interpretation,
  recommendation, and consent. It decides what to ask the scripts for and
  what to tell the human. It never holds behaviour the scripts lack.
- The seam objects are the **settled spec** and — once artwork repair
  lands — the **repair consent record**.

Rule of thumb when placing new work: *capability → scripts; judgment →
the skill.*

## Code conventions

- Free software only — no paid tools or paid APIs anywhere in the
  pipeline; dependencies arrive via `uv` from PyPI, nothing else.
- No absolute paths in code — derive paths from the repo root at runtime.
- The project profile for this repo lives in the conductor repo at
  `projects/print3d/`; specs live in the conductor repo under `specs/`
  (e.g. `specs/print3d-bootstrap/`), not in this repo.
