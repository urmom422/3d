---
name: make-3d
description: Convert a 2D image into a print-ready 3D model via a user interview, then run and explain the quality checks.
---

# make-3d

Front door for turning one source image into a print-ready design under
`designs/<slug>/`. Run this as an interview: one question at a time, each
with a recommended default the user can accept with a plain "yes" or by
overriding it. **Never invent a spec value the user didn't confirm** —
every value below must be offered and accepted, not assumed silently.

## 1. Interview

Ask these in order. After each answer, restate what you recorded before
moving on.

1. **Object type.** "This pipeline currently builds one object type:
   **keychain**. Use keychain?" (v1 has no other registered type — the
   CLI's `--type` registry is the seam for future ones, so this is asked
   even though there's only one right answer today.)
2. **Design name.** "What should this design be called?" Default: the
   image filename's stem. Becomes `--name`, which the CLI slugifies
   (lowercased, ASCII-folded, non-alphanumerics → dashes — e.g. "My Cat"
   → `my-cat`) into the folder under `designs/`. Tell the user the name
   will be slugified; don't promise it verbatim.
3. **Size.** "What's the longest XY dimension of the finished part, in
   mm? Default **50mm**." The bed is 180×180mm; keep parts comfortably
   under that (`--size`'s effective max is the bed minus margin).
4. **Base thickness.** "Base thickness, in mm? Default **3mm**."
5. **Detail mode.** "How should the second colour be expressed —
   **raised** (default, 0.8mm relief), **recessed**, or **none**
   (single colour)?" Only `raised` produces a second, separately
   AMS-mappable object; `recessed` (a pocket cut into the base) and
   `none` both export a single object, so they print single-colour.
6. **AMS Lite colors.** Only if detail mode is `raised` (skip for
   `recessed` and `none` — nothing to map). "Which AMS Lite colors do you
   want for the base and the detail? No default — name your two loaded
   filament colors." Note: the CLI itself has no color flags — colors are
   assigned later in Bambu Studio when you map the 3MF's two named
   objects (`base`, `detail`) to AMS slots. This question is asked anyway
   so the next-steps guidance below can name the colors you chose.
7. **Hole placement.** "Hanging-hole placement — **auto** (default,
   top-centre with a small tab fallback if the silhouette can't hold the
   margin there), or an explicit position (X Y in mm)?"

## 2. Run the CLI

Map the confirmed answers to real flags — do not use any flag not listed
in `uv run make3d --help`:

```
uv run make3d <image> --type keychain --size <N> --thickness <N> \
  --detail raised|recessed|none [--hole-position <X> <Y>] --name <name> \
  [--photo]
```

- `--hole-diameter` only if the user asked to override the 5.2mm default
  (not part of the standard interview above).
- `--hole-position X Y` only if "explicit" was chosen in step 7; omit for
  "auto".
- `--photo` — opt-in, off by default; not part of the standard interview
  above. Add it when the source image is a phone photo of a paper
  drawing rather than flat digital artwork: it evens out lighting and
  shadows first, then treats pencil/marker strokes as ink and the paper
  as background. Ask the user whether the source image is a photo of a
  drawing before deciding, rather than assuming.
- `--name` is the design name from step 2, as typed — the CLI slugifies
  it into the folder name. Don't reconstruct the slug yourself; read it
  off the CLI's own `report: <path>` line (or see step 3 if the run
  aborts before that line prints).
- `--no-slice` is a quick pre-check (skips the OrcaSlicer slice, caps the
  verdict at `passes-level-1`); the **final** run for the user must
  slice — never hand back a `--no-slice` run as done.

Run it with `uv run ...` from the repo root, per this repo's rules.

## 3. Handle a pipeline abort

If the CLI exits nonzero **and no `designs/<slug>/` folder (or report)
was produced**, the pipeline aborted before writing anything — a trace
error (thin features), a build error (disconnected islands, hole
margin), or a slicer config error. These print an actionable message to
stderr and nothing else. Relay that message verbatim — it already states
the remedy. Where it names an artwork problem (thin features,
disconnected islands), also point the user to bolder or better-connected
artwork and offer to re-run. Skip step 4 — there is no report to read.

If a design folder and report **were** produced (even for a `failed`
verdict), continue to step 4.

## 4. Surface warnings, then read the quality report

**Mandatory, before giving any verdict:** read `report.json`'s
`warnings` (mirrored under `## Warnings` in `report.md`) and relay every
one to the user. Warnings are where the pipeline says it changed what
was asked for — a single-colour flip because the traced image had no
detail layer, a hanging tab that grew the part's footprint past the
requested size, a file moved to `_oversize/` — and an all-green checks
table can still hide one of these. Never call a result a match for the
request without checking here first.

Then report the verdict, and explain every **FAILING** check in plain
language using this table:

| Check name | Plain meaning | What the user can do |
|---|---|---|
| `wall_thickness` | Part of the outline is thinner than 1.2mm. | Thicken the artwork's thin strokes, or increase `--size`. |
| `hole_diameter` | The hanging hole didn't come out at the requested diameter (or wasn't found). | Re-check the image for a clear hole area; consider `--hole-diameter`. |
| `hole_margin` | The hole is too close to the part's edge (need ≥2mm). | Move the hole (`--hole-position`) or shrink it, or increase `--size`. |
| `bed_fit` | The finished part is bigger than the 180×180×180mm bed. | Reduce `--size`. |
| `detail_present` | The trace had a detail layer, but it's missing from the export. | Not an artwork problem — the detail was lost between tracing and export; report it as a pipeline bug. |
| `detail_stroke_width` | Detail strokes are thinner than 1.0mm and got dropped. | Bolden the detail lines in the source artwork, or increase `--size`. |
| `detail_depth_measured` | The exported relief height doesn't match what was specified. | Not an artwork problem — a pipeline bug to report. |
| `base.body_count` / `combined_stl.body_count` | That export unexpectedly split into loose pieces. | Disconnected artwork is normally refused earlier with a clear error; if this fires anyway, report it as a pipeline bug. |
| `detail.body_count` | Number of separate raised-detail regions (e.g. two eyes, two letters). | Informational, not a failure to fix — multiple regions are normal. If it's still marked failing, treat as a pipeline bug, not artwork to "connect". |
| `<object>.watertight` / `.is_volume` / `.no_degenerate_faces` / `.winding_consistent` | The exported mesh for that object is structurally broken. | A pipeline bug, not an image issue — re-run; if it recurs, flag it. |
| `export_3mf.readable` / `export_stl.readable` | That export file couldn't be loaded at all. | Every other check on that file was skipped as a result — report this as one root cause, not a wall of failures. |
| `slice_success` | OrcaSlicer itself rejected the model during slicing. | Read the `stderr_excerpt` in the report for the slicer's own explanation. |

For any check name not in this table (the object names are `base`,
`detail`, `combined_stl`), quote its `message` field verbatim — every
failure must be surfaced, not just the ones with a canned explanation.

For **passing** checks, give a brief summary too (count, plus key
measured values like size and hole diameter) so the user sees why things
passed, not only what failed.

## 5. Verdict wording

- **`print-ready`** — level 2 (the real OrcaSlicer slice) passed. Only
  ever call a design this when the report says so.
- **`passes-level-1`** — check `report.json`'s `slicer` field for why:
  - `slicer` is `null` → `--no-slice` was used, skipping slicing on
    purpose. Remedy: re-run without `--no-slice` for a final result.
  - `slicer.found` is `false` → no OrcaSlicer install was found. Use this
    exact framing: say the gate **"passes level 1 only — not
    print-ready"**, and tell the user to **install OrcaSlicer, or set
    `ORCASLICER_EXE`** to point at it, then re-run.
- **`failed`** — at least one check failed; walk through the FAILING
  checks per the table above and suggest a re-run once addressed.

## 6. Drive overflow rule

State this whenever it applies (check for `_oversize/` and
`DRIVE_LINK.md` in the design folder): any committable file over the
size limit (source image, traced SVG, 3MF, or STL — never the reports)
is moved into `designs/<slug>/_oversize/` (gitignored, not committed to
git). The exact limit appears in the warning text and in
`DRIVE_LINK.md` inside the design folder, which also names the moved
files and the Master Conductor's Drive folder link. Uploading is
**manual** — the pipeline never touches Drive itself; tell the user
which files to upload and point them at `DRIVE_LINK.md`.

## 7. Next steps on success

1. Open the exported `.3mf` in Bambu Studio.
2. **raised** only: map its two named objects (`base`, `detail`) to AMS
   Lite slots using the colors chosen in interview step 6. **recessed**
   or **none**: single object, single filament — no color mapping.
3. Slice and print — PLA defaults, 0.4mm nozzle.
4. On a single-color printer, use the exported `.stl` instead (single
   body, no color mapping needed).
