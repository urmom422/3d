# OrcaSlicer profiles — Bambu Lab A1 Mini

## Install

OrcaSlicer 2.4.2 is installed on this machine as a **portable extraction**,
not a winget/MSI install:

- `winget install --id SoftFever.OrcaSlicer -e --accept-source-agreements --accept-package-agreements`
  was tried first. It installed the WebView2 dependency fine, but the
  OrcaSlicer installer step itself failed
  (`0x800704c7 : The operation was canceled by the user`) — the installer
  needs interactive UAC elevation that isn't available non-interactively.
- Fallback used instead: the official portable zip from the OrcaSlicer
  GitHub releases
  (`https://github.com/SoftFever/OrcaSlicer/releases/download/v2.4.2/OrcaSlicer_Windows_V2.4.2_x64_portable.zip`,
  downloaded via `gh release download` against `SoftFever/OrcaSlicer`),
  extracted to:

  ```
  %LOCALAPPDATA%\Programs\OrcaSlicer\
  ```

  giving `orca-slicer.exe` at
  `C:\Users\jconkle\AppData\Local\Programs\OrcaSlicer\orca-slicer.exe`.
  There is no separate `orca-slicer-console.exe` in this build — the one
  `orca-slicer.exe` handles both GUI and CLI. Running it with no
  slice-related args (e.g. a bare `--help`) launches the full GUI and
  blocks, so always pass a real CLI job (`--slice ... --export-3mf ...`).

## Bundled profiles

Copied as-is from OrcaSlicer's own bundled Bambu system presets
(`resources\profiles\BBL\` under the OrcaSlicer install dir) into this
folder, filenames unchanged:

| File | Preset name | Role |
|---|---|---|
| `Bambu Lab A1 mini 0.4 nozzle.json` | "Bambu Lab A1 mini 0.4 nozzle" | machine |
| `0.20mm Standard @BBL A1M.json` | "0.20mm Standard @BBL A1M" | process |
| `Bambu PLA Basic @BBL A1M.json` | "Bambu PLA Basic @BBL A1M" | filament |

Each JSON has an `"inherits"` field pointing at a base system preset
(e.g. `fdm_bbl_3dp_001_common`, `fdm_process_single_0.20`,
`Bambu PLA Basic @base`). OrcaSlicer resolves those inherited settings
from its own installed profile database at run time — this works whether
the leaf JSON is loaded from OrcaSlicer's own `resources\profiles\BBL\`
or from a copy elsewhere (verified: slicing with the copies in this
`profiles\` folder produces the same output as slicing with the
originals). Nothing else was added to these files.

Provenance and license: copied unmodified from `SoftFever/OrcaSlicer`
v2.4.2, `resources/profiles/BBL/` — OrcaSlicer is AGPL-3.0, and these
presets are redistributed here under that license with their source
recorded. The copies are **pinned to the 2.4.2 bundle**: their
`inherits` targets resolve against the installed OrcaSlicer's own
preset database, so a future OrcaSlicer upgrade that renames a base
preset would break these leaves — re-copy from that version's bundle
if that happens.

## Working headless CLI invocation

Confirmed working template (**cmd.exe only** — the `^` line continuation
below is cmd syntax and breaks in PowerShell; see the PowerShell note
right after), using the profiles in this folder by path (see Windows
caveat below for how to judge success):

```
"<ORCASLICER_EXE>" ^
  --load-settings "<REPO>\profiles\Bambu Lab A1 mini 0.4 nozzle.json;<REPO>\profiles\0.20mm Standard @BBL A1M.json" ^
  --load-filaments "<REPO>\profiles\Bambu PLA Basic @BBL A1M.json" ^
  --slice 0 ^
  --export-3mf "<OUTPUT_DIR>\<name>.gcode.3mf" ^
  "<INPUT_STL_OR_3MF>"
```

**PowerShell note:** a quoted exe path is not a command by itself in
PowerShell — prefix it with `&` (the call operator), and drop the `^`
continuations in favor of a backtick (`` ` ``) or just one line. The same
invocation, single-line and PowerShell-safe:

```
& "<ORCASLICER_EXE>" --load-settings "<REPO>\profiles\Bambu Lab A1 mini 0.4 nozzle.json;<REPO>\profiles\0.20mm Standard @BBL A1M.json" --load-filaments "<REPO>\profiles\Bambu PLA Basic @BBL A1M.json" --slice 0 --export-3mf "<OUTPUT_DIR>\<name>.gcode.3mf" "<INPUT_STL_OR_3MF>"
```

Notes on the flags (same flags either way; only the shell syntax differs):

- `--load-settings` takes **both** the machine and process JSON, joined
  with `;`, in one argument.
- `--load-filaments` takes the filament JSON.
- `--slice 0` slices all plates.
- `--export-3mf <path>` — always pass a full, absolute output path here.
  Do **not** also pass `--outputdir`: combining an absolute
  `--export-3mf` path with `--outputdir` causes OrcaSlicer to concatenate
  them into an invalid path and the export fails (exit code still 0 —
  see below) with `Project export to <outputdir>/<abs path> failed` on
  stderr. That failure mode also leaves a stray `plate_1.gcode`
  (hundreds of KB) in the outputdir — so never glob an output directory
  to detect success; stat the exact `--export-3mf` path.
- The input mesh (STL or 3MF) is a bare positional argument at the end.

Exact command used to slice the acceptance-check fixture (paths as on
this machine):

```
"C:\Users\jconkle\AppData\Local\Programs\OrcaSlicer\orca-slicer.exe" --load-settings "C:\Users\jconkle\Documents\GitHub\3d\profiles\Bambu Lab A1 mini 0.4 nozzle.json;C:\Users\jconkle\Documents\GitHub\3d\profiles\0.20mm Standard @BBL A1M.json" --load-filaments "C:\Users\jconkle\Documents\GitHub\3d\profiles\Bambu PLA Basic @BBL A1M.json" --slice 0 --export-3mf "<GREEN ROOM>\out_repo.gcode.3mf" "<GREEN ROOM>\cube.stl"
```

- Exit code: `0`
- Output: `<GREEN ROOM>\out_repo.gcode.3mf`, 76375 bytes
- Embedded gcode header (`Metadata/plate_1.gcode` inside the 3MF zip):
  `; model printing time: 33m 17s; total estimated time: 41m 1s`

## Windows caveat — the exit code and stdout carry NO failure signal

Measured on this machine (OrcaSlicer 2.4.2): the exe returns **exit
code 0 in every observed case, including hard slice failures** — the
`--outputdir` conflict returns 0 with no export, and an object too big
for the bed (300 mm cube) returns 0 with no artifact at all. stdout is
empty on success AND on failure. Do not gate on either; a
`returncode != 0` check will simply never fire.

The only real success gates are:

1. **Output artifact exists at the exact `--export-3mf` path and is
   > 1 KB.** The `.gcode.3mf` is a zip; a failed slice either doesn't
   produce it or produces a near-empty stub. Never glob the output
   directory (see the stray `plate_1.gcode` trap above).
2. **Embedded gcode header parses.** Unzip the `.gcode.3mf`, extract
   `Metadata/plate_<N>.gcode`, and confirm a line like
   `; model printing time: ...; total estimated time: ...` is present
   near the top with a non-zero time. If the metadata block is missing,
   treat the slice as failed regardless of anything else.

**Capture stderr anyway — as the diagnostic, never the gate.** On the
observed failures stderr carried the only useful message
(`Slic3r::CLI::run found error, exit`, plus the malformed path in the
`--outputdir` case). Surface it in the quality report when the gates
fail.

For the Python consumer (the QC module): a plain
`subprocess.run([exe, "--load-settings", ..., ...], capture_output=True)`
argv **list** is the reliable way to invoke this — it sidesteps the
quoting traps entirely (profile filenames contain spaces and `@`; the
`;` join inside the `--load-settings` value is a statement separator in
PowerShell but is just an argument character when passed as a list
element). Do not use `shell=True` with a formatted string.

Also: never run the exe with no slicing-relevant arguments (e.g. bare
`--help`) in an automated context — it launches the full GUI and hangs
rather than printing usage and exiting.

## How the QC module should locate the exe

1. `ORCASLICER_EXE` environment variable: if set, use it verbatim and
   **hard-fail with a clear message if the file does not exist** — a
   set-but-wrong env var is a configuration error to surface, not a
   reason to fall through to guessing.

Otherwise probe, in order, until one exists:

2. `%LOCALAPPDATA%\Programs\OrcaSlicer\orca-slicer.exe` — the portable
   install location used on this machine.
3. `%ProgramFiles%\OrcaSlicer\orca-slicer.exe` — the default winget/MSI
   install location, for machines where the winget path succeeds.
4. `%ProgramFiles(x86)%\OrcaSlicer\orca-slicer.exe` — fallback for a
   32-bit-prefixed install.

If none of the above exist, do not guess further: per the spec, the QC
module records a named "OrcaSlicer not found; set ORCASLICER_EXE" warning,
skips level 2, and caps the verdict at "passes-level-1" (never
"print-ready" without a real slice). Only a set-but-wrong `ORCASLICER_EXE`
is a hard error.

One more measured note on the exit code: while all failures observed
during profile bundling returned 0, a later raw failing slice was seen to
return a nonzero code (4294967246) — so the exit code is best described
as carrying no reliable signal in either direction, which leaves the two
gates above as the only judges.
