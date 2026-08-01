"""Quality gates for a built keychain: geometry checks plus an optional slice.

This stage takes the outputs of the earlier ones -- the trace stage's
:class:`~print3d.trace.TraceResult`, the build stage's
:class:`~print3d.build.KeychainSolids`, and the files ``export_3mf``/
``export_stl`` wrote -- and turns them into a :class:`QCReport`. Checks
always run against the **exported files**, not the in-memory shapes: a
mesh that looks fine in build123d but exports corrupt is exactly the bug
this stage exists to catch.

Two levels:

* **Level 1** (always runs, pure Python/geometry, no external process):
  that both exports load at all; mesh integrity (watertight,
  winding-consistent, a valid volume, no degenerate faces) for every
  named object in the 3MF *and* the fused STL; body count (each object
  must be exactly one connected body); bed fit; wall thickness; detail
  presence, depth (as specified *and* as measured) and stroke width; hole
  diameter and margin. Upstream warnings from the trace and build stages
  are carried through into the report verbatim. Nothing here is ever
  skipped quietly: an object missing from the export fails the checks
  that needed it rather than dropping them, because a check that is
  absent from the report reads as one that passed.
* **Level 2** (runs when OrcaSlicer can be found): a real headless slice
  of the exported STL with the bundled A1 Mini profiles. Judged only by
  the two gates that are actually reliable on this platform -- the
  ``.gcode.3mf`` exists at the requested path and is bigger than 1 KB,
  and its embedded ``Metadata/plate_N.gcode`` has a non-zero print-time
  header -- because OrcaSlicer's exit code and stdout carry no failure
  signal at all (see ``profiles/README.md``). When no slicer is found,
  level 2 is skipped, a warning says so, and the verdict is capped at
  ``"passes-level-1"``: nothing is ever called ``"print-ready"`` without
  a real slice.

Wall thickness and detail stroke width are both checked by measuring the
**geometry actually being printed** -- a 2D cross-section taken from the
exported mesh itself (after the hole cut, after any hanging tab, after
chamfer trimming), not the original traced polygons -- using the same
morphological-opening technique as :mod:`print3d.trace`
(:func:`~print3d.trace.find_thin_regions`).

``run_qc(...)`` builds a :class:`QCReport`; ``write_report(...)`` renders
it to ``report.json`` and ``report.md`` in a caller-chosen directory (or
the directory the STL lives in), as two separate steps so a caller (the
future CLI) can place them wherever it likes.

``report.json`` schema (``schema_version`` bumps on a breaking change)::

    {
        "schema_version": 1,
        "slug": "<KeychainSpec.slug>",
        "verdict": "print-ready" | "passes-level-1" | "failed",
        "levels_run": [1] | [1, 2],
        "checks": [
            {
                "name": str,          # stable id, e.g. "wall_thickness",
                                       # "base.watertight", "hole_margin"
                "level": 1 | 2,
                "passed": bool,
                "measured": <JSON scalar, list, or null>,
                "expected": <JSON scalar, list, or null>,
                "message": str        # human-readable, plain language
            },
            ...
        ],
        "warnings": [str, ...],       # upstream trace/build warnings plus
                                       # any QC-level warnings (e.g. no
                                       # slicer found), in that order
        "slicer": null | {
            "found": bool,
            "exe": str | null,
            "elapsed_s": float | null,
            "stderr_excerpt": str | null
        }
    }

``slicer`` is ``null`` only when level 2 was never attempted at all
(``run_level2=False``); when it was attempted but no exe was found,
``slicer.found`` is ``false`` and the rest of the fields are ``null``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

from .build import KeychainSolids
from .spec import BED_SIZE_MM, MIN_DETAIL_DEPTH_MM, MIN_DETAIL_STROKE_MM, MIN_WALL_MM
from .trace import TraceResult, find_thin_regions

__all__ = [
    "SCHEMA_VERSION",
    "OrcaSlicerConfigError",
    "QCCheck",
    "SlicerInfo",
    "QCReport",
    "find_orcaslicer_exe",
    "run_qc",
    "write_report",
]

#: Bumped whenever the report.json shape changes in a way old readers can't
#: shrug off (a field renamed or removed; a new required field is fine to
#: leave the version alone).
SCHEMA_VERSION = 1

#: The three bundled A1 Mini presets (see profiles/README.md), located
#: relative to this file so the module works from any checkout.
_PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"
_MACHINE_PROFILE = _PROFILES_DIR / "Bambu Lab A1 mini 0.4 nozzle.json"
_PROCESS_PROFILE = _PROFILES_DIR / "0.20mm Standard @BBL A1M.json"
_FILAMENT_PROFILE = _PROFILES_DIR / "Bambu PLA Basic @BBL A1M.json"

#: Slack on hole diameter/margin measurement, in mm. Tessellation chords and
#: the mesh-section polyline both wobble at the hundredth of a mm; a tenth
#: is the smallest difference worth flagging.
_HOLE_TOL_MM = 0.1

#: Slack on the bed-fit check, in mm - the same reasoning as build.py's own
#: size tolerance.
_BED_TOL_MM = 0.1

#: How far below a recess floor the "full material" probe section is taken,
#: in mm. Only needs to clear the floor's own tessellation.
_RECESS_PROBE_MM = 0.05

#: Slack on the measured detail depth, in mm. The relief is read off the
#: exported mesh, so it carries the same tessellation wobble as every other
#: measurement here.
_DETAIL_DEPTH_TOL_MM = 0.1

#: Area below which a differenced footprint part is arithmetic noise rather
#: than geometry, in mm2. Subtracting two mesh sections that share an outline
#: leaves a rash of slivers along it -- measured at ~1e-18 mm2, twelve orders
#: of magnitude under this floor -- and every one of them would otherwise be
#: reported as a region "thinner than 1 mm". A real feature this small is
#: 1 micron across: not something the trace stage can produce and not
#: something a 0.4 mm nozzle could print either way.
_MIN_FOOTPRINT_AREA_MM2 = 1e-6

#: Cosine of the largest tilt still counted as "facing straight up" when
#: locating a recess floor in the exported mesh.
_UPWARD_NORMAL_MIN = 0.999

#: The object names ``build.export_3mf`` writes, and that the per-object
#: checks below look for. Named here rather than inline so a missing object
#: can be reported by name instead of quietly skipping its checks.
_BASE_OBJECT_NAME = "base"
_DETAIL_OBJECT_NAME = "detail"

#: Timeout handed to the OrcaSlicer subprocess by default, in seconds.
DEFAULT_SLICE_TIMEOUT_S = 300.0

#: Smallest a real ``.gcode.3mf`` export is expected to be, in bytes. A
#: failed export either doesn't appear at all or leaves a near-empty stub.
_MIN_GCODE_3MF_BYTES = 1024

_PLATE_GCODE_RE = re.compile(r"Metadata/plate_(?P<n>\d+)\.gcode$")
_TIME_HEADER_RE = re.compile(
    r";\s*model printing time:\s*(?P<model>[^;]+);\s*total estimated time:\s*(?P<total>.+)"
)
#: Duration components OrcaSlicer writes into the print-time header. ``d`` is
#: in here because a multi-day estimate reads "2d 3h 4m 5s" (or "2 days ..."),
#: and dropping the day component would turn 51 hours into 3.
_DURATION_COMPONENT_RE = re.compile(r"(\d+)\s*(d|h|m|s)")


# --- errors ------------------------------------------------------------


class OrcaSlicerConfigError(Exception):
    """``ORCASLICER_EXE`` is set but does not point at a real file.

    A set-but-wrong environment variable is a configuration mistake to
    surface loudly, not a reason to fall back to guessing at the exe's
    location.
    """


# --- report data model ---------------------------------------------------


@dataclass
class QCCheck:
    """One pass/fail measurement in a :class:`QCReport`.

    ``name`` is a stable identifier (documented in the module docstring)
    that callers -- including tests -- can look a check up by, independent
    of the human-readable ``message``.
    """

    name: str
    level: int
    passed: bool
    measured: Any
    expected: Any
    message: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "level": self.level,
            "passed": self.passed,
            "measured": self.measured,
            "expected": self.expected,
            "message": self.message,
        }


@dataclass
class SlicerInfo:
    """What level 2 knows about the OrcaSlicer run (or lack of one)."""

    found: bool
    exe: str | None = None
    elapsed_s: float | None = None
    stderr_excerpt: str | None = None

    def to_dict(self) -> dict:
        return {
            "found": self.found,
            "exe": self.exe,
            "elapsed_s": self.elapsed_s,
            "stderr_excerpt": self.stderr_excerpt,
        }


@dataclass
class QCReport:
    """The full quality report for one built keychain. See module docstring."""

    slug: str
    verdict: str
    levels_run: list[int]
    checks: list[QCCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    slicer: SlicerInfo | None = None

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "slug": self.slug,
            "verdict": self.verdict,
            "levels_run": list(self.levels_run),
            "checks": [c.to_dict() for c in self.checks],
            "warnings": list(self.warnings),
            "slicer": self.slicer.to_dict() if self.slicer is not None else None,
        }

    def to_markdown(self) -> str:
        return _render_markdown(self)


# --- public entry points ---------------------------------------------------


def run_qc(
    trace: TraceResult,
    solids: KeychainSolids,
    exported_3mf: str | Path,
    exported_stl: str | Path,
    work_dir: str | Path | None = None,
    *,
    run_level2: bool = True,
    slicer_exe: str | Path | None = None,
    slice_timeout_s: float = DEFAULT_SLICE_TIMEOUT_S,
) -> QCReport:
    """Run every quality check and return a :class:`QCReport`.

    ``exported_3mf`` and ``exported_stl`` are the files ``export_3mf`` and
    ``export_stl`` (see :mod:`print3d.build`) wrote for ``solids``; checks
    read those files back rather than trusting the in-memory shapes.
    ``work_dir`` is where level 2 writes its ``.gcode.3mf``; it defaults to
    the STL's own directory (per the storage rule: slice output is
    regenerable, so it belongs next to the model, not in the repo).

    ``run_level2=False`` skips the slice entirely (``levels_run == [1]``,
    ``slicer`` stays ``None``) -- useful for fast unit tests. Otherwise
    level 2 runs unless no OrcaSlicer install can be found, in which case
    a warning is added and the verdict is capped at ``"passes-level-1"``.

    ``slicer_exe`` overrides the discovery in :func:`find_orcaslicer_exe`
    (mainly for tests); by default the exe is discovered per
    ``profiles/README.md`` (``ORCASLICER_EXE`` env var, or the standard
    Windows install locations). Either way, a set-but-invalid path is a
    hard failure (:class:`OrcaSlicerConfigError`), not a fallback and not
    a ``FileNotFoundError`` from somewhere inside ``subprocess``.
    """
    exported_3mf = Path(exported_3mf)
    exported_stl = Path(exported_stl)
    resolved_work_dir = Path(work_dir) if work_dir is not None else exported_stl.parent

    spec = solids.spec
    warnings: list[str] = list(trace.warnings) + list(solids.warnings)
    checks: list[QCCheck] = []

    scene_meshes, scene_check = _load_scene_meshes(exported_3mf)
    combined_mesh, stl_check = _load_single_mesh(exported_stl)
    checks.append(scene_check)
    checks.append(stl_check)

    all_meshes: dict[str, trimesh.Trimesh] = dict(scene_meshes)
    if combined_mesh is not None:
        all_meshes["combined_stl"] = combined_mesh

    for name, mesh in all_meshes.items():
        checks.extend(_mesh_integrity_checks(name, mesh))
        checks.append(_body_count_check(name, mesh))

    if combined_mesh is not None:
        checks.append(_bed_fit_check(combined_mesh))

    base_mesh = scene_meshes.get(_BASE_OBJECT_NAME)
    if base_mesh is not None:
        base_footprint = _section(base_mesh, spec.base_thickness_mm / 2.0)
        checks.append(_wall_thickness_check(base_footprint))
        hole_measurement = _hole_measurements(base_footprint, solids.hole_center_mm)
        checks.append(_hole_diameter_check(hole_measurement, spec))
        checks.append(_hole_margin_check(hole_measurement, spec))
    else:
        # A missing object is not a reason to skip its checks: silently
        # dropping three of them turns a broken export into a clean report.
        checks.extend(_missing_object_checks(_BASE_OBJECT_NAME, spec))

    if spec.detail_mode != "none":
        checks.extend(_detail_checks(scene_meshes, base_mesh, spec))

    levels_run = [1]
    slicer_info: SlicerInfo | None = None
    if run_level2:
        slicer_info, slice_check, slicer_warning = _run_level2(
            exported_stl, resolved_work_dir, slicer_exe, slice_timeout_s
        )
        if slicer_warning:
            warnings.append(slicer_warning)
        if slice_check is not None:
            checks.append(slice_check)
            levels_run.append(2)

    verdict = _determine_verdict(checks, levels_run)
    return QCReport(
        slug=spec.slug,
        verdict=verdict,
        levels_run=levels_run,
        checks=checks,
        warnings=warnings,
        slicer=slicer_info,
    )


def write_report(report: QCReport, out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``report.json`` and ``report.md`` into ``out_dir``.

    A separate step from :func:`run_qc` so a caller can decide where the
    report lives (the CLI places it in the design folder) without QC
    itself knowing anything about that layout.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    md_path = out / "report.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return json_path, md_path


def find_orcaslicer_exe() -> Path | None:
    """Locate the OrcaSlicer executable, per ``profiles/README.md``.

    Order: ``ORCASLICER_EXE`` env var (hard-fails via
    :class:`OrcaSlicerConfigError` if set but not pointing at a file), then
    the standard winget/MSI and portable install locations. Returns
    ``None`` if none of those exist -- that is a normal "not installed"
    outcome, not an error.

    An ``ORCASLICER_EXE`` that is empty or only whitespace counts as
    **unset**, not as a misconfiguration: that is what a shell leaves
    behind when a wrapper script exports the variable without a value, and
    refusing to run at all in that case would be unhelpfully literal.

    The test is ``is_file()``, not ``exists()``: a variable pointing at the
    *directory* OrcaSlicer lives in is exactly the mistake to catch here,
    and it passes ``exists()`` only to explode as a ``PermissionError``
    when the subprocess tries to execute it.
    """
    env = (os.environ.get("ORCASLICER_EXE") or "").strip()
    if env:
        candidate = Path(env)
        if not candidate.is_file():
            raise OrcaSlicerConfigError(
                f"ORCASLICER_EXE is set to '{env}', but that is not a file "
                f"(it is missing, or it is a directory). Point it at the "
                f"orca-slicer executable itself, or unset the environment "
                f"variable to fall back to auto-detection."
            )
        return candidate

    for var, suffix in (
        ("LOCALAPPDATA", ("Programs", "OrcaSlicer", "orca-slicer.exe")),
        ("ProgramFiles", ("OrcaSlicer", "orca-slicer.exe")),
        ("ProgramFiles(x86)", ("OrcaSlicer", "orca-slicer.exe")),
    ):
        base = os.environ.get(var)
        if not base:
            continue
        candidate = Path(base).joinpath(*suffix)
        if candidate.is_file():
            return candidate
    return None


# --- level 2: slicing --------------------------------------------------


def _run_level2(
    exported_stl: Path,
    work_dir: Path,
    slicer_exe: str | Path | None,
    timeout_s: float,
) -> tuple[SlicerInfo, QCCheck | None, str | None]:
    if slicer_exe is not None:
        # Validated up front so an explicit override fails the same typed,
        # explained way a bad ORCASLICER_EXE does, rather than escaping as
        # a bare FileNotFoundError from deep inside subprocess.
        exe = Path(slicer_exe)
        if not exe.is_file():
            raise OrcaSlicerConfigError(
                f"slicer_exe was set to '{slicer_exe}', but that is not a "
                f"file (it is missing, or it is a directory). Point it at "
                f"the orca-slicer executable itself, or leave it unset to "
                f"fall back to auto-detection."
            )
    else:
        exe = find_orcaslicer_exe()
    if exe is None:
        return (
            SlicerInfo(found=False),
            None,
            "OrcaSlicer was not found (checked ORCASLICER_EXE and the "
            "standard install locations), so level 2 (slice) checks were "
            "skipped. The verdict is capped at 'passes-level-1' -- it is "
            "never 'print-ready' without a real slice.",
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / f"{exported_stl.stem}.gcode.3mf"
    # Delete any artifact left behind by an earlier run *before* slicing.
    # Both success gates below are measurements of this file, and a stale
    # one satisfies both no matter what the slicer does this time -- which
    # is how a failing slice gets reported as "print-ready".
    out_path.unlink(missing_ok=True)
    argv = [
        str(exe),
        "--load-settings",
        f"{_MACHINE_PROFILE};{_PROCESS_PROFILE}",
        "--load-filaments",
        str(_FILAMENT_PROFILE),
        "--slice",
        "0",
        "--export-3mf",
        str(out_path),
        str(exported_stl),
    ]

    start = time.monotonic()
    timed_out = False
    stderr = ""
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            # OrcaSlicer drops a 00000.log into whatever directory it is
            # started in. Started from the repo root that means a stray log
            # in the working tree on every run; the work dir is where the
            # rest of the regenerable slice output already goes.
            cwd=str(work_dir),
        )
        stderr = result.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        raw = exc.stderr
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        stderr = raw or ""
    elapsed = time.monotonic() - start

    passed, measured, message = _judge_slice(out_path, timed_out)
    info = SlicerInfo(
        found=True,
        exe=str(exe),
        elapsed_s=round(elapsed, 2),
        stderr_excerpt=stderr[:2000] if stderr else None,
    )
    check = QCCheck(
        name="slice_success",
        level=2,
        passed=passed,
        measured=measured,
        expected="a .gcode.3mf over 1 KB with a non-zero print-time header",
        message=message,
    )
    return info, check, None


def _judge_slice(out_path: Path, timed_out: bool) -> tuple[bool, Any, str]:
    """Apply the two measured success gates. Never the exit code or stdout."""
    if timed_out:
        return False, None, "OrcaSlicer did not finish within the timeout."
    if not out_path.exists():
        return False, None, f"No .gcode.3mf was produced at {out_path}."
    size = out_path.stat().st_size
    if size <= _MIN_GCODE_3MF_BYTES:
        return (
            False,
            size,
            f"{out_path.name} exists but is only {size} bytes -- too small "
            f"to be a real slice output.",
        )
    seconds = _extract_print_time_seconds(out_path)
    if seconds is None or seconds <= 0:
        return (
            False,
            seconds,
            "The .gcode.3mf has no valid, non-zero print-time header in "
            "its embedded gcode -- treated as a failed slice.",
        )
    return True, round(seconds, 1), f"Sliced successfully; ~{seconds:.0f}s estimated print time."


def _extract_print_time_seconds(path: Path) -> float | None:
    """Shortest plate print time in the ``.gcode.3mf``, or None if unreadable.

    **Every** plate is read, not just the first: a multi-plate export where
    one plate sliced to nothing is a failed slice, and returning the
    shortest time lets the single ``<= 0`` test in :func:`_judge_slice`
    catch it. Plates are ordered numerically for the message's sake --
    ``sorted()`` on the names alone puts ``plate_10`` before ``plate_2``,
    which is how a broken plate 2 got hidden behind a healthy plate 10.
    """
    heads: list[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            matches = [
                (int(m.group("n")), n)
                for n, m in (
                    (n, _PLATE_GCODE_RE.search(n)) for n in zf.namelist()
                )
                if m is not None
            ]
            if not matches:
                return None
            for _number, name in sorted(matches):
                with zf.open(name) as fh:
                    heads.append(fh.read(4096).decode("utf-8", errors="replace"))
    except (zipfile.BadZipFile, OSError, KeyError):
        return None

    times: list[float] = []
    for head in heads:
        match = _TIME_HEADER_RE.search(head)
        if not match:
            return None  # a plate with no header at all is not a slice
        times.append(_parse_duration(match.group("total")))
    return min(times) if times else None


def _parse_duration(text: str) -> float:
    """Seconds from an OrcaSlicer duration such as ``"2d 3h 4m 5s"``."""
    seconds_per_unit = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}
    total = 0.0
    for value, unit in _DURATION_COMPONENT_RE.findall(text):
        total += float(value) * seconds_per_unit[unit]
    return total


def _determine_verdict(checks: list[QCCheck], levels_run: list[int]) -> str:
    level1_failed = any(c.level == 1 and not c.passed for c in checks)
    if level1_failed:
        return "failed"
    if 2 in levels_run:
        level2_failed = any(c.level == 2 and not c.passed for c in checks)
        return "failed" if level2_failed else "print-ready"
    return "passes-level-1"


# --- level 1: mesh loading -------------------------------------------------


def _load_scene_meshes(
    path: Path,
) -> tuple[dict[str, trimesh.Trimesh], QCCheck]:
    """(object name -> mesh) for an exported 3MF, plus a readability check.

    A missing or corrupt export is a QC *result*, not a QC crash: the
    report is the product of this stage, and "the 3MF could not be read"
    is exactly the kind of thing it exists to say. Third-party exceptions
    (trimesh's ``ValueError`` for a missing path, ``zipfile.BadZipFile``
    for a truncated 3MF) are caught here and turned into a failed check.
    """
    try:
        scene = trimesh.load(str(path), force="scene")
        meshes: dict[str, trimesh.Trimesh] = {}
        for node in scene.graph.nodes_geometry:
            transform, geom_name = scene.graph[node]
            mesh = scene.geometry[geom_name].copy()
            mesh.apply_transform(transform)
            meshes[geom_name] = mesh
    except Exception as exc:
        return {}, _unreadable_check("export_3mf", path, exc)
    return meshes, _readable_check("export_3mf", path, sorted(meshes))


def _load_single_mesh(path: Path) -> tuple[trimesh.Trimesh | None, QCCheck]:
    """The fused STL as one mesh, plus a readability check. See above."""
    try:
        mesh = trimesh.load(str(path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError("the file holds no triangles")
    except Exception as exc:
        return None, _unreadable_check("export_stl", path, exc)
    return mesh, _readable_check("export_stl", path, int(len(mesh.faces)))


def _readable_check(name: str, path: Path, measured: Any) -> QCCheck:
    return QCCheck(
        name=f"{name}.readable",
        level=1,
        passed=True,
        measured=measured,
        expected="a readable export",
        message=f"{path.name} loaded cleanly.",
    )


def _unreadable_check(name: str, path: Path, exc: Exception) -> QCCheck:
    return QCCheck(
        name=f"{name}.readable",
        level=1,
        passed=False,
        measured=None,
        expected="a readable export",
        message=(
            f"{path} could not be read as a mesh export "
            f"({type(exc).__name__}: {exc}). Every check that needed it "
            f"was skipped."
        ),
    )


# --- level 1: mesh integrity -------------------------------------------


def _mesh_integrity_checks(name: str, mesh: trimesh.Trimesh) -> list[QCCheck]:
    watertight = bool(mesh.is_watertight)
    winding = bool(mesh.is_winding_consistent)
    volume = bool(mesh.is_volume)
    nondegenerate = mesh.nondegenerate_faces()
    degenerate_count = int((~nondegenerate).sum()) if len(nondegenerate) else 0

    return [
        QCCheck(
            name=f"{name}.watertight",
            level=1,
            passed=watertight,
            measured=watertight,
            expected=True,
            message=f"{name}: mesh {'is' if watertight else 'is NOT'} watertight.",
        ),
        QCCheck(
            name=f"{name}.winding_consistent",
            level=1,
            passed=winding,
            measured=winding,
            expected=True,
            message=(
                f"{name}: face winding "
                f"{'is' if winding else 'is NOT'} consistent."
            ),
        ),
        QCCheck(
            name=f"{name}.is_volume",
            level=1,
            passed=volume,
            measured=volume,
            expected=True,
            message=(
                f"{name}: mesh {'is' if volume else 'is NOT'} a valid, "
                f"closed volume."
            ),
        ),
        QCCheck(
            name=f"{name}.no_degenerate_faces",
            level=1,
            passed=degenerate_count == 0,
            measured=degenerate_count,
            expected=0,
            message=(
                f"{name}: no degenerate faces."
                if degenerate_count == 0
                else f"{name}: {degenerate_count} degenerate face(s) found."
            ),
        ),
    ]


def _body_count_check(name: str, mesh: trimesh.Trimesh) -> QCCheck:
    count = int(mesh.body_count)
    return QCCheck(
        name=f"{name}.body_count",
        level=1,
        passed=count == 1,
        measured=count,
        expected=1,
        message=(
            f"{name}: exactly one body, as expected."
            if count == 1
            else (
                f"{name}: split into {count} separate bodies -- this "
                f"would print as {count} loose pieces."
            )
        ),
    )


def _bed_fit_check(mesh: trimesh.Trimesh) -> QCCheck:
    extents = [float(v) for v in (mesh.bounds[1] - mesh.bounds[0])]
    fits = all(e <= b + _BED_TOL_MM for e, b in zip(extents, BED_SIZE_MM))
    measured = [round(v, 3) for v in extents]
    return QCCheck(
        name="bed_fit",
        level=1,
        passed=fits,
        measured=measured,
        expected=list(BED_SIZE_MM),
        message=(
            f"Model measures {measured[0]:.1f} x {measured[1]:.1f} x "
            f"{measured[2]:.1f} mm, within the "
            f"{BED_SIZE_MM[0]:.0f}x{BED_SIZE_MM[1]:.0f}x{BED_SIZE_MM[2]:.0f} "
            f"mm bed."
            if fits
            else (
                f"Model measures {measured[0]:.1f} x {measured[1]:.1f} x "
                f"{measured[2]:.1f} mm, which exceeds the "
                f"{BED_SIZE_MM[0]:.0f}x{BED_SIZE_MM[1]:.0f}x"
                f"{BED_SIZE_MM[2]:.0f} mm bed."
            )
        ),
    )


# --- level 1: wall thickness / detail (measured from the printed mesh) -----


def _wall_thickness_check(footprint: MultiPolygon) -> QCCheck:
    thin = find_thin_regions(footprint, MIN_WALL_MM, label="the base")
    passed = not thin
    narrowest = min((r.width_mm for r in thin), default=None)
    return QCCheck(
        name="wall_thickness",
        level=1,
        passed=passed,
        measured=round(narrowest, 3) if narrowest is not None else None,
        expected=f">= {MIN_WALL_MM} mm",
        message=(
            f"No wall thinner than {MIN_WALL_MM} mm found in the printed "
            f"base."
            if passed
            else (
                f"{len(thin)} region(s) thinner than {MIN_WALL_MM} mm: "
                + "; ".join(r.describe() for r in thin[:3])
            )
        ),
    )


def _missing_object_checks(name: str, spec) -> list[QCCheck]:
    """The base-object checks, all failed, when that object is not there.

    ``wall_thickness``, ``hole_diameter`` and ``hole_margin`` are all
    measured from one named object in the 3MF. If it is absent the honest
    report is three failures naming it, not three checks that quietly do
    not appear -- an absent check reads as a passed one to everybody
    downstream.
    """
    message = (
        f"The 3MF has no object named '{name}', so this could not be "
        f"measured. A keychain export always has one."
    )
    return [
        QCCheck(
            name="wall_thickness",
            level=1,
            passed=False,
            measured=None,
            expected=f">= {MIN_WALL_MM} mm",
            message=message,
        ),
        QCCheck(
            name="hole_diameter",
            level=1,
            passed=False,
            measured=None,
            expected=spec.hole_diameter_mm,
            message=message,
        ),
        QCCheck(
            name="hole_margin",
            level=1,
            passed=False,
            measured=None,
            expected=spec.hole_margin_mm,
            message=message,
        ),
    ]


def _detail_checks(
    scene_meshes: dict[str, trimesh.Trimesh],
    base_mesh: trimesh.Trimesh | None,
    spec,
) -> list[QCCheck]:
    """Every check about the second-colour layer, measured from the export."""
    footprint = _detail_footprint(scene_meshes, base_mesh, spec)
    present = footprint is not None and not footprint.is_empty
    return [
        _detail_depth_check(spec),
        _detail_present_check(footprint, spec),
        _detail_measured_depth_check(scene_meshes, base_mesh, footprint, spec),
        _detail_stroke_check(footprint if present else None),
    ]


def _detail_present_check(footprint: MultiPolygon | None, spec) -> QCCheck:
    present = footprint is not None and not footprint.is_empty
    return QCCheck(
        name="detail_present",
        level=1,
        passed=present,
        measured=round(footprint.area, 3) if present else None,
        expected="a detail layer in the export",
        message=(
            f"The {spec.detail_mode} detail layer is in the export "
            f"({footprint.area:.1f} mm2 of it)."
            if present
            else (
                f"A {spec.detail_mode} detail layer was asked for, but the "
                f"export has none to measure -- the second colour is "
                f"missing from the printed part."
            )
        ),
    )


def _detail_depth_check(spec) -> QCCheck:
    measured = spec.detail_height_mm
    passed = measured >= MIN_DETAIL_DEPTH_MM
    return QCCheck(
        name="detail_depth",
        level=1,
        passed=passed,
        measured=measured,
        expected=f">= {MIN_DETAIL_DEPTH_MM} mm",
        message=(
            f"Detail depth is {measured} mm, at or above the "
            f"{MIN_DETAIL_DEPTH_MM} mm minimum."
            if passed
            else (
                f"Detail depth is {measured} mm, below the "
                f"{MIN_DETAIL_DEPTH_MM} mm minimum -- it may not be "
                f"visible once printed."
            )
        ),
    )


def _detail_stroke_check(footprint: MultiPolygon | None) -> QCCheck:
    if footprint is None or footprint.is_empty:
        # Not a pass. "Nothing to measure" is only ever reached when a
        # detail layer was asked for and the export has none, and calling
        # that a clean stroke-width check is how a missing second colour
        # reaches a "print-ready" verdict.
        return QCCheck(
            name="detail_stroke_width",
            level=1,
            passed=False,
            measured=None,
            expected=f">= {MIN_DETAIL_STROKE_MM} mm",
            message=(
                "There is no detail geometry in the export to measure a "
                "stroke width on."
            ),
        )
    thin = find_thin_regions(footprint, MIN_DETAIL_STROKE_MM, label="the detail")
    passed = not thin
    narrowest = min((r.width_mm for r in thin), default=None)
    return QCCheck(
        name="detail_stroke_width",
        level=1,
        passed=passed,
        measured=round(narrowest, 3) if narrowest is not None else None,
        expected=f">= {MIN_DETAIL_STROKE_MM} mm",
        message=(
            f"No detail stroke thinner than {MIN_DETAIL_STROKE_MM} mm found."
            if passed
            else (
                f"{len(thin)} detail region(s) thinner than "
                f"{MIN_DETAIL_STROKE_MM} mm: "
                + "; ".join(r.describe() for r in thin[:3])
            )
        ),
    )


def _detail_measured_depth_check(
    scene_meshes: dict[str, trimesh.Trimesh],
    base_mesh: trimesh.Trimesh | None,
    footprint: MultiPolygon | None,
    spec,
) -> QCCheck:
    """How deep the relief actually came out, read off the exported mesh.

    :func:`_detail_depth_check` only ever restates ``spec`` back to
    itself, which cannot catch a build that produced the wrong relief.
    This one measures: the top of the raised detail body above the base,
    or -- for a recess, which has no body of its own -- how far the
    pocket floor sits below the base's top surface.
    """
    expected = spec.detail_height_mm
    measured = _measure_detail_depth_mm(scene_meshes, base_mesh, footprint, spec)
    if measured is None:
        return QCCheck(
            name="detail_depth_measured",
            level=1,
            passed=False,
            measured=None,
            expected=expected,
            message=(
                "The detail relief could not be measured in the export -- "
                "there is no detail body and no recess floor to find."
            ),
        )
    passed = abs(measured - expected) <= _DETAIL_DEPTH_TOL_MM
    return QCCheck(
        name="detail_depth_measured",
        level=1,
        passed=passed,
        measured=round(measured, 3),
        expected=expected,
        message=(
            f"The {spec.detail_mode} detail measures {measured:.2f} mm of "
            f"relief in the export (spec: {expected} mm)."
            if passed
            else (
                f"The {spec.detail_mode} detail measures {measured:.2f} mm "
                f"of relief in the export, but {expected} mm was asked "
                f"for -- the part will not look like the design."
            )
        ),
    )


def _measure_detail_depth_mm(
    scene_meshes: dict[str, trimesh.Trimesh],
    base_mesh: trimesh.Trimesh | None,
    footprint: MultiPolygon | None,
    spec,
) -> float | None:
    if spec.detail_mode == "raised":
        mesh = scene_meshes.get(_DETAIL_OBJECT_NAME)
        if mesh is None:
            return None
        return float(mesh.bounds[1][2]) - spec.base_thickness_mm

    if spec.detail_mode == "recessed":
        if base_mesh is None or footprint is None or footprint.is_empty:
            return None
        floor_z = _highest_upward_face_z(base_mesh, footprint)
        if floor_z is None:
            return None
        return spec.base_thickness_mm - floor_z

    return None


def _highest_upward_face_z(
    mesh: trimesh.Trimesh, region: MultiPolygon
) -> float | None:
    """Top of the highest upward-facing surface inside ``region``, in mm.

    A pocket's floor is the only upward-facing surface under the pocket
    (the top face has been cut away there), so this is the recess floor.
    Triangle centroids are used rather than vertices because a planar face
    tessellates to triangles whose corners all sit on its boundary -- there
    may be no vertex strictly inside the pocket at all. Ray casting would
    be the obvious tool and is deliberately not used: trimesh needs
    ``rtree`` for it, which this project does not depend on.
    """
    normals = np.asarray(mesh.face_normals)
    centers = np.asarray(mesh.triangles_center)
    upward = normals[:, 2] >= _UPWARD_NORMAL_MIN
    if not upward.any():
        return None
    best: float | None = None
    for cx, cy, cz in centers[upward]:
        if region.contains(Point(float(cx), float(cy))):
            value = float(cz)
            if best is None or value > best:
                best = value
    return best


def _detail_footprint(
    scene_meshes: dict[str, trimesh.Trimesh],
    base_mesh: trimesh.Trimesh | None,
    spec,
) -> MultiPolygon | None:
    """The 2D footprint of the detail layer as actually printed, in mm.

    Raised detail is its own exported object, so its footprint is a
    section straight through the middle of its body. A recess has no
    separate body -- it's cut into the base -- so its footprint is
    recovered by differencing a section inside the recess band (material
    minus the pocket) out of one below the recess floor (full material).

    **Both recess sections are taken below the chamfer band**, which is
    the whole trick. The chamfer bevels the outline over the top
    ``chamfer_mm`` of the part, so a section taken inside that band has a
    *smaller outline* than one below it, and the difference picks up a
    ring of "material" right round the perimeter that no recess ever cut.
    At the default 0.4 mm chamfer that fires for every legal detail height
    from 0.6 mm up to 0.8 mm -- a whole-perimeter false failure across most
    of the legal range. Keeping both planes under
    ``base_thickness - chamfer_mm`` means the two outlines are identical
    and the difference is the pocket and nothing else.

    Returns ``None`` when there is no detail layer to measure (which the
    caller reports as a failure, not as "nothing to check").
    """
    if spec.detail_mode == "raised":
        mesh = scene_meshes.get(_DETAIL_OBJECT_NAME)
        if mesh is None:
            return None
        z0, z1 = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
        return _clean_footprint(_section(mesh, (z0 + z1) / 2.0))

    if spec.detail_mode == "recessed":
        if base_mesh is None:
            return None
        planes = _recess_probe_planes(spec)
        if planes is None:
            return None
        below_z, band_z = planes
        band = _section(base_mesh, band_z)
        below = _section(base_mesh, below_z)
        return _clean_footprint(make_valid(below.difference(band)))

    return None


def _recess_probe_planes(spec) -> tuple[float, float] | None:
    """(below-the-floor Z, inside-the-recess Z) for the footprint difference.

    Both strictly below the chamfer band, and the pair only exists when
    the recess is deeper than the chamfer is wide. It is not, for a recess
    shallower than ``chamfer_mm`` -- but such a recess is under 0.4 mm
    deep, which ``detail_depth`` already fails on its own.
    """
    top = float(spec.base_thickness_mm)
    floor_z = top - float(spec.detail_height_mm)
    chamfer = (
        float(spec.chamfer_mm)
        if spec.top_edge_chamfer and spec.chamfer_mm > 0
        else 0.0
    )
    safe_top = top - chamfer
    if floor_z <= 0.0 or safe_top <= floor_z:
        return None
    band_z = (floor_z + safe_top) / 2.0
    below_z = floor_z - min(_RECESS_PROBE_MM, floor_z / 2.0)
    return below_z, band_z


def _clean_footprint(geometry: BaseGeometry) -> MultiPolygon:
    """A footprint with the arithmetic noise taken out. See the constant.

    Differencing two mesh sections that share an outline leaves a rash of
    ~1e-18 mm2 slivers along it. Each one is its own polygon, each erodes
    to nothing, and :func:`~print3d.trace.find_thin_regions` reports every
    part that erodes to nothing as "thinner than the minimum all over" --
    so an untouched difference can produce dozens of invented stroke-width
    failures. The floor is applied here, on the QC side, rather than in
    ``trace``: a caller that hands ``find_thin_regions`` a genuine 1e-18
    mm2 polygon does want to hear that it is too thin.
    """
    parts = [
        p
        for p in _polygon_parts(_as_multipolygon(make_valid(geometry)))
        if p.area > _MIN_FOOTPRINT_AREA_MM2
    ]
    return MultiPolygon(parts) if parts else _empty()


# --- level 1: hole diameter / margin (measured from the printed mesh) ------


def _hole_measurements(
    footprint: MultiPolygon, center: tuple[float, float]
) -> tuple[float, float] | None:
    """(measured diameter mm, measured margin mm) of the hole at ``center``.

    Found by locating the polygon part whose *interior* ring (a hole in
    the material) encloses ``center`` -- exact for a circular cut, unlike
    probing a fixed set of ring points. The margin is the distance from
    the centre to the part's *outer* ring minus the hole's own radius,
    the same "distance from centre to boundary" the build stage itself
    uses to decide whether a hole position is legal in the first place.
    """
    point = Point(center)
    for polygon in _polygon_parts(footprint):
        outer_only = Polygon(polygon.exterior)
        if not outer_only.contains(point):
            continue
        exterior_distance = polygon.exterior.distance(point)
        for interior in polygon.interiors:
            if Polygon(interior).contains(point):
                hole_radius = interior.distance(point)
                return 2.0 * hole_radius, exterior_distance - hole_radius
        return None  # centre sits in solid material -- no hole here
    return None


def _hole_diameter_check(
    measurement: tuple[float, float] | None, spec
) -> QCCheck:
    if measurement is None:
        return QCCheck(
            name="hole_diameter",
            level=1,
            passed=False,
            measured=None,
            expected=spec.hole_diameter_mm,
            message="Could not find the hanging hole in the printed base.",
        )
    diameter, _margin = measurement
    passed = abs(diameter - spec.hole_diameter_mm) <= _HOLE_TOL_MM
    return QCCheck(
        name="hole_diameter",
        level=1,
        passed=passed,
        measured=round(diameter, 3),
        expected=spec.hole_diameter_mm,
        message=(
            f"Hole measures {diameter:.2f} mm across "
            f"(spec: {spec.hole_diameter_mm} mm)."
        ),
    )


def _hole_margin_check(measurement: tuple[float, float] | None, spec) -> QCCheck:
    if measurement is None:
        return QCCheck(
            name="hole_margin",
            level=1,
            passed=False,
            measured=None,
            expected=spec.hole_margin_mm,
            message="Could not find the hanging hole in the printed base.",
        )
    _diameter, margin = measurement
    passed = margin >= spec.hole_margin_mm - _HOLE_TOL_MM
    return QCCheck(
        name="hole_margin",
        level=1,
        passed=passed,
        measured=round(margin, 3),
        expected=spec.hole_margin_mm,
        message=(
            f"Hole has {margin:.2f} mm of margin "
            f"(spec: {spec.hole_margin_mm} mm)."
            if passed
            else (
                f"Hole has only {margin:.2f} mm of margin, short of the "
                f"{spec.hole_margin_mm} mm spec."
            )
        ),
    )


# --- geometry plumbing -------------------------------------------------


def _section(mesh: trimesh.Trimesh, z: float) -> MultiPolygon:
    """The material present at height ``z``, as one shapely MultiPolygon.

    Loops are combined with an even-odd (symmetric-difference) rule
    rather than trimesh's own ``polygons_full``, which needs the
    ``rtree`` package -- not a dependency of this project.
    """
    cut = mesh.section(plane_origin=[0.0, 0.0, z], plane_normal=[0.0, 0.0, 1.0])
    if cut is None:
        return _empty()
    region: BaseGeometry | None = None
    for loop in cut.discrete:
        if len(loop) < 4:
            continue
        ring = make_valid(Polygon(np.asarray(loop)[:, :2]))
        if ring.is_empty or ring.area <= 0:
            continue
        region = ring if region is None else region.symmetric_difference(ring)
    if region is None or region.is_empty:
        return _empty()
    return _as_multipolygon(region)


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    out: list[Polygon] = []
    for geom in getattr(geometry, "geoms", [geometry]):
        if geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            out.append(geom)
        elif hasattr(geom, "geoms"):
            out.extend(_polygon_parts(geom))
    return out


def _as_multipolygon(geometry: BaseGeometry) -> MultiPolygon:
    if geometry is None or geometry.is_empty:
        return _empty()
    polygons = [p for p in _polygon_parts(geometry) if p.area > 0]
    return MultiPolygon(polygons) if polygons else _empty()


def _empty() -> MultiPolygon:
    return MultiPolygon()


# --- markdown rendering --------------------------------------------------


def _render_markdown(report: QCReport) -> str:
    lines = [
        f"# Quality report: {report.slug}",
        "",
        f"**Verdict: {report.verdict}**",
        "",
        f"Levels run: {', '.join(str(n) for n in report.levels_run)}",
        "",
    ]

    for level in report.levels_run:
        checks = [c for c in report.checks if c.level == level]
        if not checks:
            continue
        lines.append(f"## Level {level}")
        lines.append("")
        for check in checks:
            mark = "PASS" if check.passed else "FAIL"
            lines.append(f"- [{mark}] **{check.name}** -- {check.message}")
        lines.append("")

    if report.slicer is not None:
        lines.append("## Slicer")
        lines.append("")
        if report.slicer.found:
            lines.append(f"- exe: `{report.slicer.exe}`")
            lines.append(f"- elapsed: {report.slicer.elapsed_s}s")
            if report.slicer.stderr_excerpt:
                lines.append(f"- stderr: `{report.slicer.stderr_excerpt}`")
        else:
            lines.append("- OrcaSlicer was not found.")
        lines.append("")

    if report.warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in report.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines)
