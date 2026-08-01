"""Tests for the QC stage.

Geometry is drawn directly with shapely, the same way ``test_build.py``
does it: QC's contract starts at "a built, exported keychain", so tracing
an image first would only make the tests slower without exercising
anything QC-specific.

The "good" fixture deliberately reuses the flush-detail keychain from
``test_build.py`` (detail == the whole silhouette): the chamfer trims it,
which is a real build-stage warning to check gets surfaced, and both a
base and a detail object end up in the 3MF, exercising every per-object
check QC runs.

**Level 2 is tested without OrcaSlicer.** One test slices for real, and it
is marked ``slicer`` and off unless ``--run-slicer`` is passed (see
``conftest.py``). Everything else about level 2 -- the two success gates,
the exact output path, the fact that the exit code is never consulted --
is driven by a fake ``subprocess.run`` that fabricates whatever outcome
the test needs. That is deliberate: the gates are the part of this module
most likely to rot into "always true", and a suite that can only exercise
them by launching a GUI slicer is a suite that stops exercising them.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest
import trimesh
from shapely.geometry import MultiPolygon, box

from print3d import qc
from print3d.build import build_keychain, export_3mf, export_stl
from print3d.qc import OrcaSlicerConfigError, run_qc, write_report
from print3d.spec import KeychainSpec
from print3d.trace import TraceResult

SIZE_MM = 40.0


# --- fixtures --------------------------------------------------------------


def _spec(**overrides) -> KeychainSpec:
    values = dict(slug="sample", source_image="sample.png")
    values.update(overrides)
    return KeychainSpec(**values)


def _rounded_square(size: float = SIZE_MM, radius: float = 4.0) -> MultiPolygon:
    core = box(radius, radius, size - radius, size - radius)
    return MultiPolygon([core.buffer(radius, quad_segs=16)])


def _built(out: Path, silhouette, detail, spec, stem: str = "sample"):
    """(trace, solids, 3mf, stl) for a freshly built, freshly exported part.

    Used by every test that needs to *mutate* the spec or the export: the
    ``good`` fixture is module-scoped and shared, so nothing may touch it.
    """
    solids = build_keychain(silhouette, detail, spec)
    mf3 = export_3mf(solids, out / f"{stem}.3mf")
    stl = export_stl(solids, out / f"{stem}.stl")
    trace = TraceResult(
        silhouette=silhouette,
        detail=detail if detail is not None else MultiPolygon(),
        size_mm=SIZE_MM,
        source=Path(f"{stem}.png"),
    )
    return trace, solids, mf3, stl


@pytest.fixture(scope="module")
def good(tmp_path_factory):
    """A built, exported keychain plus the TraceResult that "produced" it.

    Detail is flush with the silhouette, so the build stage trims it back
    from the chamfered edge and leaves a warning behind - used below to
    check that build warnings reach the report. A second warning is
    stitched onto a hand-built TraceResult to check the trace side too.
    """
    out = tmp_path_factory.mktemp("good")
    square = _rounded_square()
    spec = _spec()
    solids = build_keychain(square, square, spec)
    assert any("trimmed back" in w for w in solids.warnings), solids.warnings

    mf3 = export_3mf(solids, out / "sample.3mf")
    stl = export_stl(solids, out / "sample.stl")
    trace = TraceResult(
        silhouette=square,
        detail=square,
        size_mm=SIZE_MM,
        source=Path("sample.png"),
        warnings=["TRACE_WARNING_MARKER: thin detail dropped upstream"],
    )
    return trace, solids, mf3, stl, out


# --- fake-slicer harness ---------------------------------------------------


def _fake_exe(tmp_path: Path) -> Path:
    """A file that passes the exe validation without being an executable."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    exe = tmp_path / "fake-orca-slicer.exe"
    exe.write_bytes(b"not a real executable")
    return exe


def _plate_gcode(total: str = "1h 12m 30s") -> bytes:
    return (
        "; HEADER_BLOCK_START\n"
        f"; model printing time: {total}; total estimated time: {total}\n"
        "; HEADER_BLOCK_END\n" + "; filler line for bulk\n" * 100
    ).encode("utf-8")


def _write_gcode_3mf(path: Path, plates: dict[str, bytes] | None = None) -> Path:
    """A stand-in ``.gcode.3mf``: a real zip, over 1 KB, plate gcode inside.

    Stored (not deflated) so the padding actually makes the file big: the
    size gate is measured in bytes on disk.
    """
    if plates is None:
        plates = {"Metadata/plate_1.gcode": _plate_gcode()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name, data in plates.items():
            zf.writestr(name, data)
        zf.writestr("Metadata/_padding", b"x" * 4096)
    return path


class _FakeSlicer:
    """Stands in for ``subprocess.run``; records the call, fakes the output.

    ``writes`` is handed the ``--export-3mf`` path taken from the argv QC
    actually built, so a test can write the artifact where QC asked for it,
    somewhere else, or not at all.
    """

    def __init__(self, writes=None, returncode: int = 0, stderr: str = ""):
        self.writes = writes
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        if self.writes is not None:
            self.writes(Path(argv[argv.index("--export-3mf") + 1]))
        return subprocess.CompletedProcess(argv, self.returncode, "", self.stderr)

    @property
    def cwd(self):
        return self.calls[-1][1].get("cwd")


def _slice_report(monkeypatch, good, work_dir: Path, fake: _FakeSlicer):
    """Run a full QC with ``fake`` standing in for OrcaSlicer."""
    trace, solids, mf3, stl, _out = good
    monkeypatch.setattr(qc.subprocess, "run", fake)
    return run_qc(
        trace,
        solids,
        mf3,
        stl,
        work_dir,
        run_level2=True,
        slicer_exe=_fake_exe(work_dir),
    )


def _slice_check(report):
    return next(c for c in report.checks if c.name == "slice_success")


def _by_name(report) -> dict:
    return {c.name: c for c in report.checks}


# --- level 1: the happy path -------------------------------------------


def test_good_sample_passes_level1(good):
    trace, solids, mf3, stl, out = good
    report = run_qc(trace, solids, mf3, stl, out, run_level2=False)

    assert report.levels_run == [1]
    assert report.verdict == "passes-level-1"
    failed = [c for c in report.checks if not c.passed]
    assert failed == [], failed
    assert report.slicer is None


def test_upstream_warnings_are_surfaced(good):
    trace, solids, mf3, stl, out = good
    report = run_qc(trace, solids, mf3, stl, out, run_level2=False)

    assert "TRACE_WARNING_MARKER: thin detail dropped upstream" in report.warnings
    assert any("trimmed back" in w for w in report.warnings)


# --- report schema -----------------------------------------------------


def test_report_json_schema_is_documented_and_stable(good, tmp_path):
    trace, solids, mf3, stl, out = good
    report = run_qc(trace, solids, mf3, stl, out, run_level2=False)

    json_path, md_path = write_report(report, tmp_path / "report_out")
    assert json_path.name == "report.json"
    assert md_path.name == "report.md"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(data.keys()) == {
        "schema_version",
        "slug",
        "verdict",
        "levels_run",
        "checks",
        "warnings",
        "slicer",
    }
    assert data["schema_version"] == qc.SCHEMA_VERSION
    assert data["slug"] == "sample"
    assert data["verdict"] in ("print-ready", "passes-level-1", "failed")
    assert data["levels_run"] == [1]
    assert data["slicer"] is None

    assert data["checks"], "expected at least one check"
    for check in data["checks"]:
        assert set(check.keys()) == {
            "name",
            "level",
            "passed",
            "measured",
            "expected",
            "message",
        }

    assert md_path.read_text(encoding="utf-8").startswith("# Quality report")


def test_every_mandatory_check_is_present_by_name(good):
    """The report is only worth reading if the checks are actually in it.

    A check that silently stops being emitted looks exactly like a check
    that passed, so the roster itself is pinned here as well as each
    check's behaviour being pinned by its own negative test below.
    """
    trace, solids, mf3, stl, out = good
    report = run_qc(trace, solids, mf3, stl, out, run_level2=False)
    names = set(_by_name(report))

    assert {
        "export_3mf.readable",
        "export_stl.readable",
        "bed_fit",
        "wall_thickness",
        "hole_diameter",
        "hole_margin",
        "detail_depth",
        "detail_depth_measured",
        "detail_present",
        "detail_stroke_width",
    } <= names
    for obj in ("base", "detail", "combined_stl"):
        assert {
            f"{obj}.watertight",
            f"{obj}.winding_consistent",
            f"{obj}.is_volume",
            f"{obj}.no_degenerate_faces",
            f"{obj}.body_count",
        } <= names


# --- mesh integrity ------------------------------------------------------


def test_a_broken_mesh_fails_watertight_by_name(good, tmp_path):
    """Deleting faces from the fused STL must fail the watertight check,
    and the failing check must be identifiable by name."""
    trace, solids, mf3, stl, out = good

    mesh = trimesh.load(str(stl), force="mesh")
    assert mesh.is_watertight  # sanity: the good export really is sound

    broken = mesh.copy()
    keep = np.arange(len(broken.faces) - 8)  # drop the last 8 faces
    broken.update_faces(keep)
    broken.remove_unreferenced_vertices()
    broken_path = tmp_path / "broken.stl"
    broken.export(str(broken_path))
    assert not trimesh.load(str(broken_path), force="mesh").is_watertight

    report = run_qc(trace, solids, mf3, broken_path, out, run_level2=False)
    by_name = _by_name(report)

    assert "combined_stl.watertight" in by_name
    assert by_name["combined_stl.watertight"].passed is False
    assert report.verdict == "failed"


def test_a_zero_area_triangle_fails_the_degenerate_face_check(good, tmp_path):
    trace, solids, mf3, stl, out = good
    mesh = trimesh.load(str(stl), force="mesh")

    vertices = np.vstack([mesh.vertices, [mesh.vertices[0]]])
    faces = np.vstack([mesh.faces, [[0, 0, 1]]])  # two corners in one place
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(
        str(tmp_path / "degenerate.stl")
    )

    report = run_qc(
        trace, solids, mf3, tmp_path / "degenerate.stl", out, run_level2=False
    )
    check = _by_name(report)["combined_stl.no_degenerate_faces"]

    assert check.passed is False
    assert check.measured == 1
    assert report.verdict == "failed"


def test_two_loose_bodies_in_one_stl_fail_the_body_count_check(good, tmp_path):
    trace, solids, mf3, stl, out = good

    left = trimesh.creation.box(extents=(10, 10, 3))
    right = trimesh.creation.box(extents=(10, 10, 3))
    right.apply_translation([40.0, 0.0, 0.0])
    loose = tmp_path / "loose.stl"
    trimesh.util.concatenate([left, right]).export(str(loose))

    report = run_qc(trace, solids, mf3, loose, out, run_level2=False)
    check = _by_name(report)["combined_stl.body_count"]

    assert check.passed is False
    assert check.measured == 2
    assert report.verdict == "failed"


def test_multiple_raised_detail_regions_are_legal(tmp_path):
    """Two disconnected detail regions (initials, letters, eyes) are
    legitimate two-color artwork: each prints fused to the base, so only
    the base and the fused STL keep the strict one-body expectation."""
    silhouette = _rounded_square()
    detail = MultiPolygon([box(8, 8, 16, 16), box(24, 8, 32, 16)])
    trace, solids, mf3, stl = _built(tmp_path, silhouette, detail, _spec())

    report = run_qc(trace, solids, mf3, stl, tmp_path, run_level2=False)
    by_name = _by_name(report)

    detail_check = by_name["detail.body_count"]
    assert detail_check.passed is True
    assert detail_check.measured == 2
    assert by_name["base.body_count"].passed is True
    assert by_name["combined_stl.body_count"].passed is True
    assert report.verdict == "passes-level-1"


# --- unreadable exports are a verdict, not a crash -------------------------


def test_a_missing_3mf_becomes_a_failed_check(good, tmp_path):
    trace, solids, mf3, stl, out = good
    report = run_qc(
        trace, solids, tmp_path / "not-here.3mf", stl, out, run_level2=False
    )
    check = _by_name(report)["export_3mf.readable"]

    assert check.passed is False
    assert "not-here.3mf" in check.message
    assert report.verdict == "failed"


def test_a_corrupt_3mf_becomes_a_failed_check(good, tmp_path):
    trace, solids, mf3, stl, out = good
    corrupt = tmp_path / "corrupt.3mf"
    corrupt.write_bytes(b"PK\x03\x04 this is not really a zip at all")

    report = run_qc(trace, solids, corrupt, stl, out, run_level2=False)
    check = _by_name(report)["export_3mf.readable"]

    assert check.passed is False
    assert report.verdict == "failed"


def test_a_missing_stl_becomes_a_failed_check(good, tmp_path):
    trace, solids, mf3, stl, out = good
    report = run_qc(
        trace, solids, mf3, tmp_path / "not-here.stl", out, run_level2=False
    )
    check = _by_name(report)["export_stl.readable"]

    assert check.passed is False
    assert report.verdict == "failed"


def test_a_corrupt_stl_becomes_a_failed_check(good, tmp_path):
    trace, solids, mf3, stl, out = good
    corrupt = tmp_path / "corrupt.stl"
    corrupt.write_bytes(b"solid nonsense\nthis is not an STL body\n")

    report = run_qc(trace, solids, mf3, corrupt, out, run_level2=False)

    assert _by_name(report)["export_stl.readable"].passed is False
    assert report.verdict == "failed"


# --- a missing object fails its checks rather than skipping them -----------


def test_a_3mf_without_the_base_object_fails_the_base_checks(good, tmp_path):
    trace, solids, mf3, stl, out = good
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.box(extents=(10, 10, 3)),
        geom_name="widget",
        node_name="widget",
    )
    no_base = tmp_path / "no-base.3mf"
    scene.export(str(no_base), file_type="3mf")

    report = run_qc(trace, solids, no_base, stl, out, run_level2=False)
    by_name = _by_name(report)

    for name in ("wall_thickness", "hole_diameter", "hole_margin"):
        assert name in by_name, f"{name} was dropped instead of failed"
        assert by_name[name].passed is False
        assert "base" in by_name[name].message
    assert report.verdict == "failed"


# --- bed fit ---------------------------------------------------------------


def test_an_oversize_model_fails_bed_fit_by_name(tmp_path):
    big = _rounded_square(size=200.0)
    spec = _spec(detail_enabled=False, max_dimension_mm=200.0)
    solids = build_keychain(big, None, spec)

    out = tmp_path / "big"
    mf3 = export_3mf(solids, out / "big.3mf")
    stl = export_stl(solids, out / "big.stl")
    trace = TraceResult(
        silhouette=big, detail=MultiPolygon(), size_mm=200.0, source=Path("big.png")
    )

    report = run_qc(trace, solids, mf3, stl, out, run_level2=False)
    by_name = _by_name(report)

    assert "bed_fit" in by_name
    assert by_name["bed_fit"].passed is False
    assert report.verdict == "failed"


# --- wall thickness --------------------------------------------------------


def test_a_thin_neck_fails_the_wall_thickness_check(tmp_path):
    """Measured from the exported mesh, so a sub-minimum wall must show up."""
    silhouette = MultiPolygon(
        [_rounded_square().union(box(SIZE_MM - 1.0, 19.6, SIZE_MM + 15.0, 20.4))]
    )
    spec = _spec(detail_enabled=False, max_dimension_mm=60.0)
    trace, solids, mf3, stl = _built(tmp_path, silhouette, None, spec, "neck")

    report = run_qc(trace, solids, mf3, stl, tmp_path, run_level2=False)
    check = _by_name(report)["wall_thickness"]

    assert check.passed is False, check.message
    assert check.measured is not None and check.measured < 1.2
    assert report.verdict == "failed"


# --- hole diameter and margin ----------------------------------------------


def test_a_hole_that_does_not_match_the_spec_fails_by_name(tmp_path):
    spec = _spec(detail_enabled=False)
    trace, solids, mf3, stl = _built(
        tmp_path, _rounded_square(), None, spec, "hole"
    )
    # The part is built and exported; now ask for a hole it does not have.
    solids.spec.hole_diameter_mm = 8.0

    report = run_qc(trace, solids, mf3, stl, tmp_path, run_level2=False)
    check = _by_name(report)["hole_diameter"]

    assert check.passed is False, check.message
    assert check.measured == pytest.approx(5.2, abs=0.1)
    assert report.verdict == "failed"


def test_a_hole_margin_under_the_spec_fails_by_name(tmp_path):
    spec = _spec(detail_enabled=False)
    trace, solids, mf3, stl = _built(
        tmp_path, _rounded_square(), None, spec, "margin"
    )
    solids.spec.hole_margin_mm = 20.0  # far more than the part can offer

    report = run_qc(trace, solids, mf3, stl, tmp_path, run_level2=False)
    check = _by_name(report)["hole_margin"]

    assert check.passed is False, check.message
    assert check.measured is not None and check.measured < 20.0
    assert report.verdict == "failed"


# --- detail: depth, presence and stroke width ------------------------------


def test_a_shallow_detail_depth_fails_the_spec_check(tmp_path):
    spec = _spec(detail_height_mm=0.4)
    trace, solids, mf3, stl = _built(
        tmp_path, _rounded_square(), MultiPolygon([box(10, 10, 30, 30)]), spec, "shallow"
    )

    report = run_qc(trace, solids, mf3, stl, tmp_path, run_level2=False)
    by_name = _by_name(report)

    assert by_name["detail_depth"].passed is False
    assert by_name["detail_depth"].measured == 0.4
    # The relief really is 0.4 mm, so the *measured* check agrees with the
    # export - it is the spec that is out of bounds.
    assert by_name["detail_depth_measured"].passed is True
    assert report.verdict == "failed"


def test_the_measured_detail_depth_reads_the_export_not_the_spec(tmp_path):
    spec = _spec(detail_height_mm=0.8)
    trace, solids, mf3, stl = _built(
        tmp_path, _rounded_square(), MultiPolygon([box(10, 10, 30, 30)]), spec, "relief"
    )
    # A 0.8 mm relief was built and exported; now claim 1.5 mm was wanted.
    solids.spec.detail_height_mm = 1.5

    report = run_qc(trace, solids, mf3, stl, tmp_path, run_level2=False)
    check = _by_name(report)["detail_depth_measured"]

    assert check.passed is False, check.message
    assert check.measured == pytest.approx(0.8, abs=0.05)
    assert report.verdict == "failed"


def test_detail_requested_but_absent_from_the_export_fails(tmp_path):
    spec = _spec()
    trace, solids, _mf3, stl = _built(
        tmp_path, _rounded_square(), MultiPolygon([box(10, 10, 30, 30)]), spec, "nodetail"
    )
    # Same spec (detail still requested), but an export with only the base.
    base_only = export_3mf(
        dataclasses.replace(solids, detail=None), tmp_path / "base-only.3mf"
    )

    report = run_qc(trace, solids, base_only, stl, tmp_path, run_level2=False)
    by_name = _by_name(report)

    assert by_name["detail_present"].passed is False
    assert by_name["detail_stroke_width"].passed is False
    assert by_name["detail_depth_measured"].passed is False
    assert report.verdict == "failed"


@pytest.mark.parametrize("detail_height_mm", [0.6, 0.7, 0.8])
@pytest.mark.parametrize("top_edge_chamfer", [True, False])
def test_a_legal_recess_passes_across_the_whole_range(
    tmp_path, detail_height_mm, top_edge_chamfer
):
    """No false failures anywhere in the legal recess-depth range.

    The recess footprint is recovered by differencing two sections of the
    base. Take the upper one inside the chamfer band and the difference
    picks up a ring right round the perimeter that no recess ever cut -
    which fired for every depth under twice the chamfer width. Leave the
    slivers in and a chamfer-free part yields dozens of 1e-18 mm2 scraps,
    each reported as its own too-thin region. Both are false, and both
    used to fail here.
    """
    spec = _spec(
        detail_recessed=True,
        detail_height_mm=detail_height_mm,
        top_edge_chamfer=top_edge_chamfer,
    )
    trace, solids, mf3, stl = _built(
        tmp_path, _rounded_square(), MultiPolygon([box(10, 10, 30, 30)]), spec, "recess"
    )

    report = run_qc(trace, solids, mf3, stl, tmp_path, run_level2=False)
    failed = [(c.name, c.message) for c in report.checks if not c.passed]

    assert failed == [], failed
    assert _by_name(report)["detail_present"].measured == pytest.approx(400.0, abs=1.0)
    assert _by_name(report)["detail_depth_measured"].measured == pytest.approx(
        detail_height_mm, abs=0.05
    )


def test_a_too_narrow_recess_stroke_still_fails(tmp_path):
    """The other half of the fix above: it must not just pass everything."""
    spec = _spec(detail_recessed=True, detail_height_mm=0.8)
    hairline = MultiPolygon([box(10.0, 19.75, 30.0, 20.25)])  # 0.5 mm across
    trace, solids, mf3, stl = _built(
        tmp_path, _rounded_square(), hairline, spec, "hairline"
    )

    report = run_qc(trace, solids, mf3, stl, tmp_path, run_level2=False)
    by_name = _by_name(report)

    assert by_name["detail_present"].passed is True
    assert by_name["detail_stroke_width"].passed is False, by_name[
        "detail_stroke_width"
    ].message
    assert by_name["detail_stroke_width"].measured == pytest.approx(0.5, abs=0.1)
    assert report.verdict == "failed"


def test_a_too_narrow_raised_stroke_fails(tmp_path):
    spec = _spec(detail_height_mm=0.8)
    hairline = MultiPolygon([box(10.0, 19.75, 30.0, 20.25)])
    trace, solids, mf3, stl = _built(
        tmp_path, _rounded_square(), hairline, spec, "raisedhair"
    )

    report = run_qc(trace, solids, mf3, stl, tmp_path, run_level2=False)
    check = _by_name(report)["detail_stroke_width"]

    assert check.passed is False, check.message
    assert report.verdict == "failed"


# --- level 2: slicer discovery and the capped verdict -----------------


def test_slicer_absent_caps_the_verdict_and_warns_by_name(monkeypatch, good):
    trace, solids, mf3, stl, out = good
    monkeypatch.setattr(qc, "find_orcaslicer_exe", lambda: None)

    report = run_qc(trace, solids, mf3, stl, out, run_level2=True)

    assert report.levels_run == [1]
    assert report.verdict == "passes-level-1"
    assert report.slicer is not None
    assert report.slicer.found is False
    assert any("OrcaSlicer was not found" in w for w in report.warnings)


def test_orcaslicer_exe_env_var_pointing_nowhere_is_a_hard_fail(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCASLICER_EXE", str(tmp_path / "does-not-exist.exe"))
    with pytest.raises(OrcaSlicerConfigError, match="ORCASLICER_EXE"):
        qc.find_orcaslicer_exe()


def test_orcaslicer_exe_env_var_pointing_at_a_directory_is_a_hard_fail(
    monkeypatch, tmp_path
):
    """A directory passes exists() and then explodes as a PermissionError."""
    folder = tmp_path / "OrcaSlicer"
    folder.mkdir()
    monkeypatch.setenv("ORCASLICER_EXE", str(folder))
    with pytest.raises(OrcaSlicerConfigError, match="not a file"):
        qc.find_orcaslicer_exe()


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_an_empty_orcaslicer_exe_env_var_means_unset(monkeypatch, tmp_path, value):
    """Blank is what a wrapper script leaves behind, not a misconfiguration."""
    monkeypatch.setenv("ORCASLICER_EXE", value)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path))

    assert qc.find_orcaslicer_exe() is None  # falls through, does not raise


def test_run_qc_propagates_the_env_var_hard_fail(monkeypatch, good, tmp_path):
    trace, solids, mf3, stl, out = good
    # A local, definitely-nonexistent path - not a bare drive letter (e.g.
    # "Z:\\..."): on Windows, Path.exists() on an unmapped/disconnected
    # drive letter can block for minutes resolving it, which is not what
    # this test means to exercise.
    monkeypatch.setenv("ORCASLICER_EXE", str(tmp_path / "nowhere" / "orca-slicer.exe"))
    with pytest.raises(OrcaSlicerConfigError):
        run_qc(trace, solids, mf3, stl, out, run_level2=True)


def test_a_bad_slicer_exe_override_is_the_same_typed_error(good, tmp_path):
    """Not a raw FileNotFoundError from somewhere inside subprocess."""
    trace, solids, mf3, stl, out = good
    with pytest.raises(OrcaSlicerConfigError, match="not a file"):
        run_qc(
            trace,
            solids,
            mf3,
            stl,
            out,
            run_level2=True,
            slicer_exe=tmp_path / "nowhere" / "orca-slicer.exe",
        )


def test_a_directory_slicer_exe_override_is_the_same_typed_error(good, tmp_path):
    trace, solids, mf3, stl, out = good
    with pytest.raises(OrcaSlicerConfigError, match="not a file"):
        run_qc(
            trace, solids, mf3, stl, out, run_level2=True, slicer_exe=tmp_path
        )


def test_run_level2_false_never_touches_the_slicer(monkeypatch, good):
    """Belt and braces: run_level2=False must not even probe for an exe."""
    trace, solids, mf3, stl, out = good

    def _boom():
        raise AssertionError("find_orcaslicer_exe should not be called")

    monkeypatch.setattr(qc, "find_orcaslicer_exe", _boom)
    report = run_qc(trace, solids, mf3, stl, out, run_level2=False)
    assert report.levels_run == [1]


# --- level 2: the two success gates, driven by a fake slicer --------------


def test_a_slice_that_produces_nothing_fails_and_sinks_the_verdict(
    monkeypatch, good, tmp_path
):
    """Level 1 is spotless here; the verdict must still be "failed"."""
    fake = _FakeSlicer(writes=None)
    report = _slice_report(monkeypatch, good, tmp_path, fake)

    assert report.levels_run == [1, 2]
    assert [c.name for c in report.checks if not c.passed] == ["slice_success"]
    assert "No .gcode.3mf was produced" in _slice_check(report).message
    assert report.verdict == "failed"


def test_a_slice_artifact_under_1kb_fails(monkeypatch, good, tmp_path):
    fake = _FakeSlicer(writes=lambda out: out.write_bytes(b"stub"))
    report = _slice_report(monkeypatch, good, tmp_path, fake)

    check = _slice_check(report)
    assert check.passed is False
    assert check.measured == 4
    assert report.verdict == "failed"


def test_a_zero_print_time_header_fails(monkeypatch, good, tmp_path):
    fake = _FakeSlicer(
        writes=lambda out: _write_gcode_3mf(
            out, {"Metadata/plate_1.gcode": _plate_gcode("0s")}
        )
    )
    report = _slice_report(monkeypatch, good, tmp_path, fake)

    check = _slice_check(report)
    assert check.passed is False
    assert "print-time header" in check.message
    assert report.verdict == "failed"


def test_a_gcode_3mf_with_no_plate_gcode_fails(monkeypatch, good, tmp_path):
    fake = _FakeSlicer(writes=lambda out: _write_gcode_3mf(out, {}))
    report = _slice_report(monkeypatch, good, tmp_path, fake)

    assert _slice_check(report).passed is False
    assert report.verdict == "failed"


def test_a_slice_artifact_written_where_qc_asked_passes(monkeypatch, good, tmp_path):
    fake = _FakeSlicer(writes=_write_gcode_3mf)
    report = _slice_report(monkeypatch, good, tmp_path, fake)

    check = _slice_check(report)
    assert check.passed is True, check.message
    assert check.measured == pytest.approx(4350.0)  # 1h 12m 30s
    assert report.levels_run == [1, 2]
    assert report.verdict == "print-ready"
    assert report.slicer.found is True


def test_an_artifact_at_some_other_path_does_not_count(monkeypatch, good, tmp_path):
    """The gate is the exact --export-3mf path, not "a .gcode.3mf somewhere"."""
    fake = _FakeSlicer(
        writes=lambda out: _write_gcode_3mf(out.with_name("elsewhere.gcode.3mf"))
    )
    report = _slice_report(monkeypatch, good, tmp_path, fake)

    assert _slice_check(report).passed is False
    assert report.verdict == "failed"


def test_a_stale_artifact_from_a_previous_run_does_not_pass_the_slice(
    monkeypatch, good, tmp_path
):
    """The single nastiest failure mode: a false "print-ready".

    A real failing slice was observed leaving the previous run's
    ``.gcode.3mf`` in place. Both success gates are measurements of that
    file, so unless it is deleted first they both pass and QC reports a
    part that never sliced as print-ready.
    """
    trace, solids, mf3, stl, _out = good
    stale = _write_gcode_3mf(tmp_path / f"{Path(stl).stem}.gcode.3mf")
    assert stale.exists() and stale.stat().st_size > 1024

    fake = _FakeSlicer(writes=None)  # this run produces nothing at all
    report = _slice_report(monkeypatch, good, tmp_path, fake)

    assert _slice_check(report).passed is False, "a stale artifact was accepted"
    assert report.verdict == "failed"
    assert not stale.exists()


def test_the_exit_code_is_never_the_gate(monkeypatch, good, tmp_path):
    """A real failure returned 4294967246; other failures returned 0.

    So the exit code is read in neither direction: a wild return code with
    a good artifact passes, and a clean 0 with no artifact fails.
    """
    loud = _FakeSlicer(writes=_write_gcode_3mf, returncode=4294967246)
    assert _slice_check(_slice_report(monkeypatch, good, tmp_path, loud)).passed

    quiet = _FakeSlicer(writes=None, returncode=0)
    assert not _slice_check(
        _slice_report(monkeypatch, good, tmp_path / "second", quiet)
    ).passed


def test_a_single_zero_time_plate_fails_however_the_plates_are_numbered(
    monkeypatch, good, tmp_path
):
    """Plate 2 is broken and plate 10 is fine; lexical order hides that."""
    fake = _FakeSlicer(
        writes=lambda out: _write_gcode_3mf(
            out,
            {
                "Metadata/plate_2.gcode": _plate_gcode("0s"),
                "Metadata/plate_10.gcode": _plate_gcode("1h"),
            },
        )
    )
    report = _slice_report(monkeypatch, good, tmp_path, fake)

    assert _slice_check(report).passed is False
    assert report.verdict == "failed"


def test_the_slicer_is_started_in_the_work_dir(monkeypatch, good, tmp_path):
    """OrcaSlicer writes 00000.log into its CWD; keep that out of the repo."""
    fake = _FakeSlicer(writes=_write_gcode_3mf)
    _slice_report(monkeypatch, good, tmp_path, fake)

    assert fake.cwd == str(tmp_path)
    argv = fake.calls[-1][0]
    assert "--outputdir" not in argv  # export path is exact, never a dir
    assert argv[argv.index("--export-3mf") + 1].endswith(".gcode.3mf")


def test_print_time_parsing_keeps_the_day_component():
    assert qc._parse_duration("3h") == 3 * 3600
    assert qc._parse_duration("2 days 3h") == 2 * 86400 + 3 * 3600
    assert qc._parse_duration("1d 2h 3m 4s") == 86400 + 2 * 3600 + 3 * 60 + 4


# --- level 2: a real slice (off unless --run-slicer) ----------------------


@pytest.mark.slicer
def test_real_slice_passes_both_gates(good, tmp_path):
    trace, solids, mf3, stl, _out = good
    if qc.find_orcaslicer_exe() is None:
        pytest.skip("OrcaSlicer is not installed on this machine")

    # OrcaSlicer drops a 00000.log into its working directory. Snapshot the
    # one in the process CWD (the repo root, when run the usual way) so the
    # assertion below is about *this* run rather than about whatever the
    # machine happened to have lying around.
    cwd_log = Path.cwd() / "00000.log"
    before = cwd_log.stat().st_mtime_ns if cwd_log.exists() else None

    report = run_qc(trace, solids, mf3, stl, tmp_path, run_level2=True)

    assert report.levels_run == [1, 2]
    slice_check = _slice_check(report)
    assert slice_check.passed is True, slice_check.message
    assert report.slicer.found is True
    assert report.slicer.elapsed_s is not None and report.slicer.elapsed_s > 0
    assert report.verdict == "print-ready"

    # The slicer's own log belongs with the slice output, not in the repo.
    after = cwd_log.stat().st_mtime_ns if cwd_log.exists() else None
    assert after == before, f"OrcaSlicer wrote {cwd_log} instead of the work dir"
    assert (tmp_path / "00000.log").exists(), "expected the log in the work dir"
