"""Tests for the CLI (``make3d``).

Images are drawn with Pillow, the same way ``test_trace.py`` does it, so
the fixtures are readable in the diff and the suite stays fast (no real
OrcaSlicer). Every test that runs the pipeline passes ``--no-slice`` and
redirects ``--designs-root`` to a ``tmp_path``, so the suite never touches
the real ``designs/`` folder and never launches a slicer.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from print3d import cli
from print3d.qc import QCCheck, QCReport

SIZE_MM = 50.0


# --- fixtures ----------------------------------------------------------


@pytest.fixture
def sample_image(tmp_path):
    """A bold grey disc with a smaller, darker inner bar (the detail)."""
    image = Image.new("L", (300, 300), 255)
    draw = ImageDraw.Draw(image)
    draw.ellipse([20, 20, 279, 279], fill=160)
    draw.rectangle([90, 130, 209, 170], fill=20)
    path = tmp_path / "sample.png"
    image.save(path)
    return path


@pytest.fixture
def hairline_image(tmp_path):
    """1 px strokes: at the default 50 mm size, every wall is unprintable."""
    image = Image.new("L", (300, 300), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 279, 279], outline=0, width=1)
    draw.line([20, 20, 279, 279], fill=0, width=1)
    path = tmp_path / "hairline.png"
    image.save(path)
    return path


@pytest.fixture
def flat_image(tmp_path):
    """One dark shape, no lighter/darker split: traces to an empty detail layer."""
    image = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse([20, 20, 279, 279], fill=(10, 10, 10, 255))
    path = tmp_path / "flat.png"
    image.save(path)
    return path


@pytest.fixture
def photo_image(tmp_path):
    """A phone-photo-style fixture: an illumination gradient plus a bold
    ink ring, the same shape as ``sample_image`` but lit unevenly."""
    w, h = 320, 320
    yy, xx = np.mgrid[0:h, 0:w]
    gradient = 235 - 70 * (xx / w) - 40 * (yy / h)
    image = Image.fromarray(gradient.astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.ellipse([70, 70, 250, 250], outline=(15, 15, 15), width=16)
    path = tmp_path / "photo.png"
    image.save(path)
    return path


def _run(image, *extra, designs_root, exit_ok=True):
    rc = cli.main(
        [str(image), "--type", "keychain", "--no-slice", "--designs-root", str(designs_root), *extra]
    )
    if exit_ok:
        assert rc == 0
    return rc


# --- full pipeline -------------------------------------------------------


def test_full_pipeline_no_slice_writes_a_complete_design_folder(sample_image, tmp_path):
    designs_root = tmp_path / "designs"
    rc = _run(sample_image, "--name", "Sample Keychain!", designs_root=designs_root)
    assert rc == 0

    design_dir = designs_root / "sample-keychain"
    assert design_dir.is_dir()
    names = {p.name for p in design_dir.iterdir()}
    assert names == {
        sample_image.name,
        "traced.svg",
        "sample-keychain.3mf",
        "sample-keychain.stl",
        "report.json",
        "report.md",
    }
    # No staging leftover once a run has succeeded.
    assert not (designs_root / ".staging-sample-keychain").exists()

    report = json.loads((design_dir / "report.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "passes-level-1"
    assert report["levels_run"] == [1]
    assert all(c["passed"] for c in report["checks"]), report["checks"]


def test_summary_is_printed_to_stdout(sample_image, tmp_path, capsys):
    designs_root = tmp_path / "designs"
    _run(sample_image, "--name", "printed", designs_root=designs_root)
    out = capsys.readouterr().out
    assert "verdict: passes-level-1" in out
    assert "report:" in out
    assert "passes-level-1" in out  # the --no-slice cap note mentions it


# --- registry dispatch -----------------------------------------------------


def test_registry_starts_with_only_keychain():
    assert set(cli.REGISTRY) == {"keychain"}
    assert cli.REGISTRY["keychain"] is cli._build_keychain


def test_unknown_type_lists_registered_types_and_exits_nonzero(tmp_path, capsys):
    # The type is checked before the image even needs to exist, so a bogus
    # path here does not muddy what is under test.
    rc = cli.main([str(tmp_path / "nope.png"), "--type", "nonsense"])
    captured = capsys.readouterr()
    assert rc != 0
    assert "keychain" in captured.err
    assert "nonsense" in captured.err


def test_keychain_resolves_through_the_registry_not_an_ifelse(
    monkeypatch, sample_image, tmp_path
):
    """Spy on the registered keychain builder: if it never runs, dispatch
    happened some other way (an if/else branch bypassing the registry)."""
    calls = []
    original = cli.REGISTRY["keychain"]

    def spy(request):
        calls.append(request)
        return original(request)

    monkeypatch.setitem(cli.REGISTRY, "keychain", spy)
    designs_root = tmp_path / "designs"
    rc = _run(sample_image, designs_root=designs_root)
    assert rc == 0
    assert len(calls) == 1


def test_a_dummy_registered_type_is_dispatched_too(monkeypatch, sample_image, tmp_path):
    """A type the CLI has never heard of still runs, purely via the registry."""
    calls = []

    def dummy_builder(request):
        calls.append(request)
        return cli._build_keychain(request)

    monkeypatch.setitem(cli.REGISTRY, "widget", dummy_builder)
    designs_root = tmp_path / "designs"
    rc = cli.main(
        [str(sample_image), "--type", "widget", "--no-slice", "--designs-root", str(designs_root)]
    )
    assert rc == 0
    assert len(calls) == 1


# --- exit codes ------------------------------------------------------------


def test_missing_required_arguments_is_a_bad_argument_exit():
    with pytest.raises(SystemExit) as caught:
        cli.main([])
    assert caught.value.code not in (0, None)


def test_thin_feature_image_exits_nonzero_with_an_actionable_message(
    hairline_image, tmp_path, capsys
):
    designs_root = tmp_path / "designs"
    rc = cli.main(
        [str(hairline_image), "--type", "keychain", "--no-slice", "--designs-root", str(designs_root)]
    )
    captured = capsys.readouterr()
    assert rc != 0
    assert "thinner than" in captured.err
    assert "Traceback" not in captured.err
    # Nothing was written: the error happened before any folder was created.
    assert not designs_root.exists()


def test_a_failed_qc_verdict_exits_nonzero(sample_image, tmp_path):
    """A part sized past the bed fails QC's bed_fit check, no exception raised."""
    designs_root = tmp_path / "designs"
    rc = cli.main(
        [
            str(sample_image),
            "--type",
            "keychain",
            "--no-slice",
            "--size",
            "200",
            "--name",
            "oversize",
            "--designs-root",
            str(designs_root),
        ]
    )
    assert rc != 0
    report = json.loads(
        (designs_root / "oversize" / "report.json").read_text(encoding="utf-8")
    )
    assert report["verdict"] == "failed"
    bed_fit = next(c for c in report["checks"] if c["name"] == "bed_fit")
    assert not bed_fit["passed"]


# --- F2: --size/--thickness/--hole-diameter must be positive ---------------


@pytest.mark.parametrize("flag", ["--size", "--thickness", "--hole-diameter"])
@pytest.mark.parametrize("bad_value", ["0", "-5"])
def test_non_positive_numeric_flags_exit_2_with_no_traceback(
    sample_image, tmp_path, capsys, flag, bad_value
):
    designs_root = tmp_path / "designs"
    with pytest.raises(SystemExit) as caught:
        cli.main(
            [
                str(sample_image),
                "--type",
                "keychain",
                "--no-slice",
                flag,
                bad_value,
                "--designs-root",
                str(designs_root),
            ]
        )
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert not designs_root.exists()


@pytest.mark.parametrize("flag", ["--size", "--thickness", "--hole-diameter"])
@pytest.mark.parametrize("bad_value", ["inf", "-inf", "nan"])
def test_non_finite_numeric_flags_exit_2_with_no_traceback(
    sample_image, tmp_path, capsys, flag, bad_value
):
    designs_root = tmp_path / "designs"
    with pytest.raises(SystemExit) as caught:
        cli.main(
            [
                str(sample_image),
                "--type",
                "keychain",
                "--no-slice",
                flag,
                bad_value,
                "--designs-root",
                str(designs_root),
            ]
        )
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert not designs_root.exists()


def test_hole_position_inf_exits_2_with_no_traceback(sample_image, tmp_path, capsys):
    designs_root = tmp_path / "designs"
    with pytest.raises(SystemExit) as caught:
        cli.main(
            [
                str(sample_image),
                "--type",
                "keychain",
                "--no-slice",
                "--hole-position",
                "inf",
                "0",
                "--designs-root",
                str(designs_root),
            ]
        )
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert not designs_root.exists()


def test_size_zero_bad_float_string_also_exits_2(sample_image, tmp_path, capsys):
    designs_root = tmp_path / "designs"
    with pytest.raises(SystemExit) as caught:
        cli.main(
            [
                str(sample_image),
                "--type",
                "keychain",
                "--no-slice",
                "--size",
                "not-a-number",
                "--designs-root",
                str(designs_root),
            ]
        )
    assert caught.value.code == 2
    assert "Traceback" not in capsys.readouterr().err


# --- empty-detail flip -------------------------------------------------


def test_empty_detail_flips_detail_enabled_off_and_warns(flat_image, tmp_path):
    designs_root = tmp_path / "designs"
    rc = _run(flat_image, "--name", "flat", designs_root=designs_root)
    assert rc == 0

    report = json.loads(
        (designs_root / "flat" / "report.json").read_text(encoding="utf-8")
    )
    # spec.detail_mode ended up "none", so QC never emits a detail_present
    # check at all - if it were still "raised" with nothing to show, this
    # is exactly the check that would sink the verdict to "failed".
    assert not any(c["name"] == "detail_present" for c in report["checks"])
    assert report["verdict"] == "passes-level-1"
    # This is the CLI's *own* flip warning text (from _build_keychain), not
    # trace.py's separate "single-colour" warning - both happen to contain
    # the word "single-colour", so a substring check on that alone would
    # pass even with the CLI's own warnings.append/extend deleted entirely.
    # Assert on text unique to the CLI's warning instead.
    assert any(
        "was requested, but the traced image has no detail layer" in w
        for w in report["warnings"]
    ), report["warnings"]


def test_explicit_detail_none_does_not_need_the_flip_warning(sample_image, tmp_path):
    """--detail none on an image that *has* detail: no flip, no CLI warning."""
    designs_root = tmp_path / "designs"
    rc = _run(sample_image, "--detail", "none", "--name", "none-mode", designs_root=designs_root)
    assert rc == 0
    report = json.loads(
        (designs_root / "none-mode" / "report.json").read_text(encoding="utf-8")
    )
    assert not any("was requested" in w for w in report["warnings"])


# --- --photo flag wiring -----------------------------------------------


def test_photo_flag_runs_the_pipeline_on_a_photographed_drawing(
    photo_image, tmp_path
):
    designs_root = tmp_path / "designs"
    rc = _run(
        photo_image, "--photo", "--name", "photo-mode", designs_root=designs_root
    )
    assert rc == 0
    report = json.loads(
        (designs_root / "photo-mode" / "report.json").read_text(encoding="utf-8")
    )
    # _run always passes --no-slice, so level 1 is the ceiling here:
    # anything else (print-ready included) would mean the gate broke.
    assert report["verdict"] == "passes-level-1"


def test_without_photo_flag_the_same_lit_image_fails_qc(photo_image, tmp_path):
    """--photo is opt-in: read as flat artwork, the unflattened illumination
    gradient itself crosses the ink thresholds and sinks the verdict, where
    ``--photo`` (above) gets the very same file to a passing one."""
    designs_root = tmp_path / "designs"
    rc = _run(
        photo_image,
        "--name",
        "no-photo-mode",
        designs_root=designs_root,
        exit_ok=False,
    )
    assert rc == 1
    report = json.loads(
        (designs_root / "no-photo-mode" / "report.json").read_text(encoding="utf-8")
    )
    assert report["verdict"] == "failed"


# --- slug derivation, ASCII-folding and reserved names ----------------------


def test_slug_derives_from_the_image_filename_when_no_name_is_given(
    sample_image, tmp_path
):
    designs_root = tmp_path / "designs"
    _run(sample_image, designs_root=designs_root)
    expected = cli._slugify(sample_image.stem)
    assert (designs_root / expected).is_dir()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("My Cool Keychain", "my-cool-keychain"),
        ("  leading/trailing  ", "leading-trailing"),
        ("UPPER_CASE!!", "upper-case"),
        ("", "design"),
        # F1: non-ASCII folds to ASCII instead of surviving into the slug.
        ("café", "cafe"),
        ("日本語", "design"),  # nothing ASCII survives -> the empty-input fallback
        ("café日本", "cafe"),
        # F8: Windows reserved device names get prefixed.
        ("nul", "design-nul"),
        ("NUL", "design-nul"),
        ("con", "design-con"),
        ("COM1", "design-com1"),
        ("lpt9", "design-lpt9"),
        ("nullable", "nullable"),  # not an exact match, left alone
    ],
)
def test_slugify(raw, expected):
    assert cli._slugify(raw) == expected


def test_rerunning_the_same_slug_overwrites_the_folder_cleanly(sample_image, tmp_path):
    designs_root = tmp_path / "designs"
    args = ("--name", "dup")
    assert _run(sample_image, *args, designs_root=designs_root) == 0

    design_dir = designs_root / "dup"
    stray = design_dir / "leftover_from_a_previous_run.txt"
    stray.write_text("stale", encoding="utf-8")

    assert _run(sample_image, *args, designs_root=designs_root) == 0
    assert not stray.exists(), "overwrite must clear the folder, not merge into it"
    assert (design_dir / "dup.3mf").exists()
    assert not (designs_root / ".staging-dup").exists()


# --- F1: non-ASCII names/filenames never crash a successful run ------------


def test_non_ascii_name_ascii_folds_to_a_safe_slug(sample_image, tmp_path):
    designs_root = tmp_path / "designs"
    rc = _run(sample_image, "--name", "café日本", designs_root=designs_root)
    assert rc == 0
    design_dir = designs_root / "cafe"
    assert design_dir.is_dir()
    # The slug (and therefore every path derived from it) is pure ASCII.
    design_dir.name.encode("ascii")


def test_non_ascii_image_stem_ascii_folds_when_no_name_is_given(tmp_path):
    image = Image.new("L", (300, 300), 255)
    draw = ImageDraw.Draw(image)
    draw.ellipse([20, 20, 279, 279], fill=160)
    draw.rectangle([90, 130, 209, 170], fill=20)
    path = tmp_path / "câfé日本.png"
    image.save(path)

    designs_root = tmp_path / "designs"
    rc = _run(path, designs_root=designs_root)
    assert rc == 0
    expected = cli._slugify(path.stem)
    expected.encode("ascii")
    assert (designs_root / expected).is_dir()


def test_harden_console_encoding_reconfigures_streams_with_replace(monkeypatch):
    calls = []

    class FakeStream:
        def reconfigure(self, **kwargs):
            calls.append(kwargs)

    fake_out, fake_err = FakeStream(), FakeStream()
    monkeypatch.setattr(cli.sys, "stdout", fake_out)
    monkeypatch.setattr(cli.sys, "stderr", fake_err)
    cli._harden_console_encoding()
    assert calls == [{"errors": "replace"}, {"errors": "replace"}]


def test_harden_console_encoding_skips_streams_without_reconfigure(monkeypatch):
    class NoReconfigure:
        pass

    monkeypatch.setattr(cli.sys, "stdout", NoReconfigure())
    monkeypatch.setattr(cli.sys, "stderr", NoReconfigure())
    # Must not raise even though neither stream supports reconfigure().
    cli._harden_console_encoding()


# --- F3: a failing pipeline never destroys an existing design, and never ---
# --- leaves a stray empty folder behind -------------------------------------


def _bad_hole_position_args():
    # Way outside a 50 mm part: HoleMarginError (a BuildError) every time,
    # deterministically, with no dependency on trace/mesh internals.
    return ("--hole-position", "1000", "1000")


def test_failing_rerun_leaves_the_previous_good_folder_intact(sample_image, tmp_path):
    designs_root = tmp_path / "designs"
    assert _run(sample_image, "--name", "keepme", designs_root=designs_root) == 0

    design_dir = designs_root / "keepme"
    before = {p.name: (design_dir / p.name).read_bytes() for p in design_dir.iterdir()}
    assert before  # sanity: the good run really did write files

    rc = cli.main(
        [
            str(sample_image),
            "--type",
            "keychain",
            "--no-slice",
            "--name",
            "keepme",
            "--designs-root",
            str(designs_root),
            *_bad_hole_position_args(),
        ]
    )
    assert rc == 1

    after = {p.name: (design_dir / p.name).read_bytes() for p in design_dir.iterdir()}
    assert after == before, "the old design folder must be untouched by a failing re-run"
    assert not (designs_root / ".staging-keepme").exists()


def test_fresh_failing_run_leaves_no_stray_empty_folder(sample_image, tmp_path, capsys):
    designs_root = tmp_path / "designs"
    rc = cli.main(
        [
            str(sample_image),
            "--type",
            "keychain",
            "--no-slice",
            "--name",
            "neverexisted",
            "--designs-root",
            str(designs_root),
            *_bad_hole_position_args(),
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert not (designs_root / "neverexisted").exists()
    assert not (designs_root / ".staging-neverexisted").exists()


# --- F7: --designs-root pointing at a file --------------------------------


def test_designs_root_pointing_at_a_file_is_a_clean_error(sample_image, tmp_path, capsys):
    not_a_dir = tmp_path / "not_a_dir"
    not_a_dir.write_text("i am a file", encoding="utf-8")

    rc = cli.main(
        [str(sample_image), "--type", "keychain", "--no-slice", "--designs-root", str(not_a_dir)]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "Traceback" not in captured.err
    assert str(not_a_dir) in captured.err


# --- F8: a reserved-device-name slug builds cleanly (real filesystem ops) --


def test_reserved_device_name_slug_builds_without_exploding(sample_image, tmp_path):
    designs_root = tmp_path / "designs"
    rc = _run(sample_image, "--name", "nul", designs_root=designs_root)
    assert rc == 0
    assert (designs_root / "design-nul").is_dir()
    # "nul" is a Windows device name, not a real path: os.path.exists()
    # reports it as always "existing" even though nothing was ever written
    # there. is_dir() is the meaningful check -- no directory was created.
    assert not (designs_root / "nul").is_dir()


# --- the >10MB commit rule: oversized artifacts are MOVED, not deleted -----


def test_oversized_artifacts_are_moved_to_oversize_and_recorded_in_drive_link(
    monkeypatch, sample_image, tmp_path
):
    monkeypatch.setattr(cli, "_MAX_COMMITTABLE_BYTES", 10)
    designs_root = tmp_path / "designs"
    rc = _run(sample_image, "--name", "big", designs_root=designs_root)
    assert rc == 0

    design_dir = designs_root / "big"
    # Oversized artifacts are gone from the top level...
    assert not (design_dir / "big.3mf").exists()
    assert not (design_dir / "big.stl").exists()
    # ...but survive, in full, under _oversize/.
    oversize_dir = design_dir / "_oversize"
    assert (oversize_dir / "big.3mf").is_file()
    assert (oversize_dir / "big.stl").is_file()

    drive_link = (design_dir / "DRIVE_LINK.md").read_text(encoding="utf-8")
    assert "designs/big/_oversize/big.3mf" in drive_link
    assert "designs/big/_oversize/big.stl" in drive_link
    assert "drive.google.com" in drive_link
    # No hardcoded "10 MB" - the text must reflect _MAX_COMMITTABLE_BYTES.
    assert "10 MB" not in drive_link

    # report.json/report.md and any regenerable slice output are untouched:
    # only the four committable artifact kinds are ever candidates.
    report_json = design_dir / "report.json"
    report_md = design_dir / "report.md"
    assert report_json.is_file()
    assert report_md.is_file()

    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert any("big.3mf" in w and "moved to designs/big/_oversize" in w for w in report["warnings"])
    assert any("big.stl" in w and "moved to designs/big/_oversize" in w for w in report["warnings"])


def test_artifacts_under_the_limit_never_produce_a_drive_link(sample_image, tmp_path):
    designs_root = tmp_path / "designs"
    _run(sample_image, "--name", "small", designs_root=designs_root)
    assert not (designs_root / "small" / "DRIVE_LINK.md").exists()
    assert not (designs_root / "small" / "_oversize").exists()


# --- F6: every FAILED check is always printed, allowlist or not ------------


def _fake_report(*, verdict, checks, warnings=()):
    return QCReport(
        slug="fake",
        verdict=verdict,
        levels_run=[1],
        checks=list(checks),
        warnings=list(warnings),
        slicer=None,
    )


def test_summary_always_prints_a_failed_check_outside_the_allowlist(tmp_path, capsys):
    report = _fake_report(
        verdict="failed",
        checks=[
            QCCheck(
                name="base.watertight",
                level=1,
                passed=False,
                measured=False,
                expected=True,
                message="base mesh is not watertight",
            ),
            QCCheck(
                name="bed_fit",
                level=1,
                passed=True,
                measured=40.0,
                expected=180.0,
                message="fits the bed",
            ),
        ],
    )
    cli._print_summary(report, tmp_path / "somewhere", no_slice=False)
    out = capsys.readouterr().out
    assert "[FAIL] base.watertight: base mesh is not watertight" in out
    assert "[ok] bed_fit: fits the bed" in out


def test_summary_does_not_print_an_ok_check_outside_the_allowlist(tmp_path, capsys):
    report = _fake_report(
        verdict="print-ready",
        checks=[
            QCCheck(
                name="base.watertight",
                level=1,
                passed=True,
                measured=True,
                expected=True,
                message="base mesh is watertight",
            ),
        ],
    )
    cli._print_summary(report, tmp_path / "somewhere", no_slice=False)
    out = capsys.readouterr().out
    assert "base.watertight" not in out


# --- F9(a): the --no-slice cap note is suppressed on a failed verdict ------


def test_no_slice_note_is_suppressed_when_the_verdict_is_failed(sample_image, tmp_path, capsys):
    designs_root = tmp_path / "designs"
    rc = cli.main(
        [
            str(sample_image),
            "--type",
            "keychain",
            "--no-slice",
            "--size",
            "200",
            "--name",
            "toobig",
            "--designs-root",
            str(designs_root),
        ]
    )
    assert rc != 0
    out = capsys.readouterr().out
    assert "verdict: failed" in out
    assert "note: ran with --no-slice" not in out


def test_no_slice_note_still_shown_when_verdict_passes(sample_image, tmp_path, capsys):
    designs_root = tmp_path / "designs"
    _run(sample_image, "--name", "fine", designs_root=designs_root)
    out = capsys.readouterr().out
    assert "note: ran with --no-slice" in out


# --- CLI options reach the spec ---------------------------------------------


def test_thickness_and_hole_diameter_flags_reach_the_built_spec(
    monkeypatch, sample_image, tmp_path
):
    designs_root = tmp_path / "designs"
    calls = []
    original = cli.REGISTRY["keychain"]

    def spy(request):
        built = original(request)
        calls.append(built.spec)
        return built

    monkeypatch.setitem(cli.REGISTRY, "keychain", spy)
    rc = cli.main(
        [
            str(sample_image),
            "--type",
            "keychain",
            "--no-slice",
            "--thickness",
            "4.5",
            "--hole-diameter",
            "6.0",
            "--name",
            "tuned",
            "--designs-root",
            str(designs_root),
        ]
    )

    assert rc == 0
    assert len(calls) == 1
    assert calls[0].base_thickness_mm == 4.5
    assert calls[0].hole_diameter_mm == 6.0
