"""Command-line entry point for print3d (the ``make3d`` script).

One command runs the whole pipeline: trace -> build -> QC -> a design
folder under ``designs/<slug>/`` (source image copy, traced SVG, 3MF,
STL, ``report.md``, ``report.json``).

**Registry seam.** Object types (v1: only ``keychain``) dispatch through
:data:`REGISTRY`, a plain ``{name: builder}`` dict. ``--type`` looks the
name up there rather than through an if/else on the flag, so a future
object type is one new entry (see :class:`BuildRequest` /
:class:`BuiltDesign` for the shape a builder must honour), not a rewrite
of this module. An unknown ``--type`` is reported with the list of names
actually registered.

**Repo / designs-root resolution.** Design folders live under
``<repo root>/designs/<slug>/``. The repo root is derived from this
file's own location -- ``src/print3d/cli.py`` sits two directories below
the checkout root in this project's src-layout, so
``Path(__file__).resolve().parents[2]`` is the root, no hardcoded
absolute path involved. ``main()``/``_run()`` also accept an explicit
``--designs-root`` override so tests can redirect writes to a tmp
directory without monkeypatching module internals.

**Overwrite semantics.** Re-running the same slug replaces the whole
design folder, but never destructively: the new run is assembled in a
staging directory (``designs/.staging-<slug>/``) and only swapped into
``designs/<slug>/`` -- replacing whatever was there, including a stale
``DRIVE_LINK.md`` from an earlier, different run -- once the pipeline has
fully succeeded. A failing re-run (an exception, not a merely-failed QC
verdict) leaves the previous good folder completely untouched and no
stray empty folder behind.

**The >10MB rule.** Enforced once, generically, on the committable
artifacts (source image copy, ``traced.svg``, 3MF, STL -- never the
report files or gitignored slice output) right before the report is
written (:func:`_enforce_size_limit`): any file over the limit is moved
into a gitignored ``_oversize/`` subfolder, and its new location recorded
in ``DRIVE_LINK.md``, instead of being committed or deleted. Nothing
about the rule is specific to any one artifact kind, and with this
program's current defaults it should never actually fire -- it exists to
guard a future, larger model.

**Exit codes.** 0 when the QC verdict is ``print-ready`` or
``passes-level-1``; nonzero for a ``failed`` verdict, a pipeline error
(:class:`~print3d.trace.TraceError` incl. ``ThinFeatureError``, the
:class:`~print3d.build.BuildError` family, or
:class:`~print3d.qc.OrcaSlicerConfigError` -- each printed as a plain
message, no traceback), an unknown ``--type``, a missing image file, or
bad arguments (argparse's own usual ``SystemExit(2)``).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import shutil
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from print3d.build import BuildError, build_keychain, export_3mf, export_stl
from print3d.qc import OrcaSlicerConfigError, run_qc, write_report
from print3d.spec import KeychainSpec
from print3d.trace import TraceError, TraceResult, trace_image

__all__ = [
    "REGISTRY",
    "BuildRequest",
    "BuiltDesign",
    "main",
]

#: Largest artifact ever written into a design folder for commit, in bytes.
#: A module-level constant (rather than a literal buried in a function) so
#: tests can force the DRIVE_LINK.md path with a tiny fake threshold.
_MAX_COMMITTABLE_BYTES = 10 * 1024 * 1024

#: Where the Chief Conductor uploads anything too big to commit. Manual
#: upload only -- the pipeline never touches Drive itself.
_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1IxXcWLzxpwOyFO2FQLSh7zQSMs_W3wai"
    "?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto"
)

#: Mirrors of KeychainSpec's own field defaults, so the CLI's --size /
#: --thickness / --hole-diameter defaults can never drift from the spec
#: they end up filling in.
_SPEC_DEFAULTS = {f.name: f.default for f in dataclasses.fields(KeychainSpec)}

#: Check names worth an [ok] line in the human summary, in the order
#: printed. Every FAILED check is always printed regardless of this list
#: (see :func:`_print_summary`) -- the allowlist only trims the noise on
#: the happy path.
_SUMMARY_CHECKS = (
    "wall_thickness",
    "hole_diameter",
    "hole_margin",
    "bed_fit",
    "detail_present",
    "slice_success",
)

#: Windows reserved device names (case-insensitive, no extension needed to
#: trip them up). A slug that collides with one of these can't be created
#: as a directory on Windows -- ``rmtree``/``mkdir`` fail with a
#: ``PermissionError`` deep enough that it looks like a crash, not a bad
#: name.
_RESERVED_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


# --- the object-type registry -----------------------------------------------


@dataclass(frozen=True)
class BuildRequest:
    """Everything a registry entry needs to build and export one design.

    ``args`` is the parsed CLI namespace, handed through unmodified so a
    future object type can read its own flags (added to the parser
    alongside ``keychain``'s) without ``cli.py`` needing to know about
    them ahead of time.
    """

    args: argparse.Namespace
    slug: str
    source_image: Path
    trace: TraceResult
    design_dir: Path


@dataclass(frozen=True)
class BuiltDesign:
    """What a registry entry hands back after building *and* exporting.

    ``solids`` only needs to look like what :func:`print3d.qc.run_qc`
    reads off it today (a ``.spec`` with ``.slug``, and ``.warnings``) --
    that is the seam a future, structurally different object type has to
    honour, or QC gets its own generalisation on that type's issue.
    """

    spec: Any
    solids: Any
    mf3_path: Path
    stl_path: Path
    warnings: list[str] = field(default_factory=list)


#: name -> builder. A builder takes one :class:`BuildRequest` and returns
#: one :class:`BuiltDesign`: interview-free spec construction from CLI
#: flags, the build, and the export, all owned by the entry. ``--type``
#: is resolved by looking a name up here -- never by branching on the
#: flag's value -- so adding an object type is one new entry.
#:
#: Honest caveat: QC (:mod:`print3d.qc`) is keychain-specific in v1 --
#: it reads a builder's output through the ``spec``/``solids`` shape
#: :class:`BuiltDesign` documents, but its checks (hole diameter, wall
#: thickness measured off a base+detail footprint, ...) assume a
#: keychain's geometry. A future object type needs a type-aware QC route
#: of its own in addition to its registry entry here -- this registry is
#: the CLI's dispatch seam, not yet a full plugin system that also
#: generalises QC.
REGISTRY: dict[str, Callable[[BuildRequest], BuiltDesign]] = {}


def _build_keychain(request: BuildRequest) -> BuiltDesign:
    """The ``keychain`` registry entry: spec from flags, build, export."""
    args = request.args
    warnings: list[str] = []

    detail_enabled = args.detail != "none"
    if detail_enabled and request.trace.detail.is_empty:
        # CRITICAL: a spec left asking for detail the trace could not find
        # sinks the QC verdict on detail_present even though there is
        # nothing wrong with the model. Flip it off here, before the
        # build ever sees it; the trace stage's own warning explaining
        # *why* there is no detail is already in trace.warnings.
        detail_enabled = False
        warnings.append(
            f"--detail {args.detail} was requested, but the traced image "
            f"has no detail layer to work with, so the design was built "
            f"single-colour instead."
        )

    spec = KeychainSpec(
        slug=request.slug,
        source_image=str(request.source_image),
        max_dimension_mm=args.size,
        base_thickness_mm=args.thickness,
        detail_enabled=detail_enabled,
        detail_recessed=(args.detail == "recessed"),
        hole_diameter_mm=args.hole_diameter,
        hole_position_mm=tuple(args.hole_position) if args.hole_position else None,
    )
    solids = build_keychain(request.trace, spec=spec)
    mf3_path = export_3mf(solids, request.design_dir / f"{request.slug}.3mf")
    stl_path = export_stl(solids, request.design_dir / f"{request.slug}.stl")
    return BuiltDesign(
        spec=spec,
        solids=solids,
        mf3_path=mf3_path,
        stl_path=stl_path,
        warnings=warnings,
    )


REGISTRY["keychain"] = _build_keychain


# --- paths -------------------------------------------------------------


def _repo_root() -> Path:
    """The checkout root, derived from this file's own location.

    ``src/print3d/cli.py`` -> ``print3d`` -> ``src`` -> repo root: no
    absolute path is ever hardcoded, per the repo's own rule.
    """
    return Path(__file__).resolve().parents[2]


def _default_designs_root() -> Path:
    return _repo_root() / "designs"


def _slugify(text: str) -> str:
    """A filesystem- and git-safe slug: lowercase, ``-``-separated, ASCII.

    Non-ASCII input is folded to its closest ASCII form first (NFKD
    decomposition, e.g. ``e-acute`` -> ``e`` + a combining mark, then the
    combining mark and anything else outside ASCII is dropped) so a
    console using a narrow encoding like cp1252 can always print the
    resulting path -- a slug is never allowed to smuggle a character the
    terminal can't render. If nothing ASCII survives (e.g. the input is
    entirely CJK), the slug falls back to ``"design"`` exactly as it does
    for empty input.

    A result that collides with a Windows reserved device name (``nul``,
    ``con``, ``com1``, ...) is prefixed rather than used as-is: those
    names can't be created as directories on Windows, no matter the
    extension.
    """
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    out = []
    prev_dash = False
    for ch in folded.strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    slug = "".join(out).strip("-") or "design"
    if slug in _RESERVED_DEVICE_NAMES:
        slug = f"design-{slug}"
    return slug


def _positive_float(raw: str) -> float:
    """An ``argparse`` ``type=`` for flags that must be a positive number.

    ``--size 0`` (or negative) used to reach ``trace_image`` and blow up
    on a bare ``ValueError`` there, printing a raw traceback. Rejecting it
    here instead turns it into argparse's own usual, traceback-free
    ``SystemExit(2)`` with a message naming the flag and the bad value.

    ``inf``/``nan`` are rejected too (``math.isfinite``): ``float("inf")``
    parses cleanly and is ``> 0``, so it would otherwise sail through this
    check and reach the geometry stages, which do not expect a
    non-finite dimension.
    """
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float value: '{raw}'") from exc
    if not math.isfinite(value) or not value > 0:
        raise argparse.ArgumentTypeError(
            f"must be a positive, finite number, got '{raw}'"
        )
    return value


def _finite_float(raw: str) -> float:
    """An ``argparse`` ``type=`` for flags that accept any real number but
    ``inf``/``nan`` (unlike :func:`_positive_float`, negative and zero are
    legal here -- a hole position is a coordinate, not a size).
    """
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float value: '{raw}'") from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"must be a finite number, got '{raw}'")
    return value


# --- the >10MB commit rule ---------------------------------------------


def _enforce_size_limit(
    staging_dir: Path, slug: str, artifact_paths: list[Path], warnings: list[str]
) -> None:
    """Move any oversized *committable* artifact into ``_oversize/``.

    Applies only to the artifacts this design folder actually commits to
    git -- the copied source image, ``traced.svg``, the 3MF and the STL
    (``artifact_paths``) -- never ``report.json``/``report.md`` (not
    written yet when this runs; see :func:`_run`) and never a regenerable
    file like ``*.gcode.3mf``/``*.log`` (already gitignored, so there is
    nothing to protect them from).

    Oversized files are *moved*, not deleted: dropping them used to leave
    ``DRIVE_LINK.md`` pointing the Chief Conductor at a file that no
    longer existed anywhere. They land in ``_oversize/`` inside the
    staging dir (gitignored, so the rest of the design folder stays
    committable) and survive there for manual upload.
    """
    oversized = sorted(
        p for p in artifact_paths if p.is_file() and p.stat().st_size > _MAX_COMMITTABLE_BYTES
    )
    if not oversized:
        return

    limit_mb = _MAX_COMMITTABLE_BYTES / 1e6
    oversize_dir = staging_dir / "_oversize"
    oversize_dir.mkdir(exist_ok=True)

    lines = [
        "# Files too large to commit",
        "",
        f"These exceeded the {limit_mb:.1f} MB commit limit and were moved "
        f"into `_oversize/` (gitignored) instead of being committed. Upload "
        f"them manually to the Chief Conductor's Drive folder: "
        f"{_DRIVE_FOLDER_URL}",
        "",
    ]
    for path in oversized:
        size_mb = path.stat().st_size / 1e6
        shutil.move(str(path), str(oversize_dir / path.name))
        real_location = f"designs/{slug}/_oversize/{path.name}"
        lines.append(f"- `{real_location}` ({size_mb:.1f} MB)")
        warnings.append(
            f"{path.name} is {size_mb:.1f} MB, over the {limit_mb:.1f} MB "
            f"commit limit -- moved to {real_location}; see DRIVE_LINK.md."
        )

    (staging_dir / "DRIVE_LINK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- the pipeline --------------------------------------------------------


def _promote_staging(staging_dir: Path, design_dir: Path) -> None:
    """Swap a fully-built staging dir into its final ``design_dir``.

    Only ever called after the whole pipeline has succeeded. Any previous
    folder at ``design_dir`` is replaced atomically-enough for a CLI tool:
    removed only once the new one is ready to take its place, never
    before.
    """
    if design_dir.exists():
        shutil.rmtree(design_dir)
    design_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging_dir), str(design_dir))


def _run(args: argparse.Namespace) -> int:
    builder = REGISTRY.get(args.type)
    if builder is None:
        registered = ", ".join(sorted(REGISTRY)) or "(none registered)"
        print(
            f"Unknown --type '{args.type}'. Registered object types: "
            f"{registered}.",
            file=sys.stderr,
        )
        return 2

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"No such image: {image_path}", file=sys.stderr)
        return 2

    root = args.designs_root if args.designs_root is not None else _default_designs_root()
    root = Path(root)
    if root.exists() and not root.is_dir():
        print(
            f"--designs-root {root} exists and is not a directory.",
            file=sys.stderr,
        )
        return 2

    slug = _slugify(args.name) if args.name else _slugify(image_path.stem)
    design_dir = root / slug
    # A dot-prefixed name distinct from any real slug, so a failed run
    # never leaves anything visible under the slug it was trying to
    # build -- and so it can never collide with ``design_dir`` itself.
    staging_dir = root / f".staging-{slug}"

    try:
        trace = trace_image(
            image_path,
            size_mm=args.size,
            detail=(args.detail != "none"),
            photo=args.photo,
        )
    except TraceError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    # Everything from here builds in a staging dir, never touching
    # design_dir until the whole pipeline has succeeded (see
    # _promote_staging). Clear any leftover from a previous crashed run
    # first, so it never gets mistaken for this run's own output.
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    request = BuildRequest(
        args=args,
        slug=slug,
        source_image=image_path,
        trace=trace,
        design_dir=staging_dir,
    )
    try:
        built = builder(request)
        shutil.copy2(image_path, staging_dir / image_path.name)
        trace.to_svg(staging_dir / "traced.svg")
        report = run_qc(
            trace,
            built.solids,
            built.mf3_path,
            built.stl_path,
            work_dir=staging_dir,
            run_level2=not args.no_slice,
        )
    except BuildError as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        print(str(exc), file=sys.stderr)
        return 1
    except OrcaSlicerConfigError as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:
        # Not one of the pipeline's own typed errors -- still must not
        # leave a stray, half-built staging dir behind. Re-raised
        # unchanged: an unexpected exception is a real bug to see the
        # traceback for, not one to swallow.
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    # CLI-originated warnings (the detail-flip warning is the only one
    # today) never reached report.json before -- they only ever went to
    # stdout. run_qc already folded trace.warnings/build-solids.warnings
    # in; this adds the ones the CLI itself generated.
    report.warnings.extend(w for w in built.warnings if w not in report.warnings)

    # The >10MB rule runs before write_report so its own warnings land in
    # the on-disk report too, and before promotion so it only ever
    # touches the staging dir.
    artifact_paths = [
        staging_dir / image_path.name,
        staging_dir / "traced.svg",
        built.mf3_path,
        built.stl_path,
    ]
    _enforce_size_limit(staging_dir, slug, artifact_paths, report.warnings)

    write_report(report, staging_dir)
    _promote_staging(staging_dir, design_dir)

    _print_summary(report, design_dir, no_slice=args.no_slice)

    return 0 if report.verdict in ("print-ready", "passes-level-1") else 1


def _print_summary(report, design_dir: Path, *, no_slice: bool) -> None:
    print(f"verdict: {report.verdict}")
    for check in report.checks:
        if check.passed:
            # The allowlist trims the happy-path noise -- 25 checks passing
            # silently is fine. A failure is never trimmed: an allowlist
            # miss must not be able to hide a real failure from stdout.
            if check.name in _SUMMARY_CHECKS:
                print(f"  [ok] {check.name}: {check.message}")
        else:
            print(f"  [FAIL] {check.name}: {check.message}")

    if report.warnings:
        print("warnings:")
        for w in report.warnings:
            print(f"  - {w}")

    if no_slice and report.verdict != "failed":
        print("note: ran with --no-slice; verdict caps at 'passes-level-1'.")
    print(f"report: {design_dir / 'report.md'}")


# --- argument parsing ------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make3d",
        description=(
            "Trace an image, build a print-ready 3D model, and quality-"
            "check it, in one step."
        ),
    )
    parser.add_argument("image", help="Path to the source image (PNG/JPG).")
    parser.add_argument(
        "--type",
        required=True,
        help=(
            "Object type to build; resolved through the type registry. "
            f"Registered: {', '.join(sorted(REGISTRY))}."
        ),
    )
    parser.add_argument(
        "--size",
        type=_positive_float,
        default=_SPEC_DEFAULTS["max_dimension_mm"],
        help="Longest XY dimension of the finished part, in mm (default: %(default)s).",
    )
    parser.add_argument(
        "--thickness",
        type=_positive_float,
        default=_SPEC_DEFAULTS["base_thickness_mm"],
        help="Base thickness, in mm (default: %(default)s).",
    )
    parser.add_argument(
        "--detail",
        choices=("raised", "recessed", "none"),
        default="raised",
        help="How the second-colour layer is expressed (default: %(default)s).",
    )
    parser.add_argument(
        "--hole-diameter",
        type=_positive_float,
        default=_SPEC_DEFAULTS["hole_diameter_mm"],
        help="Hanging-hole diameter, in mm (default: %(default)s).",
    )
    parser.add_argument(
        "--hole-position",
        type=_finite_float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Explicit hole centre in model mm; default lets the builder place it.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Design slug; default is derived from the image filename.",
    )
    parser.add_argument(
        "--no-slice",
        action="store_true",
        help="Skip the OrcaSlicer level-2 check; verdict caps at 'passes-level-1'.",
    )
    parser.add_argument(
        "--photo",
        action="store_true",
        help=(
            "Image is a phone photo of a paper drawing, not flat artwork: "
            "even out the lighting and shadows first, then treat pencil or "
            "marker strokes as ink and the paper as background. Off by "
            "default."
        ),
    )
    parser.add_argument(
        "--designs-root",
        type=Path,
        default=None,
        help=(
            "Write design folders under this directory instead of "
            "<repo root>/designs (mainly for tests)."
        ),
    )
    return parser


def _harden_console_encoding() -> None:
    """Belt-and-braces: an unencodable character must never crash a run.

    ``_slugify`` ASCII-folds so a slug itself is always safe to print, but
    other text that reaches stdout/stderr (an error message quoting a
    source filename, for one) is not under the CLI's control. cp1252 (the
    default Windows console encoding) raises ``UnicodeEncodeError`` on
    anything outside its range; reconfiguring both streams with
    ``errors="replace"`` turns that into a printed replacement character
    instead of a crash *after* the pipeline has already finished its
    work. Cheap, and safe even when the stream doesn't support
    reconfiguration (e.g. it has been replaced with something else in a
    test) -- that case is just skipped.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    """Entry point registered as the ``make3d`` console script."""
    _harden_console_encoding()
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
