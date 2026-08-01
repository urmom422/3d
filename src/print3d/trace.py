"""Image tracing: 2D raster image to vector silhouette/detail geometry.

The trace stage is the front of the pipeline. It turns a bitmap into two
shapely geometries, both in **millimetres**, both sharing one origin:

* ``silhouette`` - the outline of the object the printer has to make solid;
* ``detail`` - the regions that later become the raised (or recessed)
  second-colour layer, always a subset of the silhouette.

Tracing is done with potracer (a pure-Python port of potrace) run directly
on threshold masks, so no SVG round-trip is needed to get geometry. An SVG
*is* still written for the design folder, but as an output artefact
(:meth:`TraceResult.to_svg`) rather than as an intermediate.

The stage is also the first quality gate. A silhouette with any real
feature narrower than :data:`~print3d.spec.MIN_WALL_MM` cannot be printed
at all, so it raises :class:`ThinFeatureError`. Detail strokes narrower
than :data:`~print3d.spec.MIN_DETAIL_STROKE_MM` would simply vanish in the
print, so they are dropped and reported in ``TraceResult.warnings``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import potrace
from PIL import Image
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid

from .spec import BED_SIZE_MM, MIN_DETAIL_STROKE_MM, MIN_WALL_MM

__all__ = [
    "MIN_DETAIL_STROKE_MM",
    "MIN_WALL_MM",
    "ThinFeatureError",
    "ThinRegion",
    "TraceError",
    "TraceResult",
    "UnsupportedImageError",
    "find_thin_regions",
    "trace_detail",
    "trace_image",
    "trace_silhouette",
]

# --- tuning constants ------------------------------------------------------

#: Default longest XY dimension of the finished part, in mm.
DEFAULT_SIZE_MM = 50.0

#: Pixels lighter than this fraction of white are treated as background when
#: the image has no alpha channel. Deliberately generous: mid-greys are part
#: of the object, only near-white is "nothing here".
DEFAULT_SILHOUETTE_THRESHOLD = 0.85

#: Pixels darker than this fraction of white become detail regions.
DEFAULT_DETAIL_THRESHOLD = 0.5

#: Douglas-Peucker tolerance applied after scaling to mm. A 0.4 mm nozzle
#: cannot resolve anything finer, so 0.05 mm keeps every printable feature
#: while shedding the per-pixel wobble potrace leaves behind.
DEFAULT_SIMPLIFY_MM = 0.05

#: Speckle suppression handed to potrace, in pixels squared.
DEFAULT_SPECKLE_PX = 2

#: Bezier flattening resolution: line segments emitted per traced curve.
_BEZIER_STEPS = 12

#: A morphological opening rounds sharp convex corners off, leaving a
#: crescent of "lost" area at each one. For a square corner rounded by
#: radius r that crescent is exactly r**2 * (1 - pi/4). Anything at or below
#: that, times a safety factor, is corner noise rather than a thin feature.
_CORNER_SLIVER_COEFF = 1.0 - math.pi / 4.0

#: Headroom over the analytic corner crescent. At the default 1.2 mm wall
#: this puts the cut-off at ~0.19 mm2, comfortably above the ~0.08 mm2
#: slivers measured on square corners and far below the ~0.5 mm2 a genuine
#: sub-millimetre neck loses.
_SLIVER_HEADROOM = 2.5

#: Alpha at or above which a pixel counts as "not see-through" when deciding
#: whether an image uses its alpha channel at all.
_ALPHA_OPAQUE = 128

#: Fraction of pixels that must be meaningfully transparent before the alpha
#: channel is trusted as the silhouette. One stray anti-aliased pixel must
#: not turn a disc into the whole canvas.
_MIN_TRANSPARENT_FRACTION = 0.01

#: Above this detail-to-silhouette area ratio the "two-colour" split is
#: degenerate: the detail layer is just the silhouette again.
_DEGENERATE_DETAIL_RATIO = 0.95

#: Longest part the machine could ever print, used to tell "print it bigger"
#: apart from "this artwork will never print".
_MAX_PRINTABLE_MM = max(BED_SIZE_MM)

#: Margin on a suggested size. Scaling a feature to *exactly* the minimum
#: wall leaves it on the wrong side of the gate (measured: a neck suggested
#: at 89.6 mm still fails at 89.6 mm), and advice that does not work is
#: worse than no advice, so aim 10% past the line.
_SIZE_SAFETY = 1.1


# --- errors ----------------------------------------------------------------


class TraceError(Exception):
    """Base class for every failure raised by the trace stage."""


class UnsupportedImageError(TraceError):
    """The input file is not a raster image this stage can read."""


@dataclass(frozen=True)
class ThinRegion:
    """One patch of geometry that is narrower than the printable minimum.

    ``width_mm`` is a *measured* estimate of how wide the offending material
    actually is (largest disc that fits inside it), not a guess from its
    area. Callers use it to work out what size would clear the feature.
    """

    label: str
    bounds_mm: tuple[float, float, float, float]
    area_mm2: float
    width_mm: float = 0.0
    centroid_mm: tuple[float, float] = (0.0, 0.0)
    whole_part: bool = False

    def describe(self) -> str:
        cx, cy = self.centroid_mm
        if self.whole_part:
            # A bounding box is useless here - it is the whole part. Name
            # where the part sits and how much of it there is instead.
            return (
                f"{self.label}, centred at x {cx:.1f} mm, y {cy:.1f} mm "
                f"({self.area_mm2:.1f} mm2 of material, about "
                f"{self.width_mm:.2f} mm wide)"
            )
        x0, y0, x1, y1 = self.bounds_mm
        return (
            f"{self.label} around x {x0:.1f}-{x1:.1f} mm, "
            f"y {y0:.1f}-{y1:.1f} mm (about {self.width_mm:.2f} mm wide)"
        )


class ThinFeatureError(TraceError):
    """The silhouette has features the printer physically cannot make.

    Carries the offending :class:`ThinRegion` list and the minimum wall
    width that was applied, so callers (the CLI, the QC report) can render
    their own message instead of re-parsing this one.

    The suggested size is *derived from the measurement*, not from a fixed
    multiplier: the narrowest region has to grow to ``min_width_mm``, and
    everything scales with the part, so the size that clears it is
    ``min_width_mm / measured_width * size_mm``. When that lands beyond what
    the machine can print, the honest answer is that no size will do and the
    artwork itself has to change - :attr:`needs_bolder_artwork` says so.
    """

    def __init__(
        self,
        regions: list[ThinRegion],
        min_width_mm: float,
        size_mm: float,
    ) -> None:
        self.regions = regions
        self.min_width_mm = min_width_mm
        self.size_mm = size_mm
        self.suggested_size_mm = _size_that_clears(
            regions, min_width_mm, size_mm
        )
        self.needs_bolder_artwork = (
            self.suggested_size_mm is None
            or self.suggested_size_mm > _MAX_PRINTABLE_MM
        )

        where = "; ".join(r.describe() for r in regions[:3])
        if len(regions) > 3:
            where += f"; and {len(regions) - 3} more"
        if self.needs_bolder_artwork:
            if self.suggested_size_mm is None:
                need = ""
            else:
                need = (
                    f" Clearing it would need about "
                    f"{self.suggested_size_mm:.0f} mm across, past the "
                    f"{_MAX_PRINTABLE_MM:.0f} mm the printer can build."
                )
            advice = (
                f"{need} This artwork has hairline features that will not "
                f"print at any keychain size - no --size clears them, "
                f"because they get thinner in step with everything else. "
                f"Use a bolder image with thicker strokes."
            )
        else:
            advice = (
                f" Try a larger --size (around "
                f"{self.suggested_size_mm:.0f} mm would clear it) or a "
                f"bolder image with thicker strokes."
            )
        super().__init__(
            f"The outline of your image has parts thinner than "
            f"{min_width_mm} mm at the printed size ({size_mm:.0f} mm "
            f"across), so the printer cannot make them: {where}."
            f"{advice}"
        )


def _size_that_clears(
    regions: list[ThinRegion], min_width_mm: float, size_mm: float
) -> float | None:
    """Smallest overall size at which every flagged region clears the gate.

    Everything scales with the part, so a region measured at ``w`` mm when
    the part is ``size_mm`` across reaches ``min_width_mm`` at
    ``size_mm * min_width_mm / w``; the worst region sets the answer, plus
    :data:`_SIZE_SAFETY` so the suggestion actually passes when retried.

    ``None`` when nothing could be measured (a degenerate region with no
    measurable width at all), which is itself a "no size will help" answer.
    """
    widths = [r.width_mm for r in regions if r.width_mm > 0.0]
    if not widths:
        return None
    narrowest = min(widths)
    if narrowest >= min_width_mm:  # pragma: no cover - defensive
        return size_mm
    needed = size_mm * (min_width_mm / narrowest) * _SIZE_SAFETY
    return float(math.ceil(needed))


# --- result ----------------------------------------------------------------


@dataclass
class TraceResult:
    """Traced geometry for one design, in millimetres.

    Both geometries share an origin at (0, 0) with Y pointing up, so they
    can be handed straight to the solid builder. ``detail`` is always
    contained by ``silhouette`` and may be empty.
    """

    silhouette: MultiPolygon
    detail: MultiPolygon
    size_mm: float
    source: Path
    warnings: list[str] = field(default_factory=list)

    @property
    def width_mm(self) -> float:
        x0, _, x1, _ = self.silhouette.bounds
        return x1 - x0

    @property
    def height_mm(self) -> float:
        _, y0, _, y1 = self.silhouette.bounds
        return y1 - y0

    def to_svg(self, path: str | Path, *, include_detail: bool = True) -> Path:
        """Write the traced outline to an SVG file, in mm user units.

        This is the artefact that gets committed to ``designs/<slug>/``; it
        is not used as an intermediate by the rest of the pipeline.
        """
        out = Path(path)
        w, h = self.width_mm, self.height_mm
        parts = [
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{w:.3f}mm" height="{h:.3f}mm" '
            f'viewBox="0 0 {w:.3f} {h:.3f}">',
            # SVG's Y axis points down; our geometry's points up.
            f'<g transform="translate(0,{h:.3f}) scale(1,-1)">',
            '<path id="silhouette" fill="#000000" fill-rule="evenodd" '
            f'd="{_svg_path_data(self.silhouette)}"/>',
        ]
        if include_detail and not self.detail.is_empty:
            parts.append(
                '<path id="detail" fill="#cc2222" fill-rule="evenodd" '
                f'd="{_svg_path_data(self.detail)}"/>'
            )
        parts.append("</g></svg>\n")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(parts), encoding="utf-8")
        return out


# --- public entry points ---------------------------------------------------


def trace_image(
    image_path: str | Path,
    size_mm: float = DEFAULT_SIZE_MM,
    *,
    detail: bool = True,
    silhouette_threshold: float = DEFAULT_SILHOUETTE_THRESHOLD,
    detail_threshold: float = DEFAULT_DETAIL_THRESHOLD,
    simplify_mm: float = DEFAULT_SIMPLIFY_MM,
    min_wall_mm: float = MIN_WALL_MM,
    min_detail_stroke_mm: float = MIN_DETAIL_STROKE_MM,
    speckle_px: int = DEFAULT_SPECKLE_PX,
) -> TraceResult:
    """Trace ``image_path`` into printable silhouette and detail geometry.

    ``size_mm`` is the longest XY dimension of the finished silhouette;
    everything is scaled uniformly to hit it. Raises
    :class:`ThinFeatureError` if the silhouette has printable features
    narrower than ``min_wall_mm``.
    """
    source = Path(image_path)
    if size_mm <= 0:
        raise ValueError("size_mm must be positive")

    sil_mask, detail_mask = _load_masks(
        source, silhouette_threshold, detail_threshold
    )

    # Two very different failures used to share one message. Separate them:
    # "there is nothing here" needs different artwork, "what is here is
    # smaller than the speckle filter" needs a smaller filter.
    if not sil_mask.any():
        raise TraceError(
            f"No artwork found in {source.name}: every pixel looks like "
            f"background. Use an image with a solid, dark shape on a light "
            f"or transparent background."
        )
    silhouette_px = _trace_mask(sil_mask, speckle_px=speckle_px)
    if silhouette_px.is_empty:
        raise TraceError(
            f"Everything in {source.name} was removed as speckle: the marks "
            f"in it are all smaller than the {speckle_px} px2 speckle "
            f"filter. Pass a smaller speckle_px (0 disables it), or scan / "
            f"export the artwork at a higher resolution."
        )

    to_mm = _scaling(silhouette_px, size_mm)
    silhouette = _clean(to_mm(silhouette_px), simplify_mm)
    # Simplification nudges the outline by a few hundredths of a mm, so the
    # final fit to size happens afterwards: the requested size is exact.
    fit = _fit_transform(silhouette.bounds, size_mm)
    silhouette = _as_multipolygon(fit(silhouette))

    thin = find_thin_regions(silhouette, min_wall_mm, label="the outline")
    if thin:
        raise ThinFeatureError(thin, min_wall_mm, size_mm)

    warnings: list[str] = []
    detail_geom = _empty()
    if detail:
        detail_px = _trace_mask(detail_mask, speckle_px=speckle_px)
        if not detail_px.is_empty:
            detail_geom = fit(_clean(to_mm(detail_px), simplify_mm))
            detail_geom = _as_multipolygon(
                detail_geom.intersection(silhouette)
            )
            # A single flat dark shape traces to detail == silhouette. That
            # is not a two-colour design, it is one colour twice; say so
            # rather than emitting a detail layer identical to the base.
            ratio = (
                detail_geom.area / silhouette.area
                if silhouette.area > 0
                else 0.0
            )
            if ratio > _DEGENERATE_DETAIL_RATIO:
                detail_geom = _empty()
                warnings.append(
                    f"The whole image is one dark shape ({ratio * 100:.0f}% "
                    f"of the outline), so there is no separate detail layer "
                    f"to raise - printing single-colour. Use an image with "
                    f"lighter and darker areas if you want two colours."
                )
            else:
                detail_geom = _drop_thin_parts(
                    detail_geom, min_detail_stroke_mm, warnings
                )
        if detail_geom.is_empty and not warnings:
            warnings.append(
                "No detail regions were found, so the design will be a "
                "single-colour silhouette. Use an image with darker "
                "markings inside the shape if you want a two-colour print."
            )

    return TraceResult(
        silhouette=silhouette,
        detail=detail_geom,
        size_mm=size_mm,
        source=source,
        warnings=warnings,
    )


def trace_silhouette(
    image_path: str | Path, size_mm: float = DEFAULT_SIZE_MM, **kwargs
) -> MultiPolygon:
    """Convenience wrapper: just the silhouette, in mm."""
    _reject_detail_kwarg("trace_silhouette", kwargs, "never traces detail")
    return trace_image(image_path, size_mm, detail=False, **kwargs).silhouette


def trace_detail(
    image_path: str | Path, size_mm: float = DEFAULT_SIZE_MM, **kwargs
) -> MultiPolygon:
    """Convenience wrapper: just the detail regions, in mm."""
    _reject_detail_kwarg("trace_detail", kwargs, "always traces detail")
    return trace_image(image_path, size_mm, detail=True, **kwargs).detail


def _reject_detail_kwarg(name: str, kwargs: dict, why: str) -> None:
    """Turn a ``detail=`` collision into an explanation, not a stack trace."""
    if "detail" in kwargs:
        raise TypeError(
            f"{name}() does not accept 'detail' - it {why}. Call "
            f"trace_image(..., detail=...) if you need to choose."
        )


# --- thin-feature check ----------------------------------------------------


def find_thin_regions(
    geometry: BaseGeometry,
    min_width_mm: float = MIN_WALL_MM,
    label: str = "the outline",
) -> list[ThinRegion]:
    """Return the parts of ``geometry`` narrower than ``min_width_mm``.

    The test is a morphological **opening**: erode by half the minimum
    width, then dilate back by the same amount. Anything the erosion ate
    and the dilation could not restore was narrower than ``min_width_mm``
    somewhere - that covers thin plates, thin necks between two thick
    blobs, and thin spikes alike. Concavities are untouched by an opening,
    so decorative notches and cut-outs are never flagged.

    The dilation also rounds off sharp convex corners, which leaves tiny
    sliver artefacts. Those are bounded analytically - a square corner
    rounded by radius r loses exactly ``r**2 * (1 - pi/4)`` - so the cut-off
    is that crescent plus headroom (:data:`_SLIVER_HEADROOM`), about
    0.19 mm2 at a 1.2 mm wall. That sits just above the ~0.08 mm2 measured
    on real square corners and well below the ~0.5 mm2 that a sub-millimetre
    neck between two thick blobs loses, so genuine thin necks are caught.
    """
    if geometry.is_empty:
        return []
    radius = min_width_mm / 2.0
    sliver_mm2 = _sliver_area_mm2(min_width_mm)
    single = _count_parts(geometry) == 1

    found: list[ThinRegion] = []
    for index, part in enumerate(_parts(geometry), start=1):
        name = label if single else f"{label}, part {index}"
        core = part.buffer(-radius, join_style="round")
        if core.is_empty:
            found.append(
                _thin_region(
                    f"{name} (thinner than {min_width_mm} mm all over)",
                    part,
                    min_width_mm,
                    whole_part=True,
                )
            )
            continue
        lost = part.difference(core.buffer(radius, join_style="round"))
        for patch in _parts(lost):
            if patch.area > sliver_mm2:
                found.append(_thin_region(name, patch, min_width_mm))
    found.sort(key=lambda r: r.area_mm2, reverse=True)
    return found


def _sliver_area_mm2(min_width_mm: float) -> float:
    """Area below which a lost patch is corner round-off, not a feature."""
    radius = min_width_mm / 2.0
    return _SLIVER_HEADROOM * _CORNER_SLIVER_COEFF * radius * radius


def _thin_region(
    name: str,
    patch: Polygon,
    min_width_mm: float,
    *,
    whole_part: bool = False,
) -> ThinRegion:
    centroid = patch.centroid
    return ThinRegion(
        label=name,
        bounds_mm=_round4(patch.bounds),
        area_mm2=patch.area,
        width_mm=_measure_width_mm(patch, min_width_mm),
        centroid_mm=(round(float(centroid.x), 4), round(float(centroid.y), 4)),
        whole_part=whole_part,
    )


def _measure_width_mm(
    geometry: BaseGeometry, cap_mm: float, steps: int = 14
) -> float:
    """Estimate how wide ``geometry`` actually is, in mm.

    The widest disc that fits inside the patch, found by bisecting on
    "erode by r and something survives". For a patch that a morphological
    opening removed, that diameter *is* the offending neck/stroke width.
    Approximate by design - a couple of hundredths of a mm is plenty to
    decide what size would clear it - but never rounded the wrong way: the
    bisection keeps the last radius that survived, so the answer is a lower
    bound on the true width and the size it implies is never too small.
    """
    if geometry.is_empty:
        return 0.0
    hi = cap_mm / 2.0
    if not geometry.buffer(-hi, join_style="round").is_empty:
        return cap_mm  # at least as wide as the minimum; not the culprit
    lo = 0.0
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        if geometry.buffer(-mid, join_style="round").is_empty:
            hi = mid
        else:
            lo = mid
    return 2.0 * lo


def _drop_thin_parts(
    geometry: MultiPolygon, min_width_mm: float, warnings: list[str]
) -> MultiPolygon:
    """Remove the too-narrow *parts of* the detail, warning about them.

    Not per connected component: letterforms join their hairlines to their
    bold strokes, so an all-or-nothing test on each component either keeps
    an unprintable tail or throws away a whole legible glyph. This is the
    same morphological opening the silhouette gate uses - erode by half the
    minimum stroke, dilate back - which deletes the thin *portions* and
    leaves the thick ones, whether or not they are connected.

    Every removal lands in ``warnings`` with the area it took away, so the
    QC report can surface it.
    """
    if geometry.is_empty:
        return _empty()
    radius = min_width_mm / 2.0
    opened = _as_multipolygon(
        make_valid(
            geometry.buffer(-radius, join_style="round")
            .buffer(radius, join_style="round")
            .intersection(geometry)
        )
    )
    lost = geometry.difference(opened)
    sliver_mm2 = _sliver_area_mm2(min_width_mm)
    removed = [p for p in _parts(lost) if p.area > sliver_mm2]
    if not removed:
        # Only corner round-off came off. Keep the detail exactly as traced
        # rather than handing back needlessly rounded corners.
        return _as_multipolygon(geometry)

    removed_mm2 = sum(p.area for p in removed)
    warnings.append(
        f"Removed {len(removed)} detail area(s) narrower than "
        f"{min_width_mm} mm ({removed_mm2:.2f} mm2 in total) - strokes that "
        f"fine disappear at this size. Print larger, or thicken those "
        f"markings in the source image, if you need them."
    )
    return opened


# --- image loading ---------------------------------------------------------


def _load_masks(
    source: Path, silhouette_threshold: float, detail_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``source`` and return (silhouette mask, detail mask).

    Both masks are boolean arrays, True where there is ink. The silhouette
    comes from the alpha channel when the image has one and actually uses
    it, and from luminance otherwise. Detail is always luminance-based, and
    always clipped to the silhouette.
    """
    if source.suffix.lower() == ".svg":
        raise UnsupportedImageError(
            "SVG input is not supported - this stage traces raster images "
            "only, and there is no SVG importer planned for v1. Export your "
            "drawing as a PNG (300 px or more across) and trace that "
            "instead."
        )
    if not source.exists():
        raise UnsupportedImageError(f"No such image: {source}")
    try:
        image = Image.open(source)
        image.load()
    except OSError as exc:  # unreadable, or not an image at all
        raise UnsupportedImageError(
            f"Could not read {source.name} as an image ({exc}). PNG and JPG "
            f"files work best."
        ) from exc

    rgba = image.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"))
    # Composite over white so transparent pixels read as background.
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    gray = np.array(Image.alpha_composite(white, rgba).convert("L"))

    # "Has an alpha channel" is not "uses transparency": a single
    # anti-aliased corner pixel at alpha 249 must not switch the whole
    # silhouette over to the alpha channel (which would then read the entire
    # opaque canvas as the part). Demand real, deliberate transparency.
    transparent_fraction = float((alpha < _ALPHA_OPAQUE).mean())
    uses_alpha = transparent_fraction >= _MIN_TRANSPARENT_FRACTION
    if uses_alpha:
        # ``silhouette_threshold`` is honoured here too, as its opacity dual:
        # luminance keeps pixels darker than ``threshold`` of white, so alpha
        # keeps pixels more opaque than ``1 - threshold`` of solid. At the
        # 0.85 default that admits anything above alpha 38, i.e. all but the
        # faintest fringe.
        silhouette_mask = alpha > (255.0 * (1.0 - silhouette_threshold))
    else:
        silhouette_mask = gray < (255.0 * silhouette_threshold)

    detail_mask = (gray < (255.0 * detail_threshold)) & silhouette_mask
    return silhouette_mask, detail_mask


# --- potrace plumbing ------------------------------------------------------


def _trace_mask(mask: np.ndarray, *, speckle_px: int) -> MultiPolygon:
    """Trace a boolean ink mask into polygons in pixel coordinates.

    ``potrace.Bitmap`` inverts whatever it is handed and then traces the
    True areas, so the *background* mask is what goes in.
    """
    if not mask.any():
        return _empty()
    bitmap = potrace.Bitmap(np.ascontiguousarray(~mask))
    path = bitmap.trace(turdsize=speckle_px)

    shells: list[Polygon] = []
    holes: list[Polygon] = []
    for curve in path:
        ring = _flatten(curve)
        if len(ring) < 4:
            continue
        polygon = Polygon(ring)
        if polygon.area <= 0:
            continue
        # ``curve._path.sign`` is potracer-private (no public accessor
        # exists for a curve's winding direction) and could break on a
        # potracer upgrade; pyproject pins potracer>=0.0.4 for this reason.
        # True means an outer boundary, False means a hole.
        (shells if curve._path.sign else holes).append(polygon)

    if not shells:
        return _empty()

    # Attach every hole to the smallest shell that covers it, so nested
    # islands (a dot inside a ring inside a disc) survive intact. Whole-ring
    # containment rather than a probe point: potrace happily emits hairline
    # slivers whose bounding box swallows half the image, and a single
    # sample point can land right on one of them.
    shells.sort(key=lambda p: p.area)
    assigned: dict[int, list] = {id(s): [] for s in shells}
    for hole in holes:
        for shell in shells:
            if shell.covers(hole):
                assigned[id(shell)].append(hole.exterior.coords)
                break

    built = [
        Polygon(shell.exterior.coords, assigned[id(shell)]) for shell in shells
    ]
    return _as_multipolygon(unary_union([make_valid(p) for p in built]))


def _flatten(curve) -> list[tuple[float, float]]:
    """Flatten one potrace curve into a closed polyline."""
    start = curve.start_point
    current = (start.x, start.y)
    points = [current]
    for segment in curve:
        if segment.is_corner:
            points.append((segment.c.x, segment.c.y))
            points.append((segment.end_point.x, segment.end_point.y))
        else:
            p0 = current
            p1 = (segment.c1.x, segment.c1.y)
            p2 = (segment.c2.x, segment.c2.y)
            p3 = (segment.end_point.x, segment.end_point.y)
            for step in range(1, _BEZIER_STEPS + 1):
                t = step / _BEZIER_STEPS
                u = 1.0 - t
                points.append(
                    (
                        u**3 * p0[0]
                        + 3 * u * u * t * p1[0]
                        + 3 * u * t * t * p2[0]
                        + t**3 * p3[0],
                        u**3 * p0[1]
                        + 3 * u * u * t * p1[1]
                        + 3 * u * t * t * p2[1]
                        + t**3 * p3[1],
                    )
                )
        current = (segment.end_point.x, segment.end_point.y)
    return points


# --- geometry helpers ------------------------------------------------------


def _scaling(silhouette_px: MultiPolygon, size_mm: float):
    """Build the pixel-space to mm transform.

    Uniform scale so the silhouette's longest side is exactly ``size_mm``,
    Y flipped (image rows run downwards, model space runs up), and the
    silhouette's bounding box moved to the origin. The same transform is
    applied to the detail geometry so the two stay registered.
    """
    x0, y0, x1, y1 = silhouette_px.bounds
    span = max(x1 - x0, y1 - y0)
    if span <= 0:
        raise TraceError("Traced artwork has no area.")
    factor = size_mm / span

    def transform(geometry: BaseGeometry) -> BaseGeometry:
        flipped = affinity.scale(
            geometry, xfact=factor, yfact=-factor, origin=(0.0, 0.0)
        )
        return affinity.translate(flipped, xoff=-x0 * factor, yoff=y1 * factor)

    return transform


def _fit_transform(bounds, size_mm: float):
    """Scale/translate so ``bounds`` sits at the origin, ``size_mm`` across.

    Applied to the silhouette and the detail alike, from the silhouette's
    bounds, so the two stay registered to one another.
    """
    x0, y0, x1, y1 = bounds
    span = max(x1 - x0, y1 - y0)
    if span <= 0:
        raise TraceError("Traced artwork has no area.")
    factor = size_mm / span

    def transform(geometry: BaseGeometry) -> BaseGeometry:
        scaled = affinity.scale(
            geometry, xfact=factor, yfact=factor, origin=(0.0, 0.0)
        )
        return affinity.translate(
            scaled, xoff=-x0 * factor, yoff=-y0 * factor
        )

    return transform


def _clean(geometry: BaseGeometry, simplify_mm: float) -> MultiPolygon:
    """Simplify to nozzle resolution and repair any self-intersections."""
    cleaned = make_valid(geometry)
    if simplify_mm > 0:
        cleaned = make_valid(
            cleaned.simplify(simplify_mm, preserve_topology=True)
        )
    return _as_multipolygon(cleaned)


def _as_multipolygon(geometry: BaseGeometry) -> MultiPolygon:
    """Normalise anything shapely hands back into a MultiPolygon of areas."""
    if geometry.is_empty:
        return _empty()
    polygons = [part for part in _parts(geometry) if part.area > 0]
    return MultiPolygon(polygons) if polygons else _empty()


def _parts(geometry: BaseGeometry) -> list[Polygon]:
    """Connected polygon components, ignoring empties, points and lines."""
    if geometry.is_empty:
        return []
    out: list[Polygon] = []
    for geom in getattr(geometry, "geoms", [geometry]):
        if geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            out.append(geom)
        elif hasattr(geom, "geoms"):
            out.extend(_parts(geom))
    return out


def _count_parts(geometry: BaseGeometry) -> int:
    return len(_parts(geometry))


def _empty() -> MultiPolygon:
    return MultiPolygon()


def _round4(bounds) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (round(float(v), 4) for v in bounds)
    return (x0, y0, x1, y1)


def _svg_path_data(geometry: BaseGeometry) -> str:
    """Render a MultiPolygon as one SVG path, even-odd filled."""
    chunks: list[str] = []
    for polygon in _parts(geometry):
        for ring in [polygon.exterior, *polygon.interiors]:
            coords = [
                (x, y)
                for x, y in ring.coords
                if not (math.isnan(x) or math.isnan(y))
            ]
            if len(coords) < 4:
                continue
            head = f"M {coords[0][0]:.4f},{coords[0][1]:.4f}"
            body = " ".join(f"L {x:.4f},{y:.4f}" for x, y in coords[1:-1])
            chunks.append(f"{head} {body} Z")
    return " ".join(chunks)
