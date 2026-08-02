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

# --- photo mode (``--photo``) -----------------------------------------------
#
# A phone photo of a paper drawing is not the clean, flat artwork the rest
# of this module assumes: paper shading and lighting gradients make "how
# light is this pixel" a function of where on the page it sits, not just of
# whether there is ink there. Photo mode fixes that up *before* the mask
# logic above ever runs, so everything past it - including the degenerate
# single-shade case - stays exactly as it already was.

#: Shorter side, in pixels, that photo mode works at. A phone photo is
#: 3000-4000 px on its short side; tracing gains nothing from that. At the
#: largest part this machine can print (~250 mm) 1200 px is still ~0.2 mm
#: per pixel, half a 0.4 mm nozzle width, and at the default 50 mm part it
#: is 0.04 mm per pixel - finer than :data:`DEFAULT_SIMPLIFY_MM`, so those
#: extra pixels are simplified away moments later regardless. Anything
#: bigger is downscaled (Lanczos) before cleanup, which also bounds the
#: cost of everything downstream: the whole photo path is O(pixels) with
#: the pixel count capped here. One knock-on to know about: pixel-domain
#: parameters like ``speckle_px`` apply at this working size, not at the
#: photo's native size, so on a downscaled 12 MP photo one speckle unit
#: covers correspondingly more of the page - stronger noise suppression,
#: which is what a noisy phone photo wants anyway.
_PHOTO_MAX_SHORT_SIDE_PX = 1200

#: Fraction of the shorter side used as the blur radius for the
#: illumination field (see :func:`_illumination_field`). Lighting across a
#: sheet of paper varies slowly, so the field wants a wide window: wide
#: enough to reach across a big filled shape and pick up the paper on the
#: far side, narrow enough to still follow a real shadow edge. A sixth of
#: the short side, applied as three box passes, reaches half the image.
_PHOTO_FIELD_FRACTION = 1.0 / 6.0

#: Minimum field radius, in pixels, for images small enough that the
#: fraction above would round down to a window narrower than a stroke.
_PHOTO_FIELD_MIN_PX = 8

#: A pixel is provisionally ink when it is at least this much darker than
#: the local paper level around it - i.e. below 80% of the field. Paper
#: shading and sensor noise move a pixel by a few percent; a pencil line
#: moves it by tens of percent. The gap between those two is what this
#: sits in, and it is what lets a blank, unevenly lit page come back with
#: *no* ink rather than with its dim half called ink.
_PHOTO_INK_RATIO = 0.80

#: How many times the field is re-estimated with the ink masked out. One
#: pass already fixes the large-filled-shape case; a second lets the ink
#: mask it was estimated from be refined against the flattened image, which
#: tightens stroke edges. Beyond that, measured changes are nil.
_PHOTO_FIELD_PASSES = 2

#: Weight of the "no paper anywhere near here" fallback in the field's
#: normalised convolution, in units of paper pixels. Deep inside a shape
#: bigger than the blur can see across there is no local paper to
#: interpolate from and the ratio would divide near-zero by near-zero;
#: this pulls those pixels to the page's overall paper level instead.
#: Tiny, so anywhere with real paper support is unaffected.
_PHOTO_FIELD_FALLBACK_WEIGHT = 1e-3

#: Minimum gap, on the flattened 0-255 scale, between the mean brightness of
#: what Otsu calls "paper" and what it calls "ink" before that split is
#: trusted as real strokes rather than residual shading. Otsu always
#: returns *some* split, even across a perfectly blank, evenly-lit page -
#: measured at ~2 there from illumination-correction noise alone, against
#: ~45+ for the faintest real pencil tested. This sits well clear of both.
_PHOTO_MIN_CONTRAST = 20.0

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
    photo: bool = False,
) -> TraceResult:
    """Trace ``image_path`` into printable silhouette and detail geometry.

    ``size_mm`` is the longest XY dimension of the finished silhouette;
    everything is scaled uniformly to hit it. Raises
    :class:`ThinFeatureError` if the silhouette has printable features
    narrower than ``min_wall_mm``.

    ``photo=True`` runs a cleanup pass on the loaded grayscale *before* any
    of the above: it flattens uneven paper shading/lighting and turns
    pencil/marker strokes into ink-black on paper-white (see
    :func:`_flatten_photo`). That cleaned image then feeds the exact same
    threshold logic as any other input - a photographed drawing is a single
    shade of ink, so the detail layer typically comes back empty or
    degenerate, which is the existing, expected behaviour for flat artwork,
    not a photo-specific special case.
    """
    source = Path(image_path)
    if size_mm <= 0:
        raise ValueError("size_mm must be positive")

    sil_mask, detail_mask = _load_masks(
        source, silhouette_threshold, detail_threshold, photo=photo
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
    source: Path,
    silhouette_threshold: float,
    detail_threshold: float,
    *,
    photo: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``source`` and return (silhouette mask, detail mask).

    Both masks are boolean arrays, True where there is ink. The silhouette
    comes from the alpha channel when the image has one and actually uses
    it, and from luminance otherwise. Detail is always luminance-based, and
    always clipped to the silhouette.

    ``photo=True`` replaces the loaded luminance with :func:`_flatten_photo`'s
    cleaned-up version before either mask is built, and - because
    transparency has no meaning for a photograph of a physical piece of
    paper - forces the alpha channel to be ignored entirely, even if the
    file happens to carry one. A phone camera never produces one, but a
    photo run through an image editor first might; treating it as opaque is
    the documented behaviour rather than an accident of which branch runs.
    Photo mode also caps the resolution it works at, so the masks it
    returns may be smaller than the file on disk - see
    :data:`_PHOTO_MAX_SHORT_SIDE_PX`. Nothing downstream cares: the
    silhouette is scaled to ``size_mm`` from its own traced extent.
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

    if photo:
        gray = _flatten_photo(gray, source)
        uses_alpha = False
    else:
        # "Has an alpha channel" is not "uses transparency": a single
        # anti-aliased corner pixel at alpha 249 must not switch the whole
        # silhouette over to the alpha channel (which would then read the
        # entire opaque canvas as the part). Demand real, deliberate
        # transparency.
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


def _flatten_photo(gray: np.ndarray, source: Path) -> np.ndarray:
    """Turn a photographed drawing into clean ink-on-paper, in place of raw
    grayscale.

    Four steps, all numpy + PIL:

    1. **Shrink to a working size.** Photo mode caps the shorter side at
       :data:`_PHOTO_MAX_SHORT_SIDE_PX` (see there for why that loses
       nothing printable). Everything below - and every stage after this
       function - then runs on a bounded number of pixels, which is what
       keeps a 12-megapixel phone photo down to a couple of seconds.
    2. **Estimate the illumination.** :func:`_illumination_field` works out
       what each pixel's paper would have looked like with no ink on it,
       reading *only* pixels that currently look like paper and
       interpolating across the ones that look like ink.
    3. **Flatten it out.** Dividing the grayscale by that field (rescaled
       back to 0-255) cancels the shading: a stroke on a dim corner of the
       page and the same stroke on a bright one come out equally dark.
    4. **Threshold.** Otsu's method (:func:`_otsu_threshold`) finds the
       graylevel that best splits the flattened image into two populations
       - ink and paper - with no fixed cutoff to tune per photo.

    Step 2 is the part worth spelling out. The obvious illumination
    estimate - blur the photo, or take the brightest pixel in a window
    around each one - reads the ink as if it were paper. For thin strokes
    that barely matters. For the shapes children actually draw, a filled
    heart or sun wider than the window, it is fatal: the shape's own
    darkness lands in the estimate of the paper underneath it, dividing
    cancels it against itself, and the middle of the shape comes back
    reading as blank page. So the field is a **normalised convolution**
    instead: blur the paper pixels' values and blur the paper mask, then
    divide one by the other. Ink pixels contribute nothing to either, so
    the field over a filled shape is interpolated from the paper around
    it - exactly the lighting the paper under the shape would have had.

    The result is handed back as a plain grayscale array (ink at 0, paper
    at 255) so every threshold downstream of this function - silhouette,
    detail, both - runs exactly as it does for any other input.
    """
    work = _photo_working_copy(gray).astype(np.float64)
    height, width = work.shape
    radius = max(
        _PHOTO_FIELD_MIN_PX,
        int(min(height, width) * _PHOTO_FIELD_FRACTION),
    )

    # First guess at the paper: a plain blur, ink included. It is wrong
    # over big filled shapes - dragged down by the very ink it is meant to
    # ignore - but it is wrong in the *safe* direction, because a shape
    # dark enough to drag the local average down is also dark enough to
    # fall well under _PHOTO_INK_RATIO of it. That is all the first ink
    # mask has to get right; the masked passes below do the rest.
    field = _box_blur(work, radius)
    for _ in range(_PHOTO_FIELD_PASSES):
        paper = work >= field * _PHOTO_INK_RATIO
        field = _illumination_field(work, paper, radius)

    flattened = np.clip(work / np.maximum(field, 1.0) * 255.0, 0.0, 255.0)

    threshold = _otsu_threshold(flattened)
    # ``threshold`` names a histogram *bin*, and that bin belongs to the
    # dark side of the split (:func:`_otsu_threshold` counts it there), so
    # the cut is one bin above it. A strict ``<`` instead drops every pixel
    # that lands exactly on the split level - which on clean, low-noise
    # artwork is not a stray pixel but a whole region of even ink.
    ink_mask = flattened < threshold + 1.0
    # Otsu always returns *a* split, even across a blank page - the residual
    # noise from illumination correction still has a darker and a lighter
    # half. A real stroke reads much darker than the paper around it even
    # after flattening; residual noise does not. Demanding a minimum gap
    # between the two sides' means tells the two cases apart.
    contrast = (
        float(flattened[~ink_mask].mean()) - float(flattened[ink_mask].mean())
        if ink_mask.any() and (~ink_mask).any()
        else 0.0
    )

    if not ink_mask.any() or contrast < _PHOTO_MIN_CONTRAST:
        raise TraceError(
            f"No pencil or marker strokes were found in {source.name} after "
            f"correcting for the lighting. Try again with brighter, more "
            f"even light and no shadow falling across the page, or draw "
            f"with a darker pen or pencil - going over the lines a second "
            f"time so they stand out from the paper."
        )

    return np.where(ink_mask, 0, 255).astype(np.uint8)


def _photo_working_copy(gray: np.ndarray) -> np.ndarray:
    """Downscale ``gray`` to the photo-mode working size, if it is bigger.

    Lanczos, because it is the resampler that keeps a thin pencil line
    looking like a line rather than a dotted one. Images already at or
    under the cap are handed straight back untouched, so the small
    hand-scanned artwork case is bit-for-bit what it always was.
    """
    height, width = gray.shape
    short_side = min(height, width)
    if short_side <= _PHOTO_MAX_SHORT_SIDE_PX:
        return gray
    scale = _PHOTO_MAX_SHORT_SIDE_PX / short_side
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    shrunk = Image.fromarray(gray, mode="L").resize(size, Image.LANCZOS)
    return np.asarray(shrunk, dtype=np.uint8)


def _illumination_field(
    values: np.ndarray, paper: np.ndarray, radius: int
) -> np.ndarray:
    """The paper's brightness everywhere, read only from ``paper`` pixels.

    A normalised convolution: blur the paper pixels' values, blur the paper
    mask itself, divide. Where a pixel has paper around it the answer is
    the local average of that paper; where it does not - the middle of a
    filled-in shape - the weight tails off and the answer is dominated by
    whatever paper is nearest, which is the right interpolation to make.

    Well inside a shape wider than the blur can see across there is no
    paper support at all, so both sums approach zero and the ratio would be
    numerical noise. :data:`_PHOTO_FIELD_FALLBACK_WEIGHT` adds a whisper of
    "the page's overall paper level" to both halves, which those pixels
    then fall back to and every other pixel ignores.
    """
    fallback = float(values[paper].mean()) if paper.any() else float(values.mean())
    weight = _box_blur(paper.astype(np.float64), radius)
    total = _box_blur(np.where(paper, values, 0.0), radius)
    return (total + _PHOTO_FIELD_FALLBACK_WEIGHT * fallback) / (
        weight + _PHOTO_FIELD_FALLBACK_WEIGHT
    )


def _box_blur(values: np.ndarray, radius: int, passes: int = 3) -> np.ndarray:
    """Blur ``values`` by averaging over a square window, ``passes`` times.

    Three box passes approximate a Gaussian closely enough for something as
    smooth as a lighting field (it is how PIL's own ``GaussianBlur`` is
    built), and a box average over running sums costs the same *whatever
    the window size* - one pass over the pixels per axis, no multiply by
    window area. That is the difference that matters here: the window is a
    sixth of the image, and the first version of this cleanup used a rank
    filter that big, which cost pixels times window squared and took the
    better part of an hour on a real phone photo.

    Edges replicate rather than fade to black, so paper at the margin of
    the page is not mistaken for shadow.
    """
    out = np.asarray(values, dtype=np.float64)
    for _ in range(passes):
        for axis in (0, 1):
            out = _box_blur_axis(out, radius, axis)
    return out


def _box_blur_axis(values: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """One box average of ``values`` along ``axis``, via a running sum."""
    moved = np.moveaxis(values, axis, -1)
    padded = np.pad(moved, ((0, 0), (radius, radius)), mode="edge")
    window = 2 * radius + 1
    running = np.cumsum(padded, axis=-1)
    running = np.concatenate(
        (np.zeros(running.shape[:-1] + (1,)), running), axis=-1
    )
    averaged = (running[..., window:] - running[..., :-window]) / window
    return np.moveaxis(averaged, -1, axis)


def _otsu_threshold(values: np.ndarray) -> float:
    """Otsu's method: the graylevel splitting ``values`` into two
    populations with the least combined within-population variance.

    Computed from the image's own 256-bin histogram, so no per-photo
    calibration and no extra dependency beyond numpy. Standard formulation:
    for every candidate split, the "cost" is each side's population times
    its variance; Otsu shows that is minimised exactly where the *between*-
    population variance below is maximised, which needs only running sums
    over the histogram rather than recomputing a variance per candidate.
    """
    hist, _ = np.histogram(values, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:  # pragma: no cover - defensive, values is never empty
        return 128.0

    levels = np.arange(256, dtype=np.float64)
    cum_weight = np.cumsum(hist)
    cum_intensity = np.cumsum(hist * levels)
    total_intensity = cum_intensity[-1]

    weight_bg = cum_weight
    weight_fg = total - cum_weight
    # Levels with zero population on one side can't be a real split; give
    # them zero between-class variance instead of a division warning.
    valid = (weight_bg > 0) & (weight_fg > 0)
    mean_bg = np.zeros_like(cum_weight)
    mean_fg = np.zeros_like(cum_weight)
    mean_bg[valid] = cum_intensity[valid] / weight_bg[valid]
    mean_fg[valid] = (total_intensity - cum_intensity[valid]) / weight_fg[valid]

    between = np.zeros_like(cum_weight)
    between[valid] = (
        weight_bg[valid] * weight_fg[valid] * (mean_bg[valid] - mean_fg[valid]) ** 2
    )
    return float(np.argmax(between))


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
