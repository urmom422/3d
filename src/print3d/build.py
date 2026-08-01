"""Solid construction: 2D geometry to a print-ready 3D model.

Wraps build123d (extrude, hole subtract, chamfer) and exports via trimesh
(3MF path needs networkx + lxml; py-lib3mf has no Windows wheels — lib3mf
itself arrives transitively via build123d). Base and detail layers are
built as separate objects for AMS two-color mapping.

The stage takes the trace stage's two shapely geometries — silhouette and
detail, both in mm, both sharing an origin at the silhouette's bounding-box
corner with Y up — and returns :class:`KeychainSolids`: a base body, an
optional detail body, and the warnings collected on the way. Z = 0 is the
print bed, so the bottom face sits flat on it and nothing needs moving in
the slicer.

Three things here are load-bearing enough to spell out:

* **Sizing belongs to the trace stage.** Geometry arrives already scaled to
  ``spec.max_dimension_mm``; the builder never rescales it, because scaling
  after the wall-thickness gate would invalidate the gate. The only thing
  that can grow the footprint is a hole tab, and that is warned about.
* **The hole's margin is a contract, not a preference.** A keychain whose
  ring tears out is not a keychain, so the builder either finds a spot with
  a full ring of material around the hole, grows a tab to make one, or
  raises :class:`HoleMarginError`. It never emits a thin-lipped hole.
* **The chamfer is cosmetic, and it never touches the hole.** It is cut
  into the bare base prism *before* the hanging hole, so the hole is a
  straight 5.2 mm cylinder at every height and the margin contract survives.
  Traced outlines are hundreds of short segments and OCCT's chamfer gives up
  on some of them. That is a warning, never a failure: a keychain with a
  sharp top lip still prints.
* **The silhouette must be one connected part.** The trace stage keeps
  islands on purpose, but a keychain that traces to two islands prints as
  two loose pieces, so the builder refuses it with
  :class:`DisconnectedSilhouetteError` rather than exporting a bag of
  bodies. Interior rings inside one part (a donut) are fine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import trimesh
from build123d import Axis, Compound, Face, Location, Shape, Solid, Vector, Wire
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid

from print3d.spec import KeychainSpec

__all__ = [
    "BuildError",
    "DEFAULT_TESSELLATION_MM",
    "DisconnectedSilhouetteError",
    "EmptyGeometryError",
    "HoleMarginError",
    "KeychainSolids",
    "build_keychain",
    "export_3mf",
    "export_stl",
]


# --- tuning constants ------------------------------------------------------

#: Linear deflection handed to OCCT's tessellator, in mm. Well under what a
#: 0.4 mm nozzle can resolve, and an order of magnitude finer than the
#: 0.05 mm the trace stage already simplified to, so the mesh is faithful to
#: the solid without turning a keychain into a million triangles (this
#: machine has ~8 GB of RAM; a finer setting buys nothing and costs a lot).
DEFAULT_TESSELLATION_MM = 0.02

#: Angular deflection, in radians. Governs how finely the hole cylinder and
#: the chamfer's curved faces are faceted.
DEFAULT_ANGULAR_TOLERANCE = 0.2

#: Slack on the hole-margin test, in mm. Absorbs the last-bit noise in
#: shapely's offsetting; a micron of margin is not a real-world difference.
_MARGIN_TOL_MM = 1e-6

#: How far below the top of the searchable region the automatic hole probe
#: runs, in mm. Sampling exactly at the apex of an eroded polygon usually
#: yields a single vertex whose X is wherever the outline happened to peak;
#: dropping a hair lets the probe cross a whole horizontal span and pick the
#: X nearest the centre, which is what "top-centre" is supposed to mean.
_PROBE_DROP_MM = 0.01

#: Extra radius on a fallback tab, in mm, over the bare hole-plus-margin.
#: Shapely approximates a circle with chords that fall *inside* the true
#: circle, so a tab drawn at exactly the required radius measures a few
#: ten-thousandths short. Draw it slightly proud instead of loosening the
#: margin test.
_TAB_OVERSIZE_MM = 0.05

#: Segments per quarter circle when drawing a tab. High enough that the tab
#: reads as round in the print, not as a polygon.
_TAB_QUAD_SEGS = 64

#: How far a fallback tab's centre sits beyond the silhouette's topmost
#: material, as a fraction of the tab radius. Under 1.0 by construction, so
#: the tab always overlaps real material and the union stays one body.
_TAB_OVERLAP_FRACTION = 0.6

#: Vertical slack when selecting "the edges on the top face", in mm.
_TOP_FACE_TOL_MM = 1e-4

#: How far past the model the hole cylinder is stretched, in mm, so the cut
#: is unambiguously through rather than tangent at either end.
_CUT_OVERSHOOT_MM = 1.0

#: Slack on the "did the part come out bigger than asked for" check, in mm.
#: Tessellation and shapely's chord approximation both wobble at the third
#: decimal; a tenth of a millimetre is the smallest difference worth a word.
_SIZE_TOL_MM = 0.1

#: Smallest area of trimmed detail worth warning about, in mm^2. Clipping
#: the detail to the chamfered top face rounds convex corners by a hair even
#: when nothing really overhangs, and that is not news.
_TRIM_REPORT_MM2 = 0.01

#: Ring points closer together than this are treated as one, in mm. OCCT
#: refuses to build an edge between coincident vertices, and simplified
#: traces do emit the odd duplicate.
_POINT_MERGE_MM = 1e-7


# --- errors ----------------------------------------------------------------


class BuildError(Exception):
    """Base class for every failure raised by the build stage."""


class EmptyGeometryError(BuildError):
    """There is no silhouette to extrude."""


class DisconnectedSilhouetteError(BuildError):
    """The outline is several separate islands, not one connected part.

    The trace stage preserves islands deliberately, so this is a normal thing
    for artwork to do - and a fatal thing for a keychain, which would come
    off the plate in pieces. Refused here rather than exported as several
    loose bodies sharing one object name.
    """

    def __init__(self, part_count: int) -> None:
        self.part_count = part_count
        super().__init__(
            f"This artwork traces to {part_count} disconnected islands, not "
            f"one outline. A printed keychain would fall into "
            f"{part_count} pieces, so the build was refused. Use artwork "
            f"with a single connected outline (or join the islands in the "
            f"source image). A ring or other interior hole inside one "
            f"connected outline is fine; separate blobs are not."
        )


class HoleMarginError(BuildError):
    """The hanging hole cannot be placed with the required margin.

    Raised rather than shipping a hole with a thin lip, which is the one
    failure mode that reliably destroys a keychain in a pocket. Carries the
    numbers so the CLI and the QC report can phrase their own advice.
    """

    def __init__(
        self,
        *,
        required_mm: float,
        diameter_mm: float,
        margin_mm: float,
        requested: tuple[float, float] | None,
        available_mm: float | None,
        size_mm: float,
    ) -> None:
        self.required_mm = required_mm
        self.diameter_mm = diameter_mm
        self.margin_mm = margin_mm
        self.requested = requested
        self.available_mm = available_mm
        self.size_mm = size_mm

        need = (
            f"A {diameter_mm} mm hole with {margin_mm} mm of material all "
            f"round needs a solid disc {required_mm * 2:.1f} mm across"
        )
        if requested is None:
            where = (
                f"{need}, and nowhere in this outline has one at "
                f"{size_mm:.0f} mm across."
            )
            advice = (
                " Print it larger (--size), use artwork with a thicker area "
                "near the top, or allow a hanging tab so the hole can sit in "
                "a small added lug."
            )
        else:
            x, y = requested
            got = (
                "the outline ends before that"
                if available_mm is None
                else f"there is only {available_mm:.2f} mm to the nearest edge"
            )
            where = (
                f"{need}, but at the requested hole position "
                f"(x {x:.1f} mm, y {y:.1f} mm) {got}."
            )
            advice = (
                " Move the hole further inside the outline, use a smaller "
                "hole or margin, or leave the position unset and let the "
                "builder place it."
            )
        super().__init__(where + advice)


# --- result ----------------------------------------------------------------


@dataclass
class KeychainSolids:
    """The built bodies for one keychain, in mm, bottom face at Z = 0.

    ``detail`` is a *separate* body rather than part of the base: that
    separation is what lets Bambu Studio map the two to different AMS Lite
    slots. It is None for a recessed or single-colour design, where there is
    only ever one object.
    """

    base: Shape
    detail: Shape | None
    spec: KeychainSpec
    hole_center_mm: tuple[float, float]
    warnings: list[str] = field(default_factory=list)
    tolerance_mm: float = DEFAULT_TESSELLATION_MM

    @property
    def detail_mode(self) -> str:
        """How the second colour ended up expressed: see KeychainSpec."""
        if self.detail is not None:
            return "raised"
        return "recessed" if self.spec.detail_recessed else "none"

    def named_objects(self) -> list[tuple[str, Shape]]:
        """(name, shape) pairs, in the order they go into the 3MF."""
        objects: list[tuple[str, Shape]] = [("base", self.base)]
        if self.detail is not None:
            objects.append(("detail", self.detail))
        return objects

    def shapes(self) -> list[Shape]:
        return [shape for _, shape in self.named_objects()]

    def __iter__(self) -> Iterator[Shape]:
        return iter(self.shapes())


# --- public entry points ---------------------------------------------------


def build_keychain(
    silhouette: BaseGeometry,
    detail: BaseGeometry | None = None,
    spec: KeychainSpec | None = None,
    *,
    tolerance_mm: float = DEFAULT_TESSELLATION_MM,
) -> KeychainSolids:
    """Build a keychain solid (base + optional detail layer) from traced geometry.

    ``silhouette`` may also be a :class:`~print3d.trace.TraceResult`, in
    which case ``detail`` is taken from it. Returns the build123d solid(s)
    ready for export.

    Raises :class:`EmptyGeometryError` when there is nothing to extrude,
    :class:`DisconnectedSilhouetteError` when the outline is several loose
    islands, :class:`HoleMarginError` when the hanging hole cannot keep
    ``spec.hole_margin_mm`` of material around it, and plain
    :class:`BuildError` for a spec whose numbers cannot describe a solid.
    """
    silhouette, detail = _unpack_inputs(silhouette, detail)
    if spec is None:
        raise TypeError("build_keychain() needs a KeychainSpec")
    _check_spec(spec)

    silhouette = _as_multipolygon(make_valid(silhouette))
    if silhouette.is_empty:
        raise EmptyGeometryError(
            "There is no silhouette to build: the traced outline is empty. "
            "Trace an image first, or check that the trace stage did not "
            "drop everything as speckle."
        )
    detail = _as_multipolygon(make_valid(detail)) if detail is not None else _empty()
    if spec.detail_mode == "none":
        detail = _empty()
    else:
        # The trace stage promises detail is a subset of the silhouette, but
        # this stage is also called with hand-built geometry (tests, future
        # CLI overrides), and a detail island hanging off the edge would
        # extrude into thin air.
        detail = _as_multipolygon(make_valid(detail.intersection(silhouette)))

    warnings: list[str] = []
    tx0, ty0, tx1, ty1 = silhouette.bounds
    traced_largest = max(tx1 - tx0, ty1 - ty0)
    base_2d, hole_center = _place_hole(silhouette, spec, warnings)
    _check_connected(base_2d)

    base = _extrude(base_2d, 0.0, spec.base_thickness_mm)

    # The chamfer goes on now, while the base is still a bare prism: the
    # only edges at the top are the outline's own, so the hanging hole cut
    # below cannot be caught up in it. See _chamfer_top.
    chamfered = False
    if spec.top_edge_chamfer and spec.chamfer_mm > 0:
        base, chamfered = _chamfer_top(base, spec.chamfer_mm, warnings)

    detail_solid: Shape | None = None
    if not detail.is_empty:
        if spec.detail_mode == "recessed":
            # No top-face clipping here: a recess reaching the chamfered
            # edge takes material away rather than hanging over air, so it
            # is a cosmetic nick in the bevel, not an unsupported lip.
            pocket = _extrude(
                detail,
                spec.base_thickness_mm - spec.detail_height_mm,
                spec.detail_height_mm,
            )
            base = base.cut(pocket)
        else:
            if chamfered:
                detail = _clip_to_top_face(
                    detail, base_2d, spec.chamfer_mm, warnings
                )
            if detail.is_empty:
                warnings.append(
                    "Every raised detail region sat on the chamfered top "
                    "edge, so there is no second-colour object left - "
                    "printing single-colour."
                )
            else:
                detail_solid = _extrude(
                    detail, spec.base_thickness_mm, spec.detail_height_mm
                )

    top_z = spec.base_thickness_mm + (
        spec.detail_height_mm if detail_solid is not None else 0.0
    )
    hole = _hole_cylinder(hole_center, spec.hole_diameter_mm / 2.0, top_z)
    base = base.cut(hole)
    if detail_solid is not None:
        cut_detail = detail_solid.cut(hole)
        # A detail layer that lived entirely where the hole now is leaves
        # nothing behind; emitting an empty second object would break the
        # AMS mapping, so drop it and say so.
        detail_solid = cut_detail if _has_volume(cut_detail) else None
        if detail_solid is None:
            warnings.append(
                "The detail layer was entirely inside the hanging hole, so "
                "there is no second-colour object left - printing "
                "single-colour."
            )

    _check_built_size(base, spec, traced_largest, warnings)

    return KeychainSolids(
        base=base,
        detail=detail_solid,
        spec=spec,
        hole_center_mm=hole_center,
        warnings=warnings,
        tolerance_mm=tolerance_mm,
    )


def export_3mf(solids: KeychainSolids, out_path: Path | str) -> Path:
    """Export base/detail solids as a multi-object 3MF (AMS color mapping).

    One file, one object per body, each named ("base", "detail") so Bambu
    Studio shows two pickable objects to assign filaments to. A recessed or
    single-colour design writes one object.
    """
    named = _named_objects(solids)
    tolerance = solids.tolerance_mm
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    scene = trimesh.Scene()
    for name, shape in named:
        scene.add_geometry(
            _to_mesh(shape, tolerance), geom_name=name, node_name=name
        )
    scene.export(str(out), file_type="3mf")
    return out


def export_stl(solids: KeychainSolids, out_path: Path | str) -> Path:
    """Export a combined STL alongside the 3MF.

    STL has no notion of separate objects, so the bodies are fused into one
    watertight solid rather than shipped as two overlapping shells: a fused
    body is what every downstream tool expects from an STL, and it is what
    keeps ``is_volume`` meaningful.
    """
    named = _named_objects(solids)
    tolerance = solids.tolerance_mm
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    shapes = [shape for _, shape in named]
    combined = shapes[0]
    for shape in shapes[1:]:
        try:
            # clean() drops the now-internal face where the two bodies met,
            # which is the difference between one solid and two shells in a
            # trenchcoat.
            combined = combined.fuse(shape).clean()
        except Exception:  # pragma: no cover - OCCT fuse failure is rare
            combined = Compound(children=list(_solids_of(combined)) + list(
                _solids_of(shape)
            ))
    _to_mesh(combined, tolerance).export(str(out), file_type="stl")
    return out


# --- hole placement --------------------------------------------------------


def _place_hole(
    silhouette: MultiPolygon, spec: KeychainSpec, warnings: list[str]
) -> tuple[MultiPolygon, tuple[float, float]]:
    """Return (base outline, hole centre) with the margin contract honoured.

    The base outline is normally the silhouette unchanged; it grows a tab
    only when nothing inside the silhouette can hold the hole.
    """
    required = spec.hole_diameter_mm / 2.0 + spec.hole_margin_mm
    x0, _, x1, y1 = silhouette.bounds
    target_x = (x0 + x1) / 2.0
    size_mm = max(x1 - x0, y1 - silhouette.bounds[1])

    if spec.hole_position_mm is not None:
        center = (
            float(spec.hole_position_mm[0]),
            float(spec.hole_position_mm[1]),
        )
        if _margin_mm(silhouette, center) is None or not _margin_ok(
            silhouette, center, required
        ):
            raise HoleMarginError(
                required_mm=required,
                diameter_mm=spec.hole_diameter_mm,
                margin_mm=spec.hole_margin_mm,
                requested=center,
                available_mm=_margin_mm(silhouette, center),
                size_mm=size_mm,
            )
        return silhouette, center

    found = _auto_hole_center(silhouette, required, target_x)
    if found is not None:
        return silhouette, found

    if not spec.hole_tab_allowed:
        raise HoleMarginError(
            required_mm=required,
            diameter_mm=spec.hole_diameter_mm,
            margin_mm=spec.hole_margin_mm,
            requested=None,
            available_mm=None,
            size_mm=size_mm,
        )

    base_2d, center = _with_hole_tab(silhouette, required, target_x)
    new_x0, new_y0, new_x1, new_y1 = base_2d.bounds
    warnings.append(
        f"No part of the outline is thick enough to hold a "
        f"{spec.hole_diameter_mm} mm hole with {spec.hole_margin_mm} mm of "
        f"margin, so a round hanging tab was added at the top. The part is "
        f"now {new_x1 - new_x0:.1f} x {new_y1 - new_y0:.1f} mm rather than "
        f"{x1 - x0:.1f} x {y1 - silhouette.bounds[1]:.1f} mm."
    )
    return base_2d, center


def _auto_hole_center(
    silhouette: MultiPolygon, required: float, target_x: float
) -> tuple[float, float] | None:
    """Highest point inside the silhouette that holds a full margin ring.

    Erode the silhouette by the required radius: what survives is exactly
    the set of legal hole centres. Take the top of that set (a keychain
    hangs from its top edge) and, along it, the X nearest the middle.
    Returns None when nothing survives.
    """
    inner = _as_multipolygon(make_valid(silhouette.buffer(-required, join_style="round")))
    if inner.is_empty:
        return None

    ix0, iy0, ix1, iy1 = inner.bounds
    probe_y = iy1 - min(_PROBE_DROP_MM, max((iy1 - iy0) / 1000.0, 0.0))
    span = LineString([(ix0 - 1.0, probe_y), (ix1 + 1.0, probe_y)])
    best: tuple[float, tuple[float, float]] | None = None
    for x in _candidate_x(inner.intersection(span), target_x):
        center = (x, probe_y)
        if _margin_ok(silhouette, center, required):
            distance = abs(x - target_x)
            if best is None or distance < best[0]:
                best = (distance, center)
    if best is not None:
        return best[1]

    # Degenerate erosion (a sliver so thin the probe line missed it). Any
    # legal centre beats none.
    fallback = inner.representative_point()
    candidate = (float(fallback.x), float(fallback.y))
    return candidate if _margin_ok(silhouette, candidate, required) else None


def _candidate_x(section: BaseGeometry, target_x: float) -> list[float]:
    """X positions worth testing along a horizontal section of the region.

    Both the section's endpoints and, for each sub-segment, the point on it
    nearest ``target_x`` - so a wide flat top yields the centre rather than
    a corner.
    """
    if section.is_empty:
        return []
    out: list[float] = []
    for part in getattr(section, "geoms", [section]):
        coords = list(getattr(part, "coords", []))
        for (ax, _), (bx, _) in zip(coords, coords[1:]):
            lo, hi = min(ax, bx), max(ax, bx)
            out.append(min(max(target_x, lo), hi))
        out.extend(float(c[0]) for c in coords)
    return out


def _with_hole_tab(
    silhouette: MultiPolygon, required: float, target_x: float
) -> tuple[MultiPolygon, tuple[float, float]]:
    """Grow a round hanging tab off the top so the hole has somewhere to go.

    The tab is a disc of exactly the required radius (plus a hair for
    shapely's chord approximation), centred beyond the silhouette's topmost
    material by a fraction of that radius - so it always overlaps the part,
    the union is one body, and the hole at its centre has a full ring.
    """
    x0, y0, x1, y1 = silhouette.bounds
    probe_y = y1 - min(_PROBE_DROP_MM, max((y1 - y0) / 1000.0, 0.0))
    span = LineString([(x0 - 1.0, probe_y), (x1 + 1.0, probe_y)])
    candidates = _candidate_x(silhouette.intersection(span), target_x)
    anchor_x = (
        min(candidates, key=lambda x: abs(x - target_x))
        if candidates
        else target_x
    )

    radius = required + _TAB_OVERSIZE_MM
    center = (anchor_x, probe_y + radius * _TAB_OVERLAP_FRACTION)
    disc = Point(center).buffer(radius, quad_segs=_TAB_QUAD_SEGS)
    grown = _as_multipolygon(make_valid(unary_union([silhouette, disc])))
    return grown, center


def _margin_mm(geometry: BaseGeometry, center: tuple[float, float]) -> float | None:
    """Distance from ``center`` to the nearest edge, or None if outside.

    Measured against the boundary rings directly rather than by testing a
    buffered disc for containment: shapely draws circles as inscribed
    polygons, which would quietly report a hole as fitting when it does not.
    """
    point = Point(center)
    if not geometry.contains(point):
        return None
    return float(geometry.boundary.distance(point))


def _margin_ok(
    geometry: BaseGeometry, center: tuple[float, float], required: float
) -> bool:
    available = _margin_mm(geometry, center)
    return available is not None and available >= required - _MARGIN_TOL_MM


# --- shapely to build123d --------------------------------------------------


def _extrude(geometry: MultiPolygon, z0: float, height: float) -> Shape:
    """Extrude a 2D region upward from ``z0``, holes and all."""
    solids = [
        Solid.extrude(face, Vector(0, 0, height))
        for face in _faces(geometry)
    ]
    if not solids:
        raise EmptyGeometryError(
            "Nothing to extrude: the region has no polygon with area."
        )
    shape = solids[0] if len(solids) == 1 else Compound(children=solids)
    return shape.moved(Location((0, 0, z0))) if z0 else shape


def _faces(geometry: MultiPolygon) -> list[Face]:
    """One build123d face per shapely polygon, interiors kept as holes.

    Interiors are the whole point of doing this by hand: a donut logo whose
    hole is silently filled is a different design, so the exterior and every
    interior ring becomes a wire and the face is built from all of them.
    """
    faces: list[Face] = []
    for polygon in _parts(geometry):
        outer = _wire(polygon.exterior.coords)
        if outer is None:
            continue
        inners = [w for w in (_wire(r.coords) for r in polygon.interiors) if w]
        faces.append(Face(outer, inners))
    return faces


def _wire(coords: Iterable[tuple[float, float]]) -> Wire | None:
    """Closed wire from a shapely ring, or None if it has no real extent.

    Shapely repeats the first point to close a ring and simplification can
    leave near-duplicates behind; OCCT will not build an edge between two
    coincident vertices, so both are dropped here.
    """
    points: list[tuple[float, float]] = []
    for x, y in coords:
        x, y = float(x), float(y)
        if math.isnan(x) or math.isnan(y):
            continue
        if points and math.dist(points[-1], (x, y)) <= _POINT_MERGE_MM:
            continue
        points.append((x, y))
    while len(points) > 1 and math.dist(points[0], points[-1]) <= _POINT_MERGE_MM:
        points.pop()
    if len(points) < 3:
        return None
    return Wire.make_polygon([Vector(x, y, 0.0) for x, y in points], close=True)


def _hole_cylinder(
    center: tuple[float, float], radius: float, top_z: float
) -> Solid:
    """A cylinder that pierces the whole model, overshooting both ends."""
    height = top_z + 2.0 * _CUT_OVERSHOOT_MM
    return Solid.make_cylinder(radius, height).moved(
        Location((center[0], center[1], -_CUT_OVERSHOOT_MM))
    )


def _chamfer_top(
    shape: Shape, width: float, warnings: list[str]
) -> tuple[Shape, bool]:
    """Chamfer the top edges of the bare base prism; report whether it took.

    **Called before the hanging hole is cut, on purpose.** "Every edge at the
    top Z" is the only selection OCCT offers here, and it cannot tell the
    hole's rim from the outline: chamfering afterwards opened a 5.2 mm hole
    to 6.0 mm at the top face and cut the enforced 2 mm ring to 1.2 mm,
    silently. On a bare prism the only top edges are the outline's own -
    exterior and any interior rings - so cutting the hole afterwards leaves
    it a straight 5.2 mm cylinder at every height.

    The outer chamfer does still narrow the top *face* by ``width`` all
    round, so a hole near the rim measures ``width`` less material at the
    very top surface. That is inherent to any chamfer and is accepted: the
    ring that carries the keyring load is the full-thickness one below the
    chamfer band, and it keeps the whole margin. ``_check_spec`` keeps
    ``chamfer_mm`` well under the margin so the thinning stays cosmetic.

    Traced outlines are long chains of short segments meeting at shallow
    angles, and OCCT's chamfer algorithm does fail on some of them. That
    must never sink a build: the part prints perfectly well with a sharp top
    lip, so a failure is reported and the unchamfered shape handed back. The
    returned flag says whether the chamfer actually happened, because the
    raised detail is clipped back to the chamfered face only when it did.
    """
    try:
        top = shape.bounding_box().max.Z
        edges = shape.edges().filter_by_position(
            Axis.Z, top - _TOP_FACE_TOL_MM, top + _TOP_FACE_TOL_MM
        )
        if not edges:
            return shape, False
        chamfered = shape.chamfer(width, None, edges)
    except Exception as exc:
        warnings.append(
            f"Could not chamfer the top edge of this outline ({exc}); the "
            f"model was built with a sharp top edge instead. This is "
            f"cosmetic - the part still prints."
        )
        return shape, False
    if not _has_volume(chamfered):  # pragma: no cover - defensive
        warnings.append(
            "Chamfering the top edge produced an empty solid, so the model "
            "was built with a sharp top edge instead."
        )
        return shape, False
    return chamfered, True


def _clip_to_top_face(
    detail: MultiPolygon,
    base_2d: MultiPolygon,
    chamfer_mm: float,
    warnings: list[str],
) -> MultiPolygon:
    """Trim raised detail back to the base's chamfered top face.

    The chamfer eats a band ``chamfer_mm`` wide out of the top face all
    round, so detail drawn flush to the outline would sit over air - a lip
    of second-colour material with nothing under it, which the slicer either
    droops or supports. Eroding the outline by the chamfer width gives
    exactly the surviving top face; the round join is a hair more generous
    to the chamfer than OCCT's mitred corners, which is the safe direction.
    """
    before = detail.area
    top_face = make_valid(base_2d.buffer(-chamfer_mm))
    clipped = _as_multipolygon(make_valid(detail.intersection(top_face)))
    trimmed = before - clipped.area
    if trimmed > _TRIM_REPORT_MM2:
        warnings.append(
            f"{trimmed:.2f} mm2 of raised detail reached the chamfered top "
            f"edge and was trimmed back by {chamfer_mm} mm so no "
            f"second-colour material overhangs air."
        )
    return clipped


# --- meshing and export ----------------------------------------------------


def _to_mesh(shape: Shape, tolerance: float) -> trimesh.Trimesh:
    """Tessellate a build123d shape into a watertight trimesh mesh.

    OCCT tessellates each face independently, so the raw output has one
    vertex per face corner and no shared edges at all - which reads as a
    shell full of holes. Welding coincident vertices is therefore not a
    tidy-up, it is what makes the mesh a solid; normals are fixed
    afterwards so ``is_volume`` (the QC stage's print-ready predicate)
    holds.
    """
    vertices, triangles = shape.tessellate(tolerance, DEFAULT_ANGULAR_TOLERANCE)
    mesh = trimesh.Trimesh(
        vertices=np.array([(v.X, v.Y, v.Z) for v in vertices], dtype=np.float64),
        faces=np.array(triangles, dtype=np.int64),
        process=True,
    )
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if not (mesh.is_winding_consistent and mesh.is_volume):
        mesh.fix_normals()
    if mesh.is_volume and mesh.volume < 0:  # pragma: no cover - defensive
        mesh.invert()
    return mesh


# --- input plumbing --------------------------------------------------------


def _named_objects(solids: KeychainSolids) -> list[tuple[str, Shape]]:
    """(name, shape) pairs to export, or a clear error about the argument.

    The exporters take a :class:`KeychainSolids` and nothing else: the naming
    is half the point of the 3MF (Bambu Studio maps filaments by object
    name), and there is no second producer of these bodies to accommodate.
    """
    if not isinstance(solids, KeychainSolids):
        raise TypeError(
            f"export expects a KeychainSolids from build_keychain(), got "
            f"{type(solids).__name__}."
        )
    return solids.named_objects()


def _unpack_inputs(
    silhouette, detail
) -> tuple[BaseGeometry, BaseGeometry | None]:
    """Accept either two geometries or a whole TraceResult."""
    if hasattr(silhouette, "silhouette") and hasattr(silhouette, "detail"):
        trace = silhouette
        return trace.silhouette, trace.detail if detail is None else detail
    return silhouette, detail


def _check_spec(spec: KeychainSpec) -> None:
    """Refuse specs whose numbers cannot describe a solid.

    Every message names the field it is about, because these arrive from
    CLI overrides and the caller's next move is to change one number. The
    alternative to checking is not "it works anyway": a zero detail height
    reaches OCCT as a bare Standard_ConstructionError, and a negative one
    quietly buries the second-colour body inside the base.
    """
    if spec.base_thickness_mm <= 0:
        raise BuildError(
            f"base_thickness_mm must be positive, got {spec.base_thickness_mm}."
        )
    if spec.hole_diameter_mm <= 0:
        raise BuildError(
            f"hole_diameter_mm must be positive, got {spec.hole_diameter_mm}."
        )
    if spec.hole_margin_mm < 0:
        raise BuildError(
            f"hole_margin_mm cannot be negative, got {spec.hole_margin_mm}. "
            f"Use 0 to allow a hole right up to the edge."
        )
    if spec.detail_mode != "none" and spec.detail_height_mm <= 0:
        raise BuildError(
            f"detail_height_mm must be positive when the detail layer is "
            f"enabled, got {spec.detail_height_mm}. Set detail_enabled="
            f"False for a single-colour part."
        )
    if spec.detail_mode == "recessed" and (
        spec.detail_height_mm >= spec.base_thickness_mm
    ):
        raise BuildError(
            f"A recessed detail {spec.detail_height_mm} mm deep would cut "
            f"straight through a {spec.base_thickness_mm} mm base. Make the "
            f"base thicker or the recess shallower."
        )
    if spec.chamfer_mm < 0:
        raise BuildError(
            f"chamfer_mm cannot be negative, got {spec.chamfer_mm}."
        )
    if spec.top_edge_chamfer and spec.chamfer_mm > 0:
        # A chamfer is a bevel on an edge, not a redesign: at these sizes it
        # would eat the whole hole margin or the whole base.
        if spec.chamfer_mm >= spec.hole_margin_mm:
            raise BuildError(
                f"chamfer_mm ({spec.chamfer_mm}) must be smaller than "
                f"hole_margin_mm ({spec.hole_margin_mm}): a chamfer that "
                f"wide would take the whole ring of material the hole "
                f"margin exists to guarantee. Use a smaller chamfer, a "
                f"bigger margin, or top_edge_chamfer=False."
            )
        if spec.chamfer_mm >= spec.base_thickness_mm:
            raise BuildError(
                f"chamfer_mm ({spec.chamfer_mm}) must be smaller than "
                f"base_thickness_mm ({spec.base_thickness_mm}): a chamfer "
                f"that deep would consume the whole base."
            )


def _check_connected(base_2d: MultiPolygon) -> None:
    """Refuse an outline that is several loose islands.

    Run after the tab step, because a tab can be exactly the bridge that
    joins two islands into one part. Overlapping or touching polygons are
    unioned first, so this counts connected material rather than however
    many rings the trace happened to emit; an interior ring inside one part
    is one part.
    """
    parts = _parts(unary_union(base_2d))
    if len(parts) > 1:
        raise DisconnectedSilhouetteError(len(parts))


def _check_built_size(
    base: Shape,
    spec: KeychainSpec,
    traced_largest: float,
    warnings: list[str],
) -> None:
    """Warn - never rescale - when the part came out over the asked-for size.

    Sizing belongs to the trace stage, and rescaling here would invalidate
    the wall-thickness gate it already applied, so this is strictly a
    report. ``traced_largest`` is the incoming silhouette's largest side, so
    a hanging tab is named as the cause only when the tab is actually what
    pushed the part over.
    """
    bounds = base.bounding_box()
    width = bounds.max.X - bounds.min.X
    depth = bounds.max.Y - bounds.min.Y
    largest = max(width, depth)
    if largest <= spec.max_dimension_mm + _SIZE_TOL_MM:
        return
    because = (
        " because a hanging tab was added to hold the hole"
        if traced_largest <= spec.max_dimension_mm + _SIZE_TOL_MM
        else ""
    )
    warnings.append(
        f"The finished part measures {width:.1f} x {depth:.1f} mm, which "
        f"exceeds the requested maximum dimension of "
        f"{spec.max_dimension_mm:.1f} mm{because}. Nothing was rescaled - "
        f"sizing belongs to the trace stage - so check it still fits the "
        f"plate and your keyring."
    )


# --- small geometry helpers ------------------------------------------------


def _has_volume(shape: Shape | None) -> bool:
    if shape is None:
        return False
    try:
        return bool(shape.volume > 0)
    except Exception:  # pragma: no cover - defensive
        return False


def _solids_of(shape: Shape) -> list[Solid]:
    found = shape.solids()
    return list(found) if found else [shape]


def _parts(geometry: BaseGeometry) -> list[Polygon]:
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


def _as_multipolygon(geometry: BaseGeometry) -> MultiPolygon:
    if geometry.is_empty:
        return _empty()
    polygons = [part for part in _parts(geometry) if part.area > 0]
    return MultiPolygon(polygons) if polygons else _empty()


def _empty() -> MultiPolygon:
    return MultiPolygon()
