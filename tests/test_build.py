"""Tests for the build stage.

Geometry is drawn here with shapely rather than traced from a bitmap: the
builder's contract starts at "two polygons in mm", so tracing first would
only make every test slower and couple it to potrace's output. The shapes
are deliberately plain (a rounded square, a donut, a bar too thin to hold a
hole) so a failing assertion points at the builder, not at the fixture.

The fixture is 40 mm, not the 50 mm default of ``max_dimension_mm``, so the
size assertions compare against a number that was actually asked for rather
than agreeing with a default twice over.

Mesh assertions go through a **planar section** rather than ray-casting
`contains`: sectioning a watertight mesh at a given Z and testing the
resulting shapely polygons is exact, which matters when the thing under
test is a 2 mm margin.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.validation import make_valid

from print3d.build import (
    BuildError,
    DisconnectedSilhouetteError,
    HoleMarginError,
    build_keychain,
    export_3mf,
    export_stl,
)
from print3d.spec import KeychainSpec

#: Fixture size, deliberately not the 50 mm ``max_dimension_mm`` default.
SIZE_MM = 40.0
MID_MM = SIZE_MM / 2.0

#: Where the island detail sits: low in the silhouette, clear of the hole.
DETAIL_CENTER = (MID_MM, 14.0)
DETAIL_RADIUS = 7.0


# --- fixtures --------------------------------------------------------------


def _spec(**overrides) -> KeychainSpec:
    values = dict(slug="sample", source_image="sample.png")
    values.update(overrides)
    return KeychainSpec(**values)


def _rounded_square(size: float = SIZE_MM, radius: float = 4.0) -> MultiPolygon:
    """A ``size`` x ``size`` square with rounded corners, origin at (0, 0)."""
    core = box(radius, radius, size - radius, size - radius)
    return MultiPolygon([core.buffer(radius, quad_segs=16)])


def _island_detail() -> MultiPolygon:
    """A disc low in the silhouette, clear of the automatic hole position."""
    return MultiPolygon(
        [Point(*DETAIL_CENTER).buffer(DETAIL_RADIUS, quad_segs=32)]
    )


def _donut(size: float = SIZE_MM) -> MultiPolygon:
    """A square with a big round hole - the "does an interior ring survive" case."""
    shape = box(0, 0, size, size).difference(
        Point(size / 2, size / 2).buffer(size / 5, quad_segs=32)
    )
    return MultiPolygon([shape])


def _thin_bar() -> MultiPolygon:
    """40 x 6 mm: printable, but nowhere near thick enough for hole + margin."""
    return MultiPolygon([box(0.0, 0.0, SIZE_MM, 6.0)])


# --- helpers ---------------------------------------------------------------


def _scene_meshes(path) -> dict[str, trimesh.Trimesh]:
    """Load a 3MF and return {object name: mesh}, node transforms applied."""
    scene = trimesh.load(str(path), force="scene")
    meshes: dict[str, trimesh.Trimesh] = {}
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        mesh = scene.geometry[geom_name].copy()
        mesh.apply_transform(transform)
        meshes[geom_name] = mesh
    return meshes


def _section(mesh: trimesh.Trimesh, z: float):
    """The material present at height ``z``, as one shapely geometry.

    The closed loops the section returns are assembled here with an even-odd
    rule rather than via trimesh's ``polygons_full``, which needs ``rtree``
    - not a dependency of this project, and not one worth adding for a test
    helper. Symmetric difference *is* even-odd fill, so a loop inside a loop
    becomes a hole and a loop inside that becomes an island again.
    """
    cut = mesh.section(plane_origin=[0.0, 0.0, z], plane_normal=[0.0, 0.0, 1.0])
    assert cut is not None, f"mesh has no material at z={z}"
    region = None
    for loop in cut.discrete:
        if len(loop) < 4:
            continue
        ring = make_valid(Polygon(np.asarray(loop)[:, :2]))
        if ring.is_empty or ring.area <= 0:
            continue
        region = ring if region is None else region.symmetric_difference(ring)
    assert region is not None and not region.is_empty, f"empty section at z={z}"
    return region


def _hole_diameter(region, center) -> float:
    """Diameter of the hole at ``center`` in a sectioned region, in mm."""
    return 2.0 * float(region.boundary.distance(Point(center)))


def _ring_points(center, radius: float, count: int = 32):
    cx, cy = center
    return [
        Point(cx + radius * math.cos(a), cy + radius * math.sin(a))
        for a in np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    ]


def _build(**spec_overrides):
    return build_keychain(
        _rounded_square(), _island_detail(), _spec(**spec_overrides)
    )


@pytest.fixture(scope="module")
def raised(tmp_path_factory):
    """The default design - raised detail - built and exported once."""
    out = tmp_path_factory.mktemp("raised")
    solids = build_keychain(_rounded_square(), _island_detail(), _spec())
    mf3 = export_3mf(solids, out / "sample.3mf")
    stl = export_stl(solids, out / "sample.stl")
    return solids, mf3, stl


# --- the default design ----------------------------------------------------


def test_raised_detail_writes_both_files(raised):
    _, mf3, stl = raised
    assert mf3.exists() and mf3.stat().st_size > 0
    assert stl.exists() and stl.stat().st_size > 0
    assert isinstance(trimesh.load(str(stl)), trimesh.Trimesh)
    assert trimesh.load(str(mf3), force="scene") is not None


def test_3mf_holds_base_and_detail_as_separate_named_objects(raised):
    _, mf3, _ = raised
    meshes = _scene_meshes(mf3)
    assert sorted(meshes) == ["base", "detail"]


def test_detail_sits_on_top_of_the_base(raised):
    solids, mf3, _ = raised
    meshes = _scene_meshes(mf3)
    base_top = float(meshes["base"].bounds[1][2])
    detail_low = float(meshes["detail"].bounds[0][2])
    detail_top = float(meshes["detail"].bounds[1][2])

    assert base_top == pytest.approx(solids.spec.base_thickness_mm, abs=1e-3)
    assert detail_low == pytest.approx(base_top, abs=1e-3)
    assert detail_top == pytest.approx(
        base_top + solids.spec.detail_height_mm, abs=1e-3
    )


def test_bottom_face_sits_on_the_bed(raised):
    _, mf3, _ = raised
    assert float(_scene_meshes(mf3)["base"].bounds[0][2]) == pytest.approx(
        0.0, abs=1e-6
    )


def test_xy_bounds_match_the_requested_size(raised):
    _, mf3, _ = raised
    low, high = _scene_meshes(mf3)["base"].bounds
    width, depth = float(high[0] - low[0]), float(high[1] - low[1])
    assert max(width, depth) == pytest.approx(SIZE_MM, abs=0.1)


def test_a_part_inside_the_size_limit_says_nothing_about_size(raised):
    solids, _, _ = raised
    assert not any("maximum dimension" in w for w in solids.warnings), (
        solids.warnings
    )


def test_every_exported_object_is_a_printable_volume(raised):
    _, mf3, stl = raised
    for name, mesh in _scene_meshes(mf3).items():
        assert mesh.is_volume, f"{name} is not a closed, consistent volume"
        assert mesh.volume > 0
    combined = trimesh.load(str(stl))
    assert combined.is_volume
    assert combined.volume > 0


def test_the_stl_is_one_fused_body_not_two_shells(raised):
    """The STL must be a single solid, not base and detail shipped as-is.

    ``body_count`` is the assertion that bites: two shells stacked face to
    face are each watertight, so ``is_volume`` alone is happy to sign off a
    compound. Volume is checked too, but only as a "nothing was double
    counted" guard - base and detail merely touch, so a correct fuse weighs
    exactly what the two bodies weigh, never less.
    """
    _, mf3, stl = raised
    meshes = _scene_meshes(mf3)
    parts = meshes["base"].volume + meshes["detail"].volume

    combined = trimesh.load(str(stl))
    assert combined.body_count == 1, "the STL is a bag of loose shells"
    assert combined.is_volume
    assert combined.volume == pytest.approx(parts, rel=1e-5)


# --- the top-edge chamfer --------------------------------------------------


def test_the_top_edge_is_chamfered(raised):
    """The top face is inset from the body by the chamfer width, all round.

    Pinned by measurement rather than by "a chamfer method was called": a
    chamfer that quietly does nothing is the failure this catches.
    """
    solids, mf3, _ = raised
    spec = solids.spec
    mesh = _scene_meshes(mf3)["base"]

    body = _section(mesh, spec.base_thickness_mm / 2.0)
    top = _section(mesh, spec.base_thickness_mm - 1e-3)

    body_w = body.bounds[2] - body.bounds[0]
    top_w = top.bounds[2] - top.bounds[0]
    assert body_w - top_w == pytest.approx(2 * spec.chamfer_mm, abs=0.02)
    assert top.area < body.area, "the top face was not inset at all"


def test_the_hanging_hole_is_not_chamfered(raised):
    """The hole stays 5.2 mm at every height, including the very top face.

    The chamfer used to be cut after the hole, which swept up the hole's rim
    with the outline: the hole opened to ~6 mm at the top and the enforced
    2 mm ring dropped to 1.2 mm without a word.
    """
    solids, mf3, _ = raised
    spec = solids.spec
    mesh = _scene_meshes(mf3)["base"]

    for z in (0.05, spec.base_thickness_mm / 2.0, spec.base_thickness_mm - 1e-3):
        measured = _hole_diameter(_section(mesh, z), solids.hole_center_mm)
        assert measured == pytest.approx(spec.hole_diameter_mm, abs=0.05), (
            f"hole measures {measured:.3f} mm at z={z}"
        )


def test_a_chamfer_that_would_eat_the_margin_is_refused():
    with pytest.raises(BuildError, match="chamfer_mm"):
        _build(chamfer_mm=2.0)  # == hole_margin_mm


def test_a_legal_large_chamfer_still_leaves_the_hole_alone(tmp_path):
    """The 1.0 mm override that used to leave a 0.0024 mm knife edge."""
    solids = _build(chamfer_mm=1.0)
    mesh = _scene_meshes(export_3mf(solids, tmp_path / "chamfered.3mf"))["base"]
    spec = solids.spec
    for z in (spec.base_thickness_mm / 2.0, spec.base_thickness_mm - 1e-3):
        assert _hole_diameter(
            _section(mesh, z), solids.hole_center_mm
        ) == pytest.approx(spec.hole_diameter_mm, abs=0.05)


# --- detail modes ----------------------------------------------------------


def test_detail_none_exports_a_single_object(tmp_path):
    solids = _build(detail_enabled=False)
    assert solids.detail is None
    assert solids.detail_mode == "none"
    meshes = _scene_meshes(export_3mf(solids, tmp_path / "plain.3mf"))
    assert list(meshes) == ["base"]
    assert meshes["base"].is_volume


def test_recessed_detail_is_one_object_that_lost_material(tmp_path):
    plain = _build(detail_enabled=False)
    recessed = _build(detail_recessed=True)
    assert recessed.detail is None
    assert recessed.detail_mode == "recessed"

    meshes = _scene_meshes(export_3mf(recessed, tmp_path / "recessed.3mf"))
    assert list(meshes) == ["base"]
    assert meshes["base"].is_volume

    plain_mesh = _scene_meshes(export_3mf(plain, tmp_path / "plain.3mf"))["base"]
    pocket_mm3 = _island_detail().area * recessed.spec.detail_height_mm
    assert meshes["base"].volume == pytest.approx(
        plain_mesh.volume - pocket_mm3, rel=0.02
    )


def test_recess_is_cut_into_the_top_face_only(tmp_path):
    recessed = _build(detail_recessed=True)
    mesh = _scene_meshes(export_3mf(recessed, tmp_path / "recessed.3mf"))["base"]
    depth = recessed.spec.detail_height_mm

    just_below_top = _section(mesh, recessed.spec.base_thickness_mm - depth / 2)
    assert not just_below_top.contains(Point(*DETAIL_CENTER))
    below_the_pocket = _section(mesh, recessed.spec.base_thickness_mm - depth * 2)
    assert below_the_pocket.contains(Point(*DETAIL_CENTER))


def test_a_recess_deeper_than_the_base_is_refused():
    with pytest.raises(BuildError, match="straight through"):
        _build(detail_recessed=True, detail_height_mm=3.0)


def test_raised_detail_never_overhangs_the_chamfered_edge(tmp_path):
    """Detail drawn flush to the outline is trimmed, not left over air.

    A second-colour lip hanging over the chamfer band has nothing under it:
    the slicer either droops it or grows support into the chamfer. The
    trimmed area is reported, because losing detail silently is its own bug.
    """
    solids = build_keychain(_rounded_square(), _rounded_square(), _spec())
    trims = [w for w in solids.warnings if "trimmed back" in w]
    assert len(trims) == 1, solids.warnings
    assert "mm2" in trims[0]

    meshes = _scene_meshes(export_3mf(solids, tmp_path / "flush.3mf"))
    thickness = solids.spec.base_thickness_mm
    base_top = _section(meshes["base"], thickness - 1e-3)
    detail = _section(meshes["detail"], thickness + solids.spec.detail_height_mm / 2)

    overhang = detail.difference(base_top.buffer(1e-3))
    assert overhang.area < 0.05, f"{overhang.area:.2f} mm2 of detail over air"


# --- the hanging hole ------------------------------------------------------


def test_the_hole_goes_all_the_way_through(raised):
    solids, mf3, _ = raised
    mesh = _scene_meshes(mf3)["base"]
    radius = solids.spec.hole_diameter_mm / 2.0
    for z in (0.05, 1.5, solids.spec.base_thickness_mm - 0.05):
        material = _section(mesh, z)
        assert not material.intersects(
            Point(solids.hole_center_mm).buffer(radius * 0.9)
        ), f"hole is blocked at z={z}"


@pytest.mark.parametrize("where", ["middle", "under the chamfer"])
def test_the_hole_keeps_its_ring_of_margin(raised, where):
    """The full ring of material, at mid-height and near the top surface.

    "Near the top" stops just under the chamfer band on purpose. The outer
    chamfer narrows the top *face* by ``chamfer_mm`` all round - that is
    inherent to a chamfer and accepted - so the ring that carries the
    keyring is the full-thickness one directly below it.
    """
    solids, mf3, _ = raised
    spec = solids.spec
    z = (
        spec.base_thickness_mm / 2
        if where == "middle"
        else spec.base_thickness_mm - spec.chamfer_mm - 0.05
    )
    material = _section(_scene_meshes(mf3)["base"], z)
    inner = spec.hole_diameter_mm / 2.0 + 0.05
    outer = spec.hole_diameter_mm / 2.0 + spec.hole_margin_mm - 0.05
    for radius in (inner, outer, (inner + outer) / 2):
        for point in _ring_points(solids.hole_center_mm, radius):
            assert material.contains(point), (
                f"no material at {radius:.2f} mm from the hole centre at z={z}"
            )


def test_the_hole_is_placed_near_the_top_centre(raised):
    solids, _, _ = raised
    x, y = solids.hole_center_mm
    required = solids.spec.hole_diameter_mm / 2.0 + solids.spec.hole_margin_mm
    assert x == pytest.approx(MID_MM, abs=0.5)
    assert y == pytest.approx(SIZE_MM - required, abs=0.2)


def test_an_explicit_position_without_margin_is_refused():
    with pytest.raises(HoleMarginError) as caught:
        _build(hole_position_mm=(2.0, MID_MM))
    assert "requested hole position" in str(caught.value)
    assert "2.0 mm" in str(caught.value)  # the margin it could not honour


def test_an_explicit_position_outside_the_outline_is_refused():
    with pytest.raises(HoleMarginError, match="outline ends before that"):
        _build(hole_position_mm=(-10.0, -10.0))


def test_an_explicit_position_with_room_is_honoured():
    solids = _build(hole_position_mm=(MID_MM, 30.0))
    assert solids.hole_center_mm == (MID_MM, 30.0)


def test_a_thin_outline_grows_a_tab_to_hold_the_hole(tmp_path):
    solids = build_keychain(_thin_bar(), None, _spec(detail_enabled=False))
    assert any("tab" in w for w in solids.warnings), solids.warnings

    mesh = _scene_meshes(export_3mf(solids, tmp_path / "bar.3mf"))["base"]
    assert mesh.is_volume
    material = _section(mesh, solids.spec.base_thickness_mm / 2)
    assert len(list(getattr(material, "geoms", [material]))) == 1  # one body

    spec = solids.spec
    outer = spec.hole_diameter_mm / 2.0 + spec.hole_margin_mm - 0.05
    for point in _ring_points(solids.hole_center_mm, outer):
        assert material.contains(point)
    assert not material.intersects(
        Point(solids.hole_center_mm).buffer(spec.hole_diameter_mm / 2.0 * 0.9)
    )


def test_a_thin_outline_with_tabs_refused_raises_instead(tmp_path):
    with pytest.raises(HoleMarginError, match="nowhere in this outline"):
        build_keychain(
            _thin_bar(), None, _spec(detail_enabled=False, hole_tab_allowed=False)
        )


# --- size against the spec -------------------------------------------------


def test_a_part_bigger_than_the_spec_is_warned_about():
    """Trace owns sizing, so an oversize part is reported, never rescaled."""
    solids = build_keychain(
        _rounded_square(50.0), None, _spec(detail_enabled=False, max_dimension_mm=40.0)
    )
    sized = [w for w in solids.warnings if "maximum dimension" in w]
    assert len(sized) == 1, solids.warnings
    assert "50.0" in sized[0] and "40.0" in sized[0]
    assert "tab" not in sized[0]

    low, high = solids.base.bounding_box().min, solids.base.bounding_box().max
    assert (high.X - low.X) == pytest.approx(50.0, abs=0.1), "it was rescaled"


def test_a_tab_that_pushes_the_part_oversize_says_so():
    tall_bar = MultiPolygon([box(0.0, 0.0, 6.0, 30.0)])
    solids = build_keychain(
        tall_bar, None, _spec(detail_enabled=False, max_dimension_mm=30.0)
    )
    sized = [w for w in solids.warnings if "maximum dimension" in w]
    assert len(sized) == 1, solids.warnings
    assert "hanging tab" in sized[0]
    assert "30.0" in sized[0]


# --- polygons with holes ---------------------------------------------------


def test_a_donut_silhouette_keeps_its_hole(tmp_path):
    solids = build_keychain(_donut(), None, _spec(detail_enabled=False))
    mesh = _scene_meshes(export_3mf(solids, tmp_path / "donut.3mf"))["base"]
    assert mesh.is_volume

    material = _section(mesh, solids.spec.base_thickness_mm / 2)
    assert not material.contains(Point(MID_MM, MID_MM)), "the donut hole was filled"
    assert material.contains(Point(2.0, 2.0)), "the donut body went missing"
    for point in _ring_points((MID_MM, MID_MM), SIZE_MM / 5 + 2.0):
        assert material.contains(point), "the ring around the hole is broken"


def test_a_donut_with_a_detail_ring_still_exports_two_objects(tmp_path):
    inner, outer = SIZE_MM / 5 + 2.0, SIZE_MM / 5 + 6.0
    detail = MultiPolygon(
        [
            Point(MID_MM, MID_MM)
            .buffer(outer, quad_segs=32)
            .difference(Point(MID_MM, MID_MM).buffer(inner, quad_segs=32))
        ]
    )
    solids = build_keychain(_donut(), detail, _spec())
    meshes = _scene_meshes(export_3mf(solids, tmp_path / "donut2.3mf"))
    assert sorted(meshes) == ["base", "detail"]
    assert all(mesh.is_volume for mesh in meshes.values())

    material = _section(meshes["detail"], solids.spec.base_thickness_mm + 0.4)
    assert not material.contains(Point(MID_MM, MID_MM))


# --- disconnected outlines -------------------------------------------------


def test_two_islands_are_refused():
    """Two blobs would print as two loose pieces, so the build stops."""
    islands = MultiPolygon([box(0, 0, 15, 20), box(25, 0, SIZE_MM, 20)])
    with pytest.raises(DisconnectedSilhouetteError) as caught:
        build_keychain(islands, None, _spec(detail_enabled=False))
    message = str(caught.value)
    assert "2 disconnected islands" in message
    assert "fall into 2 pieces" in message
    assert "single connected outline" in message


def test_a_ring_with_a_loose_dot_inside_it_is_refused():
    """An interior ring is fine; something floating inside it is not."""
    ring = box(0, 0, SIZE_MM, SIZE_MM).difference(
        Point(MID_MM, MID_MM).buffer(12.0, quad_segs=32)
    )
    dot = Point(MID_MM, MID_MM).buffer(4.0, quad_segs=32)
    with pytest.raises(DisconnectedSilhouetteError, match="2 disconnected"):
        build_keychain(
            MultiPolygon([ring, dot]), None, _spec(detail_enabled=False)
        )


def test_a_single_donut_is_still_connected_and_builds():
    solids = build_keychain(_donut(), None, _spec(detail_enabled=False))
    assert solids.base.volume > 0


# --- spec validation -------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        (dict(base_thickness_mm=0.0), "base_thickness_mm"),
        (dict(base_thickness_mm=-1.0), "base_thickness_mm"),
        (dict(hole_diameter_mm=0.0), "hole_diameter_mm"),
        (dict(hole_margin_mm=-5.0), "hole_margin_mm"),
        (dict(detail_height_mm=0.0), "detail_height_mm"),
        (dict(detail_height_mm=-0.8), "detail_height_mm"),
        (dict(chamfer_mm=-1.0), "chamfer_mm"),
        (dict(chamfer_mm=2.5), "chamfer_mm"),
        (dict(chamfer_mm=3.0, hole_margin_mm=4.0), "chamfer_mm"),
    ],
)
def test_an_impossible_spec_value_is_refused_by_name(overrides, field):
    """Each bad number is refused, and the message names the field to change."""
    with pytest.raises(BuildError) as caught:
        _build(**overrides)
    assert field in str(caught.value)


def test_a_zero_detail_height_is_fine_when_the_detail_is_off():
    solids = _build(detail_enabled=False, detail_height_mm=0.0)
    assert solids.detail is None


def test_a_chamfer_wider_than_the_margin_is_fine_when_the_chamfer_is_off():
    solids = _build(top_edge_chamfer=False, chamfer_mm=5.0)
    assert solids.base.volume > 0


# --- inputs ----------------------------------------------------------------


def test_a_trace_result_can_be_passed_whole():
    from pathlib import Path

    from print3d.trace import TraceResult

    trace = TraceResult(
        silhouette=_rounded_square(),
        detail=_island_detail(),
        size_mm=SIZE_MM,
        source=Path("sample.png"),
    )
    solids = build_keychain(trace, spec=_spec())
    assert solids.detail is not None


def test_an_empty_silhouette_is_refused():
    from print3d.build import EmptyGeometryError

    with pytest.raises(EmptyGeometryError, match="no silhouette"):
        build_keychain(MultiPolygon(), None, _spec())


def test_the_exporters_only_take_built_solids(raised, tmp_path):
    """The exporters name their objects, so they take a KeychainSolids."""
    solids, _, _ = raised
    for export in (export_3mf, export_stl):
        with pytest.raises(TypeError, match="KeychainSolids"):
            export(solids.base, tmp_path / "loose.3mf")
