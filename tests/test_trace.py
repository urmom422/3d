"""Tests for the trace stage.

Every fixture is drawn here with Pillow rather than committed as a binary,
so the shapes under test are readable in the diff. Keep them small
(~200-400 px): potracer is pure Python and slow on big bitmaps.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon

from print3d.spec import MIN_DETAIL_STROKE_MM, MIN_WALL_MM
from print3d.trace import (
    ThinFeatureError,
    TraceError,
    TraceResult,
    UnsupportedImageError,
    _PHOTO_MAX_SHORT_SIDE_PX,
    _flatten_photo,
    find_thin_regions,
    trace_detail,
    trace_image,
    trace_silhouette,
)

SIZE_MM = 50.0


# --- fixtures --------------------------------------------------------------


def _save(image: Image.Image, tmp_path, name: str):
    path = tmp_path / name
    image.save(path)
    return path


def _photo_drawing(tmp_path, name="photo.png", size=(360, 360), noise_std=4.0):
    """A synthetic "phone photo of a paper drawing": known stroke geometry,
    an uneven-illumination gradient, and mild sensor noise on top -- built
    in the test rather than committed as a binary, per this module's own
    convention (see the module docstring), and with a ground-truth ink mask
    returned alongside so segmentation quality can be measured directly
    (IoU, background false-positive rate) instead of merely "did it trace".
    """
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    # Bright top-left corner fading to dim bottom-right: a desk lit from one
    # side, the classic phone-photo lighting defect.
    gradient = 235.0 - 70.0 * (xx / w) - 40.0 * (yy / h)
    base = Image.fromarray(gradient.astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(base)
    ink = (20, 20, 20)
    stroke = max(4, int(w * 0.035))
    draw.ellipse(
        [w * 0.25, h * 0.25, w * 0.75, h * 0.75], outline=ink, width=stroke
    )
    draw.ellipse([w * 0.375, h * 0.40, w * 0.45, h * 0.475], fill=ink)
    draw.ellipse([w * 0.55, h * 0.40, w * 0.625, h * 0.475], fill=ink)
    draw.arc(
        [w * 0.375, h * 0.475, w * 0.625, h * 0.65],
        start=20,
        end=160,
        fill=ink,
        width=max(3, int(w * 0.025)),
    )

    ground_truth = Image.new("L", size, 0)
    gt_draw = ImageDraw.Draw(ground_truth)
    gt_draw.ellipse(
        [w * 0.25, h * 0.25, w * 0.75, h * 0.75], outline=255, width=stroke
    )
    gt_draw.ellipse([w * 0.375, h * 0.40, w * 0.45, h * 0.475], fill=255)
    gt_draw.ellipse([w * 0.55, h * 0.40, w * 0.625, h * 0.475], fill=255)
    gt_draw.arc(
        [w * 0.375, h * 0.475, w * 0.625, h * 0.65],
        start=20,
        end=160,
        fill=255,
        width=max(3, int(w * 0.025)),
    )
    ground_truth_mask = np.array(ground_truth) > 0

    arr = np.array(base.convert("L")).astype(np.int16)
    rng = np.random.default_rng(0)
    noisy = np.clip(arr + rng.normal(0, noise_std, arr.shape), 0, 255).astype(
        np.uint8
    )
    path = _save(Image.fromarray(noisy, mode="L"), tmp_path, name)
    return path, ground_truth_mask


def _filled_shape_photo(tmp_path, name="filled_photo.png", size=(300, 300)):
    """A photographed drawing whose subject is one big *filled* shape.

    Same uneven lighting as ``_photo_drawing``, but the ink is a solid disc
    covering most of the page rather than thin strokes -- the case where an
    illumination estimate that cannot tell ink from paper quietly hollows
    the shape out. Returns the file and its ground-truth ink mask.
    """
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    gradient = 235.0 - 70.0 * (xx / w) - 40.0 * (yy / h)
    base = Image.fromarray(gradient.astype(np.uint8), mode="L").convert("RGB")
    box = [w * 0.2, h * 0.2, w * 0.8, h * 0.8]
    ImageDraw.Draw(base).ellipse(box, fill=(25, 25, 25))

    ground_truth = Image.new("L", size, 0)
    ImageDraw.Draw(ground_truth).ellipse(box, fill=255)

    arr = np.array(base.convert("L")).astype(np.int16)
    rng = np.random.default_rng(1)
    noisy = np.clip(arr + rng.normal(0, 4.0, arr.shape), 0, 255).astype(
        np.uint8
    )
    path = _save(Image.fromarray(noisy, mode="L"), tmp_path, name)
    return path, np.array(ground_truth) > 0


@pytest.fixture
def bold_logo(tmp_path):
    """A bold two-tone logo: mid-grey disc with dark bar and dark dot.

    Grey (160) is inside the silhouette threshold but outside the detail
    threshold, so the silhouette is the whole disc while the detail is just
    the bar and the dot.
    """
    image = Image.new("L", (300, 300), 255)
    draw = ImageDraw.Draw(image)
    draw.ellipse([20, 20, 279, 279], fill=160)
    draw.rectangle([70, 130, 229, 170], fill=20)
    draw.ellipse([130, 60, 169, 99], fill=20)
    return _save(image, tmp_path, "bold_logo.png")


@pytest.fixture
def hairline_logo(tmp_path):
    """1 px strokes: at 50 mm across, every wall is far under 1.2 mm."""
    image = Image.new("L", (300, 300), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 279, 279], outline=0, width=1)
    draw.line([20, 20, 279, 279], fill=0, width=1)
    return _save(image, tmp_path, "hairline.png")


@pytest.fixture
def necked_logo(tmp_path):
    """Two thick blobs joined by a 3 px neck - thin, but not everywhere."""
    image = Image.new("L", (320, 200), 255)
    draw = ImageDraw.Draw(image)
    draw.ellipse([10, 40, 129, 159], fill=0)
    draw.ellipse([190, 40, 309, 159], fill=0)
    draw.rectangle([129, 98, 190, 101], fill=0)
    return _save(image, tmp_path, "necked.png")


@pytest.fixture
def blob_with_hairline_detail(tmp_path):
    """Solid grey blob carrying one bold bar and one 1 px hairline."""
    image = Image.new("L", (300, 300), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 279, 279], fill=160)
    draw.rectangle([60, 100, 239, 130], fill=0)  # bold: keep
    draw.line([60, 200, 239, 200], fill=0, width=1)  # hairline: drop
    return _save(image, tmp_path, "mixed_detail.png")


@pytest.fixture
def blob_with_connected_hairline(tmp_path):
    """A bold bar with a hairline tail growing straight out of it.

    One connected component, half of it printable and half of it not - the
    letterform case. An all-or-nothing per-component test either keeps the
    unprintable tail or throws the whole glyph away.
    """
    image = Image.new("L", (300, 300), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 279, 279], fill=160)
    draw.rectangle([60, 100, 200, 130], fill=0)  # bold bar
    draw.line([200, 115, 260, 115], fill=0, width=1)  # hairline, attached
    return _save(image, tmp_path, "connected_hairline.png")


@pytest.fixture
def donut(tmp_path):
    """A ring with a free-floating dot inside its hole.

    Pins hole reconstruction *and* nested-island reconstruction: the dot is
    inside the hole, which is inside the ring, so a naive point-in-shell
    test loses it or swallows it.
    """
    image = Image.new("L", (300, 300), 255)
    draw = ImageDraw.Draw(image)
    draw.ellipse([10, 10, 289, 289], fill=160)  # outer disc, 280 px across
    draw.ellipse([80, 80, 219, 219], fill=255)  # hole, 140 px across
    draw.ellipse([130, 130, 169, 169], fill=160)  # island, 40 px across
    return _save(image, tmp_path, "donut.png")


def _svg_path_points(d_attribute: str) -> list[tuple[float, float]]:
    """Parse the ``M x,y L x,y ... Z`` data this module emits."""
    points = []
    for token in d_attribute.replace("M", " ").replace("L", " ").replace(
        "Z", " "
    ).split():
        x, y = token.split(",")
        points.append((float(x), float(y)))
    return points


# --- silhouette ------------------------------------------------------------


def test_bold_logo_traces_to_valid_positive_area_polygons(bold_logo):
    result = trace_image(bold_logo, SIZE_MM)

    assert isinstance(result, TraceResult)
    assert isinstance(result.silhouette, MultiPolygon)
    assert not result.silhouette.is_empty
    assert result.silhouette.is_valid
    assert result.silhouette.area > 0
    for part in result.silhouette.geoms:
        assert part.is_valid
        assert part.area > 0

    # A disc of diameter 50 mm: roughly pi * 25**2, allow for tracing slop.
    assert result.silhouette.area == pytest.approx(1963.0, rel=0.05)


def test_scaling_hits_the_requested_size(bold_logo):
    result = trace_image(bold_logo, SIZE_MM)
    x0, y0, x1, y1 = result.silhouette.bounds

    assert max(x1 - x0, y1 - y0) == pytest.approx(SIZE_MM, abs=0.1)
    # Origin is the silhouette's bounding box corner.
    assert x0 == pytest.approx(0.0, abs=1e-6)
    assert y0 == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("size_mm", [30.0, 65.0])
def test_scaling_is_uniform_at_other_sizes(bold_logo, size_mm):
    result = trace_image(bold_logo, size_mm)
    x0, y0, x1, y1 = result.silhouette.bounds

    assert max(x1 - x0, y1 - y0) == pytest.approx(size_mm, abs=0.1)
    assert result.width_mm == pytest.approx(result.height_mm, abs=0.2)


def test_silhouette_convenience_wrapper(bold_logo):
    geometry = trace_silhouette(bold_logo, SIZE_MM)
    assert isinstance(geometry, MultiPolygon)
    assert geometry.area > 0


# --- thin-feature gate -----------------------------------------------------


def test_hairline_image_raises_thin_feature_error(hairline_logo):
    with pytest.raises(ThinFeatureError) as excinfo:
        trace_image(hairline_logo, SIZE_MM)

    message = str(excinfo.value)
    assert "1.2" in message
    assert excinfo.value.min_width_mm == MIN_WALL_MM
    assert excinfo.value.regions
    # Actionable for a hobbyist: says what to do, not just what broke.
    assert "--size" in message


def test_thin_feature_error_is_a_trace_error(hairline_logo):
    with pytest.raises(TraceError):
        trace_image(hairline_logo, SIZE_MM)


def test_thin_neck_is_caught_even_though_the_blobs_are_thick(necked_logo):
    with pytest.raises(ThinFeatureError) as excinfo:
        trace_image(necked_logo, SIZE_MM)
    assert "1.2" in str(excinfo.value)


def test_a_big_enough_size_rescues_a_borderline_image(necked_logo):
    # Same artwork at bed size: the 3 px neck now clears 1.2 mm, so the
    # gate is measuring printed width and not just counting thin pixels.
    result = trace_image(necked_logo, 180.0, detail=False)
    assert result.silhouette.area > 0
    assert find_thin_regions(result.silhouette, MIN_WALL_MM) == []


def test_find_thin_regions_ignores_decorative_concavities():
    from shapely.geometry import Point, box

    plate = box(0, 0, 50, 50).difference(Point(25, 50).buffer(10, quad_segs=64))
    assert find_thin_regions(plate, MIN_WALL_MM) == []


def test_find_thin_regions_flags_a_thin_spike():
    from shapely.geometry import Point, box
    from shapely.ops import unary_union

    spiky = unary_union([Point(0, 0).buffer(10, quad_segs=64), box(-0.3, 9, 0.3, 16)])
    regions = find_thin_regions(spiky, MIN_WALL_MM)
    assert regions
    assert regions[0].area_mm2 > 0


# --- thin-feature gate: the sliver cut-off boundary -------------------------
#
# The cut-off has to sit between two measured populations: round-off slivers
# left at sharp convex corners (~0.08 mm2) and real sub-millimetre necks
# (~0.2 mm2 and up). These two tests pin both sides of it.


@pytest.mark.parametrize(
    "neck_width_mm, neck_length_mm",
    [(0.2, 1.0), (0.6, 0.8), (0.8, 2.0), (0.9, 0.8)],
)
def test_a_sub_millimetre_neck_between_thick_blobs_is_flagged(
    neck_width_mm, neck_length_mm
):
    from shapely.geometry import box
    from shapely.ops import unary_union

    half = neck_width_mm / 2.0
    dumbbell = unary_union(
        [
            box(0, 0, 10, 10),
            box(10, 5 - half, 10 + neck_length_mm, 5 + half),
            box(10 + neck_length_mm, 0, 20 + neck_length_mm, 10),
        ]
    )
    regions = find_thin_regions(dumbbell, MIN_WALL_MM)

    assert regions, f"{neck_width_mm} mm neck was not flagged"
    # The measurement must not overstate the width, or the size it implies
    # would be too small to help.
    assert 0 < regions[0].width_mm <= neck_width_mm + 1e-6


def test_sharp_square_corners_are_not_mistaken_for_thin_features():
    from shapely.geometry import box
    from shapely.ops import unary_union

    # Four 90-degree corners: the dilation rounds every one of them off and
    # leaves a crescent. None of them is a thin feature.
    assert find_thin_regions(box(0, 0, 20, 20), MIN_WALL_MM) == []
    # Eight more corners, still nothing narrower than 12 mm anywhere.
    tabbed = unary_union([box(0, 0, 20, 20), box(20, 4, 26, 16)])
    assert find_thin_regions(tabbed, MIN_WALL_MM) == []


# --- thin-feature gate: the advice it gives ---------------------------------


def test_suggested_size_is_measured_and_actually_clears_the_gate(necked_logo):
    with pytest.raises(ThinFeatureError) as excinfo:
        trace_image(necked_logo, SIZE_MM)
    error = excinfo.value

    assert not error.needs_bolder_artwork
    assert error.suggested_size_mm > SIZE_MM
    # The whole point: retracing at the suggested size must pass. The old
    # fixed 1.5x formula suggested 75 mm here, which does not.
    rescued = trace_image(necked_logo, error.suggested_size_mm, detail=False)
    assert find_thin_regions(rescued.silhouette, MIN_WALL_MM) == []


def test_hairline_artwork_is_told_no_size_will_help(hairline_logo):
    with pytest.raises(ThinFeatureError) as excinfo:
        trace_image(hairline_logo, SIZE_MM)
    error = excinfo.value

    # This artwork's thin features scale with the image, so they never
    # clear: promising a magic --size would be confidently wrong.
    assert error.needs_bolder_artwork
    assert "bolder image" in str(error)
    assert "any keychain size" in str(error)


def test_thin_regions_report_a_measured_width(necked_logo):
    with pytest.raises(ThinFeatureError) as excinfo:
        trace_image(necked_logo, SIZE_MM)
    region = excinfo.value.regions[0]

    assert 0 < region.width_mm < MIN_WALL_MM
    assert f"{region.width_mm:.2f}" in region.describe()


def test_a_wholly_thin_part_is_located_by_centroid_not_bounding_box():
    from shapely.geometry import box

    # A long thin bar: its bounding box is the whole bar, so quoting it
    # tells the user nothing they can act on. Name where it sits instead.
    region = find_thin_regions(box(10, 20, 40, 20.5), MIN_WALL_MM)[0]

    assert region.whole_part
    text = region.describe()
    assert "centred at" in text
    assert "x 25.0 mm" in text and "y 20.2 mm" in text


# --- detail ----------------------------------------------------------------


def test_detail_regions_lie_inside_the_silhouette(bold_logo):
    result = trace_image(bold_logo, SIZE_MM, detail=True)

    assert not result.detail.is_empty
    assert result.detail.is_valid
    assert result.detail.area > 0
    assert result.detail.area < result.silhouette.area
    # Allow a hair of tolerance for the simplification pass.
    assert result.detail.difference(result.silhouette.buffer(1e-6)).area < 1e-6
    assert len(result.detail.geoms) == 2  # the bar and the dot


def test_detail_can_be_switched_off(bold_logo):
    result = trace_image(bold_logo, SIZE_MM, detail=False)
    assert result.detail.is_empty
    assert result.warnings == []


def test_hairline_detail_is_dropped_with_a_warning(blob_with_hairline_detail):
    result = trace_image(blob_with_hairline_detail, SIZE_MM)

    assert not result.detail.is_empty  # the bold bar survives
    assert len(result.detail.geoms) == 1
    assert result.warnings
    assert str(MIN_DETAIL_STROKE_MM) in result.warnings[0]
    assert find_thin_regions(result.detail, MIN_DETAIL_STROKE_MM) == []


def test_detail_convenience_wrapper(bold_logo):
    geometry = trace_detail(bold_logo, SIZE_MM)
    assert isinstance(geometry, MultiPolygon)
    assert len(geometry.geoms) == 2
    assert geometry.area == pytest.approx(
        trace_image(bold_logo, SIZE_MM).detail.area
    )


@pytest.mark.parametrize("wrapper", [trace_silhouette, trace_detail])
def test_convenience_wrappers_explain_the_detail_collision(wrapper, bold_logo):
    # Passing detail= used to blow up with "trace_image() got multiple
    # values for keyword argument 'detail'", which tells a caller nothing -
    # so assert on the explanation, not on words that message also has.
    with pytest.raises(TypeError) as excinfo:
        wrapper(bold_logo, SIZE_MM, detail=True)
    message = str(excinfo.value)
    assert "does not accept 'detail'" in message
    assert wrapper.__name__ in message
    assert "got multiple values" not in message


def test_a_hairline_attached_to_a_bold_stroke_is_partly_removed(
    blob_with_connected_hairline,
):
    """The letterform case: one component, half printable, half not."""
    result = trace_image(blob_with_connected_hairline, SIZE_MM)

    # The bold bar survives...
    assert not result.detail.is_empty
    assert result.detail.area > 100.0
    # ...and the hairline tail does not.
    assert find_thin_regions(result.detail, MIN_DETAIL_STROKE_MM) == []
    x0, _, x1, _ = result.detail.bounds
    assert x1 - x0 < 30.0  # the tail reached past 40 mm before removal

    # And the removal is reported, with the area it took, so the QC report
    # can surface it.
    assert result.warnings
    assert str(MIN_DETAIL_STROKE_MM) in result.warnings[0]
    assert "mm2" in result.warnings[0]


def test_a_solid_dark_shape_has_no_detail_layer_and_says_so(tmp_path):
    """detail == silhouette is one colour twice, not a two-colour design."""
    image = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse([20, 20, 279, 279], fill=(10, 10, 10, 255))
    path = _save(image, tmp_path, "solid_dark_on_alpha.png")

    result = trace_image(path, SIZE_MM)

    assert result.silhouette.area == pytest.approx(1963.0, rel=0.05)
    assert result.detail.is_empty
    assert result.warnings
    assert "one dark shape" in result.warnings[0]
    assert "single-colour" in result.warnings[0]


# --- svg output ------------------------------------------------------------


def test_to_svg_writes_a_millimetre_sized_file(bold_logo, tmp_path):
    result = trace_image(bold_logo, SIZE_MM)
    out = result.to_svg(tmp_path / "out" / "trace.svg")

    assert out.exists()
    text = out.read_text(encoding="utf-8")
    ElementTree.fromstring(text)  # well-formed XML
    assert text.startswith("<svg")
    assert f'width="{result.width_mm:.3f}mm"' in text
    assert f'height="{result.height_mm:.3f}mm"' in text
    assert 'id="silhouette"' in text
    assert 'id="detail"' in text


def _svg_paths(result, tmp_path):
    out = result.to_svg(tmp_path / "out" / "trace.svg")
    root = ElementTree.fromstring(out.read_text(encoding="utf-8"))
    group = root[0]
    paths = {p.get("id"): p.get("d") for p in group}
    return group, paths


def test_to_svg_path_data_matches_the_geometry(bold_logo, tmp_path):
    result = trace_image(bold_logo, SIZE_MM)
    _, paths = _svg_paths(result, tmp_path)
    points = _svg_path_points(paths["silhouette"])

    # The first coordinate is the first vertex of the first ring, verbatim.
    first_ring = list(result.silhouette.geoms[0].exterior.coords)
    assert points[0] == (
        pytest.approx(round(first_ring[0][0], 4), abs=5e-5),
        pytest.approx(round(first_ring[0][1], 4), abs=5e-5),
    )
    # Every emitted vertex is a real vertex of the geometry, not a rescaled
    # or re-origined one: the path data is in mm, in model space.
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    assert min(xs) == pytest.approx(0.0, abs=0.01)
    assert min(ys) == pytest.approx(0.0, abs=0.01)
    assert max(xs) == pytest.approx(result.width_mm, abs=0.01)
    assert max(ys) == pytest.approx(result.height_mm, abs=0.01)


def test_to_svg_flips_y_with_a_transform_not_with_the_coordinates(
    bold_logo, tmp_path
):
    """The Y flip must actually be applied, and applied exactly once."""
    result = trace_image(bold_logo, SIZE_MM)
    group, paths = _svg_paths(result, tmp_path)
    height = result.height_mm

    assert group.get("transform") == (
        f"translate(0,{height:.3f}) scale(1,-1)"
    )

    # The bold_logo's detail dot sits above the bar in model space (Y up).
    # After the transform it must sit *nearer the top of the canvas*, i.e.
    # at a smaller SVG Y - that is the flip doing its job.
    dot_y_model = max(y for _, y in _svg_path_points(paths["detail"]))
    dot_y_svg = height - dot_y_model
    assert dot_y_model > height / 2  # above centre in model space
    assert dot_y_svg < height / 2  # above centre on screen too

    # And the raw path data is *not* pre-flipped: its own top edge is the
    # geometry's top edge, not its mirror.
    top_model = max(y for _, y in _svg_path_points(paths["silhouette"]))
    assert top_model == pytest.approx(height, abs=0.01)


def test_to_svg_emits_hole_rings_for_a_donut(donut, tmp_path):
    result = trace_image(donut, SIZE_MM, detail=False)
    _, paths = _svg_paths(result, tmp_path)

    # Ring outer + ring hole + island = three subpaths, even-odd filled.
    assert paths["silhouette"].count("M ") == 3
    assert 'fill-rule="evenodd"' in (
        result.to_svg(tmp_path / "again.svg").read_text(encoding="utf-8")
    )


# --- holes and nested islands ----------------------------------------------


def test_a_donut_keeps_its_hole_and_the_island_inside_it(donut):
    result = trace_image(donut, SIZE_MM, detail=False)

    # Two parts: the ring, and the free-floating dot inside its hole.
    assert len(result.silhouette.geoms) == 2
    ring = max(result.silhouette.geoms, key=lambda p: p.area)
    island = min(result.silhouette.geoms, key=lambda p: p.area)
    assert len(ring.interiors) == 1
    assert len(island.interiors) == 0

    # The island really is inside the ring's hole, not overlapping it.
    from shapely.geometry import Polygon

    hole = Polygon(ring.interiors[0])
    assert hole.contains(island)
    assert not ring.intersects(island)

    # Analytic areas: the source is 280 px across and scaled to 50 mm, so
    # 1 px = 50/280 mm. Outer 280 px, hole 140 px, island 40 px diameter.
    mm_per_px = SIZE_MM / 280.0
    import math

    outer = math.pi * (140 * mm_per_px) ** 2
    hole_area = math.pi * (70 * mm_per_px) ** 2
    island_area = math.pi * (20 * mm_per_px) ** 2

    assert ring.area == pytest.approx(outer - hole_area, rel=0.02)
    assert island.area == pytest.approx(island_area, rel=0.06)
    assert result.silhouette.area == pytest.approx(
        outer - hole_area + island_area, rel=0.02
    )


# --- photo mode --------------------------------------------------------


def test_flatten_photo_recovers_strokes_from_uneven_lighting(tmp_path):
    """Ground-truth IoU/false-positive check on the cleaned mask itself --
    proves segmentation correctness, not merely that something traces."""
    path, ground_truth_mask = _photo_drawing(tmp_path)
    gray = np.array(Image.open(path).convert("L"))

    cleaned = _flatten_photo(gray, path)
    ink_mask = cleaned < 128

    intersection = np.logical_and(ground_truth_mask, ink_mask).sum()
    union = np.logical_or(ground_truth_mask, ink_mask).sum()
    iou = intersection / union
    background_false_positive = (
        np.logical_and(~ground_truth_mask, ink_mask).sum()
        / (~ground_truth_mask).sum()
    )

    assert iou > 0.85
    assert background_false_positive < 0.02


def test_flatten_photo_keeps_a_big_filled_shape_solid(tmp_path):
    """A filled-in shape wider than the lighting estimate can see across
    must keep its middle as ink.

    The interesting failure here is silent: an illumination estimate that
    reads the ink as if it were paper puts the shape's own darkness into
    the estimate of the page underneath it, so dividing cancels the shape
    against itself and the middle comes back as blank page. The geometry
    is then wrong -- a ring instead of a disc -- with nothing raised as an
    error. Kids fill shapes in, so this is a core input, not an edge case.
    """
    path, ground_truth_mask = _filled_shape_photo(tmp_path)
    gray = np.array(Image.open(path).convert("L"))

    ink_mask = _flatten_photo(gray, path) < 128

    intersection = np.logical_and(ground_truth_mask, ink_mask).sum()
    union = np.logical_or(ground_truth_mask, ink_mask).sum()
    assert intersection / union > 0.95

    # The middle of the disc is the part that cancels itself out when the
    # estimate is wrong, so pin it explicitly rather than trusting an
    # overall score to notice a hole.
    height, width = ink_mask.shape
    centre = ink_mask[
        height // 2 - 20 : height // 2 + 20, width // 2 - 20 : width // 2 + 20
    ]
    assert centre.all()

    recovered = np.logical_and(ground_truth_mask, ink_mask).sum() / (
        ground_truth_mask.sum()
    )
    assert recovered > 0.95


def test_flatten_photo_caps_its_working_resolution(tmp_path):
    """Photo cleanup must bound the pixel count it works at.

    A phone photo is 12 megapixels; every stage from here on scales with
    that, so photo mode shrinks the image to a stated cap first (see
    ``_PHOTO_MAX_SHORT_SIDE_PX``). This pins the mechanism rather than a
    wall-clock time, which would be a flaky thing to assert.
    """
    oversized = _PHOTO_MAX_SHORT_SIDE_PX + 600
    path, _ = _filled_shape_photo(
        tmp_path, name="oversized.png", size=(oversized, oversized * 4 // 3)
    )
    gray = np.array(Image.open(path).convert("L"))

    cleaned = _flatten_photo(gray, path)

    assert min(cleaned.shape) <= _PHOTO_MAX_SHORT_SIDE_PX
    # Shrunk, not cropped or squashed: 3:4 in, 3:4 out.
    height, width = cleaned.shape
    assert width / height == pytest.approx(3 / 4, abs=0.01)


def test_flatten_photo_leaves_a_small_image_at_its_own_size(tmp_path):
    """The cap only ever shrinks: artwork already under it is worked on at
    the resolution it came in at, not resampled for the sake of it."""
    path, _ = _photo_drawing(tmp_path, size=(360, 360))
    gray = np.array(Image.open(path).convert("L"))

    assert _flatten_photo(gray, path).shape == gray.shape


def test_flatten_photo_rejects_a_mask_with_no_real_contrast():
    """An all-dark (or otherwise structureless) cleanup must fail loudly,
    not silently hand back a mask that is really just noise."""
    uniform = np.full((200, 200), 200, dtype=np.uint8)
    with pytest.raises(TraceError):
        _flatten_photo(uniform, Path("uniform.png"))


def test_flatten_photo_rejects_an_all_dark_image():
    """Nothing to distinguish as "ink" when every pixel is already dark --
    must not be reported as a successful, all-foreground segmentation."""
    all_dark = np.full((200, 200), 10, dtype=np.uint8)
    with pytest.raises(TraceError):
        _flatten_photo(all_dark, Path("dark.png"))


def test_photo_mode_traces_a_lit_and_shadowed_drawing(tmp_path):
    path, _ = _photo_drawing(tmp_path)
    result = trace_image(path, SIZE_MM, photo=True)

    assert not result.silhouette.is_empty
    assert result.silhouette.area > 0
    # A photographed pencil/marker drawing is one shade of ink: cleanup
    # collapses it to bilevel, so the existing "one dark shape, no separate
    # detail layer" path fires -- expected, not a bug (see trace_image's
    # own docstring for ``photo``).
    assert result.detail.is_empty
    assert result.warnings
    assert "single-colour" in result.warnings[0]


def test_photo_flag_is_required_to_flatten_the_lighting(tmp_path):
    """--photo is strictly opt-in: the same file traced without it must not
    quietly get the same cleanup applied under the hood."""
    path, _ = _photo_drawing(tmp_path)

    flattened_result = trace_image(path, SIZE_MM, photo=True)
    assert not flattened_result.silhouette.is_empty

    # Without flattening, the illumination gradient itself crosses the
    # silhouette threshold and shreds the outline into a mess of hairline
    # slivers -- proof the default code path is genuinely untouched by the
    # new cleanup, not quietly also-flattened under the hood.
    with pytest.raises(ThinFeatureError):
        trace_image(path, SIZE_MM, photo=False, detail=False)


def test_photo_false_matches_not_passing_photo_at_all(bold_logo):
    default = trace_image(bold_logo, SIZE_MM)
    explicit_off = trace_image(bold_logo, SIZE_MM, photo=False)

    assert default.silhouette.equals(explicit_off.silhouette)
    assert default.detail.equals(explicit_off.detail)
    assert default.warnings == explicit_off.warnings


def test_photo_mode_reports_no_strokes_found_in_the_error_style(tmp_path):
    w, h = 300, 300
    yy, xx = np.mgrid[0:h, 0:w]
    gradient = (230 - 60 * (xx / w) - 30 * (yy / h)).astype(np.uint8)
    path = _save(
        Image.fromarray(gradient, mode="L"), tmp_path, "blank_photo.png"
    )

    with pytest.raises(TraceError) as excinfo:
        trace_image(path, SIZE_MM, photo=True)
    message = str(excinfo.value).lower()
    # Plain-language, actionable, matching the rest of this module's voice.
    assert "brighter" in message
    assert "pen" in message or "pencil" in message


def test_photo_mode_ignores_alpha_even_when_present(tmp_path):
    """A photographed drawing has no meaningful transparency: photo=True
    must not switch over to the alpha channel just because the file
    happens to carry one."""
    w, h = 320, 320
    yy, xx = np.mgrid[0:h, 0:w]
    gradient = (235 - 60 * (xx / w) - 30 * (yy / h)).astype(np.uint8)
    image = Image.fromarray(gradient, mode="L").convert("RGBA")
    draw = ImageDraw.Draw(image)
    draw.ellipse([80, 80, 240, 240], outline=(20, 20, 20, 255), width=14)
    # A meaningfully transparent corner: if alpha were consulted (as it
    # would be for a non-photo input, see _MIN_TRANSPARENT_FRACTION) this
    # would read as "mostly background" and starve the silhouette.
    for x in range(30):
        for y in range(30):
            image.putpixel((x, y), (255, 255, 255, 0))
    path = _save(image, tmp_path, "photo_with_alpha.png")

    result = trace_image(path, SIZE_MM, photo=True, detail=False)
    assert not result.silhouette.is_empty
    assert result.silhouette.area > 0


# --- input handling --------------------------------------------------------


def test_svg_input_is_reported_as_unsupported(tmp_path):
    path = tmp_path / "art.svg"
    path.write_text("<svg/>", encoding="utf-8")
    with pytest.raises(UnsupportedImageError) as excinfo:
        trace_image(path, SIZE_MM)

    message = str(excinfo.value)
    # Honest about the state of things and actionable about the way round.
    assert "PNG" in message
    assert "no SVG importer planned" in message


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(UnsupportedImageError):
        trace_image(tmp_path / "nope.png", SIZE_MM)


def test_blank_image_reports_no_artwork(tmp_path):
    blank = _save(Image.new("L", (200, 200), 255), tmp_path, "blank.png")
    with pytest.raises(TraceError) as excinfo:
        trace_image(blank, SIZE_MM)
    assert "every pixel looks like background" in str(excinfo.value)


def test_speckle_only_image_is_not_misreported_as_blank(tmp_path):
    """"Nothing here" and "all of it was filtered out" are different bugs."""
    image = Image.new("L", (200, 200), 255)
    image.putpixel((100, 100), 0)  # a single dark pixel: 1 px2 of artwork
    path = _save(image, tmp_path, "speck.png")

    with pytest.raises(TraceError) as excinfo:
        trace_image(path, SIZE_MM, speckle_px=4)

    message = str(excinfo.value)
    assert "speckle" in message
    assert "speckle_px" in message  # says which knob to turn
    assert "every pixel looks like background" not in message


def test_transparent_png_uses_its_alpha_channel(tmp_path):
    """A white shape on transparency must not read as "all background"."""
    image = Image.new("RGBA", (240, 240), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse([20, 20, 219, 219], fill=(255, 255, 255, 255))
    path = _save(image, tmp_path, "alpha.png")

    result = trace_image(path, SIZE_MM)
    assert result.silhouette.area == pytest.approx(1963.0, rel=0.05)


def test_one_stray_translucent_pixel_does_not_hijack_the_silhouette(tmp_path):
    """An opaque image with one alpha-249 pixel is not a transparent image.

    Reproduced defect: that single pixel switched the silhouette over to the
    alpha channel, which read the entire 50x50 mm canvas as the part.
    """
    image = Image.new("RGBA", (300, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse([20, 20, 279, 279], fill=(0, 0, 0, 255))
    image.putpixel((0, 0), (255, 255, 255, 249))
    path = _save(image, tmp_path, "stray_alpha.png")

    result = trace_image(path, SIZE_MM, detail=False)

    assert result.silhouette.area == pytest.approx(1963.0, rel=0.05)
    assert result.silhouette.area < 2100.0  # not the 2500 mm2 whole canvas


def test_real_transparency_still_switches_to_the_alpha_channel(tmp_path):
    """Well past the 1% threshold: a disc on transparency, no luminance cue."""
    image = Image.new("RGBA", (240, 240), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse([20, 20, 219, 219], fill=(255, 255, 255, 255))
    path = _save(image, tmp_path, "very_alpha.png")

    result = trace_image(path, SIZE_MM, detail=False)
    assert result.silhouette.area == pytest.approx(1963.0, rel=0.05)


def test_negative_size_is_rejected(bold_logo):
    with pytest.raises(ValueError):
        trace_image(bold_logo, -5.0)
