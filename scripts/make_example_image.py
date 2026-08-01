"""Generate the source PNG for the repo's own example design.

Draws a rounded badge (one connected, chunky silhouette) with "3D" in a
heavy sans-serif font as the inner detail layer -- dark badge text on a
lighter badge fill, so the trace stage's alpha-based silhouette and
luminance-based detail split it the way ``designs/example-*/`` expects
(see ``CLAUDE.md`` / issue 09 in this repo's program notes).

**Why this is committed.** ``make3d`` traces whatever PNG it is given;
this script is the *source of that PNG*, checked in so the example
design's input is reproducible from the repo alone, not from a binary
blob nobody can regenerate or audit. Running it twice must always
produce byte-identical output: no randomness, no timestamps, no
machine-dependent state beyond the font file itself (see below).

**Geometry.** A single rounded rectangle badge (one connected region --
required: the pipeline refuses a silhouette with disconnected islands)
filled with a light-ish grey, well above the luminance detail threshold
so the badge fill itself is never mistaken for detail. "3D" is drawn on
top of it in a dark grey, comfortably below the luminance detail
threshold, using a heavy/bold TrueType font at a large point size so
every stroke clears the pipeline's minimum printable widths at the
default 50 mm keychain size (badge silhouette wall >= 1.2 mm; "3D"
detail stroke >= 1.0 mm -- see ``print3d.spec.MIN_WALL_MM`` /
``MIN_DETAIL_STROKE_MM``). The canvas is drawn at a much higher pixel
density than the print, so both minimums land far inside their gates
rather than skating the edge.

**Font resolution.** Tries a short, fixed-order list of heavy/bold
TrueType fonts commonly present on Windows (Impact first -- about as
heavy as a system font gets -- then Arial Bold as a fallback). If none
of those files exist on this machine, falls back to a hand-drawn block
"3D" built entirely from rectangles/pie slices, so the script still
produces a valid, deterministic badge with no external font dependency
at all. Either path is deterministic given a fixed machine: same font
file in, same glyph outlines out.

Usage::

    uv run python scripts/make_example_image.py [output_path.png]

With no argument, writes to a fixed path under the system temp
directory (this machine's "green room": scratch space outside the
repo, never committed from here -- the committed copy lives inside
``designs/example-badge/`` once ``make3d`` has run).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --- canvas -----------------------------------------------------------------

#: Pixel canvas size. Large relative to the printed part (50 mm default)
#: so every minimum-width gate in the pipeline is cleared with generous
#: headroom instead of skating the edge.
CANVAS_W, CANVAS_H = 1000, 800

#: Badge (silhouette) bounding box within the canvas: one connected,
#: chunky rounded rectangle. Comfortably inset from the canvas edge.
BADGE_BOX = (60, 60, 940, 740)
BADGE_RADIUS = 110

#: Badge fill: light-ish grey. Must stay above the detail luminance
#: threshold (127.5, i.e. 255 * 0.5) so the badge itself never reads as
#: "detail" -- only the darker "3D" glyphs should.
BADGE_FILL = 200

#: "3D" fill: dark grey, comfortably below the detail luminance
#: threshold, so it traces as the raised/recessed detail layer.
TEXT_FILL = 25

#: A thin darker ring just inside the badge edge -- purely decorative,
#: part of the silhouette fill (same connected region), keeps the badge
#: from reading as a flat, characterless blob without risking a thin
#: wall (it is well inside the outer edge, not a separate thin sliver).
RING_INSET = 26
RING_FILL = 170

#: Bold/heavy TrueType fonts to try, in order, then a hand-drawn
#: fallback if none exist on this machine. Windows ships Impact and
#: Arial Bold by default, which is why they lead the list.
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\impact.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\verdanab.ttf",
)

#: Large point size: at this canvas density, comfortably clears the
#: 1.0 mm minimum detail stroke width at the default 50 mm print size.
FONT_SIZE = 420


def _rgba(grey: int) -> tuple[int, int, int, int]:
    """A fully-opaque grey as an explicit RGBA tuple.

    On an RGBA canvas, ``ImageDraw`` fills given as a bare int only set
    the RGB channels and leave alpha untouched (i.e. still transparent,
    invisible) -- every fill/outline color must be a 4-tuple here so the
    drawn shapes are actually opaque.
    """
    return (grey, grey, grey, 255)


def _load_font() -> ImageFont.FreeTypeFont | None:
    """First existing font from :data:`_FONT_CANDIDATES`, or ``None``.

    ``None`` signals the caller to fall back to hand-drawn block
    letters -- keeps this script fully self-contained even on a machine
    with none of the candidate fonts installed.
    """
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, FONT_SIZE)
    return None


def _draw_badge(draw: ImageDraw.ImageDraw) -> None:
    draw.rounded_rectangle(BADGE_BOX, radius=BADGE_RADIUS, fill=_rgba(BADGE_FILL))
    x0, y0, x1, y1 = BADGE_BOX
    inset_box = (x0 + RING_INSET, y0 + RING_INSET, x1 - RING_INSET, y1 - RING_INSET)
    inset_radius = max(BADGE_RADIUS - RING_INSET, 1)
    draw.rounded_rectangle(
        inset_box, radius=inset_radius, outline=_rgba(RING_FILL), width=18
    )


def _draw_text_with_font(
    image: Image.Image, draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont
) -> tuple[float, float, float, float]:
    text = "3D"
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx, cy = CANVAS_W / 2, CANVAS_H / 2
    pos = (cx - w / 2 - bbox[0], cy - h / 2 - bbox[1])
    draw.text(pos, text, font=font, fill=_rgba(TEXT_FILL))
    abs_bbox = draw.textbbox(pos, text, font=font)
    return abs_bbox


def _draw_text_fallback(draw: ImageDraw.ImageDraw) -> tuple[float, float, float, float]:
    """Hand-drawn block "3D", used only if no candidate font is found.

    Built from rectangles/pie slices so every stroke width is chosen
    directly rather than inherited from a font's design -- keeps the
    same "well past the minimum" headroom as the font path.
    """
    stroke = 70
    text_fill = _rgba(TEXT_FILL)
    badge_fill = _rgba(BADGE_FILL)

    # Block "3": three horizontal bars plus two vertical end-caps,
    # forming a chunky digital-style "3".
    x0, y0 = 280, 190
    x1, y1 = 470, 610
    draw.rounded_rectangle([x0, y0, x1, y0 + stroke], radius=stroke // 2, fill=text_fill)
    draw.rounded_rectangle(
        [x0, (y0 + y1) // 2 - stroke // 2, x1, (y0 + y1) // 2 + stroke // 2],
        radius=stroke // 2,
        fill=text_fill,
    )
    draw.rounded_rectangle([x0, y1 - stroke, x1, y1], radius=stroke // 2, fill=text_fill)
    draw.rounded_rectangle(
        [x1 - stroke, y0, x1, (y0 + y1) // 2 + stroke // 2],
        radius=stroke // 2,
        fill=text_fill,
    )
    draw.rounded_rectangle(
        [x1 - stroke, (y0 + y1) // 2 - stroke // 2, x1, y1],
        radius=stroke // 2,
        fill=text_fill,
    )

    # Block "D": a vertical bar plus a thick rounded outline.
    dx0, dy0 = 540, 190
    dx1, dy1 = 730, 610
    draw.rounded_rectangle(
        [dx0, dy0, dx0 + stroke, dy1], radius=stroke // 2, fill=text_fill
    )
    draw.pieslice([dx0, dy0, dx1, dy1], start=-90, end=90, fill=text_fill)
    inner = stroke
    draw.pieslice(
        [dx0 + inner, dy0 + inner, dx1 - inner, dy1 - inner],
        start=-90,
        end=90,
        fill=badge_fill,
    )
    draw.rectangle([dx0 + inner, dy0, (dx0 + dx1) // 2, dy1], fill=badge_fill)
    draw.rectangle([dx0, dy0, dx0 + stroke, dy1], fill=text_fill)
    return (float(x0), float(y0), float(dx1), float(dy1))


#: How far the connector bar's top edge reaches up into the glyphs'
#: bottoms, in px. Must be well past 0 so the raster ink genuinely
#: overlaps (a bar merely touching the baseline at one tangent pixel
#: would be exactly the kind of hairline joint the pipeline's own
#: thin-feature gate exists to catch) -- generous here on purpose.
_CONNECTOR_OVERLAP_PX = 60

#: Connector bar thickness, in px. Independent of the overlap above:
#: this is how tall the bar itself is, not how far it reaches upward.
_CONNECTOR_THICKNESS_PX = 55


def _draw_connector(
    draw: ImageDraw.ImageDraw, text_bounds: tuple[float, float, float, float]
) -> None:
    """A baseline bar under "3D" that fuses the two glyphs into one wordmark.

    Purely an aesthetic choice: a baseline rule under the letters reads
    as a designed logo rather than two floating glyphs. The pipeline does
    NOT require it -- disconnected raised-detail regions are legal
    (``detail.body_count`` reports the region count; each region prints
    fused to the base), so "3D" without the bar also produces a
    print-ready design, just with two detail regions instead of one.
    The bar spans both glyphs and overlaps up into their bottoms by
    :data:`_CONNECTOR_OVERLAP_PX` so the ink merges before tracing.
    """
    x0, y0, x1, y1 = text_bounds
    top = y1 - _CONNECTOR_OVERLAP_PX
    bottom = top + _CONNECTOR_THICKNESS_PX
    pad = 20
    draw.rounded_rectangle(
        [x0 - pad, top, x1 + pad, bottom],
        radius=_CONNECTOR_THICKNESS_PX / 2,
        fill=_rgba(TEXT_FILL),
    )


def make_example_image(out_path: Path) -> Path:
    """Draw the badge and save it to ``out_path``. Returns ``out_path``."""
    image = Image.new("RGBA", (CANVAS_W, CANVAS_H), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    _draw_badge(draw)
    font = _load_font()
    if font is not None:
        text_bounds = _draw_text_with_font(image, draw, font)
    else:
        text_bounds = _draw_text_fallback(draw)
    _draw_connector(draw, text_bounds)

    # The badge fill must be fully opaque so the alpha-based silhouette
    # (background stays transparent) traces as one solid, connected
    # region -- re-flatten alpha to pure 0/255 in case any anti-aliased
    # edge pixel from rounded_rectangle/text landed at a partial value.
    r, g, b, a = image.split()
    a = a.point(lambda v: 255 if v >= 128 else 0)
    image = Image.merge("RGBA", (r, g, b, a))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def _default_output_path() -> Path:
    return Path(tempfile.gettempdir()) / "print3d-example" / "example-badge-source.png"


def main(argv: list[str]) -> int:
    out_path = Path(argv[1]) if len(argv) > 1 else _default_output_path()
    make_example_image(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
