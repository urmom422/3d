"""Design spec for a print3d object.

Defines the KeychainSpec dataclass: the interview-derived, override-able
parameters that drive tracing and solid construction for a keychain design.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Printer minimums (Bambu Lab A1 Mini, 0.4 mm nozzle, PLA) --------------
#
# These are hard physical limits of the machine, not design preferences, so
# they live here as module constants rather than as KeychainSpec fields: the
# trace, build and QC stages all measure against the same numbers.

#: Thinnest wall the printer can lay down reliably, in mm. Anything narrower
#: in the printed silhouette is rejected outright by the trace stage.
MIN_WALL_MM = 1.2

#: Thinnest raised/recessed detail stroke that survives printing, in mm.
#: Narrower detail is dropped with a warning rather than failing the design.
MIN_DETAIL_STROKE_MM = 1.0

#: Shallowest detail relief that is still visible after printing, in mm.
MIN_DETAIL_DEPTH_MM = 0.6

#: Build volume of the A1 Mini, in mm (X, Y, Z).
BED_SIZE_MM = (180.0, 180.0, 180.0)


#: Width of the top-edge chamfer, in mm. Small on purpose: it is there to
#: knock the printed lip off, not to restyle the part, and every extra
#: tenth eats into the wall the trace stage already signed off on.
DEFAULT_CHAMFER_MM = 0.4


@dataclass
class KeychainSpec:
    """Parameters for a single keychain design.

    Defaults (see program notes / profiles) are applied by the caller;
    this dataclass only carries the resolved values for one design.
    """

    slug: str
    source_image: str
    base_thickness_mm: float = 3.0
    detail_height_mm: float = 0.8
    detail_recessed: bool = False
    hole_diameter_mm: float = 5.2
    hole_margin_mm: float = 2.0
    max_dimension_mm: float = 50.0
    top_edge_chamfer: bool = True

    # --- build-stage parameters --------------------------------------------
    #
    # Added for the solid builder. All default to the behaviour the earlier
    # stages already assumed, so an existing KeychainSpec keeps its meaning.

    #: False builds a single-colour part: the traced detail regions are
    #: ignored entirely rather than raised or recessed.
    detail_enabled: bool = True

    #: Chamfer width, in mm. Only consulted when ``top_edge_chamfer``. The
    #: build stage refuses a chamfer as wide as ``hole_margin_mm`` or as deep
    #: as ``base_thickness_mm``: past those it stops being a bevel on an edge
    #: and starts eating the part.
    chamfer_mm: float = DEFAULT_CHAMFER_MM

    #: Hole centre in model mm, or None to place it automatically near the
    #: top of the silhouette. An explicit position is never quietly moved:
    #: if it cannot hold the margin the build fails instead.
    hole_position_mm: tuple[float, float] | None = None

    #: Whether the builder may grow a small circular tab off the top of the
    #: silhouette when no point inside it can hold a full ring of margin.
    #: False turns that case into an error instead.
    hole_tab_allowed: bool = True

    @property
    def detail_mode(self) -> str:
        """``"raised"``, ``"recessed"`` or ``"none"``.

        The three ways the second colour can be expressed, derived from the
        two flags rather than stored, so the flags cannot disagree with it.
        """
        if not self.detail_enabled:
            return "none"
        return "recessed" if self.detail_recessed else "raised"
