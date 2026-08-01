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
