"""Solid construction: 2D geometry to a print-ready 3D model.

Wraps build123d (extrude, hole subtract, chamfer) and exports via trimesh
(3MF path needs networkx + lxml; py-lib3mf has no Windows wheels — lib3mf
itself arrives transitively via build123d). Base and detail layers are
built as separate objects for AMS two-color mapping.
"""

from __future__ import annotations

from pathlib import Path

from shapely.geometry.base import BaseGeometry

from print3d.spec import KeychainSpec


def build_keychain(silhouette: BaseGeometry, detail: BaseGeometry | None, spec: KeychainSpec):
    """Build a keychain solid (base + optional detail layer) from traced geometry.

    Returns the build123d solid(s) ready for export.
    """
    raise NotImplementedError


def export_3mf(solids, out_path: Path) -> None:
    """Export base/detail solids as a multi-object 3MF (AMS color mapping)."""
    raise NotImplementedError


def export_stl(solids, out_path: Path) -> None:
    """Export a combined STL alongside the 3MF."""
    raise NotImplementedError
