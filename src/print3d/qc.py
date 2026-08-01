"""Quality gates for generated designs.

Level 1: geometry checks (trimesh is_volume, shapely wall-thickness check
via buffer(-0.6): any region thinner than 1.2mm fails).
Level 2: slice checks (OrcaSlicer CLI headless; judged by exit code, output
.gcode.3mf existence/size, and embedded gcode header -- never stdout, it is
unreliable on Windows). A design is "print-ready" only once level 2 passes.
"""

from __future__ import annotations

from pathlib import Path


def check_level1(mesh_path: Path) -> bool:
    """Run level-1 geometry checks on an exported mesh.

    Returns True if the mesh passes (is a valid volume, no walls thinner
    than 1.2mm).
    """
    raise NotImplementedError


def check_level2(model_path: Path) -> bool:
    """Run level-2 slice checks via OrcaSlicer CLI.

    Returns True if the sliced output is print-ready (exit code 0, output
    .gcode.3mf exists with nonzero size, embedded gcode header present).
    """
    raise NotImplementedError
