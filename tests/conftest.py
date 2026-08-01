"""Shared pytest configuration.

Registers the ``slicer`` marker used by the one real-OrcaSlicer integration
test in ``test_qc.py``, and -- more importantly -- keeps that test *off* by
default.

A bare ``pytest -q`` should not launch a GUI slicer. On a machine where
OrcaSlicer is installed it took the run from seconds to a slice, and
OrcaSlicer writes a ``00000.log`` into whatever directory it was started
in, so the repo picked up an untracked file every time anyone ran the
suite. Opt in with ``--run-slicer`` when you actually mean to slice.

Marker registration and the option both normally belong in
``pyproject.toml``'s ``[tool.pytest.ini_options]``, but that file is
off-limits here, so this uses the other supported route: conftest hooks.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slicer",
        action="store_true",
        default=False,
        help=(
            "Run the tests marked 'slicer', which launch a real OrcaSlicer "
            "headless slice. Off by default: they are slow and they need "
            "OrcaSlicer installed."
        ),
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slicer: runs a real OrcaSlicer headless slice (off unless "
        "--run-slicer is given; skipped when no OrcaSlicer install can be "
        "found).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slicer"):
        return
    skip = pytest.mark.skip(reason="needs --run-slicer (launches OrcaSlicer)")
    for item in items:
        if "slicer" in item.keywords:
            item.add_marker(skip)
