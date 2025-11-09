"""
Compatibility wrapper that re-exports seeding utilities from the
standalone ``track_propagation`` package.
"""

from track_propagation.seed import track_from_seed_hits, select_seed_hits  # noqa: F401

__all__ = ["track_from_seed_hits", "select_seed_hits"]


