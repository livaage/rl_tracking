"""
Compatibility wrapper around the standalone ``track_propagation`` package.

Existing RL Tracking code can continue importing ``HitHolder`` from
``rl_tracking.physics.hit_holder``, while other projects are free to
depend directly on ``track_propagation``.
"""

from track_propagation.hit_holder import HitHolder  # noqa: F401

__all__ = ["HitHolder"]


