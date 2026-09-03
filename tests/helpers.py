"""Synthetic CSI-like signal generators used across the test suite.

Every signal produced here is synthetic test-fixture data with known,
constructed properties (e.g. "a sine wave at exactly 0.3 Hz") used to
verify the signal-processing math -- never presented as, or derived
from, real captured WiFi CSI.
"""

from __future__ import annotations

from typing import List

import numpy as np

from wisense.core.connection import CSIFrame

RNG_SEED = 1234


def _rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


def make_still_frames(n_frames: int = 40, n_sub: int = 16, base: float = 10.0, noise_std: float = 0.05) -> List[CSIFrame]:
    """Frames with only tiny sensor noise -- an "empty room" fixture."""
    rng = _rng()
    frames = []
    for i in range(n_frames):
        amp = base + rng.normal(0, noise_std, size=n_sub)
        frames.append(CSIFrame(timestamp=float(i) * 0.1, amplitude=amp))
    return frames


def make_motion_frames(n_frames: int = 40, n_sub: int = 16, base: float = 10.0, motion_std: float = 3.0) -> List[CSIFrame]:
    """Frames with large random amplitude fluctuations -- an "occupied,
    moving" fixture."""
    rng = _rng()
    frames = []
    for i in range(n_frames):
        amp = base + rng.normal(0, motion_std, size=n_sub)
        frames.append(CSIFrame(timestamp=float(i) * 0.1, amplitude=amp))
    return frames


def make_breathing_frames(
    n_frames: int, sample_rate_hz: float, breathing_hz: float, n_sub: int = 8, amplitude_scale: float = 2.0, noise_std: float = 0.05
) -> List[CSIFrame]:
    """Frames whose aggregate amplitude follows a clean sine wave at
    ``breathing_hz`` -- used to verify
    :func:`wisense.vitals.breathing.estimate_breathing_rate` recovers a
    known-frequency periodicity."""
    rng = _rng()
    t = np.arange(n_frames) / sample_rate_hz
    base_signal = 10.0 + amplitude_scale * np.sin(2 * np.pi * breathing_hz * t)
    frames = []
    for i in range(n_frames):
        amp = base_signal[i] + rng.normal(0, noise_std, size=n_sub)
        frames.append(CSIFrame(timestamp=float(t[i]), amplitude=amp))
    return frames


def make_fall_frames(
    n_pre: int = 10, n_post: int = 20, n_sub: int = 16, base: float = 10.0, drop: float = 15.0, noise_std: float = 0.1
) -> List[CSIFrame]:
    """Frames simulating: quiet baseline -> sudden large amplitude spike
    -> settling into a still, shifted level (the fall signature)."""
    rng = _rng()
    frames = []
    t = 0.0
    dt = 0.05
    for _ in range(n_pre):
        amp = base + rng.normal(0, noise_std, size=n_sub)
        frames.append(CSIFrame(timestamp=t, amplitude=amp))
        t += dt
    # the spike frame itself
    amp = base + drop + rng.normal(0, noise_std, size=n_sub)
    frames.append(CSIFrame(timestamp=t, amplitude=amp))
    t += dt
    # settle to a new, still, lower level (lying on the ground)
    settled = base - drop * 0.6
    for _ in range(n_post):
        amp = settled + rng.normal(0, noise_std, size=n_sub)
        frames.append(CSIFrame(timestamp=t, amplitude=amp))
        t += dt
    return frames


def make_walking_frames(n_frames: int = 60, n_sub: int = 16, base: float = 10.0, gait_hz: float = 1.5, sample_rate_hz: float = 20.0, amplitude_scale: float = 4.0) -> List[CSIFrame]:
    """Frames whose aggregate amplitude oscillates periodically at a
    walking-gait-like frequency, with per-subcarrier noise on top."""
    rng = _rng()
    t = np.arange(n_frames) / sample_rate_hz
    base_signal = base + amplitude_scale * np.sin(2 * np.pi * gait_hz * t)
    frames = []
    for i in range(n_frames):
        amp = base_signal[i] + rng.normal(0, 0.3, size=n_sub)
        frames.append(CSIFrame(timestamp=float(t[i]), amplitude=amp))
    return frames
