"""Environment calibration.

Collects a baseline of CSI statistics from a room in a known state
(normally: empty/unoccupied) and packages it into a
:class:`CalibrationProfile` that downstream detectors (presence, fall,
activity) accept as an optional parameter to adapt their thresholds to
the deployment environment, instead of relying on fixed global
constants that would behave inconsistently across rooms, furniture
layouts, and hardware.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wisense.core.connection import CSISource
from wisense.core.filters import filter_majority_subcarrier_count, frames_to_amplitude_matrix, normalize_frame_amplitude
from wisense.exceptions import CalibrationError

logger = logging.getLogger("wisense.core.calibration")

_MIN_CALIBRATION_FRAMES = 10


@dataclass
class CalibrationProfile:
    """Baseline statistics captured during calibration, consumed by
    detector modules to normalize/threshold their features against this
    specific environment rather than a fixed global constant.

    Attributes
    ----------
    baseline_mean:
        Per-subcarrier mean amplitude during the calibration window,
        shape ``(n_subcarriers,)``.
    baseline_std:
        Per-subcarrier amplitude standard deviation during the
        calibration window, shape ``(n_subcarriers,)``.
    baseline_variance_p95:
        95th percentile, across subcarriers, of per-subcarrier temporal
        variance during calibration. Used as the "empty room" reference
        level that presence/fall detectors compare live variance
        against.
    n_frames:
        Number of frames the profile was computed from.
    n_subcarriers:
        Number of subcarriers the profile applies to. Detectors should
        reject (or resample) frame windows whose subcarrier count
        doesn't match this.
    duration_seconds:
        Wall-clock duration the calibration capture spanned.
    created_at:
        Unix timestamp when the profile was created.
    """

    baseline_mean: np.ndarray
    baseline_std: np.ndarray
    baseline_variance_p95: float
    n_frames: int
    n_subcarriers: int
    duration_seconds: float
    created_at: float

    def variance_threshold(self, n_sigmas: float = 4.0) -> float:
        """A recommended variance threshold for presence/fall detectors:
        the empty-room 95th-percentile variance scaled up by
        ``n_sigmas`` worth of the baseline's own spread, so environments
        with naturally noisier empty-room signal (e.g. near other WiFi
        traffic) automatically get a higher, less trigger-happy
        threshold."""
        spread = float(np.mean(self.baseline_std))
        return self.baseline_variance_p95 + n_sigmas * spread


def calibrate(
    source: CSISource, duration_seconds: float = 30.0, idle_timeout_seconds: float = 3.0
) -> CalibrationProfile:
    """Collect a baseline from ``source`` for ``duration_seconds`` and
    return a :class:`CalibrationProfile`.

    ``source`` must already be connected (this function does not call
    ``connect``/``disconnect`` itself, so it composes cleanly with
    ``with source:`` blocks and with sources already in use elsewhere).
    Intended to be run while the monitored space is in a known,
    typically unoccupied, state.

    Parameters
    ----------
    source: an already-connected CSISource.
    duration_seconds: total wall-clock window to collect a baseline over.
    idle_timeout_seconds:
        If no frame has been received for this long (tracked from the
        *last successfully received frame*, not from the start of
        calibration), stop early rather than waiting out the full
        ``duration_seconds``. This matters on real serial/network links,
        where momentary gaps between individual frames are normal and
        must not be confused with the source having gone silent for
        good -- only a *sustained* gap this long ends calibration early.

    Raises
    ------
    CalibrationError
        If fewer than 10 frames are collected in the given duration
        (not enough to compute meaningful statistics), or if the frames
        collected have inconsistent subcarrier counts.

    Examples
    --------
    >>> from wisense.core import FileCSISource, calibrate
    >>> with FileCSISource("capture.csv") as src:  # doctest: +SKIP
    ...     profile = calibrate(src, duration_seconds=5)
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if idle_timeout_seconds <= 0:
        raise ValueError("idle_timeout_seconds must be positive")

    frames = []
    start = time.monotonic()
    deadline = start + duration_seconds
    last_frame_at = start
    while time.monotonic() < deadline:
        frame = source.read_frame()
        if frame is None:
            # For file sources this means end-of-file; for live sources
            # it means "nothing available right now" -- a momentary gap
            # is normal (serial/network jitter) and must not end
            # calibration on its own. Only a *sustained* gap -- no frame
            # received for idle_timeout_seconds, measured from the last
            # frame we actually got -- means the source has genuinely
            # gone quiet.
            if frames and (time.monotonic() - last_frame_at) > idle_timeout_seconds:
                break
            continue
        frames.append(frame)
        last_frame_at = time.monotonic()

    if len(frames) < _MIN_CALIBRATION_FRAMES:
        raise CalibrationError(
            f"only collected {len(frames)} frames during "
            f"{duration_seconds}s calibration window; need at least "
            f"{_MIN_CALIBRATION_FRAMES}. Check the source is connected "
            "and actively producing frames."
        )

    # Drop any stray mixed-bandwidth frames before stacking into a
    # matrix (see filter_majority_subcarrier_count docstring), then
    # apply the same per-frame AGC-mitigation normalization used at
    # detection time -- calibration thresholds must be computed in the
    # same normalized space detectors will compare live data against,
    # or the threshold is calibrated against the wrong scale.
    frames = filter_majority_subcarrier_count(frames)
    if len(frames) < _MIN_CALIBRATION_FRAMES:
        raise CalibrationError(
            f"only {len(frames)} frames remained after discarding "
            "mixed-bandwidth outliers; need at least "
            f"{_MIN_CALIBRATION_FRAMES}. This can happen if the capture "
            "environment has heavy mixed-bandwidth WiFi traffic."
        )

    matrix = frames_to_amplitude_matrix(frames)  # (n_frames, n_subcarriers)
    matrix = normalize_frame_amplitude(matrix)
    baseline_mean = matrix.mean(axis=0)
    baseline_std = matrix.std(axis=0)
    per_subcarrier_variance = matrix.var(axis=0)
    baseline_variance_p95 = float(np.percentile(per_subcarrier_variance, 95))

    actual_duration = frames[-1].timestamp - frames[0].timestamp
    if actual_duration <= 0:
        actual_duration = duration_seconds

    profile = CalibrationProfile(
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        baseline_variance_p95=baseline_variance_p95,
        n_frames=len(frames),
        n_subcarriers=matrix.shape[1],
        duration_seconds=actual_duration,
        created_at=time.time(),
    )
    logger.info(
        "Calibration complete: %d frames, %d subcarriers, variance_p95=%.4f",
        profile.n_frames,
        profile.n_subcarriers,
        profile.baseline_variance_p95,
    )
    return profile
