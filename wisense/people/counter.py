"""Occupancy counting.

Method: multipath/frequency-diversity clustering. Different subcarriers
respond to WiFi multipath along different frequency-dependent paths
through the room, so distinct moving bodies tend to imprint distinct
dominant motion frequencies onto different subcarrier groups. This
function:

1. Splits the subcarrier set into ``n_groups`` contiguous groups
   (spatial/frequency diversity).
2. For each group, computes an aggregate amplitude time series and
   flags it "active" if its temporal variance clears an activity
   threshold (i.e. this group is actually seeing motion, not just
   empty-room noise).
3. For each active group, finds its dominant motion frequency via FFT
   in the 0.1-2.0 Hz band (covers everything from slow shifting to
   brisk walking-in-place gait).
4. Greedily clusters the active groups' dominant frequencies within a
   tolerance band -- groups whose dominant frequency is close together
   are assumed to be responding to the *same* moving body, so the
   number of clusters becomes the occupancy estimate.

Genuinely hard sub-problem -- read the accuracy caveats below
--------------------------------------------------------------
Distinguishing individual people from WiFi CSI alone, with a single
link and no array/angle-of-arrival hardware, is an open research
problem. This heuristic can meaningfully distinguish "empty" from
"occupied" and often "one person" from "multiple people", but its
accuracy degrades quickly beyond 2-3 simultaneous occupants, when
people are stationary (no motion frequency to key off of), or when
people move in synchrony (their dominant frequencies collapse into one
cluster, undercounting). Treat the return value as a coarse estimate,
not a reliable exact count, and validate against your specific
environment before relying on it.

A specific physical objection worth addressing directly: in
single-antenna, single-link WiFi, all subcarriers physically pass
through the *same* space and reflect off *all* occupants at once --
splitting subcarriers into groups is a frequency-domain split, not a
spatial one, and does not give each group a physically separate view
of the room the way multiple antennas or receivers would. The
underlying assumption here -- that different subcarrier groups end up
dominated by different movers' frequency signatures often enough to
be useful -- is a heuristic, not a guaranteed spatial separation, and
it can fail in exactly the ways described above. It has not been
validated against real multi-occupant hardware captures. If it turns
out to perform no better than chance in practice, the honest fix is to
narrow this function's scope down to presence/no-presence rather than
attempting a count, and that's on the table depending on what testing
shows.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from wisense.core.calibration import CalibrationProfile
from wisense.core.connection import CSIFrame
from wisense.core.filters import filter_majority_subcarrier_count, frames_to_amplitude_matrix, normalize_frame_amplitude
from wisense.exceptions import InsufficientSignalError

logger = logging.getLogger("wisense.people.counter")

_MIN_FRAMES = 16
_MOTION_BAND_HZ = (0.1, 2.0)
_FREQ_CLUSTER_TOLERANCE_HZ = 0.15
_FALLBACK_ACTIVITY_VARIANCE = 1.0


def count_occupants(
    frame_window: List[CSIFrame],
    sample_rate_hz: float,
    n_groups: int = 4,
    calibration: Optional[CalibrationProfile] = None,
) -> int:
    """Estimate the number of people in the sensed space from a CSI
    frame window.

    Parameters
    ----------
    frame_window: frames covering the analysis period.
    sample_rate_hz: capture rate ``frame_window`` was recorded at, in Hz.
    n_groups:
        Number of contiguous subcarrier groups to split into for
        frequency-diversity analysis. More groups can resolve more
        distinct occupants but each group has a noisier per-group
        signal; 4 is a reasonable default for a typical 52-114
        subcarrier CSI capture.
    calibration:
        Optional baseline to scale the per-group activity threshold to
        this environment; falls back to an uncalibrated constant if
        omitted.

    Returns
    -------
    Estimated occupant count (``int``, >= 0). See the module docstring
    for accuracy caveats -- this is a coarse estimate.

    Raises
    ------
    InsufficientSignalError
        If fewer than 16 frames are provided (not enough to resolve a
        motion-band FFT peak with reasonable frequency resolution).
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if n_groups < 1:
        raise ValueError("n_groups must be >= 1")
    if len(frame_window) < _MIN_FRAMES:
        raise InsufficientSignalError(
            f"count_occupants requires at least {_MIN_FRAMES} frames, got {len(frame_window)}"
        )

    frame_window = filter_majority_subcarrier_count(frame_window)
    if len(frame_window) < _MIN_FRAMES:
        raise InsufficientSignalError(
            f"count_occupants requires at least {_MIN_FRAMES} frames after "
            f"discarding mixed-bandwidth outliers, got {len(frame_window)}"
        )

    matrix = frames_to_amplitude_matrix(frame_window)  # (n_frames, n_sub)
    matrix = normalize_frame_amplitude(matrix)  # mitigate AGC gain jumps -- see docstring
    n_frames, n_sub = matrix.shape
    n_groups = min(n_groups, n_sub)

    if calibration is not None:
        activity_threshold = calibration.baseline_variance_p95 * 1.2
    else:
        activity_threshold = _FALLBACK_ACTIVITY_VARIANCE
        logger.debug(
            "count_occupants running without a CalibrationProfile; using "
            "uncalibrated fallback activity threshold %.3f.",
            activity_threshold,
        )

    group_boundaries = np.array_split(np.arange(n_sub), n_groups)
    dominant_freqs: List[float] = []

    window_fn = np.hanning(n_frames) if n_frames > 1 else np.ones(n_frames)
    n_fft = max(1024, int(2 ** np.ceil(np.log2(max(n_frames, 2) * 4))))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
    band_mask = (freqs >= _MOTION_BAND_HZ[0]) & (freqs <= _MOTION_BAND_HZ[1])

    for cols in group_boundaries:
        if cols.size == 0:
            continue
        series = matrix[:, cols].mean(axis=1)
        variance = float(series.var())
        if variance <= activity_threshold:
            continue  # this group isn't seeing motion -- skip it

        detrended = series - series.mean()
        spectrum = np.fft.rfft(detrended * window_fn, n=n_fft)
        power = np.abs(spectrum) ** 2
        if not np.any(band_mask):
            continue
        band_power = power[band_mask]
        band_freqs = freqs[band_mask]
        peak_idx = int(np.argmax(band_power))
        dominant_freqs.append(float(band_freqs[peak_idx]))

    if not dominant_freqs:
        return 0

    count = _cluster_count(sorted(dominant_freqs), _FREQ_CLUSTER_TOLERANCE_HZ)
    logger.debug("count_occupants: dominant_freqs=%s -> estimate=%d", dominant_freqs, count)
    return count


def _cluster_count(sorted_values: List[float], tolerance: float) -> int:
    """Greedily count clusters in a sorted list of values: start a new
    cluster whenever the gap to the previous value exceeds
    ``tolerance``. Simple, deterministic, and dependency-free (no
    scikit-learn requirement for a one-dimensional clustering)."""
    if not sorted_values:
        return 0
    clusters = 1
    for prev, curr in zip(sorted_values, sorted_values[1:]):
        if (curr - prev) > tolerance:
            clusters += 1
    return clusters
