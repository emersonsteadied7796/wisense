"""Passive breathing-rate estimation.

Method: FFT-based periodicity detection in the human respiration band
(0.15-0.5 Hz, i.e. 9-30 breaths/minute -- the standard range used in
CSI/radar respiration-sensing literature) applied to a CSI amplitude
signal. Chest movement during breathing subtly and periodically
perturbs the multipath environment; that periodicity shows up as a
spectral peak in this band when the subject is relatively still
(e.g. sleeping, sitting) and close enough to the link for the effect to
be visible above noise.

This module deliberately returns ``None`` (with a logged reason)
instead of a fabricated number whenever the window is too short or the
signal-to-noise ratio in the respiration band is too low to support a
confident estimate -- see the ``Returns`` section of
:func:`estimate_breathing_rate`.

Hardware sensitivity caveat -- read before relying on this on an
ESP32
-----------------------------------------------------------------
Breathing-induced chest movement perturbs 2.4 GHz CSI amplitude far
less than gross body motion does (this is *why* the SNR gate exists at
all). Published CSI respiration-sensing results generally come from
more capable radios/antenna setups than a single commodity ESP32, and
this module's SNR threshold was tuned against clean synthetic sine
waves (see ``tests/vitals/test_breathing.py``), not a validated
real-world noise floor -- nobody has run this against a real ESP32 to
confirm what fraction of real attempts clear the SNR gate at typical
room distances. It is entirely plausible that, on real hardware, this
returns ``None`` far more often than the synthetic tests suggest,
particularly beyond a meter or two, through obstacles, or with a noisy
RF environment. Conversely, claiming it will *never* produce a
confident reading is also not something this project can verify one
way or the other without real-hardware testing. Treat any real-world
result -- confident estimate or ``None`` -- as data worth reporting via
an issue, not an established, benchmarked capability.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from wisense.core.connection import CSIFrame
from wisense.core.filters import filter_majority_subcarrier_count, frames_to_amplitude_matrix, normalize_frame_amplitude, select_subcarriers

logger = logging.getLogger("wisense.vitals.breathing")

RESPIRATION_BAND_HZ = (0.15, 0.5)  # 9-30 breaths/min
_MIN_CYCLES_REQUIRED = 3.0  # need this many full cycles at the *lowest* band frequency to resolve it
_MIN_SNR_DB = 15.0  # peak-to-band-median power ratio required to trust the estimate


def estimate_breathing_rate(
    frame_window: List[CSIFrame], sample_rate_hz: float, subcarrier_top_k: int = 5
) -> Optional[float]:
    """Estimate breathing rate in breaths per minute from a CSI frame
    window.

    Parameters
    ----------
    frame_window:
        Frames covering the analysis period, ideally captured at a
        roughly constant rate close to ``sample_rate_hz``. No motion
        other than breathing should be occurring in the window (the
        caller is responsible for gating this, e.g. only calling this
        after a presence detector reports stillness).
    sample_rate_hz:
        The CSI capture rate, in Hz, that ``frame_window`` was recorded
        at. Used to build the frequency axis for the FFT.
    subcarrier_top_k:
        Number of highest-temporal-variance subcarriers to average
        together before analysis (the ones most likely to be carrying
        the respiration signal).

    Returns
    -------
    Estimated breathing rate in breaths/minute, or ``None`` if no
    confident estimate could be made. ``None`` is returned (rather than
    raising) in two documented cases, both logged at ``INFO`` level with
    the specific reason:

    * The window is too short to resolve the lowest respiration-band
      frequency (needs at least
      ``_MIN_CYCLES_REQUIRED / RESPIRATION_BAND_HZ[0]`` seconds of data,
      i.e. 20 seconds by default).
    * The strongest peak found in the respiration band is not
      significantly above the surrounding spectral noise floor (SNR
      below ``_MIN_SNR_DB``), meaning the subject likely was not still
      enough, was too far from the link, or no periodic respiration
      signal is resolvable in this window.
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    # Drop stray mixed-bandwidth frames before anything else, so the
    # duration/frame-count check below reflects what will actually be
    # analyzed.
    frame_window = filter_majority_subcarrier_count(frame_window)

    n_frames = len(frame_window)
    duration_s = n_frames / sample_rate_hz
    min_duration_s = _MIN_CYCLES_REQUIRED / RESPIRATION_BAND_HZ[0]
    if duration_s < min_duration_s:
        logger.info(
            "estimate_breathing_rate: insufficient window duration "
            "(%.1fs, need >= %.1fs for %d cycles at the lowest "
            "respiration-band frequency %.2f Hz) -- returning None.",
            duration_s,
            min_duration_s,
            _MIN_CYCLES_REQUIRED,
            RESPIRATION_BAND_HZ[0],
        )
        return None

    matrix = frames_to_amplitude_matrix(frame_window)
    matrix = normalize_frame_amplitude(matrix)  # mitigate AGC gain-step artifacts -- see docstring
    reduced = select_subcarriers(matrix, method="variance_topk", top_k=subcarrier_top_k)
    signal = reduced.mean(axis=1)

    # Detrend: remove linear trend (slow drift unrelated to breathing,
    # e.g. AGC/environment drift) then window with a Hann window before
    # the FFT to reduce spectral leakage.
    t = np.arange(n_frames)
    if n_frames >= 2:
        coeffs = np.polyfit(t, signal, deg=1)
        trend = np.polyval(coeffs, t)
        signal = signal - trend
    signal = signal - signal.mean()
    window = np.hanning(n_frames) if n_frames > 1 else np.ones(n_frames)
    windowed = signal * window

    # Zero-pad for finer frequency resolution in the FFT bin spacing.
    n_fft = max(2048, int(2 ** np.ceil(np.log2(max(n_frames, 2) * 4))))
    spectrum = np.fft.rfft(windowed, n=n_fft)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)

    band_mask = (freqs >= RESPIRATION_BAND_HZ[0]) & (freqs <= RESPIRATION_BAND_HZ[1])
    if not np.any(band_mask):
        logger.info(
            "estimate_breathing_rate: sample_rate_hz=%.3f too low to "
            "resolve the respiration band -- returning None.",
            sample_rate_hz,
        )
        return None

    band_power = power[band_mask]
    band_freqs = freqs[band_mask]

    peak_idx = int(np.argmax(band_power))
    peak_power = band_power[peak_idx]
    peak_freq = band_freqs[peak_idx]

    band_median = float(np.median(band_power)) if band_power.size > 1 else 0.0
    if band_median <= 0:
        snr_db = float("inf") if peak_power > 0 else 0.0
    else:
        snr_db = 10.0 * np.log10(peak_power / band_median)

    if snr_db < _MIN_SNR_DB:
        logger.info(
            "estimate_breathing_rate: respiration-band peak SNR %.2f dB "
            "below required %.2f dB -- signal quality too low for a "
            "confident estimate; returning None.",
            snr_db,
            _MIN_SNR_DB,
        )
        return None

    breaths_per_minute = float(peak_freq * 60.0)
    return breaths_per_minute
