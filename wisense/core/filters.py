"""Signal conditioning utilities operating on CSI amplitude/phase data.

Every function here is a real, documented signal-processing routine
(not a black box) so callers can audit exactly what happens to their
data:

* :func:`amplitude_phase` -- extracts amplitude/phase from raw complex
  CSI (real + imaginary components), for sources that hand back raw IQ
  data rather than pre-split amplitude/phase.
* :func:`hampel_filter` -- outlier-robust denoising via the Hampel
  identifier (median absolute deviation over a sliding window).
* :func:`moving_average_filter` -- simple causal or centered moving
  average low-pass filter.
* :func:`select_subcarriers` -- reduces a ``(frames, subcarriers)``
  amplitude matrix to a smaller, more informative set of subcarriers
  (or a single aggregate series) for downstream detectors.
* :func:`filter_majority_subcarrier_count` -- drops frames whose
  subcarrier count doesn't match the majority (e.g. a stray
  mixed-bandwidth packet), so real hardware streams don't crash
  :func:`frames_to_amplitude_matrix`'s strict shape check.
* :func:`normalize_frame_amplitude` -- partially compensates for
  per-packet AGC (automatic gain control) gain changes by rescaling
  each frame to a common magnitude before variance-based detection.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt

from wisense.core.connection import CSIFrame


def amplitude_phase(real: np.ndarray, imag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute amplitude and (unwrapped) phase from raw complex CSI
    components.

    ``amplitude = sqrt(real**2 + imag**2)``, ``phase = atan2(imag, real)``,
    with phase unwrapped along the last axis (subcarrier axis) using
    ``numpy.unwrap`` to remove artificial :math:`2\\pi` jumps between
    adjacent subcarriers, which is standard practice before using phase
    for sensing (raw wrapped phase is not directly usable).

    Parameters
    ----------
    real, imag:
        Arrays of identical shape, e.g. ``(n_subcarriers,)`` for a
        single frame or ``(n_frames, n_subcarriers)`` for a window.

    Returns
    -------
    (amplitude, phase): tuple of arrays with the same shape as the input.
    """
    real = np.asarray(real, dtype=np.float64)
    imag = np.asarray(imag, dtype=np.float64)
    if real.shape != imag.shape:
        raise ValueError(f"real shape {real.shape} != imag shape {imag.shape}")
    amplitude = np.sqrt(real**2 + imag**2)
    phase = np.unwrap(np.arctan2(imag, real), axis=-1)
    return amplitude, phase


def hampel_filter(series: np.ndarray, window_size: int = 5, n_sigmas: float = 3.0) -> np.ndarray:
    """Apply a Hampel identifier/filter to a 1-D series.

    For each point, computes the median and median absolute deviation
    (MAD) of a symmetric window of ``window_size`` samples centered on
    it. If the point deviates from the window median by more than
    ``n_sigmas * 1.4826 * MAD`` (the ``1.4826`` factor makes MAD a
    consistent estimator of the standard deviation for normally
    distributed data), it is replaced with the window median. This is a
    standard, well-documented outlier-robust denoising technique
    (Hampel, 1974; commonly used for CSI/RSSI denoising in WiFi sensing
    pipelines because it removes spikes without smoothing out genuine
    step changes the way a plain moving average would).

    Parameters
    ----------
    series: 1-D array of samples (e.g. one subcarrier's amplitude over time).
    window_size: half-window radius in samples (full window = 2*window_size+1).
    n_sigmas: outlier threshold in (MAD-estimated) standard deviations.

    Returns
    -------
    Filtered copy of ``series`` with outliers replaced by local medians.
    """
    series = np.asarray(series, dtype=np.float64).copy()
    n = series.shape[0]
    if n == 0:
        return series
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    k = 1.4826  # scale factor from MAD to std-dev for a Gaussian
    for i in range(n):
        lo = max(0, i - window_size)
        hi = min(n, i + window_size + 1)
        window = series[lo:hi]
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        threshold = n_sigmas * k * mad
        if threshold == 0:
            # Degenerate window: every neighbour shares exactly the
            # same value, so the window's MAD is zero and *any*
            # deviation from that shared value is, by definition, an
            # outlier relative to this window.
            if series[i] != med:
                series[i] = med
            continue
        if np.abs(series[i] - med) > threshold:
            series[i] = med
    return series


def moving_average_filter(
    series: np.ndarray, window_size: int = 5, mode: Literal["causal", "centered"] = "centered"
) -> np.ndarray:
    """Apply a simple moving-average low-pass filter to a 1-D series.

    Parameters
    ----------
    series: 1-D array of samples.
    window_size: number of samples averaged together (>= 1).
    mode:
        ``"centered"`` averages ``window_size`` samples centered on
        each point (introduces no group-delay but is non-causal, i.e.
        needs future samples -- fine for offline/batch processing).
        ``"causal"`` averages the current sample and the
        ``window_size - 1`` preceding it (usable in real-time streaming,
        introduces ``(window_size-1)/2`` samples of delay).

    Returns
    -------
    Filtered copy of ``series``, same length as input. Edge samples use
    a shorter (available-data-only) window rather than zero-padding, to
    avoid biasing the filtered signal toward zero at the boundaries.
    """
    series = np.asarray(series, dtype=np.float64)
    n = series.shape[0]
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if n == 0:
        return series.copy()

    out = np.empty_like(series)
    if mode == "centered":
        half = window_size // 2
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            out[i] = series[lo:hi].mean()
    elif mode == "causal":
        for i in range(n):
            lo = max(0, i - window_size + 1)
            out[i] = series[lo : i + 1].mean()
    else:
        raise ValueError(f"mode must be 'causal' or 'centered', got {mode!r}")
    return out


def butterworth_lowpass_filter(
    series: np.ndarray, cutoff_hz: float, sample_rate_hz: float, order: int = 4
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter to a 1-D series.

    Uses ``scipy.signal.butter`` to design the filter and
    ``scipy.signal.filtfilt`` to apply it forward and backward, which
    cancels the filter's phase response (no time-shift is introduced --
    important for downstream timing-sensitive analysis like fall-event
    timestamping). This is the standard technique cited whenever "a
    Butterworth low-pass" is used for CSI denoising in the WiFi-sensing
    literature.

    Parameters
    ----------
    series: 1-D array of samples.
    cutoff_hz: -3 dB cutoff frequency in Hz.
    sample_rate_hz: sampling rate ``series`` was captured at, in Hz.
    order: filter order (higher = steeper rolloff, more ringing risk).

    Returns
    -------
    Filtered copy of ``series``, same length as input.

    Raises
    ------
    ValueError
        If ``cutoff_hz`` is not strictly between 0 and the Nyquist
        frequency (``sample_rate_hz / 2``), or if there are too few
        samples for ``filtfilt``'s default padding at this ``order``.
    """
    series = np.asarray(series, dtype=np.float64)
    nyquist = sample_rate_hz / 2.0
    if not (0 < cutoff_hz < nyquist):
        raise ValueError(f"cutoff_hz must be in (0, {nyquist}) given sample_rate_hz={sample_rate_hz}")
    min_len = 3 * (order + 1)
    if series.shape[0] <= min_len:
        raise ValueError(
            f"series too short ({series.shape[0]} samples) for a stable "
            f"order-{order} filtfilt application; need > {min_len} samples"
        )
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    return filtfilt(b, a, series)


def select_subcarriers(
    amplitude_matrix: np.ndarray,
    method: Literal["variance_topk", "mean", "index"] = "variance_topk",
    top_k: int = 10,
    indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Reduce a ``(n_frames, n_subcarriers)`` amplitude matrix to a
    smaller/aggregated representation for downstream detectors.

    Parameters
    ----------
    amplitude_matrix: shape ``(n_frames, n_subcarriers)``.
    method:
        * ``"variance_topk"`` -- selects the ``top_k`` subcarriers with
          the highest temporal variance (the subcarriers most sensitive
          to motion/multipath change, and thus most informative for
          presence/fall/activity sensing -- a standard heuristic in the
          WiFi-sensing literature for cutting a full subcarrier set down
          to the ones worth processing).
        * ``"mean"`` -- collapses all subcarriers to their per-frame
          mean, returning shape ``(n_frames, 1)``.
        * ``"index"`` -- selects the explicit ``indices`` given.

    Returns
    -------
    Array of shape ``(n_frames, k)`` where ``k`` depends on the method.
    """
    amplitude_matrix = np.asarray(amplitude_matrix, dtype=np.float64)
    if amplitude_matrix.ndim != 2:
        raise ValueError(
            f"amplitude_matrix must be 2-D (n_frames, n_subcarriers), "
            f"got shape {amplitude_matrix.shape}"
        )
    n_frames, n_sub = amplitude_matrix.shape

    if method == "mean":
        return amplitude_matrix.mean(axis=1, keepdims=True)

    if method == "index":
        if indices is None:
            raise ValueError("indices must be provided when method='index'")
        return amplitude_matrix[:, list(indices)]

    if method == "variance_topk":
        if n_frames < 2:
            # Can't compute meaningful variance from a single frame;
            # fall back to returning everything unchanged.
            return amplitude_matrix
        k = min(top_k, n_sub)
        variances = amplitude_matrix.var(axis=0)
        top_indices = np.argsort(variances)[::-1][:k]
        top_indices.sort()
        return amplitude_matrix[:, top_indices]

    raise ValueError(f"unknown method {method!r}")


def frames_to_amplitude_matrix(frames: List[CSIFrame]) -> np.ndarray:
    """Stack a list of :class:`CSIFrame` amplitude arrays into a single
    ``(n_frames, n_subcarriers)`` matrix. Raises ``ValueError`` if the
    frames don't all share the same subcarrier count (e.g. a source
    changed bandwidth mid-stream). This function stays strict on
    purpose (callers reading real hardware streams should call
    :func:`filter_majority_subcarrier_count` first -- see its
    docstring for why)."""
    if not frames:
        return np.empty((0, 0))
    n_sub = frames[0].n_subcarriers
    for f in frames:
        if f.n_subcarriers != n_sub:
            raise ValueError(
                "all frames must have the same number of subcarriers to "
                f"stack into a matrix; got {n_sub} and {f.n_subcarriers}"
            )
    return np.stack([f.amplitude for f in frames], axis=0)


def filter_majority_subcarrier_count(frames: List[CSIFrame]) -> List[CSIFrame]:
    """Drop any frames whose subcarrier count doesn't match the
    majority in ``frames``, so a single mixed-bandwidth packet (e.g. a
    legacy 802.11a/g frame with 52 subcarriers landing among mostly
    802.11n HT20 frames with 56/64) doesn't crash the whole detection
    cycle via :func:`frames_to_amplitude_matrix`'s strict shape check.

    On real WiFi hardware, a CSI capture link can see a mix of frame
    types/bandwidths from ambient traffic; this keeps the majority,
    consistent-shape frames and silently discards the rest, which is
    the right behavior for detectors that need a uniform matrix and
    can tolerate occasionally dropping a handful of frames out of a
    window, rather than raising and losing the whole window.

    Returns a new list (does not mutate ``frames``). Returns ``frames``
    unchanged if it is empty or every frame already shares one
    subcarrier count.
    """
    if not frames:
        return frames
    counts: dict = {}
    for f in frames:
        counts[f.n_subcarriers] = counts.get(f.n_subcarriers, 0) + 1
    if len(counts) == 1:
        return frames
    majority_count = max(counts, key=counts.get)
    return [f for f in frames if f.n_subcarriers == majority_count]


def normalize_frame_amplitude(
    amplitude_matrix: np.ndarray, method: Literal["mean", "l2"] = "mean", smoothing_window: int = 9
) -> np.ndarray:
    """Rescale each frame (each row) of a ``(n_frames, n_subcarriers)``
    amplitude matrix to compensate for automatic gain control (AGC)
    gain-state changes between packets, while preserving short,
    motion-driven transients as the actual sensing signal.

    Why this matters: an ESP32's radio adjusts its LNA/VGA gain
    per-packet to avoid ADC saturation, and a gain step (a routine,
    frequent event, not a sensing signal) multiplies a frame's raw CSI
    amplitude across *all* subcarriers by roughly the same scale
    factor. Left uncorrected, a sustained gain shift can look similar
    to the "sudden large amplitude change" signature that
    :mod:`wisense.fall.detector` and :mod:`wisense.activity.classifier`
    use to detect real physical events, producing false positives.

    Design choice -- normalize against a *smoothed trend*, not each
    frame's own value: dividing frame *i* by its own instantaneous
    magnitude would also partially cancel out genuine motion (a real
    spike *is* a frame with unusually high magnitude -- dividing it by
    itself suppresses exactly the signal being measured). Instead, each
    frame is divided by a moving average of the per-frame magnitude
    series (see ``smoothing_window``), which tracks slow, sustained
    scale drift (matching how AGC actually behaves -- it settles and
    holds a gain level for many packets) without reacting to a single
    frame's brief spike.

    Important limitation -- read before relying on this: this is a
    **partial mitigation, not a full AGC correction**. ESP32-CSI-Tool's
    packet format does not expose the actual LNA/VGA gain state used
    for each packet, only the resulting CSI values, so there is no way
    to invert the true gain applied -- this can only *estimate and
    smooth out* a sustained multiplicative scale drift between frames,
    which is not exactly the same thing as the actual AGC transfer
    function. It also cannot help at all with within-frame
    (subcarrier-to-subcarrier) effects, and an AGC step that happens to
    coincide with genuine fast motion cannot be distinguished from that
    motion by this function alone. Treat this as reducing, not
    eliminating, AGC-driven false positives.

    Parameters
    ----------
    amplitude_matrix: shape ``(n_frames, n_subcarriers)``.
    method:
        ``"mean"`` uses each frame's own mean amplitude as its raw
        magnitude (fast, robust to a handful of outlier subcarriers).
        ``"l2"`` uses the frame's L2 norm instead (more standard in
        some CSI preprocessing literature, more sensitive to outliers).
    smoothing_window:
        Window size (in frames) for the moving average applied to the
        per-frame magnitude series before it's used as the
        normalization reference. Larger values track slower drift
        without touching short spikes; smaller values react faster but
        risk suppressing motion transients (as with per-frame
        normalization at the extreme of ``smoothing_window=1``).

    Returns
    -------
    A new array, same shape as input. Frames whose reference magnitude
    is (near-)zero are left unscaled rather than dividing by zero.
    """
    amplitude_matrix = np.asarray(amplitude_matrix, dtype=np.float64)
    if amplitude_matrix.ndim != 2:
        raise ValueError(
            f"amplitude_matrix must be 2-D (n_frames, n_subcarriers), "
            f"got shape {amplitude_matrix.shape}"
        )
    n_frames = amplitude_matrix.shape[0]
    if n_frames == 0:
        return amplitude_matrix.copy()

    if method == "mean":
        per_frame_scale = amplitude_matrix.mean(axis=1)
    elif method == "l2":
        per_frame_scale = np.linalg.norm(amplitude_matrix, axis=1)
    else:
        raise ValueError(f"method must be 'mean' or 'l2', got {method!r}")

    if n_frames >= 3 and smoothing_window > 1:
        reference_scale = moving_average_filter(
            per_frame_scale, window_size=min(smoothing_window, n_frames), mode="centered"
        )
    else:
        # Too few frames to smooth meaningfully -- fall back to the
        # matrix's overall average as a single flat reference, which at
        # least avoids the self-cancellation problem described above.
        reference_scale = np.full(n_frames, float(per_frame_scale.mean()))

    overall_scale = (
        float(np.mean(per_frame_scale[per_frame_scale > 1e-9]))
        if np.any(per_frame_scale > 1e-9)
        else 1.0
    )
    safe_scale = np.where(reference_scale > 1e-9, reference_scale, 1.0)
    normalized = amplitude_matrix / safe_scale[:, np.newaxis] * overall_scale
    return normalized
