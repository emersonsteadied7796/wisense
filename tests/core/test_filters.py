import numpy as np
import pytest

from wisense.core.connection import CSIFrame
from wisense.core.filters import (
    amplitude_phase,
    butterworth_lowpass_filter,
    filter_majority_subcarrier_count,
    frames_to_amplitude_matrix,
    hampel_filter,
    moving_average_filter,
    normalize_frame_amplitude,
    select_subcarriers,
)


def test_amplitude_phase_known_values():
    real = np.array([[3.0, 0.0]])
    imag = np.array([[4.0, 5.0]])
    amp, phase = amplitude_phase(real, imag)
    np.testing.assert_allclose(amp, [[5.0, 5.0]])
    assert phase.shape == amp.shape


def test_amplitude_phase_shape_mismatch_raises():
    with pytest.raises(ValueError):
        amplitude_phase(np.array([1.0, 2.0]), np.array([1.0]))


def test_hampel_filter_removes_spike():
    series = np.array([1.0, 1.0, 1.0, 1.0, 50.0, 1.0, 1.0, 1.0, 1.0])
    filtered = hampel_filter(series, window_size=2, n_sigmas=3.0)
    assert filtered[4] < 5.0  # spike should be replaced
    np.testing.assert_allclose(filtered[:4], series[:4], atol=1e-9)


def test_hampel_filter_leaves_smooth_signal_unchanged():
    series = np.linspace(0, 1, 20)
    filtered = hampel_filter(series, window_size=3, n_sigmas=3.0)
    # A smooth ramp has no outliers relative to local windows beyond
    # what a symmetric window naturally smooths; ensure no huge distortion.
    assert np.max(np.abs(filtered - series)) < 0.2


def test_hampel_filter_empty_series():
    assert hampel_filter(np.array([])).shape == (0,)


def test_moving_average_centered_constant_signal():
    series = np.full(10, 5.0)
    filtered = moving_average_filter(series, window_size=3, mode="centered")
    np.testing.assert_allclose(filtered, series)


def test_moving_average_causal_introduces_delay():
    series = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
    filtered = moving_average_filter(series, window_size=3, mode="causal")
    # Causal average should lag behind the step, so at index 3 (first
    # 10.0) the average shouldn't have jumped all the way to 10 yet.
    assert filtered[3] < 10.0
    assert filtered[-1] == pytest.approx(10.0)


def test_moving_average_invalid_mode_raises():
    with pytest.raises(ValueError):
        moving_average_filter(np.array([1.0, 2.0]), window_size=1, mode="bogus")


def test_butterworth_lowpass_smooths_high_frequency_noise():
    sample_rate = 50.0
    t = np.arange(200) / sample_rate
    clean = np.sin(2 * np.pi * 0.5 * t)  # slow 0.5 Hz signal
    noisy = clean + 0.5 * np.sin(2 * np.pi * 15.0 * t)  # fast 15 Hz noise
    filtered = butterworth_lowpass_filter(noisy, cutoff_hz=2.0, sample_rate_hz=sample_rate, order=4)
    # Filtered signal should be much closer to the clean signal than
    # the noisy one was.
    err_before = np.mean((noisy - clean) ** 2)
    err_after = np.mean((filtered - clean) ** 2)
    assert err_after < err_before * 0.3


def test_butterworth_lowpass_rejects_bad_cutoff():
    with pytest.raises(ValueError):
        butterworth_lowpass_filter(np.random.default_rng(0).normal(size=50), cutoff_hz=100.0, sample_rate_hz=50.0)


def test_select_subcarriers_mean():
    matrix = np.array([[1.0, 3.0], [2.0, 4.0]])
    out = select_subcarriers(matrix, method="mean")
    np.testing.assert_allclose(out, [[2.0], [3.0]])


def test_select_subcarriers_index():
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = select_subcarriers(matrix, method="index", indices=[0, 2])
    np.testing.assert_allclose(out, [[1.0, 3.0], [4.0, 6.0]])


def test_select_subcarriers_variance_topk_picks_highest_variance():
    rng = np.random.default_rng(0)
    n_frames = 50
    low_var_col = np.full(n_frames, 5.0)
    high_var_col = rng.normal(0, 10.0, size=n_frames)
    matrix = np.stack([low_var_col, high_var_col], axis=1)
    out = select_subcarriers(matrix, method="variance_topk", top_k=1)
    np.testing.assert_allclose(out[:, 0], high_var_col)


def test_select_subcarriers_rejects_1d_input():
    with pytest.raises(ValueError):
        select_subcarriers(np.array([1.0, 2.0, 3.0]))


def test_frames_to_amplitude_matrix_shape():
    frames = [CSIFrame(timestamp=float(i), amplitude=np.array([1.0, 2.0, 3.0])) for i in range(4)]
    matrix = frames_to_amplitude_matrix(frames)
    assert matrix.shape == (4, 3)


def test_frames_to_amplitude_matrix_rejects_inconsistent_subcarriers():
    frames = [
        CSIFrame(timestamp=0.0, amplitude=np.array([1.0, 2.0])),
        CSIFrame(timestamp=1.0, amplitude=np.array([1.0, 2.0, 3.0])),
    ]
    with pytest.raises(ValueError):
        frames_to_amplitude_matrix(frames)


def test_frames_to_amplitude_matrix_empty():
    matrix = frames_to_amplitude_matrix([])
    assert matrix.shape == (0, 0)


def test_normalize_frame_amplitude_corrects_sustained_agc_step():
    # Simulate a sustained AGC gain step: 20 frames at baseline scale,
    # then 20 frames scaled up 3x (as a real AGC gain jump would
    # multiply all subcarriers by roughly the same factor), no
    # motion-driven variance within either segment.
    rng = np.random.default_rng(0)
    baseline = 10.0 + rng.normal(0, 0.05, size=(20, 8))
    gain_stepped = 3.0 * (10.0 + rng.normal(0, 0.05, size=(20, 8)))
    matrix = np.vstack([baseline, gain_stepped])

    raw_variance = matrix.mean(axis=1).var()
    normalized = normalize_frame_amplitude(matrix, smoothing_window=9)
    normalized_variance = normalized.mean(axis=1).var()

    # The AGC step should dominate raw per-frame-mean variance; after
    # normalization, that dominance should be substantially reduced.
    assert normalized_variance < raw_variance * 0.5


def test_normalize_frame_amplitude_preserves_short_spike():
    # A brief, motion-like spike (a handful of frames) should NOT be
    # cancelled out the way a sustained AGC step is -- this is the
    # whole point of smoothing against a trend rather than each
    # frame's own instantaneous value.
    rng = np.random.default_rng(1)
    quiet_before = 10.0 + rng.normal(0, 0.05, size=(15, 8))
    spike = np.full((2, 8), 25.0) + rng.normal(0, 0.05, size=(2, 8))
    quiet_after = 10.0 + rng.normal(0, 0.05, size=(15, 8))
    matrix = np.vstack([quiet_before, spike, quiet_after])

    normalized = normalize_frame_amplitude(matrix, smoothing_window=9)
    normalized_signal = normalized.mean(axis=1)

    # The spike frames should still clearly stand out from the quiet
    # baseline after normalization -- i.e. normalization corrected
    # sustained drift, not this transient.
    spike_level = normalized_signal[15:17].mean()
    baseline_level = np.concatenate([normalized_signal[:15], normalized_signal[17:]]).mean()
    assert spike_level > baseline_level * 1.5


def test_normalize_frame_amplitude_rejects_1d_input():
    with pytest.raises(ValueError):
        normalize_frame_amplitude(np.array([1.0, 2.0, 3.0]))


def test_normalize_frame_amplitude_empty():
    result = normalize_frame_amplitude(np.empty((0, 4)))
    assert result.shape == (0, 4)


def test_normalize_frame_amplitude_invalid_method():
    with pytest.raises(ValueError):
        normalize_frame_amplitude(np.ones((5, 4)), method="bogus")


def test_filter_majority_subcarrier_count_drops_minority():
    frames = [CSIFrame(timestamp=float(i), amplitude=np.zeros(8)) for i in range(10)]
    frames += [CSIFrame(timestamp=100.0, amplitude=np.zeros(6))]  # one stray mixed-bandwidth frame
    filtered = filter_majority_subcarrier_count(frames)
    assert len(filtered) == 10
    assert all(f.n_subcarriers == 8 for f in filtered)


def test_filter_majority_subcarrier_count_all_consistent_unchanged():
    frames = [CSIFrame(timestamp=float(i), amplitude=np.zeros(8)) for i in range(5)]
    filtered = filter_majority_subcarrier_count(frames)
    assert filtered == frames


def test_filter_majority_subcarrier_count_empty():
    assert filter_majority_subcarrier_count([]) == []
