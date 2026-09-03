# API Reference

## `wisense.core.connection`

### `CSIFrame`
```python
CSIFrame(timestamp: float, amplitude: np.ndarray, phase: Optional[np.ndarray] = None,
          rssi: Optional[float] = None, mac: Optional[str] = None)
```
A single CSI measurement. `.n_subcarriers` (property) returns
`amplitude.shape[0]`. Raises `FrameParseError` if `phase` is given with
a shape mismatching `amplitude`.

### `CSISource` (abstract base class)
- `connect() -> None`
- `disconnect() -> None`
- `read_frame() -> Optional[CSIFrame]` -- `None` means no more frames
  (EOF for files) or none available right now (timeout for live
  sources).
- `stream() -> Iterator[CSIFrame]`
- Context manager: `with source: ...` calls `connect`/`disconnect`.
- `.is_connected -> bool`

### `SerialCSISource(port, baudrate=921600, timeout=2.0)`
Reads ESP32-CSI-Tool-compatible CSV lines over a serial port. Requires
`pyserial`. See the class docstring for the exact wire format parsed.

### `NetworkCSISource(host, port, protocol="udp", timeout=2.0, recv_buffer_size=65536)`
Reads newline-delimited JSON CSI frames over UDP or TCP (WiSense's own
documented protocol -- see class docstring for the schema).

### `FileCSISource(path, loop=False, realtime=False)`
Replays a WiSense CSV capture file (see class docstring for the exact
format). `loop=True` restarts at EOF instead of stopping.
`realtime=True` paces replay against the frames' own timestamps rather
than replaying as fast as possible.

### `write_capture_csv(path, frames: List[CSIFrame]) -> None`
Writes a list of frames to disk in the WiSense CSV capture format.

---

## `wisense.core.stream`

### `CSIBuffer(capacity=256)`
Thread-safe ring buffer.
- `push(frame: CSIFrame) -> None`
- `snapshot(n: Optional[int] = None) -> List[CSIFrame]` -- oldest
  first; all frames if `n=None`, else the most recent `n`.
- `clear() -> None`
- `wait_for_frames(min_count, timeout=None) -> bool`
- `register_push_callback(callback) -> None` /
  `unregister_push_callback(callback) -> None`
- `fill_from_source(source, max_frames=None) -> int` -- pulls from an
  already-connected source until exhausted; returns frames pushed.
- `len(buffer)`, `.capacity`

---

## `wisense.core.filters`

- `amplitude_phase(real, imag) -> (amplitude, phase)` -- phase is
  unwrapped along the last axis.
- `hampel_filter(series, window_size=5, n_sigmas=3.0) -> np.ndarray`
- `moving_average_filter(series, window_size=5, mode="centered"|"causal") -> np.ndarray`
- `butterworth_lowpass_filter(series, cutoff_hz, sample_rate_hz, order=4) -> np.ndarray`
  -- zero-phase (`scipy.signal.filtfilt`).
- `select_subcarriers(amplitude_matrix, method="variance_topk"|"mean"|"index", top_k=10, indices=None) -> np.ndarray`
- `frames_to_amplitude_matrix(frames: List[CSIFrame]) -> np.ndarray` --
  shape `(n_frames, n_subcarriers)`; raises `ValueError` on
  inconsistent subcarrier counts.

---

## `wisense.core.calibration`

### `calibrate(source: CSISource, duration_seconds: float = 30.0) -> CalibrationProfile`
Collects a baseline from an already-connected source. Raises
`CalibrationError` if fewer than 10 frames are collected.

### `CalibrationProfile`
Fields: `baseline_mean`, `baseline_std`, `baseline_variance_p95`,
`n_frames`, `n_subcarriers`, `duration_seconds`, `created_at`.
Method: `variance_threshold(n_sigmas=4.0) -> float`.

---

## `wisense.core.events`

### `EventEmitter()`
- `on(event: str, callback) -> None`
- `off(event: str, callback) -> None`
- `emit(event: str, *args, **kwargs) -> None` -- callback exceptions
  are caught and logged, not propagated.

### `StreamWorker(source, buffer, poll_interval=0.0)`
Runs `source` in a background thread, pushing frames into `buffer`.
`start()`, `stop(timeout=5.0)`, `.is_running`.

### `Monitor(source, buffer_capacity=256, window_size=64, poll_interval_seconds=0.5, calibration=None, presence_detector=None, fall_detector=None)`
- `on_presence_change(callback)` -- `callback(PresenceResult)`, fires
  only on a True/False transition.
- `on_fall_detected(callback)` -- `callback(FallEvent)`, fires whenever
  a non-`None` event is returned.
- `start()` / `stop()`

---

## `wisense.presence`

### `PresenceDetector(model_path=None, use_registry_model=None, registry=None, variance_top_k=10)`
- `.detect(frame_window, calibration=None) -> PresenceResult`. Raises
  `InsufficientSignalError` if `len(frame_window) < 4`.
- `.uses_model -> bool`

### `PresenceResult`
Fields: `present: bool`, `confidence: float`, `timestamp: float`.

---

## `wisense.fall`

### `FallDetector(model_path=None, use_registry_model=None, registry=None, variance_top_k=10)`
- `.detect(frame_window, calibration=None) -> Optional[FallEvent]`.
  Raises `InsufficientSignalError` if `len(frame_window) < 8`. Returns
  `None` when no qualifying event is found.

### `FallEvent`
Fields: `detected_at: float`, `confidence: float`, `severity: FallSeverity`.

### `FallSeverity` (enum)
`POSSIBLE`, `LIKELY`, `CONFIRMED` -- see class docstring for the exact
criteria each tier represents.

---

## `wisense.vitals`

### `estimate_breathing_rate(frame_window, sample_rate_hz, subcarrier_top_k=5) -> Optional[float]`
Returns breaths/minute, or `None` (logged reason) if the window is too
short (< ~20s by default) or the respiration-band peak's SNR is too
low to trust.

---

## `wisense.activity`

### `ActivityClassifier(model_path=None, use_registry_model=None, registry=None, variance_top_k=10)`
- `.classify(frame_window, calibration=None) -> ActivityResult`. Raises
  `InsufficientSignalError` if `len(frame_window) < 8`.

### `ActivityResult`
Fields: `label: ActivityLabel`, `confidence: float`, `timestamp: float`.

### `ActivityLabel` (enum)
`STILL`, `WALKING`, `SITTING_DOWN`, `STANDING_UP`, `LYING_DOWN`, `UNKNOWN`.

---

## `wisense.people`

### `count_occupants(frame_window, sample_rate_hz, n_groups=4, calibration=None) -> int`
Raises `InsufficientSignalError` if `len(frame_window) < 16`. See
module docstring for accuracy caveats.

---

## `wisense.models.registry`

### `ModelRegistry(cache_dir=None, base_url=None)`
- `local_path(filename) -> Path`
- `is_cached(filename) -> bool`
- `ensure_downloaded(filename, sha256=None) -> Path` -- raises
  `ModelNotFoundError` / `ModelChecksumError`.
- `load_session(filename, sha256=None)` -- returns an
  `onnxruntime.InferenceSession`, cached per-registry-instance.
- `load_session_from_path(path)` -- bypasses cache/download entirely.

### `get_default_registry() -> ModelRegistry`
Process-wide default instance used by every detector's
`use_registry_model=` parameter.

---

## `wisense.exceptions`

`WiSenseError` (base) with subclasses: `ConnectionError`,
`FrameParseError`, `CalibrationError`, `ModelNotFoundError`
(`ModelChecksumError` extends this), `InsufficientSignalError`,
`BufferError`.
