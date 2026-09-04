# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-09-04

### Fixed

- `wisense.core.calibration.calibrate`: fixed a bug where the idle-exit
  condition measured time since the start of the whole calibration
  window instead of time since the last frame was actually received --
  on a real live source, any single momentary gap after the first
  500ms (normal serial/network jitter) would end calibration early,
  often well under the requested duration. Now takes an
  `idle_timeout_seconds` parameter (default 3.0) measured from the
  last received frame.
- `SerialCSISource`: `local_timestamp` from the firmware (microseconds
  since boot) was being parsed and then discarded in favor of
  `time.time()` at Python receipt time, which introduces jitter from
  USB-to-UART burst delivery and corrupts FFT-based analysis (e.g.
  breathing rate). Now establishes a wall-clock offset from the first
  frame and applies the firmware's own clock to every frame after
  that.

### Added

- `SerialCSISource(target_mac=...)`: optional filtering to a single
  transmitter's MAC address. Without this, a CSI-sniffing ESP32 can
  report frames from every nearby WiFi transmitter it overhears, and
  mixing CSI from different transmitters into one time series produces
  meaningless amplitude "variance" driven by which device is
  transmitting, not by motion.
- `wisense.core.filters.filter_majority_subcarrier_count`: drops
  frames whose subcarrier count doesn't match the majority in a
  window, so a stray mixed-bandwidth packet no longer crashes the
  whole detection cycle via `frames_to_amplitude_matrix`'s strict
  shape check. Now called by every detector before building a matrix.
- `wisense.core.filters.normalize_frame_amplitude`: partial mitigation
  for AGC (automatic gain control) gain-state changes between packets,
  which can otherwise look like the same "sudden amplitude change"
  signature used for fall/activity detection. Normalizes against a
  *smoothed trend* of per-frame magnitude rather than each frame's own
  instantaneous value, specifically so short motion-driven transients
  aren't also cancelled out. Documented as a partial mitigation, not a
  full AGC correction -- see its docstring for why a full correction
  isn't possible from ESP32-CSI-Tool's packet format alone. Wired into
  presence, fall, activity, people, and breathing-rate modules.
- `CalibrationProfile` and every statistical-baseline detector now
  apply the same `filter_majority_subcarrier_count` +
  `normalize_frame_amplitude` preprocessing consistently, so
  calibration-time thresholds are computed in the same normalized
  space detection-time comparisons use.

### Changed

- Strengthened the hardware-reliability caveats in
  `wisense/vitals/breathing.py` and `wisense/people/counter.py`
  docstrings, and added an explicit note to `CSIFrame.phase` that no
  statistical-baseline detector currently reads it (raw ESP32 phase is
  not sanitized/calibrated enough to threshold on directly).

These fixes were prompted by a detailed, credited third-party code
review conducted before any real-hardware testing had occurred --
several were genuine bugs in the original implementation, confirmed by
tracing the code; see the review discussion for full detail.

## [0.1.1] - 2026-09-04

### Changed

- README: fixed relative links (`LICENSE`, `docs/`, `examples/`) that
  rendered as broken 404s on the PyPI project page -- PyPI renders the
  README standalone, so links now point at absolute GitHub URLs.
- README: moved the "Not Yet Implemented" section out to a dedicated
  `ROADMAP.md`, replaced with a one-line pointer.

## [0.1.0] - 2026-09-02

### Added

- `wisense.core.connection`: `CSIFrame`, `CSISource` abstract base
  class, `SerialCSISource` (ESP32-CSI-Tool-compatible serial parsing),
  `NetworkCSISource` (UDP/TCP, WiSense JSON-lines protocol),
  `FileCSISource` (WiSense CSV capture replay), `write_capture_csv`.
- `wisense.core.stream`: thread-safe `CSIBuffer` ring buffer.
- `wisense.core.filters`: `amplitude_phase`, `hampel_filter`,
  `moving_average_filter`, `butterworth_lowpass_filter`,
  `select_subcarriers`, `frames_to_amplitude_matrix`.
- `wisense.core.calibration`: `calibrate`, `CalibrationProfile`.
- `wisense.core.events`: `EventEmitter`, `StreamWorker`, `Monitor`
  (with `on_presence_change` / `on_fall_detected` hooks).
- `wisense.presence`: `PresenceDetector`, `PresenceResult` -- variance
  thresholding baseline + optional ONNX inference path.
- `wisense.fall`: `FallDetector`, `FallEvent`, `FallSeverity` --
  spike-then-stillness baseline + optional ONNX inference path.
- `wisense.vitals`: `estimate_breathing_rate` -- FFT-based respiration
  band peak detection.
- `wisense.activity`: `ActivityClassifier`, `ActivityLabel`,
  `ActivityResult` -- heuristic baseline + optional ONNX inference
  path.
- `wisense.people`: `count_occupants` -- frequency-diversity
  clustering estimate.
- `wisense.models.registry`: `ModelRegistry` -- local cache, download,
  SHA-256 checksum verification, and `onnxruntime.InferenceSession`
  loading for optional ONNX models.
- `wisense.exceptions`: full exception hierarchy.
- Test suite (`tests/`) covering every module above, including error
  paths, plus an end-to-end integration test replaying a bundled
  synthetic fixture capture.
- `examples/presence_demo.py` (hardware-free) and
  `examples/live_esp32_demo.py` (requires a physical ESP32).
- `docs/usage.md` and `docs/api.md`.

### Known limitations

See the README's "Not Yet Implemented" section and each module's
docstring for accuracy caveats -- in particular `wisense.people` and
the activity `LYING_DOWN` label are documented as hard, low-confidence
sub-problems for the statistical baseline.
