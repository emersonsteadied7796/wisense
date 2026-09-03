# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
