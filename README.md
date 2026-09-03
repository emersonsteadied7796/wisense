# WiSense

WiFi CSI (Channel State Information) human sensing for Python --
presence, falls, breathing rate, coarse activity, and occupant count,
without a camera, without a cloud API, and without PyTorch/CUDA at
runtime.

```python
from wisense.core import FileCSISource, CSIBuffer, calibrate
from wisense.presence import PresenceDetector

with FileCSISource("capture.csv", realtime=True) as source:
    profile = calibrate(source, duration_seconds=30)  # empty-room baseline

with FileCSISource("capture.csv") as source:
    buffer = CSIBuffer(capacity=256)
    buffer.fill_from_source(source, max_frames=64)

result = PresenceDetector().detect(buffer.snapshot(), calibration=profile)
print(result.present, result.confidence)
```

## Status

**v0.1.0, alpha.** The statistical (non-ML) detection path for every
feature module below is fully implemented, tested, and works without
any model file. The optional ONNX inference upgrade path is fully
implemented against `onnxruntime.InferenceSession`, but **no trained
model files ship with this repository** -- see
[Model Files](#model-files) below.

## Install

```bash
pip install -e .
```

Requires Python 3.9+. Core dependencies: `numpy`, `scipy`,
`onnxruntime`, `pyserial`. Install extras for development or
visualization tooling:

```bash
pip install -e ".[dev]"   # pytest, ruff, mypy, black
pip install -e ".[viz]"   # matplotlib
```

## Quickstart

See [`examples/presence_demo.py`](examples/presence_demo.py) for a
complete, runnable, **hardware-free** walkthrough (it replays a small
synthetic capture bundled in `tests/fixtures/`). Run it with:

```bash
python examples/presence_demo.py
```

For a real device, see
[`examples/live_esp32_demo.py`](examples/live_esp32_demo.py), which
**requires a physical ESP32** flashed with
[ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool)-compatible
firmware, connected over USB serial.

The full walkthrough -- connecting, calibrating, every feature module,
event callbacks -- is in [`docs/usage.md`](docs/usage.md). API
reference is in [`docs/api.md`](docs/api.md).

## Feature list

| Module | What it does | Statistical baseline | ONNX upgrade path |
|---|---|---|---|
| `wisense.presence` | Binary presence detection | Variance-of-amplitude thresholding against a calibration baseline | Yes |
| `wisense.fall` | Fall event detection with severity/confidence | Sudden-amplitude-drop-then-stillness signature | Yes |
| `wisense.vitals` | Passive breathing-rate estimation | FFT peak detection in the 0.15-0.5 Hz respiration band | No (statistical-only; see docstring) |
| `wisense.activity` | Coarse activity classification | Variance + periodicity + transient-level-shift heuristics | Yes |
| `wisense.people` | Occupant count estimation | Multipath/frequency-diversity clustering | No (statistical-only; see docstring) |
| `wisense.core` | Connection, buffering, filtering, calibration, event callbacks | -- | -- |
| `wisense.models` | Model download/cache/checksum/load management | -- | -- |

Every detection call returns a structured `dataclass` (never a raw
image or unprocessed signal) -- see `docs/api.md` for each result
type's fields.

## Architecture

```
Capture Layer (Linux host or ESP32 device)
  SerialCSISource / NetworkCSISource / FileCSISource
                    |
                    v
         wisense.core
  CSIBuffer (ring buffer) -> calibration -> filters
                    |
                    v
     Feature modules (presence / fall / vitals /
       activity / people) -- statistical baseline,
       or ONNX Runtime inference if a model is configured
                    |
                    v
   Structured output (dataclasses) + event callbacks
       (on_presence_change / on_fall_detected via
        wisense.core.events.Monitor)
```

## Supported hardware / capture sources

* **`SerialCSISource`** -- ESP32 running
  [ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool)-compatible
  firmware, over USB serial. This is the only capture target this
  repository has parsing code written and tested against.
* **`NetworkCSISource`** -- UDP or TCP, using a small newline-delimited
  JSON protocol WiSense defines itself (documented in the class
  docstring) -- there is no single industry-standard network CSI wire
  format, so bridging a different capture pipeline (e.g. a Linux host
  with a CSI-capable driver) to WiSense means emitting frames in this
  format.
* **`FileCSISource`** -- replays a recorded capture from disk in the
  WiSense CSV format (documented in the class docstring, and produced
  by `wisense.core.connection.write_capture_csv`). Works fully offline,
  no hardware needed -- this is what the tests and
  `examples/presence_demo.py` use.

## Model Files

WiSense ships **no pretrained `.onnx` model weights**. This is a
deliberate design decision: it keeps the pip install small, and every
feature module works fully without any model via its statistical
baseline method (see the feature table above).

The ONNX inference path (`model_path=` / `use_registry_model=` on each
detector/classifier) is fully implemented against
`onnxruntime.InferenceSession`, including download/cache/checksum
management in `wisense.models.registry.ModelRegistry`. But **training
and publishing model weights is out of scope for this repository** --
`ModelRegistry`'s default download URL
(`DEFAULT_MODEL_BASE_URL` in `wisense/models/registry.py`) is an
intentional, clearly-marked placeholder that will not resolve. If you
train your own model:

* Point `PresenceDetector(model_path="/path/to/your/model.onnx")` (or
  the equivalent on `FallDetector` / `ActivityClassifier`) directly at
  a local file, **or**
* Host your own `.onnx` files somewhere and configure
  `ModelRegistry(base_url="https://your-host/...")`, then use
  `use_registry_model="yourmodel.onnx"`.

Each detector's module docstring documents the exact input/output
tensor contract your model needs to conform to (e.g. presence models
must output `[P(absent), P(present)]`).

No accuracy numbers are claimed anywhere in this repository for the
ONNX path, because no benchmarked model exists yet to cite one for.
The statistical baseline's behavior is exercised by the test suite
(see `tests/`) but has likewise not been benchmarked against a labeled
real-world dataset -- treat its outputs as a reasonable engineering
default, not a validated accuracy claim, and calibrate
(`wisense.core.calibrate`) for your specific environment before
relying on it.

## Not Yet Implemented

Scoped out of this v0.1.0 pass, listed here rather than left as silent
stubs:

* **Multi-sensor fusion** (combining two or more ESP32 nodes for
  larger-space coverage) -- mentioned in the project's Phase 3 roadmap,
  not implemented.
* **Home Assistant integration package** (`wisense-hass`) -- Phase 4
  roadmap item, not implemented.
* **Pretrained model zoo** -- see [Model Files](#model-files) above.
* **Non-ESP32 capture backends** -- only ESP32-CSI-Tool-compatible
  serial capture has real parsing code; other CSI-capable chipsets
  (e.g. Linux `nexmon`/Intel 5300-class tooling) are not implemented,
  since this repository has no way to validate against them without
  the hardware.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Every module has a `logging.getLogger("wisense.<module>")` logger;
WiSense never configures Python's root logger, so attach your own
handler to see output:

```python
import logging
logging.getLogger("wisense").addHandler(logging.StreamHandler())
logging.getLogger("wisense").setLevel(logging.INFO)
```

## License

MIT -- see [LICENSE](LICENSE).
