# WiSense Usage Guide

This guide walks through connecting to a CSI source, calibrating,
running each feature module, wiring up event callbacks, and a complete
end-to-end example script.

## 1. Connecting to a device

WiSense has three `CSISource` implementations, all sharing the same
interface (`connect()`, `disconnect()`, `read_frame()`, `stream()`,
and context-manager support):

```python
from wisense.core import SerialCSISource, NetworkCSISource, FileCSISource

# A physical ESP32 running ESP32-CSI-Tool-compatible firmware:
source = SerialCSISource(port="/dev/ttyUSB0", baudrate=921600)

# A network capture bridge emitting WiSense's JSON-lines protocol:
source = NetworkCSISource(host="0.0.0.0", port=9000, protocol="udp")

# A previously recorded capture file, for offline development/testing:
source = FileCSISource("capture.csv")
```

Use any of them as a context manager, which calls `connect()` /
`disconnect()` for you:

```python
with source:
    frame = source.read_frame()
    print(frame.timestamp, frame.amplitude.shape)
```

Or iterate the whole stream:

```python
with source:
    for frame in source.stream():
        ...  # process each frame as it arrives
```

### Recording a capture for later replay

```python
from wisense.core.connection import write_capture_csv

with SerialCSISource("/dev/ttyUSB0") as source:
    frames = [frame for frame, _ in zip(source.stream(), range(1000))]

write_capture_csv("my_capture.csv", frames)
```

The WiSense CSV capture format is documented in full in the
`FileCSISource` class docstring (`wisense/core/connection.py`).

## 2. Calibrating

Every detector accepts an optional `CalibrationProfile`, which adapts
thresholds to your specific room/hardware instead of relying on fixed
global constants. Run it while the space is empty:

```python
from wisense.core import calibrate

with source:
    profile = calibrate(source, duration_seconds=30)
```

`calibrate()` requires at least 10 frames collected in the given
window, and raises `wisense.exceptions.CalibrationError` if fewer are
seen (check the source is actually connected and producing frames).

Every statistical-baseline detector works *without* a calibration
profile too, falling back to a conservative, uncalibrated constant --
but results will be less reliable for your specific environment. Pass
`calibration=profile` wherever a detector accepts it once you have
one.

## 3. Buffering frames

`CSIBuffer` is a thread-safe rolling window used by every downstream
feature module:

```python
from wisense.core import CSIBuffer

buffer = CSIBuffer(capacity=256)

# Pull frames from an already-connected source until exhausted (good
# for file replay):
with FileCSISource("capture.csv") as source:
    buffer.fill_from_source(source)

# Or push frames one at a time from your own loop (good for live
# sources):
with source:
    for frame in source.stream():
        buffer.push(frame)
        if len(buffer) >= 64:
            window = buffer.snapshot()
            # ... run a detector on `window` ...
```

## 4. Running each feature module

All detectors take a `frame_window: List[CSIFrame]` and an optional
`calibration: CalibrationProfile`.

### Presence

```python
from wisense.presence import PresenceDetector

result = PresenceDetector().detect(window, calibration=profile)
print(result.present, result.confidence, result.timestamp)
```

### Fall detection

```python
from wisense.fall import FallDetector

event = FallDetector().detect(window, calibration=profile)
if event is not None:
    print(event.severity, event.confidence, event.detected_at)
```

`detect()` returns `None` when no qualifying event is found -- this is
the expected common case, not an error.

### Breathing rate

```python
from wisense.vitals import estimate_breathing_rate

rate = estimate_breathing_rate(window, sample_rate_hz=20.0)
if rate is None:
    print("no confident estimate (window too short or signal too weak)")
else:
    print(f"{rate:.1f} breaths/min")
```

Only call this on a window where the subject is relatively still (e.g.
gate it behind a `PresenceDetector` + stillness check) -- ordinary
movement will swamp the much smaller respiration signal.

### Activity classification

```python
from wisense.activity import ActivityClassifier

result = ActivityClassifier().classify(window, calibration=profile)
print(result.label, result.confidence)
```

### Occupant counting

```python
from wisense.people import count_occupants

n = count_occupants(window, sample_rate_hz=20.0, calibration=profile)
print(f"~{n} occupant(s)")
```

Read the caveats in `wisense/people/counter.py`'s module docstring --
this is a genuinely hard sub-problem and accuracy drops quickly beyond
2-3 simultaneous occupants.

### Using an ONNX model instead of the statistical baseline

`PresenceDetector`, `FallDetector`, and `ActivityClassifier` all accept
either `model_path` (a local `.onnx` file) or `use_registry_model` (a
filename managed by a `ModelRegistry`, downloaded/cached/checksummed
automatically):

```python
detector = PresenceDetector(model_path="/path/to/presence_model.onnx")
# or:
detector = PresenceDetector(use_registry_model="presence_v1.onnx")
```

See the [Model Files section of the README](../README.md#model-files)
-- no model weights ship with this repository.

## 5. Event callbacks

For live monitoring, `wisense.core.events.Monitor` wires a source to
the presence and fall detectors and exposes `on_presence_change` /
`on_fall_detected` hooks, polling in the background:

```python
from wisense.core.events import Monitor
from wisense.presence import PresenceDetector
from wisense.fall import FallDetector

monitor = Monitor(
    source,
    calibration=profile,
    presence_detector=PresenceDetector(),
    fall_detector=FallDetector(),
    poll_interval_seconds=0.5,
)

monitor.on_presence_change(lambda result: print("presence:", result.present))
monitor.on_fall_detected(lambda event: print("FALL:", event.severity))

monitor.start()   # connects the source and starts background threads
...
monitor.stop()
```

`Monitor` only fires `presence_change` on an actual state transition
(not on every poll), and fires `fall_detected` whenever the fall
detector returns a non-`None` event. Callback exceptions are caught
and logged, so a bug in your callback won't crash the monitoring loop.

For lower-level control, `EventEmitter` (the same primitive `Monitor`
uses internally) and `StreamWorker` (runs a source in a background
thread, pushing into a `CSIBuffer`) are both available directly from
`wisense.core.events`.

## 6. Complete end-to-end example

See [`examples/presence_demo.py`](../examples/presence_demo.py) for a
runnable script combining all of the above against a bundled synthetic
capture file (no hardware required), and
[`examples/live_esp32_demo.py`](../examples/live_esp32_demo.py) for
the same pattern applied to a real ESP32 over serial.

## Logging

WiSense never configures Python's root logger. Attach a handler to the
`"wisense"` namespace (or any submodule logger, e.g.
`"wisense.fall.detector"`) to see its log output:

```python
import logging
logging.getLogger("wisense").addHandler(logging.StreamHandler())
logging.getLogger("wisense").setLevel(logging.INFO)
```
