# Roadmap / Not Yet Implemented

Scoped out of the v0.1.x line, listed here rather than left as silent
stubs in the code:

* **Multi-sensor fusion** (combining two or more ESP32 nodes for
  larger-space coverage) -- mentioned in the project's Phase 3 roadmap,
  not implemented.
* **Home Assistant integration package** (`wisense-hass`) -- Phase 4
  roadmap item, not implemented.
* **Pretrained model zoo** -- see the
  [Model Files section of the README](README.md#model-files).
* **Non-ESP32 capture backends** -- only ESP32-CSI-Tool-compatible
  serial capture has real parsing code; other CSI-capable chipsets
  (e.g. Linux `nexmon`/Intel 5300-class tooling) are not implemented,
  since this repository has no way to validate against them without
  the hardware.

Contributions on any of the above are welcome -- open an issue first
to discuss approach before sending a large PR.