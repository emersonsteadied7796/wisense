"""WiSense: WiFi CSI human sensing for Python.

Detects human presence, falls, breathing rate, coarse activity, and
occupant count from a stream of WiFi Channel State Information (CSI)
data -- no camera, no cloud API, no PyTorch/CUDA required at runtime.

Quickstart
----------
>>> from wisense.core import FileCSISource, CSIBuffer
>>> from wisense.presence import PresenceDetector
>>> with FileCSISource("capture.csv") as source:  # doctest: +SKIP
...     buffer = CSIBuffer(capacity=256)
...     buffer.fill_from_source(source, max_frames=64)
...     result = PresenceDetector().detect(buffer.snapshot())
...     print(result.present, result.confidence)

See ``examples/presence_demo.py`` in the repository for a complete,
runnable, hardware-free walkthrough, and ``docs/usage.md`` for the full
guide covering calibration, every feature module, and event callbacks.

This library never configures Python's root logger; attach a handler
to the ``"wisense"`` logger namespace (or any of its children, e.g.
``"wisense.fall.detector"``) if you want to see WiSense's log output::

    import logging
    logging.getLogger("wisense").addHandler(logging.StreamHandler())
    logging.getLogger("wisense").setLevel(logging.INFO)
"""

from wisense.__version__ import __version__

__all__ = ["__version__"]
