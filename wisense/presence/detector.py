"""Binary presence detection.

Two detection paths are supported, chosen automatically based on
whether a model is configured:

1. **Statistical baseline (default, always available).** Human
   presence perturbs the WiFi multipath environment, which shows up as
   elevated temporal variance in CSI amplitude compared to an empty
   room. This is a well-established signal-processing approach in the
   WiFi-sensing literature (variance-of-amplitude thresholding). We
   compute the mean temporal variance across the most motion-sensitive
   subcarriers (via :func:`wisense.core.filters.select_subcarriers`)
   and compare it against a threshold derived from a
   :class:`~wisense.core.calibration.CalibrationProfile`, if one is
   supplied, or a conservative fixed fallback threshold otherwise
   (documented as less reliable -- see ``_FALLBACK_VARIANCE_THRESHOLD``).
2. **ONNX model inference (optional upgrade).** If a ``model_path`` (or
   registry-managed model filename) is provided, frames are instead fed
   through an ``onnxruntime.InferenceSession``. No model ships with
   WiSense -- see :mod:`wisense.models.registry`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from wisense.core.calibration import CalibrationProfile
from wisense.core.connection import CSIFrame
from wisense.core.filters import frames_to_amplitude_matrix, select_subcarriers
from wisense.exceptions import InsufficientSignalError
from wisense.models.registry import ModelRegistry, get_default_registry

logger = logging.getLogger("wisense.presence.detector")

_MIN_FRAMES = 4
# Used only when no CalibrationProfile is supplied. This is a
# conservative, un-tuned constant and is documented as such -- results
# without calibration should be treated as a rough default, not a
# reliable production threshold. Units match amplitude^2 (variance of
# raw amplitude units, which themselves depend on the capture
# hardware's scaling).
_FALLBACK_VARIANCE_THRESHOLD = 4.0


@dataclass
class PresenceResult:
    """Result of a single presence detection call.

    Attributes
    ----------
    present: whether a human presence was detected in the window.
    confidence: value in ``[0, 1]``; higher means more confident in
        ``present``'s value (not necessarily "confidence someone is
        present" -- e.g. confidence is also high when clearly empty).
    timestamp: timestamp of the most recent frame in the analyzed window.
    """

    present: bool
    confidence: float
    timestamp: float


class PresenceDetector:
    """Detects binary human presence from a window of CSI frames.

    Parameters
    ----------
    model_path:
        Path to a local ``.onnx`` model file. If given, ONNX inference
        is used instead of the statistical baseline.
    use_registry_model:
        Name of a model file to fetch via ``registry`` (download/cache
        managed automatically). Mutually exclusive with ``model_path``.
    registry:
        :class:`~wisense.models.registry.ModelRegistry` to use for
        ``use_registry_model``. Defaults to the process-wide default
        registry.
    variance_top_k:
        Number of highest-variance subcarriers to aggregate for the
        statistical baseline (ignored when using an ONNX model).

    Examples
    --------
    >>> import numpy as np
    >>> from wisense.core.connection import CSIFrame
    >>> from wisense.presence.detector import PresenceDetector
    >>> quiet = [CSIFrame(timestamp=float(i), amplitude=np.full(8, 10.0)) for i in range(20)]
    >>> PresenceDetector().detect(quiet).present
    False
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        use_registry_model: Optional[str] = None,
        registry: Optional[ModelRegistry] = None,
        variance_top_k: int = 10,
    ) -> None:
        if model_path is not None and use_registry_model is not None:
            raise ValueError("specify at most one of model_path, use_registry_model")
        self.model_path = Path(model_path) if model_path is not None else None
        self.use_registry_model = use_registry_model
        self.registry = registry or get_default_registry()
        self.variance_top_k = variance_top_k
        self._session = None

    @property
    def uses_model(self) -> bool:
        return self.model_path is not None or self.use_registry_model is not None

    def _get_session(self):
        if self._session is not None:
            return self._session
        if self.model_path is not None:
            self._session = self.registry.load_session_from_path(self.model_path)
        elif self.use_registry_model is not None:
            self._session = self.registry.load_session(self.use_registry_model)
        return self._session

    def detect(
        self, frame_window: List[CSIFrame], calibration: Optional[CalibrationProfile] = None
    ) -> PresenceResult:
        """Run presence detection over ``frame_window``.

        Raises
        ------
        InsufficientSignalError
            If fewer than 4 frames are provided (not enough to compute
            a meaningful temporal variance).
        """
        if len(frame_window) < _MIN_FRAMES:
            raise InsufficientSignalError(
                f"PresenceDetector.detect requires at least {_MIN_FRAMES} frames, "
                f"got {len(frame_window)}"
            )

        timestamp = frame_window[-1].timestamp

        if self.uses_model:
            return self._detect_with_model(frame_window, timestamp)
        return self._detect_statistical(frame_window, calibration, timestamp)

    def _detect_statistical(
        self,
        frame_window: List[CSIFrame],
        calibration: Optional[CalibrationProfile],
        timestamp: float,
    ) -> PresenceResult:
        matrix = frames_to_amplitude_matrix(frame_window)  # (n_frames, n_sub)
        reduced = select_subcarriers(matrix, method="variance_topk", top_k=self.variance_top_k)
        metric = float(reduced.var(axis=0).mean())

        if calibration is not None:
            threshold = calibration.variance_threshold()
        else:
            threshold = _FALLBACK_VARIANCE_THRESHOLD
            logger.debug(
                "PresenceDetector running without a CalibrationProfile; using "
                "fallback threshold %.3f, which is uncalibrated for this "
                "specific environment/hardware.",
                threshold,
            )

        present = metric > threshold
        # Confidence: logistic squashing of how far metric sits from the
        # threshold, scaled by the threshold itself so it's roughly
        # environment-invariant. Saturates towards 0/1 as the metric
        # moves further from the decision boundary.
        scale = max(threshold, 1e-6) / 2.0
        z = (metric - threshold) / scale
        confidence = 1.0 / (1.0 + np.exp(-abs(z)))
        confidence = float(np.clip(confidence, 0.5, 0.999))

        return PresenceResult(present=present, confidence=confidence, timestamp=timestamp)

    def _detect_with_model(self, frame_window: List[CSIFrame], timestamp: float) -> PresenceResult:
        session = self._get_session()
        matrix = frames_to_amplitude_matrix(frame_window).astype(np.float32)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: matrix[np.newaxis, ...]})
        # Documented output contract for WiSense presence ONNX models:
        # a single output tensor of shape (1, 2) giving
        # [P(absent), P(present)] softmax probabilities. Model authors
        # targeting WiSense should conform to this contract.
        probs = np.asarray(outputs[0]).reshape(-1)
        if probs.shape[0] != 2:
            raise ValueError(
                "presence ONNX model output must be shape (1, 2) "
                f"[P(absent), P(present)], got shape {np.asarray(outputs[0]).shape}"
            )
        present = bool(probs[1] > probs[0])
        confidence = float(np.max(probs))
        return PresenceResult(present=present, confidence=confidence, timestamp=timestamp)
