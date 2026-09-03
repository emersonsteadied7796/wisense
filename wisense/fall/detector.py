"""Fall event detection.

Statistical baseline method
----------------------------
Implements the sudden-amplitude-change-then-stillness signature widely
described in WiFi fall-detection literature (e.g. Wang et al.,
"E-eyes"/"WiFall"-style approaches, and subsequent CSI fall-detection
work): a fall produces a large, rapid perturbation in CSI amplitude
(the body moving quickly through the multipath environment) followed
by an unusually still period (the person on the ground, not moving),
which is what distinguishes a fall from ordinary fast movement like
walking briskly through the room (which does not end in stillness) or
sitting down normally (smaller amplitude change).

Concretely, for a frame window:

1. Reduce to a single aggregate amplitude series (mean of the
   highest-variance subcarriers, via
   :func:`wisense.core.filters.select_subcarriers`).
2. Compute the frame-to-frame difference series and find its largest
   absolute jump ("the spike").
3. If the spike magnitude clears a calibration-derived (or fallback)
   threshold, examine the segment *after* the spike for stillness
   (low variance).
4. Combine spike magnitude and post-spike stillness into a confidence
   score and a coarse severity level (``POSSIBLE`` / ``LIKELY`` /
   ``CONFIRMED``), documented per-level below.

This is a heuristic, not a validated clinical fall-detection system --
see the People/Fall accuracy caveats in the README. An optional ONNX
inference path is supported for users who supply a trained model.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from wisense.core.calibration import CalibrationProfile
from wisense.core.connection import CSIFrame
from wisense.core.filters import frames_to_amplitude_matrix, select_subcarriers
from wisense.exceptions import InsufficientSignalError
from wisense.models.registry import ModelRegistry, get_default_registry

logger = logging.getLogger("wisense.fall.detector")

_MIN_FRAMES = 8
_MIN_POST_SPIKE_FRAMES = 3
_FALLBACK_SPIKE_THRESHOLD = 6.0  # amplitude units; uncalibrated fallback, see PresenceDetector note
_FALLBACK_STILLNESS_VARIANCE = 2.0


class FallSeverity(enum.Enum):
    """Coarse confidence tiers for a detected fall event.

    * ``POSSIBLE`` -- a qualifying amplitude spike was found, but either
      there weren't enough post-spike frames to assess stillness, or
      the post-spike segment wasn't still enough to be confident this
      wasn't just fast ordinary movement.
    * ``LIKELY`` -- spike followed by a moderately still segment.
    * ``CONFIRMED`` -- spike followed by a clearly still segment
      spanning the rest of the analyzed window -- the strongest
      statistical-baseline signature this detector can report.
    """

    POSSIBLE = "possible"
    LIKELY = "likely"
    CONFIRMED = "confirmed"


@dataclass
class FallEvent:
    """A detected candidate fall event.

    Attributes
    ----------
    detected_at: timestamp of the frame at which the triggering
        amplitude spike occurred.
    confidence: value in ``[0, 1]``.
    severity: coarse tier, see :class:`FallSeverity`.
    """

    detected_at: float
    confidence: float
    severity: FallSeverity


class FallDetector:
    """Detects candidate fall events from a window of CSI frames.

    Parameters mirror :class:`~wisense.presence.detector.PresenceDetector`:
    supply ``model_path`` or ``use_registry_model`` to use an optional
    ONNX inference path instead of the statistical baseline.
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
    ) -> Optional[FallEvent]:
        """Analyze ``frame_window`` for a candidate fall signature.
        Returns ``None`` if no qualifying event is found (this is the
        expected, common case -- it is not an error).

        Raises
        ------
        InsufficientSignalError
            If fewer than 8 frames are provided.
        """
        if len(frame_window) < _MIN_FRAMES:
            raise InsufficientSignalError(
                f"FallDetector.detect requires at least {_MIN_FRAMES} frames, "
                f"got {len(frame_window)}"
            )

        if self.uses_model:
            return self._detect_with_model(frame_window)
        return self._detect_statistical(frame_window, calibration)

    def _detect_statistical(
        self, frame_window: List[CSIFrame], calibration: Optional[CalibrationProfile]
    ) -> Optional[FallEvent]:
        matrix = frames_to_amplitude_matrix(frame_window)
        reduced = select_subcarriers(matrix, method="variance_topk", top_k=self.variance_top_k)
        signal = reduced.mean(axis=1)  # (n_frames,) aggregate amplitude series
        diffs = np.diff(signal)

        if calibration is not None:
            sigma = float(np.mean(calibration.baseline_std))
            spike_threshold = max(sigma * 5.0, 1e-6)
            stillness_threshold = calibration.baseline_variance_p95 * 1.5
        else:
            spike_threshold = _FALLBACK_SPIKE_THRESHOLD
            stillness_threshold = _FALLBACK_STILLNESS_VARIANCE
            logger.debug(
                "FallDetector running without a CalibrationProfile; using "
                "uncalibrated fallback thresholds (spike=%.3f, stillness=%.3f).",
                spike_threshold,
                stillness_threshold,
            )

        spike_idx = int(np.argmax(np.abs(diffs)))
        spike_magnitude = float(np.abs(diffs[spike_idx]))

        if spike_magnitude < spike_threshold:
            return None

        post_spike = signal[spike_idx + 1 :]
        n_post = post_spike.shape[0]
        detected_at = frame_window[spike_idx + 1].timestamp

        spike_ratio = spike_magnitude / spike_threshold

        if n_post < _MIN_POST_SPIKE_FRAMES:
            confidence = float(np.clip(0.4 + 0.1 * min(spike_ratio, 2.0), 0.4, 0.65))
            return FallEvent(detected_at=detected_at, confidence=confidence, severity=FallSeverity.POSSIBLE)

        post_variance = float(np.var(post_spike))
        stillness_ratio = stillness_threshold / max(post_variance, 1e-9)  # >1 means "stiller than threshold"

        if post_variance <= stillness_threshold:
            severity = FallSeverity.CONFIRMED
            confidence = float(np.clip(0.75 + 0.05 * min(stillness_ratio, 5.0), 0.75, 0.99))
        elif post_variance <= stillness_threshold * 2.5:
            severity = FallSeverity.LIKELY
            confidence = float(np.clip(0.55 + 0.05 * min(spike_ratio, 3.0), 0.55, 0.75))
        else:
            # Spike happened but movement continued afterwards -- more
            # consistent with ordinary fast activity than a fall.
            severity = FallSeverity.POSSIBLE
            confidence = float(np.clip(0.4 + 0.05 * min(spike_ratio, 2.0), 0.4, 0.55))

        return FallEvent(detected_at=detected_at, confidence=confidence, severity=severity)

    def _detect_with_model(self, frame_window: List[CSIFrame]) -> Optional[FallEvent]:
        session = self._get_session()
        matrix = frames_to_amplitude_matrix(frame_window).astype(np.float32)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: matrix[np.newaxis, ...]})
        # Documented output contract for WiSense fall ONNX models: a
        # tensor of shape (1, 4) giving softmax probabilities over
        # [no_fall, possible, likely, confirmed].
        probs = np.asarray(outputs[0]).reshape(-1)
        if probs.shape[0] != 4:
            raise ValueError(
                "fall ONNX model output must be shape (1, 4) "
                "[no_fall, possible, likely, confirmed], got shape "
                f"{np.asarray(outputs[0]).shape}"
            )
        top_idx = int(np.argmax(probs))
        if top_idx == 0:
            return None
        severity = [None, FallSeverity.POSSIBLE, FallSeverity.LIKELY, FallSeverity.CONFIRMED][top_idx]
        confidence = float(probs[top_idx])
        detected_at = frame_window[-1].timestamp
        assert severity is not None
        return FallEvent(detected_at=detected_at, confidence=confidence, severity=severity)
