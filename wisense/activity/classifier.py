"""Coarse activity classification.

Statistical baseline method
----------------------------
Classifies a frame window into one of a fixed set of coarse activity
labels using three features computed from the aggregate CSI amplitude
signal (mean of the highest-variance subcarriers):

* **Overall variance** -- how much motion energy is present at all.
* **Zero-crossing rate of the derivative** -- how oscillatory/periodic
  the signal is (walking produces a roughly periodic gait signature;
  a single postural change does not).
* **Transient level-shift** -- whether there is a single sharp
  amplitude jump (as in :mod:`wisense.fall.detector`) followed by a
  sustained mean-level change, and whether that level moved up or
  down, which is used as a (heuristic, not physically rigorous) cue to
  distinguish sitting-down-style transitions from standing-up-style
  ones.

This is a coarse heuristic intended as an always-available baseline,
not a validated activity-recognition model -- particularly
``LYING_DOWN`` vs. a low-severity fall share almost the same
statistical signature and cannot be reliably told apart without a
trained model or additional sensors. An optional ONNX inference path
is supported for users who supply a trained classifier.
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

logger = logging.getLogger("wisense.activity.classifier")

_MIN_FRAMES = 8


class ActivityLabel(enum.Enum):
    STILL = "still"
    WALKING = "walking"
    SITTING_DOWN = "sitting_down"
    STANDING_UP = "standing_up"
    LYING_DOWN = "lying_down"
    UNKNOWN = "unknown"


@dataclass
class ActivityResult:
    label: ActivityLabel
    confidence: float
    timestamp: float


class ActivityClassifier:
    """Classifies coarse activity from a window of CSI frames.

    Parameters mirror :class:`~wisense.presence.detector.PresenceDetector`:
    supply ``model_path`` or ``use_registry_model`` to use an optional
    ONNX inference path instead of the statistical heuristic.
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

    def classify(
        self, frame_window: List[CSIFrame], calibration: Optional[CalibrationProfile] = None
    ) -> ActivityResult:
        """Classify ``frame_window`` into one of :class:`ActivityLabel`.

        Raises
        ------
        InsufficientSignalError
            If fewer than 8 frames are provided.
        """
        if len(frame_window) < _MIN_FRAMES:
            raise InsufficientSignalError(
                f"ActivityClassifier.classify requires at least {_MIN_FRAMES} frames, "
                f"got {len(frame_window)}"
            )

        timestamp = frame_window[-1].timestamp

        if self.uses_model:
            return self._classify_with_model(frame_window, timestamp)
        return self._classify_statistical(frame_window, calibration, timestamp)

    def _classify_statistical(
        self,
        frame_window: List[CSIFrame],
        calibration: Optional[CalibrationProfile],
        timestamp: float,
    ) -> ActivityResult:
        matrix = frames_to_amplitude_matrix(frame_window)
        reduced = select_subcarriers(matrix, method="variance_topk", top_k=self.variance_top_k)
        signal = reduced.mean(axis=1)
        diffs = np.diff(signal)
        n = signal.shape[0]

        if calibration is not None:
            still_threshold = calibration.baseline_variance_p95 * 1.2
            spike_threshold = max(float(np.mean(calibration.baseline_std)) * 4.0, 1e-6)
        else:
            still_threshold = 1.0
            spike_threshold = 3.0
            logger.debug(
                "ActivityClassifier running without a CalibrationProfile; "
                "using uncalibrated fallback thresholds."
            )

        overall_variance = float(signal.var())

        # STILL: essentially empty-room-level variance.
        if overall_variance <= still_threshold:
            confidence = float(np.clip(1.0 - overall_variance / max(still_threshold, 1e-9), 0.5, 0.95))
            return ActivityResult(label=ActivityLabel.STILL, confidence=confidence, timestamp=timestamp)

        # Look for a single dominant transient (posture-change signature).
        spike_idx = int(np.argmax(np.abs(diffs)))
        spike_mag = float(np.abs(diffs[spike_idx]))
        pre = signal[: spike_idx + 1]
        post = signal[spike_idx + 1 :]

        if spike_mag > spike_threshold and post.shape[0] >= 3:
            pre_mean = float(pre.mean())
            post_mean = float(post.mean())
            post_var = float(post.var())
            level_shift = post_mean - pre_mean

            if post_var <= still_threshold * 1.5:
                # Sharp transient followed by settling into a new,
                # relatively still level -- consistent with a posture
                # change rather than ongoing locomotion.
                shift_magnitude = abs(level_shift)
                confidence = float(np.clip(0.5 + 0.1 * min(spike_mag / spike_threshold, 3.0), 0.5, 0.9))
                if level_shift < 0:
                    # Signal level dropped and stayed down: consistent
                    # with moving to a lower body profile.
                    if shift_magnitude > spike_threshold * 1.5:
                        return ActivityResult(ActivityLabel.LYING_DOWN, confidence, timestamp)
                    return ActivityResult(ActivityLabel.SITTING_DOWN, confidence, timestamp)
                else:
                    return ActivityResult(ActivityLabel.STANDING_UP, confidence, timestamp)

        # No clean single-transient signature -- check for periodicity
        # (walking gait) via the derivative's zero-crossing rate.
        signs = np.sign(diffs)
        signs[signs == 0] = 1
        zero_crossings = int(np.sum(signs[:-1] != signs[1:]))
        zero_crossing_ratio = zero_crossings / max(n - 2, 1)

        if overall_variance > still_threshold and zero_crossing_ratio > 0.3:
            confidence = float(np.clip(0.4 + 0.5 * zero_crossing_ratio, 0.5, 0.9))
            return ActivityResult(ActivityLabel.WALKING, confidence, timestamp)

        return ActivityResult(ActivityLabel.UNKNOWN, confidence=0.4, timestamp=timestamp)

    def _classify_with_model(self, frame_window: List[CSIFrame], timestamp: float) -> ActivityResult:
        session = self._get_session()
        matrix = frames_to_amplitude_matrix(frame_window).astype(np.float32)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: matrix[np.newaxis, ...]})
        # Documented output contract for WiSense activity ONNX models: a
        # tensor of shape (1, 6) giving softmax probabilities in the
        # fixed order of ActivityLabel below.
        labels_order = [
            ActivityLabel.STILL,
            ActivityLabel.WALKING,
            ActivityLabel.SITTING_DOWN,
            ActivityLabel.STANDING_UP,
            ActivityLabel.LYING_DOWN,
            ActivityLabel.UNKNOWN,
        ]
        probs = np.asarray(outputs[0]).reshape(-1)
        if probs.shape[0] != len(labels_order):
            raise ValueError(
                f"activity ONNX model output must be shape (1, {len(labels_order)}), "
                f"got shape {np.asarray(outputs[0]).shape}"
            )
        top_idx = int(np.argmax(probs))
        return ActivityResult(label=labels_order[top_idx], confidence=float(probs[top_idx]), timestamp=timestamp)
