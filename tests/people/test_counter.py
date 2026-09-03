import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import make_still_frames, make_walking_frames

from wisense.core.connection import CSIFrame
from wisense.exceptions import InsufficientSignalError
from wisense.people.counter import count_occupants, _cluster_count


def test_empty_room_counts_zero():
    frames = make_still_frames(n_frames=40, n_sub=32, noise_std=0.02)
    count = count_occupants(frames, sample_rate_hz=20.0)
    assert count == 0


def test_single_walker_counts_at_least_one():
    frames = make_walking_frames(n_frames=80, n_sub=32, sample_rate_hz=20.0, gait_hz=1.2)
    count = count_occupants(frames, sample_rate_hz=20.0)
    assert count >= 1


def test_too_few_frames_raises():
    with pytest.raises(InsufficientSignalError):
        count_occupants(make_still_frames(n_frames=5), sample_rate_hz=20.0)


def test_rejects_non_positive_sample_rate():
    with pytest.raises(ValueError):
        count_occupants(make_still_frames(n_frames=20), sample_rate_hz=0)


def test_rejects_non_positive_n_groups():
    with pytest.raises(ValueError):
        count_occupants(make_still_frames(n_frames=20), sample_rate_hz=20.0, n_groups=0)


def test_cluster_count_single_cluster():
    assert _cluster_count([1.0, 1.02, 1.05], tolerance=0.15) == 1


def test_cluster_count_two_clusters():
    assert _cluster_count([1.0, 1.02, 2.0, 2.03], tolerance=0.15) == 2


def test_cluster_count_empty():
    assert _cluster_count([], tolerance=0.15) == 0


def test_count_never_negative():
    frames = make_still_frames(n_frames=40, n_sub=16)
    count = count_occupants(frames, sample_rate_hz=20.0)
    assert count >= 0
