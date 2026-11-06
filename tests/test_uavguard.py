import numpy as np

from uavguard.detect import OneClass, conformal_threshold
from uavguard.features import FS, feature_names, record_features


def _sig(freq, amp=1.0, noise=0.1, seed=0, n=4410):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    return amp * np.sin(2 * np.pi * freq * t) + noise * rng.standard_normal(n)


def test_dominant_frequency_and_feature_length():
    f = record_features(_sig(300.0))
    assert abs(f[4] - 300.0) < 2.0 and len(f) == len(feature_names())


def test_extra_tone_raises_one_class_score():
    healthy = np.array([record_features(_sig(120.0, seed=s)) for s in range(30)])
    oc = OneClass().fit(healthy[:20])
    thr = conformal_threshold(oc.score(healthy[20:]), 0.1)
    bad = np.array(
        [
            record_features(
                _sig(120.0, seed=s) + 0.8 * np.sin(2 * np.pi * 600 * np.arange(4410) / FS)
            )
            for s in range(10)
        ]
    )
    assert (oc.score(bad) > thr).mean() > 0.9


def test_conformal_threshold_is_rank_statistic():
    assert conformal_threshold(np.arange(1.0, 10.0), 0.1) == 9.0
    assert conformal_threshold(np.arange(1.0, 6.0), 0.01) == float("inf")
