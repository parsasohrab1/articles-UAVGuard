from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis

FS = 4410.0
BANDS = [
    (0, 50),
    (50, 100),
    (100, 200),
    (200, 400),
    (400, 800),
    (800, 1200),
    (1200, 1700),
    (1700, 2205),
]


def record_features(x: np.ndarray, fs: float = FS) -> np.ndarray:
    """Features of one 1-s vibration record: log RMS, kurtosis, crest factor, spectral centroid (Hz),
    dominant frequency (Hz) and the log energy in eight frequency bands (0-2205 Hz)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    rms = np.sqrt(np.mean(x**2))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    centroid = float((f * spec).sum() / spec.sum())
    dom = float(f[1:][np.argmax(spec[1:])])
    bands = [np.log10(spec[(f >= lo) & (f < hi)].sum() + 1e-18) for lo, hi in BANDS]
    return np.array(
        [
            np.log10(rms + 1e-12),
            kurtosis(x, fisher=False),
            np.max(np.abs(x)) / (rms + 1e-12),
            centroid,
            dom,
            *bands,
        ]
    )


def feature_names() -> list[str]:
    return ["log_rms", "kurtosis", "crest", "centroid_hz", "dominant_hz"] + [
        f"band_{lo}_{hi}" for lo, hi in BANDS
    ]
