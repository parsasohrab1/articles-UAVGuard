# Software Requirements Specification – UAVGuard

- SR-1 `features.record_features`: log RMS, kurtosis, crest factor, spectral centroid, dominant frequency and log energy in eight bands (0–2205 Hz) of a one-second record (Hann window); a unit test checks the dominant-frequency estimate.
- SR-2 `detect.OneClass`: standardised features, ridge-regularised Mahalanobis distance to the healthy records; `conformal_threshold`: rank-based threshold.
- SR-3 `research/run_study.py`: detection with a threshold from leave-one-out scores of the remaining healthy records (alpha 0.1), repeated leaving each healthy record out; fault-type diagnosis (logistic regression) with leave-one-file-out cross-validation; a leaky random record split for comparison.
- SR-4 Severity index = digit after "SV" in the file name; the data description does not say which index is the low severity.
- SR-5 Three unit tests.
