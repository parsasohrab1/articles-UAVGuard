# TRL assessment – UAVGuard

**Claimed level: TRL 4 at most** (public laboratory data, 11 recordings). The study does **not** support a working detector or diagnoser. Not TRL 5: offboard vibrometer, hover only, one platform, no flight logs.

Data: Zenodo 21203159 — 10 healthy and 45 faulty one-second LDV records, 11 files of 5 records. Records from one file come from the same recording.

## Detection (healthy-only one-class model, leave-one-healthy-record-out, alpha = 0.1)
False-alarm rate on the left-out healthy record: 0.10 (1 of 10, as designed). AUROC faulty versus healthy: 0.79.

| Fault type | Severity index | Share of the 5 records flagged |
|---|---|---|
| edge cut | 1 | 0.02 |
| edge cut | 2 | 0.20 |
| edge cut | 3 | 0.46 |
| crack | 1 | 0.00 |
| crack | 2 | 0.18 |
| crack | 3 | 0.12 |
| surface unbalance | 1 | 0.00 |
| surface unbalance | 2 | 0.18 |
| surface unbalance | 3 | 0.20 |

## Diagnosis of fault type (3 classes; leave-one-file-out)
Accuracy 0.31 (chance 0.33); by severity index of the held-out file: 1: 0.00, 2: 0.40, 3: 0.53.
A random split of records (not files) gives 0.67 — an overstatement caused by near-duplicate records of the same file.

## Findings and caveats
- **Healthy-only detection is weak**: at a 10 % false-alarm rate, 0–46 % of the faulty records are flagged (the best case is the edge cut with severity index 3 at 0.46; surface unbalance and crack with index 1 are not detected at all). AUROC 0.79 is moderate and is based on 10 healthy records, 2 recordings.
- **Fault-type diagnosis on an unseen recording does not work** (0.31, chance 0.33). The apparently better 0.67 with a random record split reflects the similarity of records within a file.
- With 10 healthy records the conformal calibration set has 9 elements: the threshold is essentially the largest score and the false-alarm rate is quantised to 0.1.
- Healthy variability across recordings is unknown (two healthy files only); differences between files may be session effects rather than faults.
- The dataset documentation does not define which severity index is low; "severity index" is used as given.
- The mean of the detection rate per file hides heterogeneity: the fault with the highest index is not always the easiest to detect.
