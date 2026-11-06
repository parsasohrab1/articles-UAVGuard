"""Laser Doppler vibrometry of a quadrotor (Quanser QDrone 2, hovering): propeller fault detection and diagnosis (REAL, tiny data set).

usage: python research/run_study.py <dir 'Full Dataset' with the 11 xlsx files>
11 files x 5 one-second records (4410 Hz): healthy (2 files = 10 records) and 3 fault types x 3 severities (45 records).
Severity index = the digit after 'SV' in the file name (the README does not state which index is 'low').
(A) detection: one-class Mahalanobis model fitted on healthy records only, conformal threshold (alpha = 0.1) from the other healthy records
    (leave-one-healthy-record-out, so 9 calibration records); (B) diagnosis of fault type with leave-one-file-out cross-validation.
The 5 records of a file come from the same recording: only leave-one-FILE-out evaluation avoids leakage.
"""

import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from uavguard.detect import OneClass, conformal_threshold
from uavguard.features import record_features

warnings.filterwarnings("ignore")
ALPHA = 0.1
TYPES = {"F1": "edge_cut", "F2": "crack", "F3": "surface_unbalance"}


def load(folder: str):
    recs = []
    for f in sorted(glob.glob(f"{folder}/*.xlsx")):
        name = os.path.basename(f)[:-5]
        x = pd.read_excel(f, header=None).to_numpy(float)
        if name.startswith("Healthy"):
            kind, sev = "healthy", 0
        else:
            kind, sev = TYPES[name[:2]], int(name[-1])
        for j in range(x.shape[1]):
            recs.append({"file": name, "kind": kind, "sev": sev, "F": record_features(x[:, j])})
    return recs


def auroc(pos, neg):
    return float(np.mean([[p > n for n in neg] for p in pos]) + 0.0)


def main(folder: str):
    recs = load(folder)
    F = np.array([r["F"] for r in recs])
    kinds = np.array([r["kind"] for r in recs])
    sev = np.array([r["sev"] for r in recs])
    files = np.array([r["file"] for r in recs])
    healthy = np.where(kinds == "healthy")[0]
    faulty = np.where(kinds != "healthy")[0]
    out = {
        "note": "real LDV records; 10 healthy + 45 faulty one-second records; tiny data set",
        "n_healthy": len(healthy),
        "n_faulty": len(faulty),
    }
    # (A) detection: leave-one-healthy-record-out
    fa, det = [], {}
    for h in healthy:
        train = np.setdiff1d(healthy, [h])
        oc = OneClass(ridge=1.0).fit(F[train])
        # threshold from the training healthy records themselves via leave-one-out scores
        loo = []
        for t in train:
            rest = np.setdiff1d(train, [t])
            loo.append(OneClass(1.0).fit(F[rest]).score(F[[t]])[0])
        thr = conformal_threshold(np.array(loo), ALPHA)
        fa.append(oc.score(F[[h]])[0] > thr)
        for f in sorted(set(files[faulty])):
            idx = np.where(files == f)[0]
            det.setdefault(f, []).append(float(np.mean(oc.score(F[idx]) > thr)))
    out["detection"] = {
        "false_alarm_rate_healthy_loo": float(np.mean(fa)),
        "detection_rate_by_file": {f: float(np.mean(v)) for f, v in det.items()},
    }
    # score separation (AUROC) using the model fitted on all healthy except the one scored
    sc_h, sc_f = [], []
    for h in healthy:
        oc = OneClass(1.0).fit(F[np.setdiff1d(healthy, [h])])
        sc_h.append(oc.score(F[[h]])[0])
    oc_all = OneClass(1.0).fit(F[healthy])
    sc_f = oc_all.score(F[faulty])
    out["detection"]["auroc_faulty_vs_healthy_loo"] = auroc(sc_f, np.array(sc_h))
    # (B) diagnosis of fault type, leave-one-file-out (faulty files only)
    uniq = sorted(set(files[faulty]))
    correct, by_sev = [], {1: [], 2: [], 3: []}
    for f in uniq:
        te = np.where(files == f)[0]
        tr = np.setdiff1d(faulty, te)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(
            F[tr], kinds[tr]
        )
        ok = clf.predict(F[te]) == kinds[te]
        correct += ok.tolist()
        by_sev[sev[te][0]] += ok.tolist()
    out["diagnosis_fault_type_leave_one_file_out"] = {
        "accuracy": float(np.mean(correct)),
        "chance": 1 / 3,
        "accuracy_by_severity_index": {str(k): float(np.mean(v)) for k, v in by_sev.items()},
    }
    # leakage illustration: random split of records (not files)
    rng = np.random.default_rng(0)
    accs = []
    for _ in range(50):
        perm = rng.permutation(faulty)
        tr, te = perm[: int(0.7 * len(perm))], perm[int(0.7 * len(perm)) :]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(
            F[tr], kinds[tr]
        )
        accs.append(np.mean(clf.predict(F[te]) == kinds[te]))
    out["diagnosis_random_record_split_leaky"] = float(np.mean(accs))
    Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
