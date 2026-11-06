# articles-UAVGuard

Vibration-based propeller fault detection and diagnosis for a quadrotor, on a **small real data set**: laser Doppler vibrometry records of a
Quanser QDrone 2 in hover (Zenodo 21203159: 10 healthy and 45 faulty one-second records, 4410 Hz). TRL 4 (public real laboratory data, laboratory validation only);
the result is largely negative and based on 11 recordings.

```
pip install -e ".[dev]" && python -m pytest
# data: https://zenodo.org/records/21203159 ("Full Dataset.zip", 11 xlsx files; not committed)
python research/run_study.py "<dir>/Full Dataset"
```
