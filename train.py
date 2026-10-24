"""Train the activity classifier.

  python train.py                    # synthetic data only (works without hardware)
  python train.py --source real      # recordings in data/ (made with record.py)
  python train.py --source both
"""
import argparse
import numpy as np
from config import CLASSES, DATA_DIR, FS, WIN, WIN_S, N_SUB
from engine import resample
from features import extract, preprocess
from model import fit, make_sim_dataset, save


def real_dataset():
    X, y = [], []
    for path in sorted(DATA_DIR.glob("*.csv")):
        label = path.name.split("_")[0]
        if label not in CLASSES:
            continue
        d = np.loadtxt(path, delimiter=",")
        ts, A = d[:, 0], d[:, 1:]
        dur = ts[-1] - ts[0]
        if dur < WIN_S * 0.9:
            continue
        full, _ = resample(ts, A, ts[-1], dur)
        hop = FS // 2
        starts = range(0, len(full) - WIN + 1, hop)
        if label == "fall":      # keep only windows that contain the impact (energy peak) in their first 85%
            env = np.abs(preprocess(full)).mean(axis=1)
            peak = int(np.argmax(np.convolve(env, np.ones(FS // 2), "same")))
            starts = [s for s in starts if s <= peak < s + int(0.85 * WIN)]
        for s in starts:
            X.append(extract(full[s:s + WIN])[0])
            y.append(CLASSES.index(label))
    return np.array(X), np.array(y)


ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["sim", "real", "both"], default="sim")
ap.add_argument("--n", type=int, default=400, help="synthetic windows per class")
a = ap.parse_args()

Xs, ys = [], []
if a.source in ("sim", "both"):
    x, y = make_sim_dataset(a.n)
    Xs.append(x); ys.append(y)
if a.source in ("real", "both"):
    x, y = real_dataset()
    print("real windows per class:", {c: int((y == i).sum()) for i, c in enumerate(CLASSES)})
    if len(y) == 0:
        raise SystemExit("No usable recordings in data/ - run record.py first.")
    Xs.append(x); ys.append(y)

clf, meta = fit(np.vstack(Xs), np.concatenate(ys))
meta["trained_on"] = {"sim": "simulated data", "real": "recorded ESP32 data", "both": "simulated + recorded data"}[a.source]
save(clf, meta)
print(f"Hold-out accuracy: {meta['accuracy']:.1%}  (trained on {meta['trained_on']})")
print("Confusion matrix (rows=true, cols=pred):", CLASSES)
for c, row in zip(CLASSES, meta["confusion"]):
    print(f"  {c:8s}", row)
