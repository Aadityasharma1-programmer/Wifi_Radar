"""Dataset building, training and loading of the activity classifier."""
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from config import CLASSES, MODEL_PATH
from features import extract, NAMES
from simulator import sim_window


def make_sim_dataset(n_per_class=400, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci, c in enumerate(CLASSES):
        for _ in range(n_per_class):
            X.append(extract(sim_window(c, rng))[0])
            y.append(ci)
    return np.array(X), np.array(y)


def fit(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    meta = {"accuracy": float((pred == yte).mean()),
            "confusion": confusion_matrix(yte, pred, labels=range(len(CLASSES))).tolist(),
            "importances": dict(zip(NAMES, clf.feature_importances_.round(3).tolist())),
            "n_train": int(len(ytr))}
    return clf, meta


def save(clf, meta):
    joblib.dump({"model": clf, "meta": meta}, MODEL_PATH)


def load_or_train():
    if MODEL_PATH.exists():
        d = joblib.load(MODEL_PATH)
        return d["model"], d["meta"]
    X, y = make_sim_dataset()
    clf, meta = fit(X, y)
    meta["trained_on"] = "simulated data"
    save(clf, meta)
    return clf, meta
