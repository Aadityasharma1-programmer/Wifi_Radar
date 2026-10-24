"""Synthetic CSI generator (used for demos and for training when no hardware is available).

All signals produced here are SIMULATED. They mimic the *shape* of CSI amplitude changes:
  empty    -> static multipath + noise
  sitting  -> small slow breathing modulation
  walking  -> strong 1-2 Hz gait modulation
  fall     -> sudden large burst, then a person lying still
"""
import time
import numpy as np
from config import FS, WIN, N_SUB

LYING_S = 14.0
SCRIPT = [("empty", 12.0), ("sitting", 12.0), ("walking", 12.0)]
CYCLE = sum(d for _, d in SCRIPT)
W = 2 * np.pi


class Scene:
    """Random but fixed 'room + person' parameters."""

    def __init__(self, rng):
        self.base = rng.uniform(18, 40, N_SUB)
        self.gain = rng.uniform(0.4, 1.0, N_SUB) * rng.choice([-1.0, 1.0], N_SUB)
        self.ph = rng.uniform(0, W, (3, N_SUB))
        self.f_breath = rng.uniform(0.2, 0.35)
        self.f_gait = rng.uniform(0.9, 1.8)
        self.noise = rng.uniform(0.003, 0.007)
        self.env_ph = rng.uniform(0, W)


def deviation(mode, t, sc):
    """Relative amplitude deviation, shape (len(t), N_SUB)."""
    t = np.asarray(t, float).reshape(-1, 1)
    if mode == "empty":
        return np.zeros((len(t), N_SUB))
    if mode == "sitting":
        return 0.02 * sc.gain * np.sin(W * sc.f_breath * t + sc.ph[0])
    if mode == "lying":
        return 0.012 * sc.gain * np.sin(W * sc.f_breath * t + sc.ph[0])
    if mode == "walking":
        env = 1 + 0.35 * np.sin(W * 0.17 * t + sc.env_ph)
        gait = 0.07 * np.sin(W * sc.f_gait * t + sc.ph[0]) + 0.035 * np.sin(W * 2 * sc.f_gait * t + sc.ph[1])
        return env * sc.gain * gait + 0.03 * sc.gain * np.sin(W * 0.4 * t + sc.ph[2])
    raise ValueError(mode)


def burst(tr, sc):
    """Fall impact: big, fast-decaying disturbance. tr = seconds since impact."""
    tr = np.asarray(tr, float).reshape(-1, 1)
    out = sc.gain * 0.22 * np.exp(-np.clip(tr, 0, None) / 0.35) * np.sin(W * 4.5 * tr + sc.ph[2])
    return np.where((tr >= 0) & (tr < 1.2), out, 0.0)


def to_raw(dev, sc, rng):
    """Raw amplitudes incl. noise and per-packet AGC-like gain jitter."""
    n = len(dev)
    return sc.base * (1 + dev + rng.normal(0, sc.noise, dev.shape)) * (1 + rng.normal(0, 0.03, (n, 1)))


def sim_window(label, rng):
    """One labelled training window (WIN, N_SUB)."""
    sc = Scene(rng)
    t = np.arange(WIN) / FS + rng.uniform(0, 100)
    if label != "fall":
        if rng.random() < 0.4:   # hard negatives: activity changes mid-window (label = state at the end)
            prev = rng.choice([m for m in ("empty", "sitting", "walking") if m != label])
            k = int(rng.uniform(0.5, 3.0) * FS)
            dev = np.vstack([deviation(prev, t[:k], sc), deviation(label, t[k:], sc)])
            return to_raw(dev, sc, rng)
        return to_raw(deviation(label, t, sc), sc, rng)
    pre = rng.choice(["walking", "sitting", "empty"], p=[0.6, 0.3, 0.1])
    k = int(rng.uniform(0.6, 3.0) * FS)
    dev = np.empty((WIN, N_SUB))
    dev[:k] = deviation(pre, t[:k], sc)
    dev[k:] = deviation("lying", t[k:], sc) + burst(t[k:] - t[k], sc)
    return to_raw(dev, sc, rng)


class SimSource:
    """Real-time synthetic packet stream (50 packets/s)."""
    kind = "sim"
    name = "Simulator (synthetic data)"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.sc = Scene(self.rng)
        self.manual = "auto"
        self._stop = False
        self._trigger = False
        self._fall_t = None
        self._auto_t0 = 0.0
        self._last_pos = 0.0
        self._cur = "empty"

    def set_mode(self, m):
        self.manual = m

    def trigger_fall(self):
        self._trigger = True

    def stop(self):
        self._stop = True

    def _mode(self, t):
        if self._trigger:
            self._trigger, self._fall_t = False, t
        if self._fall_t is not None:
            if t - self._fall_t < LYING_S:
                return "lying"
            self._fall_t, self._auto_t0, self._last_pos = None, t, 0.0
        if self.manual != "auto":
            return self.manual
        pos = (t - self._auto_t0) % CYCLE
        if pos < self._last_pos:          # script finished -> simulate a fall
            self._trigger = True
            self._last_pos = pos
            return self._cur
        self._last_pos = pos
        acc = 0.0
        for m, d in SCRIPT:
            acc += d
            if pos < acc:
                return m
        return "empty"

    def _sample(self, t):
        self._cur = self._mode(t)
        dev = deviation(self._cur, [t], self.sc)
        if self._fall_t is not None:
            dev = dev + burst([t - self._fall_t], self.sc)
        return to_raw(dev, self.sc, self.rng)[0]

    def packets(self):
        t0, n = time.time(), 0
        while not self._stop:
            dt = t0 + n / FS - time.time()
            if dt > 0:
                time.sleep(dt)
            yield time.time(), self._sample(n / FS)
            n += 1
