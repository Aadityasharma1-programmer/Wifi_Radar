"""Real-time engine: buffers packets, resamples, classifies, raises fall alerts."""
import threading
import time
from collections import deque

import numpy as np

from alerts import log_event, make_event, notify
from config import CLASSES, FS, HOP_S, N_SUB, WIN_S
from features import extract, preprocess

FALL_I = CLASSES.index("fall")


def resample(ts, A, t_end, seconds, n=None):
    n = n or int(seconds * FS)
    grid = t_end - seconds + np.arange(1, n + 1) * (seconds / n)
    return np.column_stack([np.interp(grid, ts, A[:, j]) for j in range(A.shape[1])]), grid


class Engine:
    def __init__(self, model, meta=None):
        self.model, self.meta = model, meta or {}
        self.lock = threading.RLock()
        self.fall_thr = 0.6
        self._thread, self.source = None, None
        self._stop = threading.Event()
        self._reset()

    def _reset(self):
        self.buf = deque()
        self.prob_hist, self.fall_hist = deque(maxlen=3), deque(maxlen=3)
        self.energy = deque(maxlen=240)
        self.alerts, self.n_alerts = [], 0
        self.probs = np.zeros(len(CLASSES))
        self.label, self.rate, self.error = None, 0.0, None
        self.fall_until = self.last_alert = 0.0

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, source):
        self.stop()
        with self.lock:
            self._reset()
        self.source = source
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(source,), daemon=True)
        self._thread.start()

    def stop(self):
        if self.source is not None:
            self.source.stop()
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=3)

    def _run(self, source):
        last = 0.0
        try:
            for t, amp in source.packets():
                if self._stop.is_set():
                    break
                with self.lock:
                    self.buf.append((t, amp))
                    while self.buf and self.buf[0][0] < t - 25:
                        self.buf.popleft()
                if t - last >= HOP_S:
                    last = t
                    self._infer(t)
        except Exception as e:  # e.g. serial port missing
            self.error = f"{type(e).__name__}: {e}"

    def _infer(self, t_now):
        with self.lock:
            items = list(self.buf)
        ts = np.array([i[0] for i in items])
        A = np.vstack([i[1] for i in items])
        self.rate = float(np.count_nonzero(ts > t_now - 2) / 2)
        if ts[-1] - ts[0] < WIN_S * 0.95:
            return
        win, _ = resample(ts, A, t_now, WIN_S)
        x, info = extract(win)
        p = self.model.predict_proba(x.reshape(1, -1))[0]
        with self.lock:
            self.prob_hist.append(p)
            self.probs = np.mean(self.prob_hist, axis=0)
            self.energy.append((t_now, info["energy"]))
            self.fall_hist.append(p[FALL_I] >= self.fall_thr)
            now = time.time()
            if sum(self.fall_hist) >= 2 and now - self.last_alert > 20:
                ev = make_event(p[FALL_I])
                self.alerts.insert(0, ev)
                self.n_alerts += 1
                self.last_alert, self.fall_until = now, now + 10
                log_event(ev)
                notify(ev)
            self.label = "fall" if now < self.fall_until else CLASSES[int(np.argmax(self.probs))]

    def snapshot(self):
        with self.lock:
            items = list(self.buf)
            snap = dict(label=self.label, probs=self.probs.copy(), rate=self.rate, error=self.error,
                        alerts=list(self.alerts), n_alerts=self.n_alerts,
                        energy=list(self.energy), running=self.running)
        heat = None
        if len(items) > FS * 2:
            ts = np.array([i[0] for i in items])
            A = np.vstack([i[1] for i in items])
            span = min(10.0, ts[-1] - ts[0])
            raw, grid = resample(ts, A, ts[-1], span, n=int(span * 25))
            heat = (grid - grid[-1], preprocess(raw))
        snap["heat"] = heat
        return snap
