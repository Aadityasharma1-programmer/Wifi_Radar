"""Record labelled CSI from the ESP32 for training on REAL data.

  python record.py --port COM5 --label walking --seconds 60
  python record.py --port COM5 --label fall --seconds 8      # one short clip per fall
Files are saved as data/<label>_<timestamp>.csv
"""
import argparse
import time
import numpy as np
from config import CLASSES, DATA_DIR
from sources import SerialSource

ap = argparse.ArgumentParser()
ap.add_argument("--port", required=True)
ap.add_argument("--baud", type=int, default=921600)
ap.add_argument("--label", required=True, choices=CLASSES)
ap.add_argument("--seconds", type=float, default=60)
a = ap.parse_args()

DATA_DIR.mkdir(exist_ok=True)
rows, t0 = [], None
print(f"Recording '{a.label}' for {a.seconds}s ... start the activity now.")
src = SerialSource(a.port, a.baud)
for t, amp in src.packets():
    t0 = t0 or t
    rows.append(np.r_[t, amp])
    if t - t0 >= a.seconds:
        src.stop()
        break
out = DATA_DIR / f"{a.label}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
np.savetxt(out, np.array(rows), delimiter=",")
print(f"Saved {len(rows)} packets ({len(rows)/a.seconds:.0f}/s) -> {out}")
