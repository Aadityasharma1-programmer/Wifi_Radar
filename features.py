"""Signal processing + feature extraction for one 4-second CSI window."""
import numpy as np
from scipy.ndimage import median_filter
from config import FS, N_SUB

NAMES = ["std_all", "mean_abs_diff", "pc1_ratio", "breath_frac", "gait_frac", "high_frac",
         "spec_entropy", "dom_freq", "blk_max", "blk_last", "blk_first_half", "blk_second_half",
         "max_over_last", "max_over_median", "argmax_pos"]


def preprocess(raw):
    """raw amplitudes (n, N_SUB) -> relative deviation from each subcarrier's own mean."""
    a = raw / (np.median(raw, axis=1, keepdims=True) + 1e-9)   # cancel per-packet gain (AGC) jitter
    a = a / (a.mean(axis=0, keepdims=True) + 1e-9) - 1.0        # remove static multipath
    return median_filter(a, size=(3, 1), mode="nearest")        # kill single-packet spikes


def extract(raw):
    a = preprocess(raw)
    n = len(a)
    ac = a - a.mean(axis=0)
    std_all = a.std(axis=0).mean()
    mad = np.abs(np.diff(a, axis=0)).mean()

    # first principal component = the dominant shared motion pattern across subcarriers
    _, s, vt = np.linalg.svd(ac, full_matrices=False)
    pc = ac @ vt[0]
    pc1 = s[0] ** 2 / (np.sum(s ** 2) + 1e-12)

    spec = np.abs(np.fft.rfft(pc * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1 / FS)
    tot = spec[1:].sum() + 1e-12
    breath = spec[(f >= 0.1) & (f < 0.6)].sum() / tot
    gait = spec[(f >= 0.6) & (f <= 4)].sum() / tot
    high = spec[f > 4].sum() / tot
    p = spec[1:] / tot
    ent = -(p * np.log(p + 1e-12)).sum() / np.log(len(p))
    dom = f[1:][np.argmax(spec[1:])]

    # 0.5 s energy envelope: fall = big peak then quiet, walking = sustained energy
    k = FS // 2
    bs = np.sqrt(ac[: n // k * k].reshape(-1, k, N_SUB).var(axis=1).mean(axis=1))
    h = len(bs) // 2
    mx, last = bs.max(), bs[-2:].mean()
    vec = np.array([std_all, mad, pc1, breath, gait, high, ent, dom,
                    mx, last, bs[:h].mean(), bs[h:].mean(),
                    mx / (last + 1e-4), mx / (np.median(bs) + 1e-4), bs.argmax() / len(bs)])
    return vec, {"energy": float(bs[-1])}
