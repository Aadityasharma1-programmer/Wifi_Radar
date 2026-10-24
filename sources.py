"""Real CSI source: ESP32 streaming 'CSI_DATA,...,[i q i q ...]' lines over USB serial."""
import time
import numpy as np
from config import VALID_IDX


def parse_csi_line(line):
    """Return per-subcarrier amplitude (N_SUB,) or None if the line is not a usable CSI frame."""
    if not line.startswith("CSI_DATA") or "[" not in line:
        return None
    try:
        vals = np.array(line.split("[", 1)[1].split("]")[0].split(), dtype=float)
    except ValueError:
        return None
    if len(vals) < 128:
        return None
    iq = vals[:128].reshape(-1, 2)             # (imag, real) pairs, LLTF = 64 subcarriers
    return np.hypot(iq[:, 0], iq[:, 1])[VALID_IDX]


class SerialSource:
    kind = "serial"

    def __init__(self, port, baud=921600):
        self.port, self.baud, self._stop = port, baud, False
        self.name = f"ESP32 on {port}"

    def stop(self):
        self._stop = True

    def packets(self):
        import serial  # pyserial
        with serial.Serial(self.port, self.baud, timeout=1) as ser:
            while not self._stop:
                line = ser.readline().decode(errors="ignore").strip()
                amp = parse_csi_line(line)
                if amp is not None:
                    yield time.time(), amp
