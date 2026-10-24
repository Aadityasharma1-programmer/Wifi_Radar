"""Shared constants for the Wi-Fi Radar pipeline."""
from pathlib import Path

FS = 50                       # target sampling rate after resampling (Hz)
WIN_S = 4.0                   # analysis window (seconds)
WIN = int(FS * WIN_S)         # 200 samples
HOP_S = 0.5                   # run the classifier every 0.5 s
N_SUB = 52                    # usable subcarriers
# ESP32 LLTF gives 64 subcarriers; drop guard bands + DC null
VALID_IDX = list(range(6, 32)) + list(range(33, 59))

CLASSES = ["empty", "sitting", "walking", "fall"]
LABELS = {
    "empty": "Empty room",
    "sitting": "Person sitting / still",
    "walking": "Person walking",
    "fall": "Fall-like movement",
}

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.joblib"
LOG_PATH = ROOT / "alerts.log"
DATA_DIR = ROOT / "data"
