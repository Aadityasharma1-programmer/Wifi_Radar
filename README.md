# 📡 Wi-Fi Radar

Privacy-conscious human presence and activity detection using **Wi-Fi Channel State Information (CSI)**
from an **ESP32** instead of cameras. A person moving through a Wi-Fi link changes the radio signal;
Python signal processing + a Random Forest turn that into **empty / sitting / walking / fall-like**,
shown on a live dashboard with fall alerts. No camera, no microphone, no images.

```
 Router ))) ~ ~ ~ ))) ESP32 --USB serial--> Python
 (Wi-Fi link crossing    (pings router      ├─ preprocess (gain-jitter removal, resample 50 Hz)
  the monitored area)     50x/s, prints     ├─ 4 s windows -> 15 features
                          CSI per reply)    ├─ Random Forest -> activity + probabilities
                                            └─ fall logic (2 of last 3 windows) -> alert
                                                    └─> Streamlit dashboard / log / webhook
```

## Quick start (no hardware needed)
```bash
pip install -r requirements.txt
python train.py                  # optional, the dashboard trains on first launch if no model exists
streamlit run dashboard.py
```
The dashboard starts in **Simulator mode** (clearly labelled): scenario "auto" cycles empty -> sitting -> walking -> fall.
Use the sidebar to force a scenario or press **Trigger simulated fall**.

## With a real ESP32 (classic ESP32, ESP-IDF 5.x)
1. Edit `WIFI_SSID` / `WIFI_PASS` in `firmware/csi_rx/main/main.c`.
2. `cd firmware/csi_rx && idf.py set-target esp32 && idf.py build flash` (note the serial port).
3. Check the stream: lines starting with `CSI_DATA,` should appear at ~50/s (`idf.py monitor`; close it afterwards).
4. Place ESP32 and router so the monitored area lies between them (opposite sides of a room works well). Keep both fixed.
5. Dashboard sidebar -> **ESP32 (serial)** -> enter port -> **Apply source**.

### Train on your own room (strongly recommended)
CSI is room-specific. A model trained on synthetic data will **not** be accurate on real signals.
```bash
python record.py --port COM5 --label empty   --seconds 90
python record.py --port COM5 --label sitting --seconds 90
python record.py --port COM5 --label walking --seconds 90
# one clip per fall onto a crash mat/mattress, ~8 s each, impact ~2 s after start. Repeat 15-30 times.
python record.py --port COM5 --label fall    --seconds 8
python train.py --source real      # or --source both
```
Vary where you stand/walk, and record on different days. Check the confusion matrix `train.py` prints.

## Files
| File | Purpose |
|---|---|
| `firmware/csi_rx/` | ESP32 firmware: join Wi-Fi, ping gateway, stream CSI over serial |
| `sources.py` | Serial CSI parser (I/Q -> amplitude, drops null subcarriers) |
| `simulator.py` | Synthetic CSI for demos/training without hardware |
| `features.py` | Preprocessing + feature extraction (energy, PCA, band power, entropy, energy envelope) |
| `model.py`, `train.py`, `record.py` | Dataset building, training, recording real data |
| `engine.py` | Real-time thread: buffer, resample, classify, fall logic |
| `alerts.py` | Alert log (`alerts.log`) + optional webhook (`WEBHOOK_URL` env var) |
| `dashboard.py` | Streamlit live dashboard |

## How fall detection works
A fall shows up as a **large short burst followed by quiet** (person lying still), while walking is
**sustained** energy. The 0.5 s energy envelope features (peak, last-block level, peak/last ratio) capture this.
Activity changes (e.g., someone starts walking) are included as hard negatives in training to reduce false alarms.
An alert needs fall probability above the threshold in 2 of the last 3 windows, with a 20 s cooldown.

## Honest limitations
- The included accuracy numbers (~96 %) are on **simulated** data; they show the pipeline works, not real-world accuracy.
- Real CSI is noisy; expect to tune thresholds and collect your own training data. Other people/pets, moving fans, and
  furniture changes affect results. One link covers roughly one room.
- Fall-*like* movement detection is not a certified safety or medical device. Do not rely on it for care of vulnerable people.
- Firmware targets the classic ESP32; ESP32-S3/C3 use a different `wifi_csi_config_t` struct.
- Sensing people in spaces you do not own or without their knowledge may be illegal; get consent.
