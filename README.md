# 📡 Wi-Fi Radar: Privacy-Preserving Human Presence & Activity Detection

> **Device-free, camera-free presence sensing and fall-like movement detection using Wi-Fi Channel State Information (CSI).**

---

## 🛑 The Problem

Modern human monitoring systems—used in elderly care, smart homes, and patient safety—face critical limitations:

1. **The Privacy Dilemma**: Traditional monitoring relies heavily on optical cameras or microphones. People do not want cameras in private, dignity-sensitive spaces such as bathrooms, bedrooms, and private living quarters—precisely where high-risk incidents (like falls) occur most frequently.
2. **Wearable Non-Compliance**: Smart watches, pendants, and emergency panic buttons require people to wear and recharge them continuously. Elderly or injured individuals frequently forget to wear them, take them off before sleeping, or cannot reach them after an accident.
3. **Environmental Blind Spots**: Optical cameras fail in complete darkness, thick steam, smoke, or when a person is occluded behind furniture or partition walls.

---

## 💡 The Solution

**Wi-Fi Radar** turns everyday Wi-Fi signals into an invisible, non-intrusive radar link. 

By analyzing microscopic disruptions in ambient radio waves between a standard Wi-Fi router and an ESP32 microcontroller, the system detects human presence, distinguishes between different activities (**Empty room**, **Sitting/Still**, **Walking**, **Fall-like movement**), and triggers real-time alerts.

* 🔒 **100% Privacy-Preserving**: No cameras, no microphones, no images, no eavesdropping.
* 🚫 **Device-Free**: The person does not need to wear or carry any gadget or sensor.
* 🌑 **Works in Any Environment**: Operates through darkness, fog, steam, and behind obstacles.

---

## 🔬 What It Senses & How It Works

### 1. The Physics: Wi-Fi as a Radar
Wi-Fi is composed of high-frequency radio waves (2.4 GHz / 5 GHz, wavelength ~12.5 cm). The human body is approximately **70% water and conductive tissue**, acting as a dielectric reflector to radio frequencies.

When a Wi-Fi router transmits packets to a receiver:
* Radio waves bounce off stationary objects (walls, furniture) creating **static multipath**.
* When a human moves, breathes, or falls, their body length and position alter the wave paths by millimeters and centimeters.
* This creates **constructive and destructive interference** (Doppler shifts, scattering, and phase variations) that directly disturbs the received signal.

```
Router ))) ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ))) ESP32 Receiver
         \       [ Human Body ]         /
          \───(bounces off person)─────/
```

### 2. CSI (Channel State Information) vs. RSSI
Standard devices only report **RSSI** (Received Signal Strength Indicator)—a coarse, single-number metric of total signal loudness that easily gets corrupted by noise. 

This prototype captures **CSI (Channel State Information)**:
* Modern Wi-Fi uses OFDM (Orthogonal Frequency-Division Multiplexing) to split one channel into dozens of separate frequency subcarriers.
* In this prototype, the receiver samples **52 active subcarriers** ~50 times per second (50 Hz).
* Each subcarrier provides raw complex **I/Q (In-phase & Quadrature)** values: $\text{Amplitude} = \sqrt{I^2 + Q^2}$.
* This yields a 52-dimensional "fingerprint" of the room's radio propagation 50 times each second.

---

## ⚙️ The Technical Pipeline

```
[ Wi-Fi Link (50 Hz) ] 
       │  (52 OFDM subcarriers, I/Q amplitudes)
       ▼
[ Preprocessing ] 
       │  ├─ Per-packet AGC gain-jitter cancellation (median normalization)
       │  ├─ Static multipath removal (subtracting stationary baseline)
       │  └─ Median filtering (spikes & noise suppression)
       ▼
[ Feature Extraction (4-second rolling window) ]
       │  ├─ PCA Coherence (dominant motion across all 52 subcarriers)
       │  ├─ Breathing Band (0.1 Hz – 0.6 Hz: chest micro-movements while sitting)
       │  ├─ Gait Band (0.6 Hz – 4.0 Hz: limb and torso swing while walking)
       │  ├─ High-frequency turbulence (> 4.0 Hz)
       │  └─ 0.5 s Energy Envelope (sudden burst vs. continuous movement)
       ▼
[ Random Forest Classifier ]
       │  └─ Predicts probabilities: Empty | Sitting | Walking | Fall-like
       ▼
[ Fall Detection Engine & Alert Filter ]
       │  └─ Requires fall probability > threshold in 2 of the last 3 windows
       │     with a 20-second cooldown to eliminate false alarms
       ▼
[ Real-Time Streamlit Dashboard / Alert Webhook ]
```

### How Fall Detection Differs From Walking
* **Walking**: Produces sustained, rhythmic energy across the gait band (0.6–4 Hz) that remains steady over several seconds.
* **Fall**: Characterized by a **sharp, high-energy impact burst** across all subcarriers, immediately followed by **prolonged quiet** (the person remaining stationary on the floor). The energy envelope features (`blk_max`, `blk_last`, `max_over_last`) isolate this pattern.

---

## 🖥️ What the Prototype Does

* **Live Streamlit Dashboard**: Displays real-time activity state badges, 52-subcarrier spectrogram heatmaps, motion energy traces, and live probability bars.
* **Dual Operation Modes**:
  1. **Simulator Mode**: Zero hardware required. Generates mathematically accurate synthetic CSI signals based on radio physics, allowing full testing of transitions and simulated falls.
  2. **Hardware Mode (ESP32)**: Connects to a physical ESP32 board over USB serial to monitor real human movement inside a real room.
* **Auditable Alerting**: Automatically logs every detected fall with timestamps and confidence scores to `alerts.log` and pushes to external webhooks (Slack, Discord, ntfy).

---

## 🚀 Quick Start

### 1. Run in Simulator Mode (No Hardware Needed)

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
python -m streamlit run dashboard.py
```

Open **http://localhost:8501** in your browser. 
* Use the sidebar to switch between `auto`, `empty`, `sitting`, or `walking`.
* Click **"Trigger simulated fall"** to test real-time fall detection and alert logging.

---

### 2. Run with Real Hardware (Classic ESP32)

#### Requirements:
* 1x Classic ESP32 development board (e.g., ESP32-WROOM-32).
* 1x Standard 2.4 GHz Wi-Fi Router.
* [ESP-IDF 5.x](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/) installed.

#### Steps:
1. Open [`firmware/csi_rx/main/main.c`](firmware/csi_rx/main/main.c) and configure your Wi-Fi credentials:
   ```c
   #define WIFI_SSID "YOUR_WIFI_SSID"
   #define WIFI_PASS "YOUR_WIFI_PASSWORD"
   ```
2. Build and flash the firmware:
   ```bash
   cd firmware/csi_rx
   idf.py set-target esp32
   idf.py build flash
   ```
3. Position the ESP32 and router on opposite sides of the room you wish to monitor.
4. Plug the ESP32 into your PC via USB.
5. In the dashboard sidebar, switch **CSI source** to **ESP32 (serial)**, enter your serial port (e.g. `COM5` or `/dev/ttyUSB0`), and click **Apply source**.

#### Calibrate for Your Room:
Because every room has unique multipath geometry, record training data in your environment:
```bash
python record.py --port COM5 --label empty   --seconds 90
python record.py --port COM5 --label sitting --seconds 90
python record.py --port COM5 --label walking --seconds 90
python record.py --port COM5 --label fall    --seconds 8     # repeat on a soft mattress 15-30 times
python train.py --source real
```

---

## 📂 Project Structure

| File | Purpose |
|---|---|
| `dashboard.py` | Streamlit real-time dashboard with live heatmaps and probability metrics |
| `engine.py` | Multi-threaded background engine: buffering, resampling, inference, and fall logic |
| `features.py` | Signal processing: AGC correction, static removal, FFT band powers, and energy envelopes |
| `model.py` / `train.py` | Random Forest classifier training, hyperparameter setup, and evaluation metrics |
| `simulator.py` | Synthetic CSI physics simulator (Doppler shifts, multipath noise, and burst profiles) |
| `sources.py` | USB serial packet parser converting raw ESP32 I/Q lines into subcarrier amplitudes |
| `record.py` | Command-line tool to record labelled CSI datasets from an ESP32 for custom room training |
| `alerts.py` | Alert dispatcher: writes to `alerts.log` and sends HTTP webhooks |
| `firmware/csi_rx/` | C firmware for classic ESP32: connects to Wi-Fi, pings gateway at 50 Hz, and streams CSI |

---

## ⚠️ Limitations & Ethical Considerations

* **Experimental Prototype**: This software detects fall-*like* motions via RF disturbance. It is an experimental engineering prototype and is **not** a certified medical safety device.
* **Room-Specific Multipath**: Radio wave propagation is sensitive to layout, walls, moving fans, and pets. A model trained in one room should be fine-tuned or retrained when deployed in a new environment.
* **Privacy & Consent**: Always obtain full consent from occupants before deploying radio-frequency presence sensing in any physical space.
