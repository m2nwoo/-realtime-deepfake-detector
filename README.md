# Real-Time Deepfake Detection System

This project analyzes webcam input in real time and estimates deepfake risk with a security-focused pipeline.  
It combines facial landmark geometry, texture frequency analysis, jawline boundary sharpness checks, and host-level security signals.

## What This Project Does

- Streams webcam video and tracks face landmarks in real time
- Computes multi-signal deepfake risk:
  - Geometric risk (blink pattern, landmark jitter, side-rotation consistency)
  - Texture risk via FFT high-frequency energy
  - Boundary risk around the jawline/neck blend region
- Includes host security monitoring:
  - Virtual camera detection/blocking heuristics
  - Remote control process detection (AnyDesk, TeamViewer, RustDesk, etc.)
- Supports hot-reload configuration and profile switching

## Tech Stack

- **Language**: Python 3
- **Core Libraries**:
  - `opencv-python`
  - `mediapipe`
  - `numpy`
  - `psutil` (used for remote process monitoring; optional fallback-safe)

## Key Features

- Real-time face tracking and risk scoring
- Security-aware fail-safe mode (`CRITICAL` state forces risk to 100%)
- Profile-based configuration:
  - `ACCURACY`: quality-first settings
  - `SPEED`: lightweight settings for lower-end devices
- Per-profile toggles for detection modules and security scans
- Debug panel + optional CSV metric logging

## Run

```bash
cd C:\Users\saga6\realtime-deepfake-detector
python -m pip install -r requirements.txt
python main.py
```

Exit keys: `Q` or `ESC`

## Configuration

Main config file: `config.json`

Important sections:
- `CURRENT_PROFILE`
- `PROFILES.<PROFILE>.TOGGLES`
- `PROFILES.<PROFILE>.SECURITY_SETTINGS`
- `PROFILES.<PROFILE>.LOG_SETTINGS`
- `PROFILES.<PROFILE>.PERFORMANCE`

Config changes are applied via hot reload during runtime.

## Notes

- This is a heuristic detector, not a final forensic-grade classifier.
- Threshold tuning is recommended per camera/environment.
- For operational use, add policy checks and validation for your deployment context.
