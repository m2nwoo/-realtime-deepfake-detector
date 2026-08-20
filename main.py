import collections
import csv
import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
NOSE_TIP_IDX = 1
LEFT_EYE_CENTER_IDX = 33
RIGHT_EYE_CENTER_IDX = 263
FACE_EDGE_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
JAWLINE_IDX = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
    377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
]
VALID_PROFILES = ("ACCURACY", "SPEED")
REMOTE_PROCESS_BLACKLIST = ["anydesk", "teamviewer", "remotedesktop", "rustdesk", "parsec", "ultraviewer", "vnc"]
VCAM_KEYWORDS = ["obs virtual", "manycam", "splitcam", "xsplit", "snap camera", "vcam", "virtual camera"]


def profile_defaults(name: str) -> dict:
    if name == "SPEED":
        return {
            "TOGGLES": {"USE_GEOMETRIC": True, "USE_TEXTURE": False, "USE_BOUNDARY": True},
            "SMOOTHING": {"EMA_ALPHA": 0.18, "WINDOW_SIZE": 18},
            "THRESHOLDS": {
                "EAR_BLINK_THRESHOLD": 0.21,
                "BLINK_MIN_BPM": 6.0,
                "BLINK_MAX_BPM": 40.0,
                "JITTER_THRESHOLD": 0.015,
                "JITTER_SCALE": 0.045,
                "SIDE_POSE_YAW_THRESHOLD": 0.28,
                "SIDE_JUMP_THRESHOLD": 0.20,
                "LOST_COUNT_MAX_TOLERANCE": 6,
                "TEXTURE_THRESHOLD": 55.0,
                "BOUNDARY_THRESHOLD": 25.0,
            },
            "WEIGHTS": {"GEOMETRIC_WEIGHT": 0.6, "TEXTURE_WEIGHT": 0.0, "BOUNDARY_WEIGHT": 0.4},
            "GEOMETRIC_WEIGHTS": {"BLINK": 0.4, "JITTER": 0.3, "SIDE": 0.3},
            "LOG_SETTINGS": {
                "SAVE_CSV": False,
                "CSV_PATH": "logs/tuning_metrics_speed.csv",
                "LOG_EVERY_FRAME": False,
                "LOG_INTERVAL_SEC": 1.0,
            },
            "PERFORMANCE": {"JAWLINE_MASK_THICKNESS": 3, "FRAME_SKIP": 2, "PROCESS_SCALE": 0.50},
            "SECURITY_SETTINGS": {
                "ENABLE_VCAM_BLOCK": True,
                "ENABLE_REMOTE_SCAN": True,
                "SCAN_INTERVAL_SEC": 3.0,
            },
            "RISK_LEVELS": {"CAUTION": 0.45, "WARNING": 0.65},
        }

    return {
        "TOGGLES": {"USE_GEOMETRIC": True, "USE_TEXTURE": True, "USE_BOUNDARY": True},
        "SMOOTHING": {"EMA_ALPHA": 0.15, "WINDOW_SIZE": 30},
        "THRESHOLDS": {
            "EAR_BLINK_THRESHOLD": 0.21,
            "BLINK_MIN_BPM": 6.0,
            "BLINK_MAX_BPM": 40.0,
            "JITTER_THRESHOLD": 0.015,
            "JITTER_SCALE": 0.045,
            "SIDE_POSE_YAW_THRESHOLD": 0.28,
            "SIDE_JUMP_THRESHOLD": 0.20,
            "LOST_COUNT_MAX_TOLERANCE": 6,
            "TEXTURE_THRESHOLD": 55.0,
            "BOUNDARY_THRESHOLD": 25.0,
        },
        "WEIGHTS": {"GEOMETRIC_WEIGHT": 0.4, "TEXTURE_WEIGHT": 0.3, "BOUNDARY_WEIGHT": 0.3},
        "GEOMETRIC_WEIGHTS": {"BLINK": 0.4, "JITTER": 0.3, "SIDE": 0.3},
        "LOG_SETTINGS": {
            "SAVE_CSV": True,
            "CSV_PATH": "logs/tuning_metrics_accuracy.csv",
            "LOG_EVERY_FRAME": False,
            "LOG_INTERVAL_SEC": 1.0,
        },
        "PERFORMANCE": {"JAWLINE_MASK_THICKNESS": 6, "FRAME_SKIP": 1, "PROCESS_SCALE": 0.65},
        "SECURITY_SETTINGS": {
            "ENABLE_VCAM_BLOCK": True,
            "ENABLE_REMOTE_SCAN": True,
            "SCAN_INTERVAL_SEC": 1.0,
        },
        "RISK_LEVELS": {"CAUTION": 0.45, "WARNING": 0.65},
    }


DEFAULT_CONFIG = {
    "CURRENT_PROFILE": "ACCURACY",
    "PROFILES": {
        "ACCURACY": profile_defaults("ACCURACY"),
        "SPEED": profile_defaults("SPEED"),
    },
    "RUNTIME": {"CONFIG_RELOAD_SECONDS": 1.0},
}


@dataclass
class DetectionState:
    prev_points: Optional[np.ndarray] = None
    eye_closed_frames: int = 0
    blink_timestamps: Deque[float] = field(default_factory=collections.deque)
    side_jump_timestamps: Deque[float] = field(default_factory=collections.deque)
    side_lost_timestamps: Deque[float] = field(default_factory=collections.deque)
    side_pose_recent_until: float = 0.0
    face_present_prev: bool = False
    landmark_loss_events: int = 0
    landmark_lost_frames: int = 0
    smoothed_total_risk: float = 0.0
    risk_window: Deque[float] = field(default_factory=collections.deque)
    cached_tex_risk: float = 0.0
    cached_texture_energy: float = 0.0
    cached_bnd_risk: float = 0.0
    cached_boundary_sharpness: float = 0.0
    cached_jawline_roi_area: int = 0
    instant_mode_until: float = 0.0


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def norm_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def smooth_ema(prev: float, cur: float, alpha: float) -> float:
    return alpha * cur + (1.0 - alpha) * prev


def smooth_window(q: Deque[float], v: float, n: int) -> float:
    q.append(v)
    while len(q) > max(1, n):
        q.popleft()
    return float(np.mean(q))


def ear(points: np.ndarray, idx: List[int]) -> float:
    p1, p2, p3, p4, p5, p6 = [points[i] for i in idx]
    h = 2.0 * norm_distance(p1, p4)
    if h < 1e-6:
        return 0.0
    return (norm_distance(p2, p6) + norm_distance(p3, p5)) / h


def trim_old(q: Deque[float], now: float, sec: float) -> None:
    while q and now - q[0] > sec:
        q.popleft()


def deep_merge(defaults: dict, override: dict) -> dict:
    out = dict(defaults)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def draw_text_shadow(frame, text, org, scale, color, th=1):
    cv2.putText(frame, text, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def draw_main_panel(frame, lines, color):
    h, w = frame.shape[:2]
    x1, y1, pw, ph = 12, 12, 430, 24 + 22 * len(lines)
    x2, y2 = min(w - 12, x1 + pw), min(h - 12, y1 + ph)
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (x1 + 9, y1 + 21 + i * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 245), 1)


def draw_critical_overlay(frame: np.ndarray, lines: List[str]):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 180), -1)
    cv2.addWeighted(ov, 0.42, frame, 0.58, 0, frame)
    for i, line in enumerate(lines):
        draw_text_shadow(frame, line, (20, 70 + i * 44), 0.95, (255, 255, 255), 2)


def draw_debug_panel(frame, profile_name: str, status: str, metric_entries: List[Tuple[str, Tuple[int, int, int]]], errors: List[str], is_error: bool):
    h, w = frame.shape[:2]
    metrics = metric_entries[:5]
    errs = errors[:3]
    lines = [f"PROFILE: {profile_name}", status] + [m[0] for m in metrics] + errs
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0] for t in lines]
    tw, th = max((s[0] for s in sizes), default=280), max((s[1] for s in sizes), default=16)
    pad_x, pad_y, gap = 10, 8, 6
    pw = tw + 2 * pad_x
    ph = 2 * pad_y + len(lines) * th + max(0, len(lines) - 1) * gap + 4
    x1, y1 = max(12, w - pw - 12), max(12, h - ph - 12)
    x2, y2 = min(w - 12, x1 + pw), min(h - 12, y1 + ph)
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.58, frame, 0.42, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255) if is_error else (80, 80, 80), 1)
    pcol = (0, 200, 255) if profile_name == "SPEED" else (255, 220, 120)
    mc = len(metrics)
    for i, t in enumerate(lines):
        y = y1 + pad_y + th + i * (th + gap)
        if i == 0:
            c = pcol
        elif i == 1:
            c = (0, 255, 255) if is_error else (220, 220, 220)
        elif i <= 1 + mc:
            c = metrics[i - 2][1]
        else:
            c = (0, 90, 255)
        draw_text_shadow(frame, t, (x1 + pad_x, y), 0.5, c, 1)


def face_bbox(points_norm: np.ndarray, shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
    h, w = shape[:2]
    if points_norm.size == 0:
        return None
    x_min = int(np.clip(np.min(points_norm[:, 0]) * w, 0, w - 1))
    y_min = int(np.clip(np.min(points_norm[:, 1]) * h, 0, h - 1))
    x_max = int(np.clip(np.max(points_norm[:, 0]) * w, 0, w - 1))
    y_max = int(np.clip(np.max(points_norm[:, 1]) * h, 0, h - 1))
    if x_max <= x_min or y_max <= y_min:
        return None
    px, py = int((x_max - x_min) * 0.08), int((y_max - y_min) * 0.12)
    x1, y1, x2, y2 = max(0, x_min - px), max(0, y_min - py), min(w - 1, x_max + px), min(h - 1, y_max + py)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


def analyze_texture(face_roi: Optional[np.ndarray], threshold: float) -> Tuple[float, float]:
    if face_roi is None or face_roi.size == 0:
        return 1.0, 0.0
    g = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = g.shape[:2]
    if h < 16 or w < 16:
        return 1.0, 0.0
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = max(2.0, min(h, w) * 0.13)
    hf = mag[(yy - cy) ** 2 + (xx - cx) ** 2 > r ** 2]
    if hf.size == 0:
        return 1.0, 0.0
    e = float(np.mean(hf) * 25.0)
    return (0.0 if e >= threshold else clamp01((threshold - e) / max(1e-6, threshold))), e


def analyze_boundary(frame: np.ndarray, points: np.ndarray, threshold: float, thickness: int) -> Tuple[float, float, int]:
    h, w = frame.shape[:2]
    jaw = points[JAWLINE_IDX]
    jaw_px = np.zeros_like(jaw, dtype=np.int32)
    jaw_px[:, 0] = np.clip((jaw[:, 0] * w).astype(np.int32), 0, w - 1)
    jaw_px[:, 1] = np.clip((jaw[:, 1] * h).astype(np.int32), 0, h - 1)
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(m, [jaw_px], isClosed=False, color=255, thickness=max(1, thickness))
    area = int(np.count_nonzero(m))
    if area < 20:
        return 1.0, 0.0, area
    lap = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    vals = lap[m > 0]
    if vals.size < 20:
        return 1.0, 0.0, area
    sharp = float(np.var(vals))
    r = 0.0 if sharp >= threshold else clamp01((threshold - sharp) / max(1e-6, threshold))
    return r, sharp, area


class FaceTracker:
    """Compatibility wrapper for MediaPipe solutions/tasks APIs."""

    def __init__(self):
        self.mode = "tasks"
        self.face_mesh = None
        self.drawing = None
        self.drawing_spec = None
        self.landmarker = None
        self.model_path = Path(".models") / "face_landmarker.task"

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.mode = "solutions"
            self.drawing = mp.solutions.drawing_utils
            self.drawing_spec = self.drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 255))
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self._init_tasks_landmarker()

    def _init_tasks_landmarker(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.model_path.exists():
            url = (
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task"
            )
            try:
                urllib.request.urlretrieve(url, str(self.model_path))
            except Exception as exc:
                raise RuntimeError(f"Face landmarker model download failed: {exc}") from exc

        base = mp.tasks.BaseOptions(model_asset_path=str(self.model_path))
        vision = mp.tasks.vision
        opts = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int, process_scale: float = 1.0) -> Optional[np.ndarray]:
        if process_scale < 0.2:
            process_scale = 0.2
        if process_scale > 1.0:
            process_scale = 1.0
        if process_scale < 0.999:
            proc = cv2.resize(frame_bgr, None, fx=process_scale, fy=process_scale, interpolation=cv2.INTER_LINEAR)
        else:
            proc = frame_bgr

        if self.mode == "solutions":
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb)
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0]
                return np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
            return None

        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_img, int(timestamp_ms))
        if result.face_landmarks:
            lm = result.face_landmarks[0]
            return np.array([[p.x, p.y] for p in lm], dtype=np.float32)
        return None

    def draw(self, frame_bgr: np.ndarray, points: Optional[np.ndarray]) -> None:
        if points is None:
            return
        h, w = frame_bgr.shape[:2]
        # lightweight drawing: sample points only
        for p in points[::4]:
            x = int(np.clip(p[0] * w, 0, w - 1))
            y = int(np.clip(p[1] * h, 0, h - 1))
            cv2.circle(frame_bgr, (x, y), 1, (0, 220, 220), -1)

    def close(self):
        try:
            if self.face_mesh is not None:
                self.face_mesh.close()
        except Exception:
            pass
        try:
            if self.landmarker is not None:
                self.landmarker.close()
        except Exception:
            pass


class ConfigManager:
    def __init__(self, path: Path):
        self.path = path
        self.last_mtime: Optional[float] = None
        self.last_check_time = 0.0
        self.fallback_error = False
        self.error_history: Deque[str] = collections.deque(maxlen=3)
        self.config = self.load_or_create()

    def _warn(self, reason: str):
        print("[WARNING] Invalid config value detected. Falling back to default/previous value.")
        self.fallback_error = True
        self.error_history.append(f"[{time.strftime('%H:%M:%S')}] {reason}")

    @staticmethod
    def _g(d, k, default):
        return d.get(k, default) if isinstance(d, dict) else default

    def _sanitize_profile(self, cand: dict, fallback: dict, name: str) -> dict:
        safe = deep_merge(profile_defaults(name), fallback if isinstance(fallback, dict) else {})
        bad: List[str] = []

        def set_bool(sec, key):
            v = self._g(self._g(cand, sec, {}), key, None)
            if isinstance(v, bool):
                safe[sec][key] = v
            else:
                bad.append(f"{name}.{sec}.{key}")

        def set_int(sec, key, mn=1):
            v = self._g(self._g(cand, sec, {}), key, None)
            if isinstance(v, int) and v >= mn:
                safe[sec][key] = v
            else:
                bad.append(f"{name}.{sec}.{key}")

        def set_float(sec, key, mn=None, mx=None, strict=False):
            v = self._g(self._g(cand, sec, {}), key, None)
            ok = isinstance(v, (int, float))
            if ok:
                vf = float(v)
                if mn is not None and ((strict and not (vf > mn)) or ((not strict) and not (vf >= mn))):
                    ok = False
                if mx is not None and not (vf <= mx):
                    ok = False
            if ok:
                safe[sec][key] = float(v)
            else:
                bad.append(f"{name}.{sec}.{key}")

        def set_str(sec, key):
            v = self._g(self._g(cand, sec, {}), key, None)
            if isinstance(v, str):
                safe[sec][key] = v
            else:
                bad.append(f"{name}.{sec}.{key}")

        for k in ("USE_GEOMETRIC", "USE_TEXTURE", "USE_BOUNDARY"):
            set_bool("TOGGLES", k)
        set_float("SMOOTHING", "EMA_ALPHA", 0.0, 1.0, True)
        set_int("SMOOTHING", "WINDOW_SIZE", 1)
        for k in ("EAR_BLINK_THRESHOLD", "BLINK_MIN_BPM", "BLINK_MAX_BPM", "JITTER_THRESHOLD", "JITTER_SCALE", "SIDE_POSE_YAW_THRESHOLD", "SIDE_JUMP_THRESHOLD", "TEXTURE_THRESHOLD", "BOUNDARY_THRESHOLD"):
            set_float("THRESHOLDS", k, 0.0, None, k in ("JITTER_SCALE", "TEXTURE_THRESHOLD", "BOUNDARY_THRESHOLD"))
        set_int("THRESHOLDS", "LOST_COUNT_MAX_TOLERANCE", 1)
        for k in ("GEOMETRIC_WEIGHT", "TEXTURE_WEIGHT", "BOUNDARY_WEIGHT"):
            set_float("WEIGHTS", k, 0.0, 1.0, False)
        for k in ("BLINK", "JITTER", "SIDE"):
            set_float("GEOMETRIC_WEIGHTS", k, 0.0, None, False)
        for k in ("SAVE_CSV", "LOG_EVERY_FRAME"):
            set_bool("LOG_SETTINGS", k)
        set_str("LOG_SETTINGS", "CSV_PATH")
        set_float("LOG_SETTINGS", "LOG_INTERVAL_SEC", 0.0, None, True)
        set_int("PERFORMANCE", "JAWLINE_MASK_THICKNESS", 1)
        set_int("PERFORMANCE", "FRAME_SKIP", 1)
        set_float("PERFORMANCE", "PROCESS_SCALE", 0.2, 1.0, False)
        set_float("RISK_LEVELS", "CAUTION", 0.0, 1.0, False)
        set_float("RISK_LEVELS", "WARNING", 0.0, 1.0, False)

        for k in ("ENABLE_VCAM_BLOCK", "ENABLE_REMOTE_SCAN"):
            set_bool("SECURITY_SETTINGS", k)
        scan_interval = self._g(self._g(cand, "SECURITY_SETTINGS", {}), "SCAN_INTERVAL_SEC", None)
        if isinstance(scan_interval, (int, float)) and float(scan_interval) >= 0.1:
            safe["SECURITY_SETTINGS"]["SCAN_INTERVAL_SEC"] = float(scan_interval)
        else:
            safe["SECURITY_SETTINGS"]["SCAN_INTERVAL_SEC"] = 1.0
            bad.append(f"{name}.SECURITY_SETTINGS.SCAN_INTERVAL_SEC")

        if sum(float(safe["WEIGHTS"][k]) for k in ("GEOMETRIC_WEIGHT", "TEXTURE_WEIGHT", "BOUNDARY_WEIGHT")) <= 0.0:
            bad.append(f"{name}.WEIGHTS_SUM")
        if bad:
            self._warn(", ".join(sorted(set(bad))[:3]))
        return safe

    def _sanitize_root(self, cand: dict, fallback: dict) -> dict:
        safe = deep_merge(DEFAULT_CONFIG, fallback if isinstance(fallback, dict) else {})
        cur = self._g(cand, "CURRENT_PROFILE", None)
        if isinstance(cur, str) and cur in VALID_PROFILES:
            safe["CURRENT_PROFILE"] = cur
        else:
            if safe.get("CURRENT_PROFILE") not in VALID_PROFILES:
                safe["CURRENT_PROFILE"] = "ACCURACY"
            self._warn("CURRENT_PROFILE")
        rt = self._g(cand, "RUNTIME", {})
        sec = self._g(rt, "CONFIG_RELOAD_SECONDS", None)
        if isinstance(sec, (int, float)) and float(sec) > 0:
            safe["RUNTIME"]["CONFIG_RELOAD_SECONDS"] = float(sec)
        else:
            self._warn("RUNTIME.CONFIG_RELOAD_SECONDS")
        profs = self._g(cand, "PROFILES", {})
        for name in VALID_PROFILES:
            raw = self._g(profs, name, {})
            safe["PROFILES"][name] = self._sanitize_profile(
                raw if isinstance(raw, dict) else {},
                self._g(safe["PROFILES"], name, profile_defaults(name)),
                name,
            )
        return safe

    def load_or_create(self) -> dict:
        if not self.path.exists():
            self.path.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
            self.last_mtime = self.path.stat().st_mtime
            return dict(DEFAULT_CONFIG)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                self._warn("config root")
                raw = {}
        except Exception:
            self._warn("Invalid JSON")
            raw = {}
        cfg = self._sanitize_root(deep_merge(DEFAULT_CONFIG, raw), DEFAULT_CONFIG)
        self.last_mtime = self.path.stat().st_mtime
        return cfg

    def maybe_reload(self) -> dict:
        period = float(self._g(self._g(self.config, "RUNTIME", {}), "CONFIG_RELOAD_SECONDS", 1.0))
        if time.time() - self.last_check_time < max(0.1, period):
            return self.config
        self.last_check_time = time.time()
        if not self.path.exists():
            self.path.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
            self.config = dict(DEFAULT_CONFIG)
            self.last_mtime = self.path.stat().st_mtime
            return self.config
        m = self.path.stat().st_mtime
        if self.last_mtime is None or m > self.last_mtime:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    self._warn("config root")
                    raw = {}
            except Exception:
                self._warn("Invalid JSON")
                raw = {}
            self.config = self._sanitize_root(deep_merge(DEFAULT_CONFIG, raw), self.config)
            self.last_mtime = m
        return self.config

    def active_profile(self) -> Tuple[str, dict]:
        cur = self._g(self.config, "CURRENT_PROFILE", "ACCURACY")
        if cur not in VALID_PROFILES:
            cur = "ACCURACY"
        prof = self._g(self._g(self.config, "PROFILES", {}), cur, profile_defaults(cur))
        return cur, prof if isinstance(prof, dict) else profile_defaults(cur)

    def debug_payload(self) -> Tuple[str, List[str], bool]:
        _, p = self.active_profile()
        sm = self._g(p, "SMOOTHING", {})
        return (
            f"Config: {'ERROR' if self.fallback_error else 'OK'} | Alpha:{float(self._g(sm, 'EMA_ALPHA', 0.15)):.2f} W:{int(self._g(sm, 'WINDOW_SIZE', 30))}",
            list(self.error_history),
            self.fallback_error or len(self.error_history) > 0,
        )


class CsvLogger:
    def __init__(self):
        self.enabled = False
        self.path = ""
        self.every = False
        self.interval = 1.0
        self.last = 0.0
        self.fh = None
        self.writer = None
        self.lock = Lock()

    def close(self):
        with self.lock:
            if self.fh is not None:
                try:
                    self.fh.close()
                except Exception:
                    pass
            self.fh = None
            self.writer = None

    def configure(self, s: dict):
        en = bool(s.get("SAVE_CSV", False))
        p = str(s.get("CSV_PATH", "logs/tuning_metrics.csv"))
        ev = bool(s.get("LOG_EVERY_FRAME", False))
        it = max(0.05, float(s.get("LOG_INTERVAL_SEC", 1.0)))
        changed = en != self.enabled or p != self.path or ev != self.every or abs(it - self.interval) > 1e-9
        if not changed:
            return
        self.close()
        self.enabled, self.path, self.every, self.interval, self.last = en, p, ev, it, 0.0
        if self.enabled:
            self._ensure_open()

    def _ensure_open(self):
        if not self.enabled or (self.fh is not None and self.writer is not None):
            return
        try:
            op = Path(self.path)
            op.parent.mkdir(parents=True, exist_ok=True)
            exists = op.exists() and op.stat().st_size > 0
            self.fh = op.open("a", newline="", encoding="utf-8")
            self.writer = csv.writer(self.fh)
            if not exists:
                self.writer.writerow(["timestamp", "total_ema_risk", "fft_high_freq_mean", "jawline_edge_variance", "jawline_roi_area", "landmark_lost_count"])
                self.fh.flush()
        except Exception as e:
            print(f"[WARNING] CSV init failed: {e}")
            self.fh = None
            self.writer = None

    def log(self, ts, total, fft_mean, jaw_var, jaw_area, lost, now):
        if not self.enabled:
            return
        if (not self.every) and (now - self.last < self.interval):
            return
        with self.lock:
            try:
                self._ensure_open()
                if self.writer is None or self.fh is None:
                    return
                self.writer.writerow([ts, round(total, 6), round(fft_mean, 6), round(jaw_var, 6), int(jaw_area), int(lost)])
                self.fh.flush()
                self.last = now
            except Exception as e:
                print(f"[WARNING] CSV write failed: {e}")


class SecurityMonitor:
    def __init__(self):
        self.psutil_ready = psutil is not None
        self.psutil_warned = False
        self.last_scan_ts = 0.0
        self.cached_remote_hits: List[str] = []
        self.cached_virtual_hit = False
        self.cached_virtual_reason = ""
        self.cached_camera_names: List[str] = []

    def _warn_psutil(self):
        if self.psutil_ready or self.psutil_warned:
            return
        self.psutil_warned = True
        print("[WARNING] psutil not installed. 원격제어 프로세스 감시 비활성화. 설치: pip install psutil")

    def _query_cam_names(self) -> List[str]:
        try:
            cmd = ["powershell", "-NoProfile", "-Command", "Get-PnpDevice -Class Camera | Select-Object -ExpandProperty FriendlyName"]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if out.returncode != 0:
                return []
            return [x.strip().lower() for x in out.stdout.splitlines() if x.strip()]
        except Exception:
            return []

    def _scan_virtual(self, cap: cv2.VideoCapture) -> Tuple[bool, str]:
        reason = ""
        names = self._query_cam_names()
        if names:
            self.cached_camera_names = names
        for n in (names or self.cached_camera_names):
            if any(k in n for k in VCAM_KEYWORDS):
                reason = f"camera_device:{n}"
                break
        if not reason and self.psutil_ready:
            try:
                for p in psutil.process_iter(["name"]):
                    nm = (p.info.get("name") or "").lower()
                    if any(k in nm for k in ["obs", "manycam", "splitcam", "xsplit", "snapcamera"]):
                        reason = f"process:{nm}"
                        break
            except Exception:
                pass
        if not reason:
            try:
                bid = int(cap.get(cv2.CAP_PROP_BACKEND))
                bname = cv2.videoio_registry.getBackendName(bid).lower()
                if "virtual" in bname:
                    reason = f"backend:{bname}"
            except Exception:
                pass
        return (len(reason) > 0), reason

    def _scan_remote(self) -> List[str]:
        if not self.psutil_ready:
            self._warn_psutil()
            return []
        hits: List[str] = []
        try:
            for p in psutil.process_iter(["name"]):
                nm = (p.info.get("name") or "").lower()
                if nm and any(k in nm for k in REMOTE_PROCESS_BLACKLIST):
                    hits.append(nm)
        except Exception:
            pass
        return sorted(set(hits))

    def evaluate(self, cap: cv2.VideoCapture, now: float, enable_vcam: bool, enable_remote: bool, scan_interval_sec: float) -> dict:
        scan_interval_sec = max(0.1, float(scan_interval_sec))
        if (now - self.last_scan_ts) >= scan_interval_sec:
            self.last_scan_ts = now
            if enable_vcam:
                self.cached_virtual_hit, self.cached_virtual_reason = self._scan_virtual(cap)
            else:
                self.cached_virtual_hit, self.cached_virtual_reason = False, ""
            if enable_remote:
                self.cached_remote_hits = self._scan_remote()
            else:
                self.cached_remote_hits = []
        remote_hit = len(self.cached_remote_hits) > 0
        critical = self.cached_virtual_hit or remote_hit
        sec_disabled = (not enable_vcam) and (not enable_remote)
        return {
            "critical": critical,
            "virtual_hit": self.cached_virtual_hit,
            "virtual_reason": self.cached_virtual_reason,
            "remote_hit": remote_hit,
            "remote_hits": self.cached_remote_hits,
            "sec_disabled": sec_disabled,
        }


def main():
    cm, st, clog, secm = ConfigManager(Path("config.json")), DetectionState(), CsvLogger(), SecurityMonitor()
    frame_idx = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. 카메라 연결/권한을 확인하세요.")

    tracker = FaceTracker()
    try:
        while True:
            cm.maybe_reload()
            profile_name, p = cm.active_profile()
            togg, sm, th = p.get("TOGGLES", {}), p.get("SMOOTHING", {}), p.get("THRESHOLDS", {})
            wg, gw, ls = p.get("WEIGHTS", {}), p.get("GEOMETRIC_WEIGHTS", {}), p.get("LOG_SETTINGS", {})
            perf, levels = p.get("PERFORMANCE", {}), p.get("RISK_LEVELS", {})
            sec_cfg = p.get("SECURITY_SETTINGS", {})
            clog.configure(ls)

            enable_vcam = bool(sec_cfg.get("ENABLE_VCAM_BLOCK", True))
            enable_remote = bool(sec_cfg.get("ENABLE_REMOTE_SCAN", True))
            scan_interval = float(sec_cfg.get("SCAN_INTERVAL_SEC", 1.0))
            if scan_interval < 0.1:
                scan_interval = 1.0

            use_geo = bool(togg.get("USE_GEOMETRIC", True))
            use_tex = bool(togg.get("USE_TEXTURE", True))
            use_bnd = bool(togg.get("USE_BOUNDARY", True))
            alpha, win = float(sm.get("EMA_ALPHA", 0.15)), int(sm.get("WINDOW_SIZE", 30))
            fskip, jthick = int(perf.get("FRAME_SKIP", 1)), int(perf.get("JAWLINE_MASK_THICKNESS", 6))
            process_scale = float(perf.get("PROCESS_SCALE", 0.65))
            ear_th, bpm_min, bpm_max = float(th.get("EAR_BLINK_THRESHOLD", 0.21)), float(th.get("BLINK_MIN_BPM", 6.0)), float(th.get("BLINK_MAX_BPM", 40.0))
            jit_th, jit_sc = float(th.get("JITTER_THRESHOLD", 0.015)), float(th.get("JITTER_SCALE", 0.045))
            yaw_th, jump_th = float(th.get("SIDE_POSE_YAW_THRESHOLD", 0.28)), float(th.get("SIDE_JUMP_THRESHOLD", 0.2))
            lost_tol, tex_th, bnd_th = int(th.get("LOST_COUNT_MAX_TOLERANCE", 6)), float(th.get("TEXTURE_THRESHOLD", 55.0)), float(th.get("BOUNDARY_THRESHOLD", 25.0))
            gsum = float(gw.get("BLINK", 0.4)) + float(gw.get("JITTER", 0.3)) + float(gw.get("SIDE", 0.3))
            if gsum <= 1e-9:
                gsum = 1.0

            ok, frame = cap.read()
            if not ok:
                continue
            frame_idx += 1
            heavy = (frame_idx % max(1, fskip)) == 0
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now, ts = time.time(), time.strftime("%H:%M:%S")

            sec_status = secm.evaluate(cap, now, enable_vcam, enable_remote, scan_interval)
            pts = tracker.process(frame, int(now * 1000), process_scale=process_scale)
            tracker.draw(frame, pts)
            face_reacquired = False

            blink_risk = jitter_risk = side_risk = geo_risk = tex_risk = bnd_risk = bpm = yaw = 0.0
            tex_energy, bnd_sharp, jaw_area = st.cached_texture_energy, st.cached_boundary_sharpness, st.cached_jawline_roi_area
            status, color, bbox = "NO FACE", (140, 140, 140), None

            if pts is not None:
                # 얼굴이 사라졌다가 다시 잡힌 첫 프레임인지 체크
                face_reacquired = (not st.face_present_prev) and (st.landmark_lost_frames > 0)
                if face_reacquired:
                    st.instant_mode_until = now + 0.5
                st.face_present_prev, st.landmark_lost_frames = True, 0
                ear_avg = 0.5 * (ear(pts, LEFT_EYE_IDX) + ear(pts, RIGHT_EYE_IDX))
                if ear_avg < ear_th:
                    st.eye_closed_frames += 1
                else:
                    if st.eye_closed_frames >= 2:
                        st.blink_timestamps.append(now)
                    st.eye_closed_frames = 0
                trim_old(st.blink_timestamps, now, 20.0)
                bpm = len(st.blink_timestamps) * 3.0
                if bpm < bpm_min:
                    blink_risk = clamp01((bpm_min - bpm) / max(1.0, bpm_min))
                elif bpm > bpm_max:
                    blink_risk = clamp01((bpm - bpm_max) / max(1.0, bpm_max))

                if st.prev_points is not None:
                    ed = norm_distance(pts[LEFT_EYE_CENTER_IDX], pts[RIGHT_EYE_CENTER_IDX])
                    if ed > 1e-6:
                        mot = np.linalg.norm(pts[FACE_EDGE_IDX] - st.prev_points[FACE_EDGE_IDX], axis=1) / ed
                        jitter_risk = clamp01((float(np.mean(mot) + 0.5 * np.std(mot)) - jit_th) / max(1e-6, jit_sc))
                        yaw = (pts[NOSE_TIP_IDX][0] - 0.5 * (pts[LEFT_EYE_CENTER_IDX][0] + pts[RIGHT_EYE_CENTER_IDX][0])) / ed
                        side_pose = abs(yaw) > yaw_th
                        if side_pose:
                            st.side_pose_recent_until = now + 0.6
                        if side_pose and float(np.max(mot)) > jump_th:
                            st.side_jump_timestamps.append(now)
                trim_old(st.side_jump_timestamps, now, 6.0)
                trim_old(st.side_lost_timestamps, now, 6.0)
                side_risk = clamp01((len(st.side_jump_timestamps) + len(st.side_lost_timestamps)) / float(max(1, lost_tol)))
                geo_risk = clamp01((float(gw.get("BLINK", 0.4)) * blink_risk + float(gw.get("JITTER", 0.3)) * jitter_risk + float(gw.get("SIDE", 0.3)) * side_risk) / gsum)
                bbox = face_bbox(pts, frame.shape)
                if use_tex and heavy:
                    x1, y1, x2, y2 = bbox if bbox else (0, 0, 0, 0)
                    roi = frame[y1:y2, x1:x2] if bbox else None
                    tex_risk, tex_energy = analyze_texture(roi, tex_th)
                    st.cached_tex_risk, st.cached_texture_energy = tex_risk, tex_energy
                else:
                    tex_risk, tex_energy = (st.cached_tex_risk, st.cached_texture_energy) if use_tex else (0.0, 0.0)
                if use_bnd and heavy:
                    bnd_risk, bnd_sharp, jaw_area = analyze_boundary(frame, pts, bnd_th, jthick)
                    st.cached_bnd_risk, st.cached_boundary_sharpness, st.cached_jawline_roi_area = bnd_risk, bnd_sharp, jaw_area
                else:
                    bnd_risk, bnd_sharp, jaw_area = (st.cached_bnd_risk, st.cached_boundary_sharpness, st.cached_jawline_roi_area) if use_bnd else (0.0, 0.0, 0)
                st.prev_points = pts
            else:
                if st.face_present_prev:
                    st.landmark_loss_events += 1
                st.face_present_prev, st.landmark_lost_frames, st.prev_points, st.eye_closed_frames = False, st.landmark_lost_frames + 1, None, 0
                if now < st.side_pose_recent_until:
                    st.side_lost_timestamps.append(now)
                trim_old(st.side_lost_timestamps, now, 6.0)
                trim_old(st.side_jump_timestamps, now, 6.0)
                side_risk = clamp01((len(st.side_jump_timestamps) + len(st.side_lost_timestamps)) / float(max(1, lost_tol)))
                geo_risk = side_risk
                tex_risk, bnd_risk = (st.cached_tex_risk if use_tex else 0.0), (st.cached_bnd_risk if use_bnd else 0.0)

            if st.landmark_lost_frames > lost_tol:
                geo_risk = tex_risk = bnd_risk = 1.0
            e_geo, e_tex, e_bnd = (geo_risk if use_geo else 0.0), (tex_risk if use_tex else 0.0), (bnd_risk if use_bnd else 0.0)
            ww_geo, ww_tex, ww_bnd = float(wg.get("GEOMETRIC_WEIGHT", 0.4)), float(wg.get("TEXTURE_WEIGHT", 0.3)), float(wg.get("BOUNDARY_WEIGHT", 0.3))
            wsum = (ww_geo if use_geo else 0.0) + (ww_tex if use_tex else 0.0) + (ww_bnd if use_bnd else 0.0)
            raw = 0.0 if wsum <= 1e-9 else clamp01((ww_geo * e_geo + ww_tex * e_tex + ww_bnd * e_bnd) / wsum)
            # 재감지 직후 0.5초 동안 즉시모드 적용, 이후 EMA/윈도우 복귀
            instant_mode_active = now < st.instant_mode_until
            if instant_mode_active:
                st.risk_window.clear()
                st.risk_window.append(raw)
                st.smoothed_total_risk = raw
            else:
                st.smoothed_total_risk = smooth_window(
                    st.risk_window,
                    smooth_ema(st.smoothed_total_risk, raw, alpha),
                    max(1, win),
                )
            total = clamp01(st.smoothed_total_risk)
            real = (1.0 - total) * 100.0

            if sec_status.get("critical", False):
                raw, total, st.smoothed_total_risk, real = 1.0, 1.0, 1.0, 0.0
            caution, warning = float(levels.get("CAUTION", 0.45)), float(levels.get("WARNING", 0.65))
            if sec_status.get("critical", False):
                status, color = "CRITICAL", (0, 0, 255)
            elif total >= warning:
                status, color = "WARNING", (0, 0, 255)
            elif total >= caution:
                status, color = "CAUTION", (0, 165, 255)
            elif pts is not None:
                status, color = "NORMAL", (0, 180, 0)

            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"얼굴 박스 | 랜드마크 손실:{st.landmark_loss_events}", (x1, max(22, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

            status_kr = {"NORMAL": "정상", "CAUTION": "주의", "WARNING": "경고", "CRITICAL": "치명", "NO FACE": "얼굴없음"}.get(status, status)
            draw_main_panel(frame, [
                f"프로파일: {profile_name}",
                f"상태: {status_kr}",
                f"실제 확률: {real:5.1f}%",
                f"최종 위험도(EMA): {total*100:5.1f}%",
                f"원시 위험도: {raw*100:5.1f}%",
                f"기하/텍스처/경계: {e_geo*100:4.1f} / {e_tex*100:4.1f} / {e_bnd*100:4.1f}",
                f"즉시모드: {'ON' if instant_mode_active else 'OFF'}",
                f"Yaw:{yaw:+.2f} BPM:{bpm:4.1f} 손실프레임:{st.landmark_lost_frames}",
                "종료: Q 또는 ESC",
            ], color)
            cv2.rectangle(frame, (3, 3), (w - 3, h - 3), color, 4)

            if sec_status.get("virtual_hit", False):
                draw_critical_overlay(frame, ["치명적 경고: 가상 카메라 감지!", "접근을 차단합니다."])
            if sec_status.get("remote_hit", False):
                hit_text = ", ".join(sec_status.get("remote_hits", [])[:2])
                msg = "치명적 경고: 원격 제어 프로그램 실행 중!"
                if hit_text:
                    msg += f" ({hit_text})"
                draw_text_shadow(frame, msg, (20, 34), 0.72, (0, 0, 255), 2)

            dbg_status, dbg_hist, dbg_err = cm.debug_payload()
            if sec_status.get("sec_disabled", False):
                sys_sec_text, sys_sec_color = "SYS_SEC: OFF", (140, 140, 140)
            elif sec_status.get("critical", False):
                sys_sec_text, sys_sec_color = "SYS_SEC: VIOLATION", (0, 60, 255)
            else:
                sys_sec_text, sys_sec_color = "SYS_SEC: SAFE", (120, 220, 120)

            entries = [
                (sys_sec_text, sys_sec_color),
                (f"Geo: {e_geo*100:4.1f}%" if use_geo else "Geo: DISABLED", (210, 210, 210) if use_geo else (140, 140, 140)),
                (f"Tex: {e_tex*100:4.1f}%" if use_tex else "Tex: DISABLED", (210, 210, 210) if use_tex else (140, 140, 140)),
                (f"Bnd: {e_bnd*100:4.1f}%" if use_bnd else "Bnd: DISABLED", (210, 210, 210) if use_bnd else (140, 140, 140)),
                (f"FFT:{tex_energy:5.1f} JawVar:{bnd_sharp:5.1f}", (180, 180, 180)),
            ]
            draw_debug_panel(frame, profile_name, dbg_status, entries, dbg_hist, dbg_err or sec_status.get("critical", False))
            cv2.imshow("Realtime Deepfake Heuristic Monitor", frame)

            clog.log(ts, total, tex_energy, bnd_sharp, jaw_area, st.landmark_lost_frames, now)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        clog.close()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
