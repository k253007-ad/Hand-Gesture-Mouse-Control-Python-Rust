"""Webcam hand tracking mouse controller."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
CONTROL_MARGIN = 0.125  # Center 75% of the camera frame controls the screen.
SCROLL_UNITS_PER_STEP = 30
FPS_CHOICES = (12, 15, 20, 24, 30, 36)
CHROME_PROFILE_PICKER_URL = "chrome://profile-picker/"

# MediaPipe's 21 hand landmarks, joined as five fingers and a palm.
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Download and verify Google's versioned model once."""
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    print("Downloading the MediaPipe hand model (first run only)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != MODEL_SHA256:
            raise RuntimeError("Downloaded hand model failed its SHA-256 check")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def finger_straightness(landmarks, mcp: int, pip: int, tip: int) -> float:
    first_x = landmarks[pip].x - landmarks[mcp].x
    first_y = landmarks[pip].y - landmarks[mcp].y
    second_x = landmarks[tip].x - landmarks[pip].x
    second_y = landmarks[tip].y - landmarks[pip].y
    length = math.hypot(first_x, first_y) * math.hypot(second_x, second_y)
    return (first_x * second_x + first_y * second_y) / max(length, 1e-6)


def is_pointing_pose(landmarks) -> bool:
    """Only the index is extended; thumb and remaining fingers are tucked."""
    if landmarks is None:
        return False
    palm = max(distance(landmarks[5], landmarks[17]), 0.01)
    return (
        finger_straightness(landmarks, 5, 6, 8) > 0.65
        and all(finger_straightness(landmarks, *finger) < 0.45
                for finger in ((9, 10, 12), (13, 14, 16), (17, 18, 20)))
        and distance(landmarks[4], landmarks[5]) < 1.45 * palm
    )


@dataclass
class GestureState:
    left_down: bool = False
    right_ready: bool = True
    double_ready: bool = True
    alt_tab_ready: bool = True
    windows_ready: bool = True
    switcher_open: bool = False
    selection_latched: bool = False
    cursor_x: float | None = None
    cursor_y: float | None = None


class HandMouse:
    def __init__(self, mouse, smoothing: float = 0.6, margin: float = CONTROL_MARGIN):
        self.mouse = mouse
        self.smoothing = smoothing
        self.margin = margin
        self.state = GestureState()
        self.screen_width, self.screen_height = mouse.size()

    def _release_left(self) -> None:
        if self.state.left_down:
            # A corner-triggered fail-safe must still release an active drag.
            try:
                self.mouse.mouseUp(button="left")
            except self.mouse.FailSafeException:
                self.mouse.FAILSAFE = False
                try:
                    self.mouse.mouseUp(button="left")
                finally:
                    self.mouse.FAILSAFE = True
            self.state.left_down = False

    def release(self, close_switcher: bool = False) -> None:
        self._release_left()
        if close_switcher and self.state.switcher_open:
            try:
                self.mouse.press("esc")
            except self.mouse.FailSafeException:
                self.mouse.FAILSAFE = False
                try:
                    self.mouse.press("esc")
                finally:
                    self.mouse.FAILSAFE = True
            self.state.switcher_open = False
        self.state.cursor_x = None
        self.state.cursor_y = None
        self.state.right_ready = True
        self.state.double_ready = True
        self.state.alt_tab_ready = True
        self.state.windows_ready = True
        self.state.selection_latched = False

    def update(self, pointer_landmarks, click_landmarks=None,
               allow_right_click=True, actions_enabled=True) -> str:
        if pointer_landmarks is None:
            self.release()
            return "Window picker open" if self.state.switcher_open else "No hand"
        if click_landmarks is None:
            click_landmarks = pointer_landmarks

        # Normalize pinch by palm width so gestures work at different distances.
        palm = max(distance(click_landmarks[5], click_landmarks[17]), 0.01)
        left_ratio = distance(click_landmarks[4], click_landmarks[8]) / palm
        middle_ratio = distance(click_landmarks[4], click_landmarks[12]) / palm
        ring_ratio = distance(click_landmarks[4], click_landmarks[16]) / palm
        pinky_ratio = distance(click_landmarks[4], click_landmarks[20]) / palm
        left_active = left_ratio < (0.52 if self.state.left_down else 0.38)
        double_active = (click_landmarks is not pointer_landmarks
                         and middle_ratio < 0.52 and left_ratio > 0.65
                         and ring_ratio > 0.58 and pinky_ratio > 0.65)
        alt_tab_active = (ring_ratio < 0.50 and left_ratio > 0.65
                          and middle_ratio > 0.60 and pinky_ratio > 0.58)
        right_active = (allow_right_click and pinky_ratio < 0.38 and left_ratio > 0.65
                        and middle_ratio > 0.60 and ring_ratio > 0.55)

        pointer_palm = max(distance(pointer_landmarks[5], pointer_landmarks[17]), 0.01)
        green_middle_ratio = distance(pointer_landmarks[4], pointer_landmarks[12]) / pointer_palm
        green_windows_active = (
            green_middle_ratio < 0.52
            and distance(pointer_landmarks[4], pointer_landmarks[8]) / pointer_palm > 0.65
            and distance(pointer_landmarks[4], pointer_landmarks[16]) / pointer_palm > 0.58
            and distance(pointer_landmarks[4], pointer_landmarks[20]) / pointer_palm > 0.65
        )

        fingertip = pointer_landmarks[8]
        x = clamp((fingertip.x - self.margin) / (1 - 2 * self.margin), 0, 1)
        y = clamp((fingertip.y - self.margin) / (1 - 2 * self.margin), 0, 1)
        # Keep one pixel away from OS corners; PyAutoGUI's fail-safe stays enabled.
        target_x = 1 + x * (self.screen_width - 3)
        target_y = 1 + y * (self.screen_height - 3)
        state = self.state
        if state.cursor_x is None:
            state.cursor_x, state.cursor_y = target_x, target_y
        else:
            travel = math.hypot(target_x - state.cursor_x, target_y - state.cursor_y)
            response = max(self.smoothing, min(0.85, travel / 250))
            state.cursor_x += response * (target_x - state.cursor_x)
            state.cursor_y += response * (target_y - state.cursor_y)

        self.mouse.moveTo(round(state.cursor_x), round(state.cursor_y))
        if green_middle_ratio > 0.70:
            state.windows_ready = True
        if green_windows_active:
            self._release_left()
            if state.windows_ready:
                if state.switcher_open:
                    self.mouse.press("esc")
                    state.switcher_open = False
                self.mouse.press("win")
                state.windows_ready = False
            return "Windows Start"
        if click_landmarks is pointer_landmarks and is_pointing_pose(pointer_landmarks):
            self._release_left()
            state.right_ready = True
            state.double_ready = True
            state.alt_tab_ready = True
            state.selection_latched = False
            return "Point only"
        if not actions_enabled:
            self._release_left()
            state.right_ready = True
            state.double_ready = True
            state.alt_tab_ready = True
            return "Scroll"
        if state.switcher_open:
            if left_active and not state.selection_latched:
                self.mouse.click(button="left")
                state.switcher_open = False
                state.selection_latched = True
                return "Window selected"
            return "Pick window: thumb+index click"
        if state.selection_latched:
            if left_active:
                return "Window selected"
            state.selection_latched = False

        if left_active and not state.left_down:
            self.mouse.mouseDown(button="left")
            state.left_down = True
        elif not left_active and state.left_down:
            self.mouse.mouseUp(button="left")
            state.left_down = False

        if double_active and state.double_ready:
            self.mouse.doubleClick(interval=0.08)
            state.double_ready = False
        elif not double_active:
            state.double_ready = True

        if alt_tab_active and state.alt_tab_ready:
            self.mouse.hotkey("ctrl", "alt", "tab")
            state.alt_tab_ready = False
            state.switcher_open = True
        elif not alt_tab_active:
            state.alt_tab_ready = True

        if right_active and state.right_ready:
            self.mouse.click(button="right")
            state.right_ready = False
        elif not right_active:
            state.right_ready = True

        if state.left_down:
            return "Drag / left button held"
        if double_active:
            return "Double click"
        if alt_tab_active:
            return "Pick window: thumb+index click"
        if right_active:
            return "Right click"
        return "Move"


def mirror_landmarks(landmarks):
    """Match unmirrored tracking coordinates to the mirrored preview."""
    return [SimpleNamespace(x=1 - point.x, y=point.y, z=point.z) for point in landmarks]


def hands_by_side(result):
    """Use MediaPipe's hand labels, keeping the highest confidence per side."""
    found = {}
    for landmarks, categories in zip(result.hand_landmarks, result.handedness):
        if not categories:
            continue
        label = categories[0].category_name
        if label in ("Right", "Left"):
            score = categories[0].score
            if label not in found or score > found[label][0]:
                found[label] = (score, mirror_landmarks(landmarks))
    return {label: item[1] for label, item in found.items()}


def preferred_side(hands, dominant="Right") -> str | None:
    if dominant in hands:
        return dominant
    other = "Left" if dominant == "Right" else "Right"
    if other in hands:
        return other
    return None


class ControlToggle:
    """Two-frame green-hand pinch toggles controls; release rearms it."""

    def __init__(self):
        self.enabled = True
        self.armed = True
        self.pinch_frames = 0
        self.release_frames = 0

    def update(self, landmarks) -> bool:
        if landmarks is None:
            self.pinch_frames = 0
            self.release_frames = 0
            return False
        if is_pointing_pose(landmarks):
            self.pinch_frames = 0
            self.release_frames += 1
            if self.release_frames >= 2:
                self.armed = True
            return False
        palm = max(distance(landmarks[5], landmarks[17]), 0.01)
        ratio = distance(landmarks[4], landmarks[20]) / palm
        if ratio < 0.42:
            self.release_frames = 0
            self.pinch_frames += 1
            if self.armed and self.pinch_frames >= 2:
                self.enabled = not self.enabled
                self.armed = False
                return True
        elif ratio > 0.62:
            self.pinch_frames = 0
            self.release_frames += 1
            if self.release_frames >= 2:
                self.armed = True
        else:
            self.pinch_frames = 0
            self.release_frames = 0
        return False


def is_scroll_pose(landmarks) -> bool:
    """Index and middle form a V while ring and pinky are folded."""
    if landmarks is None:
        return False

    palm = max(distance(landmarks[5], landmarks[17]), 0.01)
    return (
        finger_straightness(landmarks, 5, 6, 8) > 0.65
        and finger_straightness(landmarks, 9, 10, 12) > 0.65
        and finger_straightness(landmarks, 13, 14, 16) < 0.50
        and finger_straightness(landmarks, 17, 18, 20) < 0.50
        and distance(landmarks[8], landmarks[12]) > 0.35 * palm
        and distance(landmarks[4], landmarks[8]) > 0.65 * palm
        and distance(landmarks[4], landmarks[12]) > 0.65 * palm
    )


class ScrollGesture:
    def __init__(self, dead_zone_palms=0.18, steps_per_second_per_palm=12.0):
        self.dead_zone_palms = dead_zone_palms
        self.steps_per_second_per_palm = steps_per_second_per_palm
        self.anchor_x = None
        self.anchor_y = None
        self.anchor_palm = None
        self.current_x = None
        self.current_y = None
        self.last_time = None
        self.remainder = 0.0
        self.velocity = 0.0

    def reset(self):
        self.anchor_x = None
        self.anchor_y = None
        self.anchor_palm = None
        self.current_x = None
        self.current_y = None
        self.last_time = None
        self.remainder = 0.0
        self.velocity = 0.0

    def update(self, landmarks, now: float | None = None):
        if not is_scroll_pose(landmarks):
            self.reset()
            return False, 0
        if now is None:
            now = time.monotonic()
        # Follow the palm so bending the V's fingertips does not scroll.
        self.current_x = (landmarks[0].x + landmarks[9].x) / 2
        self.current_y = (landmarks[0].y + landmarks[9].y) / 2
        if self.anchor_y is None:
            self.anchor_x = self.current_x
            self.anchor_y = self.current_y
            self.anchor_palm = max(distance(landmarks[5], landmarks[17]), 0.01)
            self.last_time = now
            return True, 0

        # Treat the fixed point as a joystick center: farther means faster.
        displacement = (self.anchor_y - self.current_y) / self.anchor_palm
        speed = min(24.0, max(0.0, abs(displacement) - self.dead_zone_palms)
                    * self.steps_per_second_per_palm)
        velocity = math.copysign(speed, displacement) if speed else 0.0
        if velocity == 0 or self.velocity * velocity < 0:
            self.remainder = 0.0
        self.velocity = velocity
        elapsed = min(0.25, max(0.0, now - self.last_time))
        self.last_time = now
        self.remainder += velocity * elapsed
        steps = max(-3, min(3, math.trunc(self.remainder)))
        self.remainder -= steps
        return True, steps


def is_triangle(left, right, aspect_ratio: float) -> bool:
    """Index tips meet at the apex; separated thumbs make a level base."""
    if left is None or right is None:
        return False

    def scaled(point):
        return SimpleNamespace(x=point.x * aspect_ratio, y=point.y)

    l_index, r_index = scaled(left[8]), scaled(right[8])
    l_thumb, r_thumb = scaled(left[4]), scaled(right[4])
    palm = (distance(scaled(left[5]), scaled(left[17])) +
            distance(scaled(right[5]), scaled(right[17]))) / 2
    if palm < 0.025:
        return False
    apex_x = (l_index.x + r_index.x) / 2
    apex_y = (l_index.y + r_index.y) / 2
    base_x = (l_thumb.x + r_thumb.x) / 2
    base_y = (l_thumb.y + r_thumb.y) / 2
    base_width = abs(l_thumb.x - r_thumb.x)
    return (
        distance(l_index, r_index) < 0.42 * palm
        and base_width > 0.85 * palm
        and abs(l_thumb.y - r_thumb.y) < 0.40 * palm
        and base_y - apex_y > 0.55 * palm
        and abs(apex_x - base_x) < 0.35 * base_width
        and min(l_thumb.x, r_thumb.x) < apex_x < max(l_thumb.x, r_thumb.x)
    )


class TriangleTrigger:
    def __init__(self, hold_seconds: float = 0.65):
        self.hold_seconds = hold_seconds
        self.started_at = None
        self.fired = False

    def update(self, matched: bool, now: float) -> bool:
        if not matched:
            self.started_at = None
            self.fired = False
            return False
        if self.started_at is None:
            self.started_at = now
        if not self.fired and now - self.started_at >= self.hold_seconds:
            self.fired = True
            return True
        return False


def find_chrome() -> str | None:
    candidates = [
        shutil.which("chrome.exe"),
        os.path.join(os.environ.get("PROGRAMFILES", ""), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
    ]
    return next((path for path in candidates if path and os.path.isfile(path)), None)


def open_chrome_profile_picker() -> str:
    path = find_chrome()
    if not path:
        return "Chrome not found"
    try:
        subprocess.Popen([path, CHROME_PROFILE_PICKER_URL],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as error:
        print(f"Could not launch Chrome: {error}", file=sys.stderr)
        return "Chrome launch failed"
    return "Chrome profile picker opened"


class LatestFrameCamera:
    """Read continuously so slow tracking never works through old camera frames."""

    def __init__(self, capture, failure_timeout: float = 5.0):
        self.capture = capture
        self.failure_timeout = failure_timeout
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.running = True
        self.last_frame_at = time.monotonic()
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        last_good_frame = time.monotonic()
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                if time.monotonic() - last_good_frame < self.failure_timeout:
                    time.sleep(0.05)
                    continue
                with self.condition:
                    self.running = False
                    self.condition.notify_all()
                break
            last_good_frame = time.monotonic()
            with self.condition:
                self.last_frame_at = last_good_frame
                self.frame = frame
                self.sequence += 1
                self.condition.notify_all()

    def read_latest(self, previous_sequence):
        with self.condition:
            self.condition.wait_for(
                lambda: (self.frame is not None and self.sequence != previous_sequence)
                or not self.running,
                timeout=0.02,
            )
            if self.frame is None or self.sequence == previous_sequence or not self.running:
                return self.sequence, None
            return self.sequence, self.frame

    @property
    def stalled(self):
        with self.condition:
            return time.monotonic() - self.last_frame_at >= 0.25

    def close(self):
        with self.condition:
            self.running = False
            self.condition.notify_all()
        self.thread.join(timeout=1)
        self.capture.release()


def poll_control_frame(video, previous_sequence, mouse, scroll):
    """Keep input safe during capture stalls without blocking preview events."""
    sequence, frame = video.read_latest(previous_sequence)
    if frame is None and video.stalled:
        mouse.release(close_switcher=True)
        scroll.reset()
    return sequence, frame


def draw_hand(cv2, frame, landmarks, color) -> None:
    height, width = frame.shape[:2]
    points = [(int(point.x * width), int(point.y * height)) for point in landmarks]
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, points[8], 7, color, 2, cv2.LINE_AA)


def open_camera(cv2, camera_index: int):
    # DirectShow starts faster here when its frame rate is requested explicitly.
    backends = ((cv2.CAP_DSHOW, cv2.CAP_MSMF)
                if sys.platform == "win32" else (cv2.CAP_ANY,))
    for backend in backends:
        camera = cv2.VideoCapture(camera_index, backend)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return camera
        camera.release()
    return None


def run(camera_index: int, smoothing: float, tracking_width: int,
        dominant_hand: str, fps_limit: int = 20) -> int:
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        import pyautogui
    except ImportError as error:
        print(f"Missing dependency: {error}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    try:
        model_path = ensure_model()
    except (OSError, RuntimeError) as error:
        print(f"Could not prepare hand model: {error}", file=sys.stderr)
        return 1

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0
    mouse = HandMouse(pyautogui, smoothing=smoothing)
    window_name = "Hand Mouse | Space: pause | Q / Esc: quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 480, 360)
    camera_state = {}
    camera_ready = threading.Event()
    camera_cancelled = threading.Event()

    def connect_camera():
        try:
            connected = open_camera(cv2, camera_index)
            if camera_cancelled.is_set() and connected is not None:
                connected.release()
            else:
                camera_state["camera"] = connected
        except Exception as error:
            camera_state["error"] = error
        finally:
            camera_ready.set()

    threading.Thread(target=connect_camera, daemon=True).start()
    while not camera_ready.is_set():
        loading = np.full((480, 640, 3), (32, 36, 40), dtype=np.uint8)
        cv2.putText(loading, "HAND MOUSE", (55, 180), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (110, 225, 150), 2, cv2.LINE_AA)
        cv2.putText(loading, "Connecting to camera...", (58, 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(loading, "Press Q or Esc to cancel", (58, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (175, 185, 190), 1, cv2.LINE_AA)
        cv2.imshow(window_name, loading)
        if cv2.waitKey(50) & 0xFF in (ord("q"), 27):
            camera_cancelled.set()
            cv2.destroyAllWindows()
            return 0
    camera = camera_state.get("camera")
    if camera is None:
        reason = camera_state.get("error")
        print(f"Could not open camera {camera_index}: {reason or 'camera unavailable'}. "
              "Check camera access or try --camera 1.", file=sys.stderr)
        error_frame = np.full((480, 640, 3), (32, 36, 40), dtype=np.uint8)
        cv2.putText(error_frame, "Camera unavailable", (55, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (110, 170, 245), 2, cv2.LINE_AA)
        cv2.putText(error_frame, "Check Windows camera access or close other camera apps.",
                    (55, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(error_frame, "Press any key to close. Try --camera 1 if needed.",
                    (55, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (175, 185, 190), 1, cv2.LINE_AA)
        cv2.imshow(window_name, error_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 1
    video = LatestFrameCamera(camera)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    paused = False
    active_side = None
    active_click_side = None
    dominant = dominant_hand
    triangle = TriangleTrigger()
    controls = ControlToggle()
    scroll = ScrollGesture()
    gesture_status = None
    last_timestamp = -1
    last_sequence = -1
    last_frame_time = None
    last_title_update = 0.0
    fps = 0.0
    model_ms = 0.0
    window_sized = False
    precise_timer = sys.platform == "win32" and ctypes.windll.winmm.timeBeginPeriod(1) == 0
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as tracker:
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord(" "):
                    paused = not paused
                    if paused:
                        mouse.release(close_switcher=True)
                        scroll.reset()
                        triangle.update(False, time.monotonic())
                        gesture_status = None
                if key in (ord("h"), ord("H")):
                    dominant = "Left" if dominant == "Right" else "Right"
                    mouse.release()
                    scroll.reset()
                    active_side = None
                    active_click_side = None
                last_sequence, frame = poll_control_frame(video, last_sequence, mouse, scroll)
                if frame is None:
                    if not video.running:
                        print("Camera stopped supplying frames.", file=sys.stderr)
                        return 1
                    if video.stalled:
                        triangle.update(False, time.monotonic())
                        gesture_status = None
                    continue
                now = time.monotonic()
                if last_frame_time is not None:
                    instant_fps = 1 / max(now - last_frame_time, 0.001)
                    fps = 0.8 * fps + 0.2 * instant_fps if fps else instant_fps
                last_frame_time = now
                raw_frame = frame
                frame = cv2.flip(raw_frame, 1)
                if not window_sized:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, round(frame.shape[1] * 0.75),
                                     round(frame.shape[0] * 0.75))
                    window_sized = True
                status = "Paused" if paused else "No hand"
                if not paused:
                    # Track the raw image so MediaPipe labels physical hands.
                    # The preview and cursor coordinates remain mirrored.
                    tracking_height = round(raw_frame.shape[0] * tracking_width / raw_frame.shape[1])
                    small_frame = cv2.resize(raw_frame, (tracking_width, tracking_height),
                                             interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp = max(time.monotonic_ns() // 1_000_000, last_timestamp + 1)
                    last_timestamp = timestamp
                    detection_started = time.perf_counter()
                    result = tracker.detect_for_video(image, timestamp)
                    detection_ms = (time.perf_counter() - detection_started) * 1000
                    model_ms = (0.8 * model_ms + 0.2 * detection_ms
                                if model_ms else detection_ms)
                    hands = hands_by_side(result)
                    side = preferred_side(hands, dominant)
                    other = "Left" if side == "Right" else "Right"
                    click_side = other if other in hands else side
                    if side != active_side or click_side != active_click_side:
                        mouse.release()
                        scroll.reset()
                        active_side = side
                        active_click_side = click_side
                    toggled = controls.update(hands.get(side))
                    if toggled:
                        mouse.release(close_switcher=True)
                        scroll.reset()
                        triangle.update(False, time.monotonic())
                        gesture_status = None
                        status = f"Controls {'ON' if controls.enabled else 'OFF'}"
                    elif not controls.enabled:
                        triangle.update(False, time.monotonic())
                        scroll.reset()
                        status = "Controls OFF | Green thumb+pinky to enable"
                    else:
                        matched = (not is_pointing_pose(hands.get(side))
                                   and is_triangle(hands.get("Left"), hands.get("Right"),
                                                   frame.shape[1] / frame.shape[0]))
                        fired = triangle.update(matched, time.monotonic())
                        if matched:
                            mouse.release(close_switcher=True)
                            scroll.reset()
                            if fired:
                                gesture_status = open_chrome_profile_picker()
                            status = gesture_status if triangle.fired else "Triangle: hold for Chrome profiles"
                        else:
                            gesture_status = None
                            scroll_hand = (hands.get(click_side)
                                           if click_side != side and not mouse.state.switcher_open
                                           else None)
                            scrolling, scroll_steps = scroll.update(scroll_hand, now)
                            status = mouse.update(
                                hands.get(side), hands.get(click_side),
                                allow_right_click=click_side != side,
                                actions_enabled=not scrolling,
                            )
                            if scroll_steps:
                                pyautogui.scroll(scroll_steps * SCROLL_UNITS_PER_STEP)
                                status = ("Scroll up" if scroll_steps > 0 else "Scroll down")
                            if side:
                                status = (f"{side[0]} pointer / {click_side[0]} clicks: {status}"
                                          if side != click_side else f"{side} hand: {status}")
                    for label, hand in hands.items():
                        draw_hand(cv2, frame, hand,
                                  (70, 230, 120) if label == side else (230, 170, 70))
                    if scroll.anchor_y is not None:
                        height, width = frame.shape[:2]
                        anchor = (round(scroll.anchor_x * width),
                                  round(scroll.anchor_y * height))
                        current = (round(scroll.current_x * width),
                                   round(scroll.current_y * height))
                        cv2.line(frame, anchor, current, (80, 190, 245), 2, cv2.LINE_AA)
                        cv2.circle(frame, anchor, 11, (30, 145, 235), -1, cv2.LINE_AA)
                        cv2.circle(frame, anchor, 5, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, current, 5, (80, 190, 245), -1, cv2.LINE_AA)
                        direction = "UP" if scroll.velocity > 0 else "DOWN" if scroll.velocity < 0 else "STOP"
                        cv2.putText(frame, f"{direction} {abs(scroll.velocity * SCROLL_UNITS_PER_STEP):.0f}/s",
                                    (max(8, min(anchor[0] + 15, width - 115)),
                                     max(76, min(anchor[1] - 12, height - 40))),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                    (255, 255, 255), 2, cv2.LINE_AA)
                height, width = frame.shape[:2]
                cv2.rectangle(frame,
                              (round(width * CONTROL_MARGIN), round(height * CONTROL_MARGIN)),
                              (round(width * (1 - CONTROL_MARGIN)),
                               round(height * (1 - CONTROL_MARGIN))),
                              (80, 200, 245), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (0, 0), (width, 58), (25, 28, 32), -1)
                cv2.putText(frame, status, (12, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"{fps:.0f} FPS", (frame.shape[1] - 82, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Other hand: V from point = scroll | Green thumb+pinky: toggle",
                            (12, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 215, 200), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (0, height - 34), (width, height), (25, 28, 32), -1)
                cv2.putText(frame, f"{'ON' if controls.enabled else 'OFF'} | H: {dominant} pointer | Triangle: Chrome profiles | Space pause | Q quit",
                            (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 215, 200), 1, cv2.LINE_AA)
                cv2.imshow(window_name, frame)
                if now - last_title_update >= 1:
                    cv2.setWindowTitle(window_name,
                                       f"Hand Mouse | {fps:.0f} FPS | Model {model_ms:.0f} ms | Space: pause | Q / Esc: quit")
                    last_title_update = now
                remaining = 1 / fps_limit - (time.monotonic() - now)
                if remaining > 0:
                    time.sleep(remaining)
    except pyautogui.FailSafeException:
        print("Stopped by PyAutoGUI fail-safe (mouse moved to a screen corner).")
    except cv2.error as error:
        print(f"OpenCV error: {error}", file=sys.stderr)
        return 1
    finally:
        try:
            mouse.release(close_switcher=True)
        except pyautogui.FailSafeException:
            pass
        video.close()
        cv2.destroyAllWindows()
        if precise_timer:
            ctypes.windll.winmm.timeEndPeriod(1)
    return 0


def main() -> int:
    project_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
    if project_python.is_file() and Path(sys.executable).resolve() != project_python.resolve():
        return subprocess.call([str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--smoothing", type=float, default=0.6,
                        help="Cursor response from 0 (slow) to 1 (immediate); default: 0.6")
    parser.add_argument("--tracking-width", type=int, default=256,
                        help="Detection image width; lower is faster (default: 256)")
    parser.add_argument("--dominant-hand", choices=("right", "left"), default="right",
                        help="Preferred pointer hand; press H in the preview to switch")
    parser.add_argument("--fps", type=int, choices=FPS_CHOICES, default=20,
                        help="Tracking FPS limit; default: 20")
    args = parser.parse_args()
    if args.camera < 0 or not 0 < args.smoothing <= 1 or not 160 <= args.tracking_width <= 640:
        parser.error("--camera must be nonnegative, --smoothing in (0, 1], and --tracking-width in [160, 640]")
    return run(args.camera, args.smoothing, args.tracking_width,
               args.dominant_hand.title(), args.fps)


if __name__ == "__main__":
    raise SystemExit(main())
