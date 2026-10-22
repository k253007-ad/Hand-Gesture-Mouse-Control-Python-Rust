import unittest
import time
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import hand_mouse
from hand_mouse import (ControlToggle, HandMouse, LatestFrameCamera, ScrollGesture, TriangleTrigger,
                        hands_by_side, is_pointing_pose, is_scroll_pose,
                        is_triangle, preferred_side)


def hand(index_x, thumb_x, apex_y=0.30, thumb_y=0.56):
    points = [SimpleNamespace(x=index_x, y=0.48, z=0) for _ in range(21)]
    points[4] = SimpleNamespace(x=thumb_x, y=thumb_y, z=0)
    points[8] = SimpleNamespace(x=index_x, y=apex_y, z=0)
    points[5] = SimpleNamespace(x=index_x - 0.04, y=0.48, z=0)
    points[17] = SimpleNamespace(x=index_x + 0.04, y=0.48, z=0)
    return points


def v_hand(offset_y=0.0):
    points = [SimpleNamespace(x=0.5, y=0.7 + offset_y, z=0) for _ in range(21)]
    locations = {
        4: (0.25, 0.55), 5: (0.42, 0.60), 6: (0.42, 0.42), 8: (0.40, 0.23),
        9: (0.52, 0.60), 10: (0.55, 0.42), 12: (0.60, 0.23),
        13: (0.61, 0.62), 14: (0.64, 0.50), 16: (0.56, 0.62),
        17: (0.68, 0.65), 18: (0.71, 0.54), 20: (0.62, 0.68),
    }
    for index, (x, y) in locations.items():
        points[index] = SimpleNamespace(x=x, y=y + offset_y, z=0)
    return points


def pointing_hand():
    points = v_hand()
    points[12] = SimpleNamespace(x=0.55, y=0.65, z=0)
    # The tucked thumb can be near the folded pinky without toggling controls.
    points[4] = points[20]
    return points


class HandMouseTests(unittest.TestCase):
    def test_camera_poll_does_not_block_the_ui_during_a_stall(self):
        finish = threading.Event()

        class StalledCapture:
            def read(self):
                finish.wait(0.5)
                return False, None

            def release(self):
                finish.set()

        video = LatestFrameCamera(StalledCapture())
        try:
            started = time.monotonic()
            _, frame = video.read_latest(0)
            self.assertIsNone(frame)
            self.assertLess(time.monotonic() - started, 0.15)
        finally:
            finish.set()
            video.close()

    def test_stale_frames_release_input_but_normal_frame_gaps_do_not(self):
        mouse, scroll = Mock(), Mock()
        video = SimpleNamespace(read_latest=lambda seq: (seq, None), stalled=False)
        hand_mouse.poll_control_frame(video, 1, mouse, scroll)
        mouse.release.assert_not_called()
        video.stalled = True
        hand_mouse.poll_control_frame(video, 1, mouse, scroll)
        mouse.release.assert_called_once_with(close_switcher=True)
        scroll.reset.assert_called_once_with()

    def test_camera_recovers_after_transient_empty_frames(self):
        class FakeCapture:
            def __init__(self):
                self.reads = 0
                self.released = False

            def read(self):
                time.sleep(0.005)
                self.reads += 1
                if self.reads <= 2:
                    return False, None
                return True, "frame"

            def release(self):
                self.released = True

        capture = FakeCapture()
        video = LatestFrameCamera(capture, failure_timeout=1)
        try:
            deadline = time.monotonic() + 1
            sequence, frame = video.read_latest(-1)
            while frame is None and time.monotonic() < deadline:
                sequence, frame = video.read_latest(sequence)
            self.assertGreater(sequence, 0)
            self.assertEqual(frame, "frame")
            self.assertTrue(video.running)
        finally:
            video.close()
        self.assertTrue(capture.released)

    def test_index_only_green_hand_moves_without_actions_or_toggle(self):
        class FakeMouse:
            def __init__(self):
                self.events = []

            def size(self):
                return (1000, 800)

            def moveTo(self, x, y):
                self.events.append(("move", x, y))

            def mouseDown(self, button):
                self.events.append(("down", button))

            def mouseUp(self, button):
                self.events.append(("up", button))

            def click(self, button):
                self.events.append(("click", button))

            def doubleClick(self, interval):
                self.events.append(("double", interval))

            def hotkey(self, *keys):
                self.events.append(("hotkey", keys))

        green = pointing_hand()
        self.assertTrue(is_pointing_pose(green))
        toggle = ControlToggle()
        controller = HandMouse(FakeMouse())
        for _ in range(3):
            self.assertFalse(toggle.update(green))
            self.assertEqual(controller.update(green), "Point only")
        self.assertTrue(toggle.enabled)
        self.assertEqual(len(controller.mouse.events), 3)
        self.assertTrue(all(event[0] == "move" for event in controller.mouse.events))

        other = hand(0.2, 0.2)
        other[4] = other[8]
        controller.update(green, other)
        self.assertIn(("down", "left"), controller.mouse.events)

    def test_camera_prefers_media_foundation_and_falls_back(self):
        class FakeCapture:
            def __init__(self, opened):
                self.opened = opened
                self.released = False
                self.properties = []

            def isOpened(self):
                return self.opened

            def set(self, key, value):
                self.properties.append((key, value))

            def release(self):
                self.released = True

        class FakeCv2:
            CAP_MSMF = 1
            CAP_DSHOW = 2
            CAP_PROP_FRAME_WIDTH = 3
            CAP_PROP_FRAME_HEIGHT = 4
            CAP_PROP_FPS = 5
            CAP_PROP_BUFFERSIZE = 6

            def __init__(self, captures):
                self.captures = iter(captures)
                self.backends = []

            def VideoCapture(self, index, backend):
                self.backends.append((index, backend))
                return next(self.captures)

        fast = FakeCapture(True)
        cv2 = FakeCv2([fast])
        with patch.object(hand_mouse.sys, "platform", "win32"):
            self.assertIs(hand_mouse.open_camera(cv2, 0), fast)
        self.assertEqual(cv2.backends, [(0, cv2.CAP_DSHOW)])
        self.assertEqual(fast.properties,
                         [(cv2.CAP_PROP_FRAME_WIDTH, 640),
                          (cv2.CAP_PROP_FRAME_HEIGHT, 480),
                          (cv2.CAP_PROP_FPS, 30),
                          (cv2.CAP_PROP_BUFFERSIZE, 1)])

        unavailable = FakeCapture(False)
        fallback = FakeCapture(True)
        cv2 = FakeCv2([unavailable, fallback])
        with patch.object(hand_mouse.sys, "platform", "win32"):
            self.assertIs(hand_mouse.open_camera(cv2, 0), fallback)
        self.assertTrue(unavailable.released)
        self.assertEqual(cv2.backends, [(0, cv2.CAP_DSHOW), (0, cv2.CAP_MSMF)])

    def test_v_swipe_scrolls_in_both_directions_and_resets(self):
        scroll = ScrollGesture()
        self.assertTrue(is_scroll_pose(v_hand()))
        self.assertEqual(scroll.update(v_hand(), now=0.0), (True, 0))
        self.assertIsNotNone(scroll.anchor_y)
        active, up = scroll.update(v_hand(-0.14), now=0.25)
        self.assertTrue(active)
        self.assertGreater(up, 0)
        active, down = scroll.update(v_hand(0.14), now=0.50)
        self.assertTrue(active)
        self.assertLess(down, 0)
        self.assertEqual(scroll.update(None, now=0.75), (False, 0))
        self.assertIsNone(scroll.anchor_y)
        self.assertEqual(scroll.update(v_hand(), now=1.0), (True, 0))

    def test_scroll_speed_follows_distance_from_fixed_point(self):
        scroll = ScrollGesture()
        scroll.update(v_hand(), now=0.0)
        anchor = (scroll.anchor_x, scroll.anchor_y)
        palm = scroll.anchor_palm

        self.assertEqual(scroll.update(v_hand(-palm * 0.1), now=0.25), (True, 0))
        self.assertEqual(scroll.velocity, 0)
        scroll.update(v_hand(-palm * 0.5), now=0.50)
        slower = scroll.velocity
        active, faster_steps = scroll.update(v_hand(-palm), now=0.75)
        self.assertTrue(active)
        self.assertGreater(scroll.velocity, slower)
        self.assertGreater(faster_steps, 0)
        active, continuing_steps = scroll.update(v_hand(-palm), now=1.0)
        self.assertTrue(active)
        self.assertGreater(continuing_steps, 0)
        self.assertEqual(scroll.update(v_hand(), now=1.25), (True, 0))
        self.assertEqual(scroll.velocity, 0)
        self.assertEqual(scroll.update(v_hand(), now=1.50), (True, 0))
        self.assertEqual((scroll.anchor_x, scroll.anchor_y), anchor)

    def test_scroll_mode_moves_pointer_without_clicking(self):
        class FakeMouse:
            def __init__(self):
                self.events = []

            def size(self):
                return (1000, 800)

            def moveTo(self, x, y):
                self.events.append("move")

            def mouseDown(self, button):
                self.events.append("click")

        fake = FakeMouse()
        pointer = hand(0.8, 0.7)
        clicks = hand(0.2, 0.2)
        clicks[4] = clicks[8]
        HandMouse(fake).update(pointer, clicks, actions_enabled=False)
        self.assertEqual(fake.events, ["move"])

    def test_green_thumb_pinky_toggles_once_until_released(self):
        toggle = ControlToggle()
        green = hand(0.8, 0.7)
        green[20] = green[4]
        self.assertFalse(toggle.update(green))
        self.assertTrue(toggle.update(green))
        self.assertFalse(toggle.enabled)
        self.assertFalse(toggle.update(green))
        self.assertFalse(toggle.update(None))
        self.assertFalse(toggle.update(green))
        green[20] = SimpleNamespace(x=0.95, y=0.48, z=0)
        self.assertFalse(toggle.update(green))
        self.assertFalse(toggle.update(green))
        green[20] = green[4]
        self.assertFalse(toggle.update(green))
        self.assertTrue(toggle.update(green))
        self.assertTrue(toggle.enabled)

    def test_green_thumb_pinky_does_not_right_click(self):
        class FakeMouse:
            def __init__(self):
                self.events = []

            def size(self):
                return (1000, 800)

            def moveTo(self, x, y):
                self.events.append(("move", x, y))

            def click(self, button):
                self.events.append(("click", button))

        fake = FakeMouse()
        green = hand(0.8, 0.7)
        green[20] = green[4]
        HandMouse(fake).update(green, green, allow_right_click=False)
        self.assertFalse(any(event[0] == "click" for event in fake.events))

    def test_green_thumb_middle_presses_windows_once_per_pinch(self):
        class FakeMouse:
            def __init__(self):
                self.events = []

            def size(self):
                return (1000, 800)

            def moveTo(self, x, y):
                pass

            def press(self, key):
                self.events.append(("press", key))

            def doubleClick(self, interval):
                self.events.append(("double", interval))

        fake = FakeMouse()
        controller = HandMouse(fake)
        green = hand(0.8, 0.7)
        green[12] = green[4]
        controller.update(green, green)
        controller.update(green, green)
        self.assertEqual(fake.events, [("press", "win")])

        green[12] = SimpleNamespace(x=0.8, y=0.25, z=0)
        controller.update(green, green)
        green[12] = green[4]
        controller.update(green, green)
        self.assertEqual(fake.events, [("press", "win"), ("press", "win")])

    def test_green_thumb_to_index_base_does_not_click(self):
        class FakeMouse:
            def __init__(self):
                self.events = []

            def size(self):
                return (1000, 800)

            def moveTo(self, x, y):
                pass

            def click(self, button):
                self.events.append(("click", button))

        fake = FakeMouse()
        controller = HandMouse(fake)
        green = hand(0.8, 0.7)
        green[4] = green[5]
        controller.update(green, green)
        controller.update(green, green)
        controller.update(green, green)
        self.assertEqual(fake.events, [])

    def test_right_pointer_left_clicks_and_pinky_right_click(self):
        class FakeMouse:
            def __init__(self):
                self.events = []

            def size(self):
                return (1000, 800)

            def moveTo(self, x, y):
                self.events.append(("move", x, y))

            def mouseDown(self, button):
                self.events.append(("down", button))

            def mouseUp(self, button):
                self.events.append(("up", button))

            def click(self, button):
                self.events.append(("click", button))

            def doubleClick(self, interval):
                self.events.append(("double", interval))

        fake = FakeMouse()
        controller = HandMouse(fake, smoothing=1)
        right = hand(0.80, 0.80)
        left = hand(0.20, 0.20)
        # Right-hand pinch must not click when the left hand owns clicks.
        right[4] = right[8]
        controller.update(right, left)
        self.assertFalse(any(event[0] == "down" for event in fake.events))
        self.assertGreater(fake.events[0][1], 800)

        left[4] = left[8]
        controller.update(right, left)
        self.assertIn(("down", "left"), fake.events)

        left[4] = SimpleNamespace(x=0.34, y=0.56, z=0)
        left[20] = left[4]
        controller.update(right, left)
        self.assertIn(("up", "left"), fake.events)
        self.assertEqual(fake.events.count(("click", "right")), 1)
        controller.update(right, left)
        self.assertEqual(fake.events.count(("click", "right")), 1)

    def test_thumb_middle_double_click_fires_once_per_pinch(self):
        class FakeMouse:
            def __init__(self):
                self.events = []

            def size(self):
                return (1000, 800)

            def moveTo(self, x, y):
                pass

            def doubleClick(self, interval):
                self.events.append("double")

            def click(self, button):
                self.events.append(button)

            def mouseDown(self, button):
                self.events.append("down")

        fake = FakeMouse()
        controller = HandMouse(fake)
        pointer = hand(0.8, 0.7)
        clicks = hand(0.2, 0.2)
        clicks[12] = clicks[4]
        controller.update(pointer, clicks)
        controller.update(pointer, clicks)
        self.assertEqual(fake.events, ["double"])
        clicks[12] = SimpleNamespace(x=0.45, y=0.48, z=0)
        controller.update(pointer, clicks)
        clicks[12] = clicks[4]
        controller.update(pointer, clicks)
        self.assertEqual(fake.events, ["double", "double"])

    def test_thumb_ring_opens_picker_and_index_click_selects_window(self):
        class FakeMouse:
            def __init__(self):
                self.events = []

            def size(self):
                return (1000, 800)

            def moveTo(self, x, y):
                pass

            def hotkey(self, *keys):
                self.events.append(keys)

            def click(self, button):
                self.events.append(("click", button))

            def press(self, key):
                self.events.append(("press", key))

        fake = FakeMouse()
        controller = HandMouse(fake)
        pointer = hand(0.8, 0.7)
        actions = hand(0.2, 0.2)
        actions[16] = actions[4]
        controller.update(pointer, actions)
        controller.update(pointer, actions)
        self.assertEqual(fake.events, [("ctrl", "alt", "tab")])
        self.assertTrue(controller.state.switcher_open)
        controller.update(None)
        self.assertTrue(controller.state.switcher_open)
        actions[16] = SimpleNamespace(x=0.45, y=0.48, z=0)
        controller.update(pointer, actions)
        actions[4] = actions[8]
        controller.update(pointer, actions)
        controller.update(pointer, actions)
        self.assertEqual(fake.events, [("ctrl", "alt", "tab"), ("click", "left")])
        self.assertFalse(controller.state.switcher_open)
        actions[4] = SimpleNamespace(x=0.2, y=0.56, z=0)
        controller.update(pointer, actions)
        actions[16] = actions[4]
        controller.update(pointer, actions)
        self.assertEqual(fake.events[-1], ("ctrl", "alt", "tab"))
        controller.release(close_switcher=True)
        self.assertEqual(fake.events[-1], ("press", "esc"))

    def test_right_hand_wins_and_left_takes_over(self):
        self.assertEqual(preferred_side({"Left": object(), "Right": object()}), "Right")
        self.assertEqual(preferred_side({"Left": object(), "Right": object()}, "Left"), "Left")
        self.assertEqual(preferred_side({"Left": object()}), "Left")
        self.assertEqual(preferred_side({"Right": object()}, "Left"), "Right")
        self.assertIsNone(preferred_side({}))

    def test_handedness_is_mapped_to_mirrored_preview(self):
        raw = hand(0.2, 0.1)
        category = SimpleNamespace(category_name="Right", score=0.95)
        result = SimpleNamespace(hand_landmarks=[raw], handedness=[[category]])
        mapped = hands_by_side(result)
        self.assertAlmostEqual(mapped["Right"][8].x, 0.8)

    def test_triangle_requires_apex_and_base(self):
        left = hand(0.49, 0.30)
        right = hand(0.51, 0.70)
        self.assertTrue(is_triangle(left, right, 4 / 3))
        self.assertFalse(is_triangle(left, None, 4 / 3))
        self.assertFalse(is_triangle(left, hand(0.82, 0.70), 4 / 3))
        self.assertFalse(is_triangle(left, hand(0.51, 0.70, thumb_y=0.30), 4 / 3))

    def test_triangle_opens_once_until_released(self):
        trigger = TriangleTrigger(hold_seconds=0.65)
        self.assertFalse(trigger.update(True, 1.0))
        self.assertFalse(trigger.update(True, 1.5))
        self.assertTrue(trigger.update(True, 1.7))
        self.assertFalse(trigger.update(True, 3.0))
        self.assertFalse(trigger.update(False, 3.1))
        self.assertFalse(trigger.update(True, 4.0))
        self.assertTrue(trigger.update(True, 4.7))

    def test_triangle_gesture_opens_chrome_profile_picker(self):
        with (patch.object(hand_mouse, "find_chrome", return_value=r"C:\Chrome\chrome.exe"),
              patch.object(hand_mouse.subprocess, "Popen") as launch):
            self.assertEqual(hand_mouse.open_chrome_profile_picker(), "Chrome profile picker opened")
            self.assertEqual(launch.call_args.args[0],
                             [r"C:\Chrome\chrome.exe", "chrome://profile-picker/"])


if __name__ == "__main__":
    unittest.main()
