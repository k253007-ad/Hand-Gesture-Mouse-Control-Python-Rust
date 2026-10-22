"""Measure Python hand-tracking throughput without sending mouse input."""

import argparse
import statistics
import struct
import time

import cv2
import mediapipe as mp
import numpy as np

from hand_mouse import ensure_model, open_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--input", help="HMR2 RGB recording to replay instead of a live camera")
    args = parser.parse_args()
    if args.seconds <= 0 or args.seconds > 300:
        parser.error("--seconds must be between 0 and 300")
    camera = None
    recording = None
    if args.input:
        recording = open(args.input, "rb")
        magic, width, height, frame_count = struct.unpack("<4sIII", recording.read(16))
        if magic != b"HMR2" or not 1 <= width <= 4096 or not 1 <= height <= 4096 or not 1 <= frame_count <= 10_000:
            parser.error("invalid HMR2 recording")
    else:
        camera = open_camera(cv2, args.camera)
        if camera is None:
            parser.error(f"could not open camera {args.camera}")
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    timings = []
    frames = 0
    one_hand_frames = 0
    two_hand_frames = 0
    last_timestamp = -1
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as tracker:
            started = time.perf_counter()
            while (frames < frame_count if recording else
                   time.perf_counter() - started < args.seconds):
                if recording:
                    data = recording.read(width * height * 3)
                    if len(data) != width * height * 3:
                        raise RuntimeError("recording ended before its declared frame count")
                    frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
                else:
                    ok, frame = camera.read()
                    if not ok:
                        raise RuntimeError("camera stopped supplying frames")
                tracking_height = round(frame.shape[0] * 256 / frame.shape[1])
                small = cv2.resize(frame, (256, tracking_height), interpolation=cv2.INTER_AREA)
                rgb = small if recording else cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = (frames * 33 if recording else
                             max(time.monotonic_ns() // 1_000_000, last_timestamp + 1))
                last_timestamp = timestamp
                begun = time.perf_counter()
                result = tracker.detect_for_video(image, timestamp)
                if len(result.hand_landmarks) == 1:
                    one_hand_frames += 1
                elif len(result.hand_landmarks) == 2:
                    two_hand_frames += 1
                timings.append((time.perf_counter() - begun) * 1000)
                frames += 1
            elapsed = time.perf_counter() - started
    finally:
        if camera is not None:
            camera.release()
        if recording is not None:
            recording.close()
    print(f"Benchmark: frames={frames}, one_hand_frames={one_hand_frames}, "
          f"two_hand_frames={two_hand_frames}, elapsed_s={elapsed:.2f}, "
          f"app_fps={frames / elapsed:.2f}, model_ms_mean={statistics.mean(timings):.2f}")


if __name__ == "__main__":
    main()
