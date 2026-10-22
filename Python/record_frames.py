"""Record RGB webcam frames for local Python/Rust tracker comparison."""

import argparse
import struct

import cv2

from hand_mouse import open_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=90)
    args = parser.parse_args()
    if not 1 <= args.frames <= 10_000:
        parser.error("--frames must be between 1 and 10000")
    camera = open_camera(cv2, args.camera)
    if camera is None:
        parser.error(f"could not open camera {args.camera}")
    try:
        with open(args.output, "wb") as output:
            output.write(b"HMR2" + b"\0" * 12)
            width = height = captured = 0
            for _ in range(args.frames):
                ok, frame = camera.read()
                if not ok:
                    break
                frame_height, frame_width = frame.shape[:2]
                if captured and (frame_width, frame_height) != (width, height):
                    raise RuntimeError("camera resolution changed during recording")
                width, height = frame_width, frame_height
                output.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
                captured += 1
            output.seek(0)
            output.write(struct.pack("<4sIII", b"HMR2", width, height, captured))
    finally:
        camera.release()
    print(f"Recorded {captured} frames at {width}x{height} to {args.output}")


if __name__ == "__main__":
    main()
