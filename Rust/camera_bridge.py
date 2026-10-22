"""DirectShow frame source for Windows cameras that reject Media Foundation.

Frames are written to stdout as b'HMR1' + little-endian width/height + RGB888.
Only camera capture and color conversion run here; Rust performs tracking.
"""

import argparse
import errno
import struct
import sys

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, required=True)
    args = parser.parse_args()
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError(f"DirectShow could not open camera {args.camera}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    output = sys.stdout.buffer
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("DirectShow stopped supplying frames")
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output.write(struct.pack("<4sII", b"HMR1", width, height))
            output.write(rgb.tobytes())
    except OSError as error:
        # Windows may report a closed stdout pipe as EINVAL instead of EPIPE.
        if error.errno in (errno.EPIPE, errno.EINVAL):
            return 0
        raise
    finally:
        camera.release()


if __name__ == "__main__":
    raise SystemExit(main())
