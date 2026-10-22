# Hand Mouse (Rust)

Windows webcam hand tracking with 21 three-dimensional landmarks, a bounded newest-frame camera queue, ONNX Runtime inference, and gesture mouse control. Tracking, gestures, and mouse input are implemented in Rust. When Media Foundation cannot open a Windows camera, the optional DirectShow fallback uses the adjacent Python environment for frame capture.

The third landmark coordinate is relative hand depth inferred from a single camera. It is not a measured distance from the webcam.

## Build and run

After installing Rust and the C++ build tools, double-click `run.cmd`. It discovers the installed Visual Studio tools, compiles changes when needed, then launches the app. A local binary is produced under `target/`; the project does not currently distribute a standalone EXE package. Developer-specific SDK settings can go in `run.local.cmd`, which is excluded from Git.

For another Windows PC, install [Rust](https://www.rust-lang.org/tools/install), the Microsoft C++ build tools, and a Windows SDK. From this folder:

```powershell
cargo run --release
```

Start with `run.cmd --tracking-only` to preview landmarks without sending input. The preview displays the camera image and tracking box. Run `run.cmd --help` to see available options; unknown options and invalid values return an error before camera startup.

Use `cargo run --release -- --camera 1` for another webcam and `--left` for a left-hand pointer preference. Use `--tracking-only` to preview hand landmarks without sending mouse or keyboard input. Press **H** to switch the preferred pointer hand, **Space** to pause, and **Q** or **Esc** to exit. The window title shows app FPS, camera FPS, and smoothed model time. The processing loop has no FPS cap. The preview window is 75% of the camera image size, and its center box maps to the screen. The preview is a normal window so it does not cover applications you need to click.

The ONNX weights are included in `models/`; no model download is needed at launch. CPU inference is the default. Use `--cuda` only if a CUDA-capable ONNX Runtime and its matching NVIDIA dependencies are installed. CoreML is omitted from this Windows build because the Windows ONNX Runtime binary does not export its CoreML provider.

The Windows camera normally uses Media Foundation. If it cannot open the camera, the app tries `camera_bridge.py` with the Python project's `.venv` and OpenCV DirectShow. Set up the adjacent Python environment using [its instructions](../Python/README.md) to enable that fallback. This fallback captures and sends RGB frames; tracking and controls still run in Rust. To measure tracking without mouse input, run `run.cmd --benchmark-seconds 10`; timed benchmarks automatically disable mouse and keyboard output. The console prints achieved FPS and average model time when it exits.

Held input is released after 250 ms without a new camera frame. The preview continues processing keys while capture is stalled. Shutdown signals the camera worker and kills its optional DirectShow child, including when the child is blocked reading the camera. A native SDK call may remain blocked until process exit; waiting for the worker is bounded to 500 ms.

For an identical-frame model comparison, record a short RGB clip with `Python/record_frames.py`, then replay it with `Python/benchmark_tracker.py --input FILE` and `Rust/run.cmd --benchmark-file FILE`. The `.hmr2` recording contains camera images; keep it local and remove it when finished.

## Gestures

The preferred pointer hand is green. The other hand handles clicks and shortcuts when both are visible. When one hand is visible, it moves the pointer and handles its supported gestures.

| Gesture | Action |
| --- | --- |
| Only index extended on green hand | Point without green-hand gestures |
| Thumb + index | Left click; hold to drag |
| Green thumb + pinky for two frames | Toggle controls |
| Other thumb + pinky | Right click |
| Green thumb + middle | Windows Start key |
| Other thumb + middle | Double click |
| Thumb + ring | Persistent Windows window picker |
| Other hand V, then hold above/below the starting point | Continuous distance-based scrolling, 30 wheel units per step |
| Both hands form a triangle for 0.65 seconds | Open Chrome profile picker |

With only one hand, thumb + index remains the left click gesture. The former thumb-to-index-base gesture is not included.

## Implementation notes

The capture thread drops a queued frame when a newer one arrives. Inference runs on the main thread. Palm detection uses a 192×192 letterboxed image, then a 224×224 rotated crop feeds the hand landmark network. Once a hand is found, the next crop is estimated from its landmarks; palm detection runs periodically to find new hands and correct drift, or immediately when tracking is lost. The ONNX input tensors are allocated once. `OneEuro` smooths pointer movement; gestures use the current landmarks so a brief pinch is not hidden by smoothing. Click events use Windows `SendInput`. Scrolling uses the distance from the V pose's starting point, so holding the hand away from that point continues scrolling.

The camera decoder and ONNX Runtime own output buffers internally and may allocate per frame; a literal zero-allocation end-to-end loop is not supported by these APIs. The camera currently requests 640×480 MJPEG at 30 FPS, so actual FPS depends on that camera mode and CPU/GPU speed. Gesture thresholds and scroll behavior follow the Python controller, while the Rust tracker uses separate ONNX models and must still be checked with physical hands for recognition parity.

## Model attribution

The palm and landmark ONNX files come from [yakhyo/mediapipe-hand-landmark-onnx](https://github.com/yakhyo/mediapipe-hand-landmark-onnx), Apache-2.0. That repository documents that the model topology originates from Google MediaPipe and weights from PINTO0309 conversions. See [`MODEL_LICENSE.txt`](MODEL_LICENSE.txt).
