# Development

## Source setup

Install 64-bit Python 3.11 or 3.12. From the repository's `Python` folder in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\hand_mouse.py
```

If `py -3.12` is unavailable, use `python -m venv .venv`. Once the environment is ready, `run.cmd` provides a double-click start and forwards command-line options such as `--camera 1` and `--dominant-hand left`.

The app downloads Google's versioned hand model on first launch if needed, verifies its SHA-256 hash, and caches it in `models/`. Windows Camera privacy settings must permit desktop apps. Close other camera apps if the camera is busy.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -q
```

The source project keeps one entry-point module, tests in `tests/`, the hand model in `models/`, and user documentation in `docs/` and `web/`. The virtual environment, model cache, and Python bytecode are excluded from source control.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `run.cmd` says Python is missing | Create `.venv` and install `requirements.txt` using the setup commands above |
| Python reports a missing module | Run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` |
| Camera unavailable | Allow desktop camera access, close other camera apps, or try `run.cmd --camera 1` |
| Preview is slower than 20 FPS | The limit is a ceiling; see [performance notes](PERFORMANCE.md) and try `run.cmd --tracking-width 240` |
| Fingertips are unreliable | Increase `--tracking-width` to 320 or 480 and improve lighting |
| Controls are off | Touch green thumb to pinky with the other green fingers open, then release |
| Python is blocked by the device policy | Ask the device's policy administrator which Python installation or deployment they permit |

## Source material

- [MediaPipe Hand Landmarker Python API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker)
- [Google's hand model URL in its sample repo](https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/hand-landmarker.ts)
- [MediaPipe Python hand landmarker sample](https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb)
- [OpenCV Python](https://github.com/opencv/opencv-python)
- [PyAutoGUI mouse and fail-safe documentation](https://github.com/asweigart/pyautogui/blob/master/docs/quickstart.rst)

The project uses these libraries and model; it does not copy their example source code.

