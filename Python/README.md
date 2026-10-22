# Hand Mouse

Hand Mouse turns a Windows webcam into a two-hand mouse controller. MediaPipe tracks hands, OpenCV displays the preview, and PyAutoGUI sends mouse input. Camera video stays on the PC.

## Run from Python source

Install 64-bit Python 3.11 or 3.12. In PowerShell, from this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\hand_mouse.py
```

After setup, double-click `run.cmd` to start the app, or use the last PowerShell command above. The first launch downloads and verifies the hand model if it is not already in `models/`. Allow desktop apps to use the camera in Windows settings. Press **Q** or **Esc** in the preview to quit.

`hand_mouse.py` is also the direct app entry point: `python .\hand_mouse.py` uses the local `.venv` automatically when it exists. Double-clicking a `.py` file depends on Windows' Python file association; use `run.cmd` for a reliable double-click start.

If `py -3.12` is unavailable, use `python -m venv .venv`. For another webcam, run `run.cmd --camera 1` from a terminal. The default tracking cap remains **20 FPS**; the live rate can be lower when the camera or model takes more than 50 ms per frame.

This is a source-based app. Windows still launches an installed Python interpreter to run it; the project no longer builds or distributes its own EXE. If `build/` or `dist/` exists from an older version, those are obsolete generated files and can be removed. Follow the device owner's application policy when using it on a managed PC.

## Project files

| Path | Purpose |
| --- | --- |
| `hand_mouse.py` | Camera, tracking, gestures, mouse control, and command-line entry point |
| `run.cmd` | Double-click launcher for the local Python environment |
| `requirements.txt` | Python dependencies |
| `models/` | Verified MediaPipe hand model cache |
| `tests/` | Behavior and camera startup tests |
| `web/` | Visual gesture guide |
| `docs/` | Controls, setup, and performance notes |

## Documentation

- [Controls and gestures](docs/CONTROLS.md)
- [Setup and troubleshooting](docs/DEVELOPMENT.md)
- [Startup and performance](docs/PERFORMANCE.md)
- [Visual gesture guide](web/Hand%20Mouse%20Gesture%20Guide.html)

The preview keeps processing keyboard input while waiting for camera frames. Held input is released after 250 ms without a fresh frame. Close the preview or press Q/Esc to quit; Space pauses control.
