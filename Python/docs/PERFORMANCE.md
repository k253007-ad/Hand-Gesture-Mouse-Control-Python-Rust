# Startup and performance

## Frame rate

The app caps preview and tracking at **20 FPS** by default. Available `--fps` limits are 12, 15, 20, 24, 30, and 36. The cap does not guarantee the camera or model can deliver 20 frames each second. The preview title reports the measured rate and smoothed model time. If model time rises above 50 ms, a steady 20 FPS is unlikely.

Hand tracking runs on a 256-pixel-wide copy of the camera image. The preview retains the camera frame size. `--tracking-width 240` can reduce image work; `--tracking-width 320` or `480` can improve fine fingertip detail at a CPU cost. Old webcam frames are dropped so the pointer uses the latest available image.

## Measured startup bottleneck

On the development PC, opening the camera through Media Foundation sometimes took about 20 seconds. Local DirectShow probes opened in about 0.8–3.6 seconds and supplied about 30 FPS after explicitly requesting `CAP_PROP_FPS = 30` at 640×480. The Python app tries DirectShow first and falls back to Media Foundation if opening fails. Startup also includes Python imports and the camera connection.

These measurements depend on the webcam, driver, lighting, and PC load. The hand model has separately ranged from about 40 ms to over 80 ms per frame in local probes. That is why a 20 FPS cap can sometimes display a lower live rate. A smaller tracking image does not consistently remove model cost.

The Windows default timer made `cv2.waitKey(1)` take about 15.4 ms in a focused local probe. Requesting 1 ms timer resolution reduced it to about 2 ms. The app requests this resolution only while tracking, then releases it on exit.

The app runs directly from Python source. Startup time depends on the installed Python environment, imports, model cache, and camera connection. The current workflow does not create a packaged EXE release.
