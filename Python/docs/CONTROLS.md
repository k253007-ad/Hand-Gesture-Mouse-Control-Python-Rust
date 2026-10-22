# Controls and gestures

The right hand is the default pointer hand. The pointer hand is drawn green; the other hand handles clicks and shortcuts when both are visible. Press **H** in the preview to choose the left hand, or start with `python hand_mouse.py --dominant-hand left`. If only one hand is visible, it moves the pointer and handles gestures.

| Gesture or key | Action |
| --- | --- |
| Point with only the index finger | Move the mouse without executing other green-hand gestures |
| Touch thumb to index, then release | Left click on the action hand; hold and move to drag |
| Touch thumb to pinky on the green hand | Turn mouse and gesture controls on or off; separate fingers before toggling again |
| Touch thumb to pinky on the other hand | Right click when both hands are visible |
| Touch thumb to middle finger on the green hand | Press the Windows key once per pinch to open Start |
| Touch thumb to middle finger on the other hand | Double click when both hands are visible |
| Touch thumb to ring finger | Open the Windows Alt+Tab thumbnail picker |
| Point at a thumbnail, then touch thumb to index | Select that window |
| Show a V with index and middle fingers on the other hand | Start scrolling from the displayed point |
| Bring both index fingertips together at the top with thumbs spread below; hold about 0.7 seconds | Open Chrome's “Who's using Chrome?” profile picker |
| **H** | Switch preferred pointer hand |
| **Space** | Pause or resume; releases a drag |
| **Q** or **Esc** | Quit |
| Move the physical mouse to a screen corner | Emergency stop through PyAutoGUI fail-safe |

## Scrolling

Keep only the index and middle fingers extended on the non-green hand. The app marks the starting palm position. Move above that point to scroll up or below it to scroll down. Greater displacement increases scrolling velocity. Return to the point to stop; fold the V to reset the point. Each step sends 30 wheel units. Scrolling needs both hands and stops when controls are off.

## Gesture priority and layout

With only the green index finger extended, pointing takes priority even if the tucked thumb touches the pinky. Open the other fingers before using green thumb and pinky to toggle controls. The other hand can still act while the green hand points.

A triangle needs both hands. It opens Chrome once per held gesture; separate the hands before triggering it again. Mouse movement and clicking pause while the triangle is held. The thumb-ring gesture opens the persistent Windows window picker. Release the pinch, point at a thumbnail, then pinch thumb-index to select it. Press **Esc** to cancel. Pausing or closing the app also dismisses the picker.

The preview is 75% of the camera image size. Its outlined center box is 75% of the frame width and height; move the pointer finger inside that box to cover the full screen. Going beyond the box clamps the cursor to the corresponding edge.

Use a well-lit space and keep both hands fully visible. Pinch thresholds scale with palm size. `--smoothing 0.6` is the default; larger values respond faster, smaller values settle more steadily. The controller uses the primary display.
