mod capture_worker;
mod cli;
mod gesture;
mod tracking;

use anyhow::{Context, Result};
use capture_worker::{CaptureControl, CaptureWorker};
use crossbeam_channel::{bounded, RecvTimeoutError, TrySendError};
use fast_image_resize::{images::Image, PixelType, Resizer};
use gesture::{distance, pinches, pointing, triangle, Hand, Scroll, Toggle};
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{
        CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution,
    },
    Camera,
};
use std::{
    fs::File,
    io::Read,
    path::PathBuf,
    process::{Command, Stdio},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};
use tracking::{TrackedHand, Tracker};

#[cfg(target_os = "windows")]
mod os {
    use windows_sys::Win32::Foundation::POINT;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        keybd_event, GetAsyncKeyState, SendInput, INPUT, INPUT_0, INPUT_MOUSE,
        MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP,
        MOUSEEVENTF_WHEEL, MOUSEINPUT, VK_CONTROL, VK_ESCAPE, VK_LWIN, VK_MENU, VK_TAB,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetCursorPos, GetSystemMetrics, SetCursorPos, SM_CXSCREEN, SM_CYSCREEN,
    };
    const KEYUP: u32 = 0x0002;

    pub fn screen() -> (i32, i32) {
        unsafe { (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)) }
    }
    pub fn move_to(x: i32, y: i32) {
        unsafe {
            SetCursorPos(x, y);
        }
    }
    fn input(flags: u32, data: u32) -> INPUT {
        INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx: 0,
                    dy: 0,
                    mouseData: data,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }

    fn send(events: &[INPUT]) -> u32 {
        let mut sent = unsafe {
            SendInput(
                events.len() as u32,
                events.as_ptr(),
                std::mem::size_of::<INPUT>() as i32,
            )
        };
        while sent > 0 && sent < events.len() as u32 {
            let remaining = &events[sent as usize..];
            let more = unsafe {
                SendInput(
                    remaining.len() as u32,
                    remaining.as_ptr(),
                    std::mem::size_of::<INPUT>() as i32,
                )
            };
            if more == 0 {
                break;
            }
            sent += more;
        }
        if sent != events.len() as u32 {
            eprintln!(
                "Mouse input: Windows accepted {sent}/{} events",
                events.len()
            );
        }
        sent
    }

    pub fn left_down() -> bool {
        send(&[input(MOUSEEVENTF_LEFTDOWN, 0)]) == 1
    }
    pub fn left_up() {
        send(&[input(MOUSEEVENTF_LEFTUP, 0)]);
    }
    pub fn right_click() -> bool {
        let sent = send(&[
            input(MOUSEEVENTF_RIGHTDOWN, 0),
            input(MOUSEEVENTF_RIGHTUP, 0),
        ]);
        if sent % 2 == 1 {
            send(&[input(MOUSEEVENTF_RIGHTUP, 0)]);
        }
        sent == 2
    }
    pub fn left_click() -> bool {
        let sent = send(&[input(MOUSEEVENTF_LEFTDOWN, 0), input(MOUSEEVENTF_LEFTUP, 0)]);
        if sent % 2 == 1 {
            left_up();
        }
        sent == 2
    }
    pub fn double_click() -> bool {
        let sent = send(&[
            input(MOUSEEVENTF_LEFTDOWN, 0),
            input(MOUSEEVENTF_LEFTUP, 0),
            input(MOUSEEVENTF_LEFTDOWN, 0),
            input(MOUSEEVENTF_LEFTUP, 0),
        ]);
        if sent % 2 == 1 {
            left_up();
        }
        sent == 4
    }
    pub fn wheel(units: i32) {
        send(&[input(MOUSEEVENTF_WHEEL, (units * 120) as u32)]);
    }
    fn key(vk: u16, up: bool) {
        unsafe {
            keybd_event(vk as u8, 0, if up { KEYUP } else { 0 }, 0);
        }
    }
    pub fn start() {
        key(VK_LWIN, false);
        key(VK_LWIN, true);
    }
    pub fn picker() {
        key(VK_CONTROL, false);
        key(VK_MENU, false);
        key(VK_TAB, false);
        key(VK_TAB, true);
        key(VK_MENU, true);
        key(VK_CONTROL, true);
    }
    pub fn escape() {
        key(VK_ESCAPE, false);
        key(VK_ESCAPE, true);
    }
    pub fn escape_pressed() -> bool {
        unsafe { GetAsyncKeyState(VK_ESCAPE as i32) < 0 }
    }
    pub fn physical_corner() -> bool {
        let mut position = POINT { x: 0, y: 0 };
        let (width, height) = screen();
        (unsafe { GetCursorPos(&mut position) != 0 })
            && (position.x <= 0 || position.x >= width - 1)
            && (position.y <= 0 || position.y >= height - 1)
    }
}

#[cfg(not(target_os = "windows"))]
compile_error!("This mouse controller currently targets Windows desktop input.");

trait Input {
    fn screen(&self) -> (i32, i32);
    fn move_to(&mut self, x: i32, y: i32);
    fn left_down(&mut self) -> bool;
    fn left_up(&mut self);
    fn right_click(&mut self) -> bool;
    fn left_click(&mut self) -> bool;
    fn double_click(&mut self) -> bool;
    fn start(&mut self);
    fn picker(&mut self);
    fn escape(&mut self);
}

struct WinInput;

impl Input for WinInput {
    fn screen(&self) -> (i32, i32) {
        os::screen()
    }
    fn move_to(&mut self, x: i32, y: i32) {
        os::move_to(x, y);
    }
    fn left_down(&mut self) -> bool {
        os::left_down()
    }
    fn left_up(&mut self) {
        os::left_up();
    }
    fn right_click(&mut self) -> bool {
        os::right_click()
    }
    fn left_click(&mut self) -> bool {
        os::left_click()
    }
    fn double_click(&mut self) -> bool {
        os::double_click()
    }
    fn start(&mut self) {
        os::start();
    }
    fn picker(&mut self) {
        os::picker();
    }
    fn escape(&mut self) {
        os::escape();
    }
}

struct Mouse<I: Input> {
    input: I,
    left_down: bool,
    right_ready: bool,
    double_ready: bool,
    windows_ready: bool,
    picker_ready: bool,
    picker_open: bool,
    selection_latched: bool,
    cursor: Option<(f32, f32)>,
}

impl Default for Mouse<WinInput> {
    fn default() -> Self {
        Self::new(WinInput)
    }
}

impl<I: Input> Mouse<I> {
    fn new(input: I) -> Self {
        Self {
            input,
            left_down: false,
            right_ready: true,
            double_ready: true,
            windows_ready: true,
            picker_ready: true,
            picker_open: false,
            selection_latched: false,
            cursor: None,
        }
    }
    fn release(&mut self, close_picker: bool) {
        if self.left_down {
            self.input.left_up();
            self.left_down = false;
        }
        if close_picker && self.picker_open {
            self.input.escape();
            self.picker_open = false;
        }
        self.cursor = None;
        self.right_ready = true;
        self.double_ready = true;
        self.windows_ready = true;
        self.picker_ready = true;
        self.selection_latched = false;
    }

    fn update(
        &mut self,
        pointer: Option<&Hand>,
        pointer_gesture: Option<&Hand>,
        action: Option<&Hand>,
        same: bool,
        scrolling: bool,
    ) -> &'static str {
        let Some(pointer) = pointer else {
            self.release(false);
            return "No hand";
        };
        let pointer_gesture = pointer_gesture.unwrap_or(pointer);
        let hand = action.unwrap_or(pointer_gesture);
        let gestures = pinches(hand, same, self.left_down);
        let (left, double, pick, right) = (
            gestures.left,
            gestures.double,
            gestures.picker,
            gestures.right,
        );
        let ppalm = distance(pointer_gesture, 5, 17).max(0.01);
        let pratio = |tip| distance(pointer_gesture, 4, tip) / ppalm;
        let windows =
            pratio(12) < 0.52 && pratio(8) > 0.65 && pratio(16) > 0.58 && pratio(20) > 0.65;

        let (screen_w, screen_h) = self.input.screen();
        let x = ((pointer[8].x - 0.125) / 0.75).clamp(0.0, 1.0) * (screen_w - 3) as f32 + 1.0;
        let y = ((pointer[8].y - 0.125) / 0.75).clamp(0.0, 1.0) * (screen_h - 3) as f32 + 1.0;
        let (cx, cy) = self.cursor.unwrap_or((x, y));
        let travel = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
        let response = (travel / 250.0).clamp(0.60, 0.85);
        let next = (cx + response * (x - cx), cy + response * (y - cy));
        self.cursor = Some(next);
        self.input
            .move_to(next.0.round() as i32, next.1.round() as i32);

        if pratio(12) > 0.70 {
            self.windows_ready = true;
        }
        if windows {
            if self.left_down {
                self.input.left_up();
                self.left_down = false;
            }
            if self.windows_ready {
                if self.picker_open {
                    self.input.escape();
                    self.picker_open = false;
                }
                self.input.start();
                self.windows_ready = false;
            }
            return "Windows Start";
        }
        if same && pointing(pointer_gesture) {
            if self.left_down {
                self.input.left_up();
                self.left_down = false;
            }
            self.right_ready = true;
            self.double_ready = true;
            self.picker_ready = true;
            self.selection_latched = false;
            return "Point only";
        }
        if scrolling {
            if self.left_down {
                self.input.left_up();
                self.left_down = false;
            }
            self.right_ready = true;
            self.double_ready = true;
            self.picker_ready = true;
            return "Scroll";
        }
        if self.picker_open {
            if left && !self.selection_latched && self.input.left_click() {
                self.picker_open = false;
                self.selection_latched = true;
                return "Window selected";
            }
            return "Pick window";
        }
        if self.selection_latched {
            if left {
                return "Window selected";
            }
            self.selection_latched = false;
        }
        if left && !self.left_down {
            self.left_down = self.input.left_down();
        }
        if !left && self.left_down {
            self.input.left_up();
            self.left_down = false;
        }
        if double && self.double_ready && self.input.double_click() {
            self.double_ready = false;
        }
        if !double {
            self.double_ready = true;
        }
        if pick && self.picker_ready {
            self.input.picker();
            self.picker_open = true;
            self.picker_ready = false;
        }
        if !pick {
            self.picker_ready = true;
        }
        if right && self.right_ready && self.input.right_click() {
            self.right_ready = false;
        }
        if !right {
            self.right_ready = true;
        }
        if self.left_down {
            "Drag"
        } else if double {
            "Double click"
        } else if pick {
            "Pick window"
        } else if right {
            "Right click"
        } else {
            "Move"
        }
    }
}

impl<I: Input> Drop for Mouse<I> {
    fn drop(&mut self) {
        self.release(true);
    }
}

struct Frame {
    rgb: Vec<u8>,
    width: usize,
    height: usize,
    camera_fps: f32,
}

fn poll_frame<I: Input>(
    receiver: &crossbeam_channel::Receiver<Frame>,
    mouse: &mut Mouse<I>,
    scroll: &mut Scroll,
    last_received: Instant,
) -> Result<Option<Frame>> {
    match receiver.recv_timeout(Duration::from_millis(20)) {
        Ok(frame) => Ok(Some(frame)),
        Err(RecvTimeoutError::Timeout) => {
            if last_received.elapsed() >= Duration::from_millis(250) {
                mouse.release(true);
                scroll.reset();
            }
            anyhow::ensure!(
                last_received.elapsed() < Duration::from_millis(5500),
                "camera stopped supplying frames"
            );
            Ok(None)
        }
        Err(RecvTimeoutError::Disconnected) => {
            mouse.release(true);
            scroll.reset();
            anyhow::bail!("camera stopped supplying frames");
        }
    }
}

fn camera_thread_native(
    index: u32,
    sender: crossbeam_channel::Sender<Frame>,
    discard: crossbeam_channel::Receiver<Frame>,
    control: &CaptureControl,
) -> Result<()> {
    let format = CameraFormat::new(Resolution::new(640, 480), FrameFormat::MJPEG, 30);
    let exact = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(format));
    let closest = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(format));
    let mut camera = Camera::new(CameraIndex::Index(index), exact)
        .or_else(|_| Camera::new(CameraIndex::Index(index), closest))
        .with_context(|| format!("open camera {index}"))?;
    camera.open_stream()?;
    let mut last_good_frame = Instant::now();
    let mut last_decoded = None;
    let mut camera_fps = 0.0_f32;
    while !control.cancelled() {
        let frame = match camera.frame() {
            Ok(frame) => {
                last_good_frame = Instant::now();
                frame
            }
            Err(_) if last_good_frame.elapsed() < Duration::from_secs(5) => {
                thread::sleep(Duration::from_millis(25));
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let decoded = frame.decode_image::<RgbFormat>()?;
        let decoded_at = Instant::now();
        if let Some(previous) = last_decoded {
            let current = 1.0 / decoded_at.duration_since(previous).as_secs_f32().max(0.001);
            camera_fps = if camera_fps == 0.0 {
                current
            } else {
                camera_fps * 0.8 + current * 0.2
            };
        }
        last_decoded = Some(decoded_at);
        let (width, height) = decoded.dimensions();
        let next = Frame {
            rgb: decoded.into_raw(),
            width: width as usize,
            height: height as usize,
            camera_fps,
        };
        match sender.try_send(next) {
            Ok(()) => {}
            Err(TrySendError::Full(newest)) => {
                let _ = discard.try_recv();
                if sender.try_send(newest).is_err() {
                    break;
                }
            }
            Err(TrySendError::Disconnected(_)) => break,
        }
    }
    camera.stop_stream()?;
    Ok(())
}

fn camera_thread_bridge(
    index: u32,
    sender: crossbeam_channel::Sender<Frame>,
    discard: crossbeam_channel::Receiver<Frame>,
    project_dir: &std::path::Path,
    control: &CaptureControl,
) -> Result<()> {
    let script = project_dir.join("camera_bridge.py");
    let python = project_dir.join("../Python/.venv/Scripts/python.exe");
    anyhow::ensure!(
        script.is_file() && python.is_file(),
        "DirectShow bridge or Python environment is missing"
    );
    let mut command = Command::new(python);
    command
        .arg("-u")
        .arg(script)
        .arg("--camera")
        .arg(index.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    let mut output = control.attach(command.spawn().context("start DirectShow camera bridge")?)?;
    let mut last_decoded = None;
    let mut camera_fps = 0.0_f32;
    while !control.cancelled() {
        let mut header = [0u8; 12];
        output
            .read_exact(&mut header)
            .context("read DirectShow frame header")?;
        anyhow::ensure!(&header[..4] == b"HMR1", "invalid DirectShow frame header");
        let width = u32::from_le_bytes(header[4..8].try_into()?) as usize;
        let height = u32::from_le_bytes(header[8..12].try_into()?) as usize;
        anyhow::ensure!(
            (1..=4096).contains(&width) && (1..=4096).contains(&height),
            "invalid DirectShow frame dimensions"
        );
        let mut rgb = vec![0u8; width * height * 3];
        output
            .read_exact(&mut rgb)
            .context("read DirectShow frame pixels")?;
        let decoded_at = Instant::now();
        if let Some(previous) = last_decoded {
            let current = 1.0 / decoded_at.duration_since(previous).as_secs_f32().max(0.001);
            camera_fps = if camera_fps == 0.0 {
                current
            } else {
                camera_fps * 0.8 + current * 0.2
            };
        }
        last_decoded = Some(decoded_at);
        let next = Frame {
            rgb,
            width,
            height,
            camera_fps,
        };
        match sender.try_send(next) {
            Ok(()) => {}
            Err(TrySendError::Full(newest)) => {
                let _ = discard.try_recv();
                if sender.try_send(newest).is_err() {
                    break;
                }
            }
            Err(TrySendError::Disconnected(_)) => break,
        }
    }
    Ok(())
}

fn camera_thread(
    index: u32,
    sender: crossbeam_channel::Sender<Frame>,
    discard: crossbeam_channel::Receiver<Frame>,
    project_dir: PathBuf,
    control: Arc<CaptureControl>,
) -> Result<()> {
    match camera_thread_native(index, sender.clone(), discard.clone(), &control) {
        Ok(()) => Ok(()),
        Err(_) if control.cancelled() => Ok(()),
        Err(error) => {
            eprintln!("Media Foundation camera unavailable: {error:#}. Trying DirectShow.");
            camera_thread_bridge(index, sender, discard, &project_dir, &control)
                .context("DirectShow camera fallback failed")
        }
    }
}

const EDGES: &[(usize, usize)] = &[
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
];

fn dot(pixels: &mut [u32], w: usize, h: usize, x: i32, y: i32, color: u32, radius: i32) {
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy > radius * radius {
                continue;
            }
            let (xx, yy) = (x + dx, y + dy);
            if xx >= 0 && yy >= 0 && xx < w as i32 && yy < h as i32 {
                pixels[yy as usize * w + xx as usize] = color;
            }
        }
    }
}

fn line(pixels: &mut [u32], w: usize, h: usize, from: (i32, i32), to: (i32, i32), color: u32) {
    let (mut x, mut y) = from;
    let dx = (to.0 - x).abs();
    let sx = if x < to.0 { 1 } else { -1 };
    let dy = -(to.1 - y).abs();
    let sy = if y < to.1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        dot(pixels, w, h, x, y, color, 1);
        if x == to.0 && y == to.1 {
            break;
        }
        let twice = 2 * error;
        if twice >= dy {
            error += dy;
            x += sx;
        }
        if twice <= dx {
            error += dx;
            y += sy;
        }
    }
}

fn render(pixels: &mut [u32], frame: &Frame, hands: &[Option<TrackedHand>; 2], dominant: usize) {
    let (w, h) = (frame.width, frame.height);
    for y in 0..h {
        for x in 0..w {
            let source = (y * w + (w - 1 - x)) * 3;
            pixels[y * w + x] = ((frame.rgb[source] as u32) << 16)
                | ((frame.rgb[source + 1] as u32) << 8)
                | frame.rgb[source + 2] as u32;
        }
    }
    let box_color = 0xF5C850;
    let left = (w as f32 * 0.125) as i32;
    let top = (h as f32 * 0.125) as i32;
    let right = (w as f32 * 0.875) as i32;
    let bottom = (h as f32 * 0.875) as i32;
    for (a, b) in [
        ((left, top), (right, top)),
        ((right, top), (right, bottom)),
        ((right, bottom), (left, bottom)),
        ((left, bottom), (left, top)),
    ] {
        line(pixels, w, h, a, b, box_color);
    }
    for (side, tracked) in hands.iter().enumerate() {
        let Some(hand) = tracked else {
            continue;
        };
        let color = if side == dominant || hands[dominant].is_none() {
            0x52E885
        } else {
            0xEAA652
        };
        let position = |i: usize| {
            (
                (hand.points[i].x * w as f32) as i32,
                (hand.points[i].y * h as f32) as i32,
            )
        };
        for &(a, b) in EDGES {
            line(pixels, w, h, position(a), position(b), color);
        }
        for i in 0..21 {
            let (x, y) = position(i);
            dot(pixels, w, h, x, y, 0xFFFFFF, 3);
        }
        let (x, y) = position(8);
        dot(pixels, w, h, x, y, color, 6);
    }
}

fn chrome_profile_picker() -> Result<()> {
    let directories = ["PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"];
    let executable = std::env::var_os("PATH")
        .and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|dir| dir.join("chrome.exe"))
                .find(|path| path.is_file())
        })
        .or_else(|| {
            directories
                .iter()
                .filter_map(std::env::var_os)
                .map(|root| PathBuf::from(root).join("Google/Chrome/Application/chrome.exe"))
                .find(|path| path.is_file())
        })
        .context("Chrome executable not found")?;
    std::process::Command::new(executable)
        .arg("chrome://profile-picker/")
        .spawn()?;
    Ok(())
}

fn models_dir() -> Result<PathBuf> {
    let executable = std::env::current_exe().context("find executable path")?;
    let executable_dir = executable.parent().context("find executable directory")?;
    let working_dir = std::env::current_dir().context("find working directory")?;
    [
        executable_dir.join("models"),
        executable_dir.join("../../models"),
        working_dir.join("models"),
    ]
    .into_iter()
    .find(|dir| {
        dir.join("palm_detection.onnx").is_file() && dir.join("hand_landmark.onnx").is_file()
    })
    .context("hand models not found beside the executable or in the project models directory")
}

fn benchmark_file(path: &str, model_dir: &std::path::Path, cuda: bool) -> Result<()> {
    let mut input = File::open(path).with_context(|| format!("open benchmark file {path}"))?;
    let mut header = [0u8; 16];
    input.read_exact(&mut header)?;
    anyhow::ensure!(&header[..4] == b"HMR2", "invalid benchmark file header");
    let width = u32::from_le_bytes(header[4..8].try_into()?) as usize;
    let height = u32::from_le_bytes(header[8..12].try_into()?) as usize;
    let frames = u32::from_le_bytes(header[12..16].try_into()?);
    anyhow::ensure!(
        (1..=4096).contains(&width)
            && (1..=4096).contains(&height)
            && (1..=10_000).contains(&frames),
        "invalid benchmark file dimensions or frame count"
    );
    let mut tracker = Tracker::new(
        &model_dir.join("palm_detection.onnx"),
        &model_dir.join("hand_landmark.onnx"),
        cuda,
    )?;
    let mut rgb = vec![0u8; width * height * 3];
    let mut total_model_ms = 0.0_f32;
    let mut one_hand_frames = 0u32;
    let mut two_hand_frames = 0u32;
    let started = Instant::now();
    for _ in 0..frames {
        input.read_exact(&mut rgb)?;
        let detection_started = Instant::now();
        let hands = tracker.detect(&rgb, width, height)?;
        total_model_ms += detection_started.elapsed().as_secs_f32() * 1000.0;
        match hands.iter().flatten().count() {
            1 => one_hand_frames += 1,
            2 => two_hand_frames += 1,
            _ => {}
        }
    }
    let elapsed = started.elapsed().as_secs_f32();
    println!(
        "Replay: frames={frames}, one_hand_frames={one_hand_frames}, two_hand_frames={two_hand_frames}, elapsed_s={elapsed:.2}, model_ms_mean={:.2}",
        total_model_ms / frames as f32
    );
    Ok(())
}

fn main() -> Result<()> {
    let options = cli::Options::parse(std::env::args().skip(1))?;
    if options.help {
        print!("{}", cli::HELP);
        return Ok(());
    }
    let benchmark_seconds = options.benchmark_seconds;
    let camera_index = options.camera;
    let tracking_only = options.tracking_only;
    let cuda = options.cuda;
    let dominant = usize::from(options.left);
    let model_dir = models_dir()?;
    if let Some(path) = options.benchmark_file {
        return benchmark_file(&path, &model_dir, cuda);
    }
    let project_dir = model_dir
        .parent()
        .context("find Rust project directory")?
        .to_path_buf();
    let (sender, receiver) = bounded::<Frame>(1);
    let discard = receiver.clone();
    let _camera = CaptureWorker::spawn(move |control| {
        if let Err(error) = camera_thread(camera_index, sender, discard, project_dir, control) {
            eprintln!("Camera error: {error:#}");
        }
    })?;
    // Camera connection and model loading happen at the same time.
    let mut tracker = Tracker::new(
        &model_dir.join("palm_detection.onnx"),
        &model_dir.join("hand_landmark.onnx"),
        cuda,
    )?;
    let first = receiver
        .recv_timeout(Duration::from_secs(15))
        .context("camera did not provide a frame")?;
    let (width, height) = (first.width, first.height);
    let (preview_w, preview_h) = (width * 3 / 4, height * 3 / 4);
    let mut window = Window::new(
        "Hand Mouse | H: hand  Space: pause  Q: quit",
        preview_w,
        preview_h,
        WindowOptions {
            resize: false,
            scale: minifb::Scale::X1,
            topmost: false,
            ..WindowOptions::default()
        },
    )?;
    window.set_target_fps(0);
    let mut pixels = vec![0u32; width * height];
    let mut preview = vec![0u32; preview_w * preview_h];
    let mut preview_source = Image::new(width as u32, height as u32, PixelType::U8x4);
    let mut preview_target = Image::new(preview_w as u32, preview_h as u32, PixelType::U8x4);
    let mut preview_resizer = Resizer::new();
    let mut mouse = Mouse::default();
    let mut controls = Toggle::new();
    let mut scroll = Scroll::default();
    let mut dominant = dominant;
    let mut paused = false;
    let mut triangle_started = None;
    let mut triangle_fired = false;
    let mut previous_sides = (None, None);
    let mut last_status = "";
    let mut last_dominant = usize::MAX;
    let mut last_frame_start = None;
    let mut last_title_update = Instant::now();
    let mut measured_fps = 0.0_f32;
    let mut model_ms = 0.0_f32;
    let mut total_model_ms = 0.0_f32;
    let mut processed_frames = 0u32;
    let mut frames_with_one_hand = 0u32;
    let mut frames_with_two_hands = 0u32;
    let mut pending_frame = Some(first);
    let mut last_received = Instant::now();
    let mut latest_camera_fps = 0.0;
    let benchmark_started = Instant::now();
    while window.is_open() && !window.is_key_down(Key::Q) && !window.is_key_down(Key::Escape) {
        let start = Instant::now();
        if os::escape_pressed()
            || os::physical_corner()
            || benchmark_seconds
                .is_some_and(|seconds| benchmark_started.elapsed().as_secs_f32() >= seconds)
        {
            break;
        }
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            paused = !paused;
            mouse.release(true);
            scroll.reset();
        }
        if window.is_key_pressed(Key::H, KeyRepeat::No) {
            dominant = 1 - dominant;
            mouse.release(true);
            scroll.reset();
        }
        let frame = match pending_frame.take() {
            Some(frame) => frame,
            None => match poll_frame(&receiver, &mut mouse, &mut scroll, last_received)? {
                Some(frame) => frame,
                None => {
                    if last_received.elapsed() >= Duration::from_millis(250) {
                        triangle_started = None;
                        triangle_fired = false;
                    }
                    window.update();
                    continue;
                }
            },
        };
        last_received = Instant::now();
        if let Some(previous) = last_frame_start {
            let elapsed = last_received.duration_since(previous).as_secs_f32();
            let instant = 1.0 / elapsed.max(0.001);
            measured_fps = if measured_fps == 0.0 {
                instant
            } else {
                measured_fps * 0.8 + instant * 0.2
            };
        }
        last_frame_start = Some(last_received);
        latest_camera_fps = frame.camera_fps;
        if frame.width != width || frame.height != height {
            anyhow::bail!("camera resolution changed");
        }
        let hands = if paused {
            [None, None]
        } else {
            let detection_started = Instant::now();
            let hands = tracker.detect(&frame.rgb, width, height)?;
            match hands.iter().flatten().count() {
                1 => frames_with_one_hand += 1,
                2 => frames_with_two_hands += 1,
                _ => {}
            }
            let current_ms = detection_started.elapsed().as_secs_f32() * 1000.0;
            total_model_ms += current_ms;
            processed_frames += 1;
            model_ms = if model_ms == 0.0 {
                current_ms
            } else {
                model_ms * 0.8 + current_ms * 0.2
            };
            hands
        };
        if os::physical_corner() {
            break;
        }
        let side = if hands[dominant].is_some() {
            Some(dominant)
        } else if hands[1 - dominant].is_some() {
            Some(1 - dominant)
        } else {
            None
        };
        let action_side = side.map(|s| if hands[1 - s].is_some() { 1 - s } else { s });
        if previous_sides != (side, action_side) {
            mouse.release(false);
            scroll.reset();
        }
        previous_sides = (side, action_side);
        let pointer_hand = side.and_then(|i| hands[i].as_ref());
        let action_hand = action_side.and_then(|i| hands[i].as_ref());
        let pointer = pointer_hand.map(|h| &h.points);
        let pointer_gesture = pointer_hand.map(|h| &h.raw_points);
        let action_gesture = action_hand.map(|h| &h.raw_points);
        let mut status = if paused {
            "Paused"
        } else if tracking_only {
            "Tracking only"
        } else {
            "No hand"
        };
        if !paused && !tracking_only {
            if controls.update(pointer_gesture) {
                mouse.release(true);
                scroll.reset();
                status = if controls.enabled {
                    "Controls ON"
                } else {
                    "Controls OFF"
                };
            } else if !controls.enabled {
                mouse.release(true);
                scroll.reset();
                status = "Controls OFF";
            } else {
                let matched =
                    hands[0]
                        .as_ref()
                        .zip(hands[1].as_ref())
                        .is_some_and(|(right, left)| {
                            !pointer_gesture.is_some_and(pointing)
                                && triangle(
                                    &left.points,
                                    &right.points,
                                    width as f32 / height as f32,
                                )
                        });
                if matched {
                    mouse.release(true);
                    scroll.reset();
                    let began = *triangle_started.get_or_insert(start);
                    if !triangle_fired && start.duration_since(began) >= Duration::from_millis(650)
                    {
                        if let Err(error) = chrome_profile_picker() {
                            eprintln!("Chrome: {error:#}");
                        }
                        triangle_fired = true;
                    }
                    status = "Chrome profiles";
                } else {
                    triangle_started = None;
                    triangle_fired = false;
                    let scroll_hand = action_hand
                        .filter(|_| action_side != side && !mouse.picker_open)
                        .map(|h| &h.points)
                        .filter(|hand| gesture::scroll_pose(hand));
                    let scrolling = scroll_hand.is_some();
                    let wheel = scroll.update(scroll_hand, start);
                    status = mouse.update(
                        pointer,
                        pointer_gesture,
                        action_gesture,
                        action_side == side,
                        scrolling,
                    );
                    if wheel != 0 {
                        os::wheel(wheel);
                        status = "Scrolling";
                    }
                }
            }
        }
        render(&mut pixels, &frame, &hands, side.unwrap_or(dominant));
        if status != last_status
            || dominant != last_dominant
            || start.duration_since(last_title_update) >= Duration::from_secs(1)
        {
            window.set_title(&format!(
                "Hand Mouse | {measured_fps:.0} app / {:.0} camera FPS | Model {model_ms:.0} ms | {status} | {} pointer | H switch | Space pause | Q quit",
                frame.camera_fps,
                if dominant == 0 { "Right" } else { "Left" }
            ));
            last_status = status;
            last_dominant = dominant;
            last_title_update = start;
        }
        for (color, bytes) in pixels.iter().zip(
            preview_source
                .buffer_mut()
                .as_chunks_mut::<4>()
                .0
                .iter_mut(),
        ) {
            bytes.copy_from_slice(&(color | 0xFF00_0000).to_le_bytes());
        }
        preview_resizer.resize(&preview_source, &mut preview_target, None)?;
        for (bytes, color) in preview_target
            .buffer()
            .as_chunks::<4>()
            .0
            .iter()
            .zip(preview.iter_mut())
        {
            *color = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) & 0x00FF_FFFF;
        }
        window.update_with_buffer(&preview, preview_w, preview_h)?;
        if benchmark_seconds
            .is_some_and(|seconds| benchmark_started.elapsed().as_secs_f32() >= seconds)
        {
            break;
        }
        if os::escape_pressed() {
            break;
        }
    }
    mouse.release(true);
    if benchmark_seconds.is_some() {
        let elapsed = benchmark_started.elapsed().as_secs_f32();
        println!(
            "Benchmark: frames={processed_frames}, one_hand_frames={frames_with_one_hand}, two_hand_frames={frames_with_two_hands}, elapsed_s={elapsed:.2}, app_fps={:.2}, camera_fps={:.2}, model_ms_mean={:.2}",
            processed_frames as f32 / elapsed.max(0.001),
            latest_camera_fps,
            total_model_ms / processed_frames.max(1) as f32
        );
    }
    Ok(())
}

#[cfg(test)]
mod input_tests {
    use super::*;
    use glam::Vec3;

    #[derive(Default)]
    struct MockInput {
        events: Vec<&'static str>,
    }

    impl Input for MockInput {
        fn screen(&self) -> (i32, i32) {
            (1000, 800)
        }
        fn move_to(&mut self, _: i32, _: i32) {
            self.events.push("move");
        }
        fn left_down(&mut self) -> bool {
            self.events.push("left_down");
            true
        }
        fn left_up(&mut self) {
            self.events.push("left_up");
        }
        fn right_click(&mut self) -> bool {
            self.events.push("right_click");
            true
        }
        fn left_click(&mut self) -> bool {
            self.events.push("left_click");
            true
        }
        fn double_click(&mut self) -> bool {
            self.events.push("double_click");
            true
        }
        fn start(&mut self) {
            self.events.push("start");
        }
        fn picker(&mut self) {
            self.events.push("picker");
        }
        fn escape(&mut self) {
            self.events.push("escape");
        }
    }

    fn hand() -> Hand {
        let mut hand = [Vec3::ZERO; 21];
        hand[17].x = 1.0;
        for tip in [8, 12, 16, 20] {
            hand[tip].x = 0.8;
        }
        hand
    }

    fn count(events: &[&str], name: &str) -> usize {
        events.iter().filter(|&&event| event == name).count()
    }

    #[test]
    fn stalled_camera_releases_drag_without_blocking_ui() {
        let (_sender, receiver) = bounded(1);
        let mut mouse = Mouse::new(MockInput::default());
        mouse.left_down = true;
        let mut scroll = Scroll::default();
        let start = Instant::now();
        assert!(poll_frame(
            &receiver,
            &mut mouse,
            &mut scroll,
            start - Duration::from_millis(300)
        )
        .unwrap()
        .is_none());
        assert!(start.elapsed() < Duration::from_millis(150));
        assert!(!mouse.left_down);
        assert_eq!(count(&mouse.input.events, "left_up"), 1);
    }

    #[test]
    fn normal_frame_gap_preserves_drag() {
        let (_sender, receiver) = bounded(1);
        let mut mouse = Mouse::new(MockInput::default());
        mouse.left_down = true;
        assert!(poll_frame(
            &receiver,
            &mut mouse,
            &mut Scroll::default(),
            Instant::now()
        )
        .unwrap()
        .is_none());
        assert!(mouse.left_down);
    }

    #[test]
    fn left_pinch_holds_and_releases_the_button() {
        let mut mouse = Mouse::new(MockInput::default());
        let pointer = hand();
        let mut action = hand();
        action[8].x = 0.3;
        mouse.update(Some(&pointer), Some(&pointer), Some(&action), false, false);
        assert_eq!(count(&mouse.input.events, "left_down"), 1);
        action[8].x = 0.8;
        mouse.update(Some(&pointer), Some(&pointer), Some(&action), false, false);
        assert_eq!(count(&mouse.input.events, "left_up"), 1);
    }

    #[test]
    fn right_and_double_click_rearm_as_soon_as_the_pinch_ends() {
        let mut mouse = Mouse::new(MockInput::default());
        let pointer = hand();
        let mut action = hand();
        action[20].x = 0.3;
        mouse.update(Some(&pointer), Some(&pointer), Some(&action), false, false);
        mouse.update(Some(&pointer), Some(&pointer), Some(&action), false, false);
        assert_eq!(count(&mouse.input.events, "right_click"), 1);
        action[20].x = 0.43;
        mouse.update(Some(&pointer), Some(&pointer), Some(&action), false, false);
        action[20].x = 0.3;
        mouse.update(Some(&pointer), Some(&pointer), Some(&action), false, false);
        assert_eq!(count(&mouse.input.events, "right_click"), 2);
        action[20].x = 0.8;
        action[12].x = 0.5;
        mouse.update(Some(&pointer), Some(&pointer), Some(&action), false, false);
        action[12].x = 0.55;
        mouse.update(Some(&pointer), Some(&pointer), Some(&action), false, false);
        action[12].x = 0.5;
        mouse.update(Some(&pointer), Some(&pointer), Some(&action), false, false);
        assert_eq!(count(&mouse.input.events, "double_click"), 2);
    }

    #[test]
    fn picker_selects_once_and_scroll_suppresses_clicks() {
        let mut mouse = Mouse::new(MockInput::default());
        let pointer = hand();
        let mut action = hand();
        action[16].x = 0.3;
        mouse.update(Some(&pointer), Some(&pointer), Some(&action), false, false);
        assert_eq!(count(&mouse.input.events, "picker"), 1);
        action[16].x = 0.8;
        action[8].x = 0.3;
        mouse.update(Some(&pointer), Some(&pointer), Some(&action), false, false);
        mouse.update(Some(&pointer), Some(&pointer), Some(&action), false, false);
        assert_eq!(count(&mouse.input.events, "left_click"), 1);
        assert_eq!(count(&mouse.input.events, "left_down"), 0);
        action[8].x = 0.8;
        mouse.update(Some(&pointer), Some(&pointer), Some(&action), false, false);
        action[8].x = 0.3;
        mouse.update(Some(&pointer), Some(&pointer), Some(&action), false, true);
        assert_eq!(count(&mouse.input.events, "left_down"), 0);
    }

    #[test]
    fn windows_start_fires_once_per_green_pinch() {
        let mut mouse = Mouse::new(MockInput::default());
        let mut pointer = hand();
        pointer[12].x = 0.3;
        mouse.update(Some(&pointer), None, None, true, false);
        mouse.update(Some(&pointer), None, None, true, false);
        assert_eq!(count(&mouse.input.events, "start"), 1);
        pointer[12].x = 0.8;
        mouse.update(Some(&pointer), None, None, true, false);
        pointer[12].x = 0.3;
        mouse.update(Some(&pointer), None, None, true, false);
        assert_eq!(count(&mouse.input.events, "start"), 2);
    }
}
