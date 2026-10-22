//! Two-stage BlazePalm and hand-landmark ONNX inference.
//! Model geometry follows yakhyo/mediapipe-hand-landmark-onnx (Apache-2.0).
use anyhow::{Context, Result};
use glam::Vec3;
use ort::{
    execution_providers::CUDAExecutionProvider,
    memory::Allocator,
    session::{Session, SessionInputValue},
    value::Tensor,
};
use std::path::Path;

use crate::gesture::{Hand, OneEuro};

const PALM_SIZE: usize = 192;
const HAND_SIZE: usize = 224;
type Roi = (f32, f32, f32, f32);

#[derive(Clone, Copy)]
struct Palm {
    score: f32,
    cx: f32,
    cy: f32,
    width: f32,
    height: f32,
    wrist: (f32, f32),
    mcp: (f32, f32),
}

impl Palm {
    fn iou(self, other: Self) -> f32 {
        let bounds = |p: Self| {
            (
                p.cx - p.width / 2.0,
                p.cy - p.height / 2.0,
                p.cx + p.width / 2.0,
                p.cy + p.height / 2.0,
            )
        };
        let (ax1, ay1, ax2, ay2) = bounds(self);
        let (bx1, by1, bx2, by2) = bounds(other);
        let inter = (ax2.min(bx2) - ax1.max(bx1)).max(0.0) * (ay2.min(by2) - ay1.max(by1)).max(0.0);
        inter / (self.width * self.height + other.width * other.height - inter).max(1e-9)
    }

    fn scale_geometry(&mut self, weight: f32) {
        self.cx *= weight;
        self.cy *= weight;
        self.width *= weight;
        self.height *= weight;
        self.wrist.0 *= weight;
        self.wrist.1 *= weight;
        self.mcp.0 *= weight;
        self.mcp.1 *= weight;
    }

    fn add_weighted(&mut self, other: Self) {
        self.cx += other.cx * other.score;
        self.cy += other.cy * other.score;
        self.width += other.width * other.score;
        self.height += other.height * other.score;
        self.wrist.0 += other.wrist.0 * other.score;
        self.wrist.1 += other.wrist.1 * other.score;
        self.mcp.0 += other.mcp.0 * other.score;
        self.mcp.1 += other.mcp.1 * other.score;
    }
}

#[derive(Clone, Copy)]
pub struct TrackedHand {
    pub points: Hand,
    pub raw_points: Hand,
    pub score: f32,
}

pub struct Tracker {
    palm_session: Session,
    hand_session: Session,
    anchors: Vec<(f32, f32)>,
    candidates: Vec<Palm>,
    palm_input: Tensor<f32>,
    hand_input: Tensor<f32>,
    filters: [[OneEuro; 21]; 2],
    last_seen: [bool; 2],
    tracked_rois: [Option<Roi>; 2],
    frames_since_detection: u8,
}

impl Tracker {
    pub fn new(palm_path: &Path, hand_path: &Path, cuda: bool) -> Result<Self> {
        let open = |path: &Path| -> Result<Session> {
            let mut builder = Session::builder()?;
            if cuda {
                builder =
                    builder.with_execution_providers([CUDAExecutionProvider::default().build()])?;
            }
            Ok(builder.commit_from_file(path)?)
        };
        let palm_session =
            open(palm_path).with_context(|| format!("load {}", palm_path.display()))?;
        let hand_session =
            open(hand_path).with_context(|| format!("load {}", hand_path.display()))?;
        let mut anchors = Vec::with_capacity(2016);
        for (cells, repeats) in [(24, 2), (12, 6)] {
            for y in 0..cells {
                for x in 0..cells {
                    for _ in 0..repeats {
                        anchors.push((
                            (x as f32 + 0.5) / cells as f32,
                            (y as f32 + 0.5) / cells as f32,
                        ));
                    }
                }
            }
        }
        let allocator = Allocator::default();
        Ok(Self {
            palm_session,
            hand_session,
            anchors,
            candidates: Vec::with_capacity(2016),
            palm_input: Tensor::new(&allocator, [1, 3, PALM_SIZE, PALM_SIZE])?,
            hand_input: Tensor::new(&allocator, [1, 3, HAND_SIZE, HAND_SIZE])?,
            filters: [[OneEuro::default(); 21]; 2],
            last_seen: [false; 2],
            tracked_rois: [None, None],
            frames_since_detection: 0,
        })
    }

    /// The camera provides RGB888. We letterbox directly into the preallocated CHW tensor.
    fn prepare_palm(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<()> {
        let scale = PALM_SIZE as f32 / width.max(height) as f32;
        let pad_x = (PALM_SIZE as f32 - width as f32 * scale) / 2.0;
        let pad_y = (PALM_SIZE as f32 - height as f32 * scale) / 2.0;
        let mut view = self.palm_input.try_extract_tensor_mut::<f32>()?;
        let tensor = view.as_slice_mut().context("noncontiguous palm input")?;
        fill_tensor(tensor, PALM_SIZE, rgb, width, height, |x, y| {
            ((x - pad_x) / scale, (y - pad_y) / scale)
        });
        Ok(())
    }

    pub fn detect(
        &mut self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Result<[Option<TrackedHand>; 2]> {
        anyhow::ensure!(
            width > 0 && height > 0 && rgb.len() >= width * height * 3,
            "invalid RGB camera frame"
        );
        let tracked_count = self.tracked_rois.iter().flatten().count();
        let detection_interval = if tracked_count == 2 { 8 } else { 4 };
        let redetect = tracked_count == 0 || self.frames_since_detection >= detection_interval;
        let previous_rois = self.tracked_rois;
        let rois = if redetect {
            self.prepare_palm(rgb, width, height)?;
            let output = self
                .palm_session
                .run([SessionInputValue::from(self.palm_input.view())])?;
            let regressors = output[0].try_extract_tensor::<f32>()?;
            let scores = output[1].try_extract_tensor::<f32>()?;
            let reg = regressors.as_slice().context("noncontiguous palm output")?;
            let logits = scores.as_slice().context("noncontiguous palm scores")?;
            let palms = decode_palms(
                reg,
                logits,
                &self.anchors,
                width,
                height,
                &mut self.candidates,
            );
            self.frames_since_detection = 0;
            palms.map(|palm| palm.map(palm_roi))
        } else {
            self.frames_since_detection += 1;
            self.tracked_rois
        };
        let mut found: [Option<TrackedHand>; 2] = [None, None];
        self.tracked_rois = [None, None];
        for roi in rois.into_iter().flatten() {
            self.run_roi(rgb, width, height, roi, &mut found)?;
        }
        if redetect {
            for (index, roi) in previous_rois.into_iter().enumerate() {
                if found[index].is_none() {
                    if let Some(roi) = roi {
                        self.run_roi(rgb, width, height, roi, &mut found)?;
                    }
                }
            }
        }
        for (last_seen, hand) in self.last_seen.iter_mut().zip(found.iter()) {
            *last_seen = hand.is_some();
        }
        Ok(found)
    }

    fn run_roi(
        &mut self,
        rgb: &[u8],
        width: usize,
        height: usize,
        roi: Roi,
        found: &mut [Option<TrackedHand>; 2],
    ) -> Result<()> {
        let (cx, cy, side, radians) = roi;
        let cosine = radians.cos();
        let sine = radians.sin();
        let factor = side / HAND_SIZE as f32;
        let mut view = self.hand_input.try_extract_tensor_mut::<f32>()?;
        let tensor = view.as_slice_mut().context("noncontiguous hand input")?;
        fill_tensor(tensor, HAND_SIZE, rgb, width, height, |x, y| {
            let dx = (x - HAND_SIZE as f32 / 2.0) * factor;
            let dy = (y - HAND_SIZE as f32 / 2.0) * factor;
            (cx + cosine * dx - sine * dy, cy + sine * dx + cosine * dy)
        });
        drop(view);
        let result = self
            .hand_session
            .run([SessionInputValue::from(self.hand_input.view())])?;
        let screen_view = result[0].try_extract_tensor::<f32>()?;
        let presence_view = result[1].try_extract_tensor::<f32>()?;
        let handedness_view = result[2].try_extract_tensor::<f32>()?;
        let screen = screen_view
            .as_slice()
            .context("noncontiguous hand output")?;
        let presence = presence_view
            .as_slice()
            .context("noncontiguous presence output")?;
        let handedness = handedness_view
            .as_slice()
            .context("noncontiguous handedness output")?;
        let score = presence.first().copied().unwrap_or(0.0);
        if score < 0.5 || screen.len() < 63 {
            return Ok(());
        }
        // The model reports selfie-style labels on the unmirrored camera image.
        let right = handedness.first().copied().unwrap_or(0.5) > 0.5;
        let index = usize::from(!right);
        if found[index].is_some_and(|previous| previous.score >= score) {
            return Ok(());
        }
        let mut points = [Vec3::ZERO; 21];
        let mut raw_points = [Vec3::ZERO; 21];
        let mut raw_xy = [(0.0, 0.0); 21];
        let now = std::time::Instant::now();
        if !self.last_seen[index] {
            self.filters[index] = [OneEuro::default(); 21];
        }
        for (i, point) in points.iter_mut().enumerate() {
            let x = (screen[3 * i] - HAND_SIZE as f32 / 2.0) * factor;
            let y = (screen[3 * i + 1] - HAND_SIZE as f32 / 2.0) * factor;
            let image_x = cx + cosine * x - sine * y;
            let image_y = cy + sine * x + cosine * y;
            raw_xy[i] = (image_x, image_y);
            let raw = Vec3::new(
                1.0 - image_x / width as f32,
                image_y / height as f32,
                screen[3 * i + 2] * factor / width as f32,
            );
            raw_points[i] = raw;
            *point = self.filters[index][i].filter(raw, now);
        }
        self.tracked_rois[index] = landmarks_roi(&raw_xy);
        found[index] = Some(TrackedHand {
            points,
            raw_points,
            score,
        });
        Ok(())
    }
}

fn palm_roi(palm: Palm) -> (f32, f32, f32, f32) {
    let dx = palm.mcp.0 - palm.wrist.0;
    let dy = palm.mcp.1 - palm.wrist.1;
    let norm = (dx * dx + dy * dy).sqrt();
    let (ux, uy) = if norm > 0.0 {
        (dx / norm, dy / norm)
    } else {
        (0.0, -1.0)
    };
    let angle = uy.atan2(ux) + std::f32::consts::FRAC_PI_2;
    let side = palm.width.max(palm.height);
    (
        palm.cx + 0.5 * side * ux,
        palm.cy + 0.5 * side * uy,
        2.6 * side,
        angle,
    )
}

/// MediaPipe's landmark rectangle: use palm joints, align the wrist-to-PIP
/// direction, then expand the rotated box by 2x for the next video frame.
fn landmarks_roi(points: &[(f32, f32); 21]) -> Option<Roi> {
    const PARTIAL: [usize; 12] = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18];
    if PARTIAL
        .iter()
        .any(|&i| !points[i].0.is_finite() || !points[i].1.is_finite())
    {
        return None;
    }
    let wrist = points[0];
    let pip_x = (points[6].0 + points[14].0 + 2.0 * points[10].0) / 4.0;
    let pip_y = (points[6].1 + points[14].1 + 2.0 * points[10].1) / 4.0;
    let angle = (pip_y - wrist.1).atan2(pip_x - wrist.0) + std::f32::consts::FRAC_PI_2;
    let (sine, cosine) = angle.sin_cos();
    let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
    let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &i in &PARTIAL {
        min_x = min_x.min(points[i].0);
        min_y = min_y.min(points[i].1);
        max_x = max_x.max(points[i].0);
        max_y = max_y.max(points[i].1);
    }
    let axis_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0);
    (min_x, min_y) = (f32::INFINITY, f32::INFINITY);
    (max_x, max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &i in &PARTIAL {
        let dx = points[i].0 - axis_center.0;
        let dy = points[i].1 - axis_center.1;
        let x = dx * cosine + dy * sine;
        let y = -dx * sine + dy * cosine;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    let rect_width = max_x - min_x;
    let rect_height = max_y - min_y;
    let side = 2.0 * rect_width.max(rect_height);
    if !side.is_finite() || side < 2.0 {
        return None;
    }
    let local_center_x = (min_x + max_x) / 2.0;
    let local_center_y = (min_y + max_y) / 2.0 - 0.1 * rect_height;
    let cx = axis_center.0 + local_center_x * cosine - local_center_y * sine;
    let cy = axis_center.1 + local_center_x * sine + local_center_y * cosine;
    Some((cx, cy, side, angle))
}

fn decode_palms(
    reg: &[f32],
    logits: &[f32],
    anchors: &[(f32, f32)],
    width: usize,
    height: usize,
    candidates: &mut Vec<Palm>,
) -> [Option<Palm>; 2] {
    let count = anchors.len().min(logits.len()).min(reg.len() / 18);
    let scale = PALM_SIZE as f32 / width.max(height) as f32;
    let pad_x = (PALM_SIZE as f32 - width as f32 * scale) / 2.0;
    let pad_y = (PALM_SIZE as f32 - height as f32 * scale) / 2.0;
    candidates.clear();
    for i in 0..count {
        if !logits[i].is_finite() || logits[i] < 0.0 {
            continue;
        }
        let score = 1.0 / (1.0 + (-logits[i]).exp());
        let row = &reg[i * 18..][..18];
        if !row[..10].iter().all(|value| value.is_finite()) {
            continue;
        }
        let (ax, ay) = anchors[i];
        let point = |x: f32, y: f32| -> (f32, f32) {
            (
                (x + ax * PALM_SIZE as f32 - pad_x) / scale,
                (y + ay * PALM_SIZE as f32 - pad_y) / scale,
            )
        };
        let (cx, cy) = point(row[0], row[1]);
        candidates.push(Palm {
            score,
            cx,
            cy,
            width: row[2].abs() / scale,
            height: row[3].abs() / scale,
            wrist: point(row[4], row[5]),
            mcp: point(row[8], row[9]),
        });
    }
    candidates.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
    weighted_nms(candidates)
}

fn weighted_nms(candidates: &mut Vec<Palm>) -> [Option<Palm>; 2] {
    let mut selected = [None, None];
    for slot in &mut selected {
        if candidates.is_empty() {
            break;
        }
        let top = candidates.remove(0);
        let mut blended = top;
        let mut total = top.score;
        blended.scale_geometry(top.score);
        candidates.retain(|candidate| {
            if top.iou(*candidate) > 0.3 {
                blended.add_weighted(*candidate);
                total += candidate.score;
                false
            } else {
                true
            }
        });
        blended.scale_geometry(1.0 / total);
        *slot = Some(blended);
    }
    selected
}

/// Bilinear resampling directly into a channel-first tensor.
fn fill_tensor<F: Fn(f32, f32) -> (f32, f32)>(
    tensor: &mut [f32],
    size: usize,
    rgb: &[u8],
    width: usize,
    height: usize,
    map: F,
) {
    let plane = size * size;
    for y in 0..size {
        for x in 0..size {
            let (sx, sy) = map(x as f32, y as f32);
            let px = x + y * size;
            if sx <= -1.0 || sy <= -1.0 || sx >= width as f32 || sy >= height as f32 {
                for channel in 0..3 {
                    tensor[channel * plane + px] = 0.0;
                }
                continue;
            }
            let xi = sx.floor() as isize;
            let yi = sy.floor() as isize;
            let tx = sx - xi as f32;
            let ty = sy - yi as f32;
            if xi >= 0 && yi >= 0 && xi + 1 < width as isize && yi + 1 < height as isize {
                let offset = (yi as usize * width + xi as usize) * 3;
                let next_row = width * 3;
                for channel in 0..3 {
                    let a = rgb[offset + channel] as f32;
                    let b = rgb[offset + channel + 3] as f32;
                    let c = rgb[offset + channel + next_row] as f32;
                    let d = rgb[offset + channel + next_row + 3] as f32;
                    let top = a + (b - a) * tx;
                    let bottom = c + (d - c) * tx;
                    tensor[channel * plane + px] = (top + (bottom - top) * ty) / 255.0;
                }
            } else {
                for channel in 0..3 {
                    let sample = |x: isize, y: isize| {
                        if x >= 0 && y >= 0 && x < width as isize && y < height as isize {
                            rgb[(y as usize * width + x as usize) * 3 + channel] as f32
                        } else {
                            0.0
                        }
                    };
                    let a = sample(xi, yi);
                    let b = sample(xi + 1, yi);
                    let c = sample(xi, yi + 1);
                    let d = sample(xi + 1, yi + 1);
                    let top = a + (b - a) * tx;
                    let bottom = c + (d - c) * tx;
                    tensor[channel * plane + px] = (top + (bottom - top) * ty) / 255.0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn low_confidence_palm_is_ignored() {
        let mut scratch = Vec::new();
        let palms = decode_palms(&[0.0; 18], &[-10.0], &[(0.5, 0.5)], 640, 480, &mut scratch);
        assert!(palms.iter().all(Option::is_none));
    }

    #[test]
    fn resize_black_image_stays_black() {
        let rgb = [0u8; 12];
        let mut output = [1.0; 12];
        fill_tensor(&mut output, 2, &rgb, 2, 2, |x, y| (x, y));
        assert!(output.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn weighted_nms_uses_all_detection_scores() {
        let make = |cx, score| Palm {
            score,
            cx,
            cy: 10.0,
            width: 10.0,
            height: 10.0,
            wrist: (cx, 10.0),
            mcp: (cx, 5.0),
        };
        let mut candidates = vec![make(0.0, 0.9), make(1.0, 0.6), make(2.0, 0.3)];
        let selected = weighted_nms(&mut candidates);
        let palm = selected[0].unwrap();
        assert!((palm.cx - 2.0 / 3.0).abs() < 1e-5);
        assert!((palm.wrist.0 - palm.cx).abs() < 1e-5);
        assert!(selected[1].is_none());
    }

    #[test]
    fn crop_samples_at_pixel_center() {
        let mut rgb = [0u8; 27];
        rgb[12] = 255;
        let mut output = [0.0; 3];
        fill_tensor(&mut output, 1, &rgb, 3, 3, |x, y| (x + 1.0, y + 1.0));
        assert_eq!(output, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn crop_keeps_pixels_on_image_border() {
        let mut rgb = [0u8; 27];
        rgb[15] = 255;
        let mut output = [0.0; 3];
        fill_tensor(&mut output, 1, &rgb, 3, 3, |_, _| (2.0, 1.0));
        assert_eq!(output, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn tracked_roi_encloses_palm_joints() {
        let mut points = [(0.0, 0.0); 21];
        for &i in &[0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18] {
            points[i] = (100.0 + i as f32 * 2.0, 130.0 - i as f32);
        }
        points[0] = (100.0, 130.0);
        let (cx, cy, side, angle) = landmarks_roi(&points).unwrap();
        let (sine, cosine) = angle.sin_cos();
        for &i in &[0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18] {
            let dx = points[i].0 - cx;
            let dy = points[i].1 - cy;
            assert!((dx * cosine + dy * sine).abs() < side / 2.0);
            assert!((-dx * sine + dy * cosine).abs() < side / 2.0);
        }
    }
}
