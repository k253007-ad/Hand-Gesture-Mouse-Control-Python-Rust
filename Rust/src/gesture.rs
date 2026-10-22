use glam::Vec3;
use std::time::Instant;

pub type Hand = [Vec3; 21];

#[derive(Debug, PartialEq)]
pub struct Pinches {
    pub left: bool,
    pub double: bool,
    pub picker: bool,
    pub right: bool,
}

pub fn pinches(hand: &Hand, same_hand: bool, left_down: bool) -> Pinches {
    let palm = distance(hand, 5, 17).max(0.01);
    let ratio = |tip| distance(hand, 4, tip) / palm;
    let index = ratio(8);
    let middle = ratio(12);
    let ring = ratio(16);
    let pinky = ratio(20);
    Pinches {
        left: index < if left_down { 0.52 } else { 0.38 },
        double: !same_hand && middle < 0.52 && index > 0.65 && ring > 0.58 && pinky > 0.65,
        picker: ring < 0.50 && index > 0.65 && middle > 0.60 && pinky > 0.58,
        right: !same_hand && pinky < 0.38 && index > 0.65 && middle > 0.60 && ring > 0.55,
    }
}

pub fn distance(hand: &Hand, a: usize, b: usize) -> f32 {
    hand[a].truncate().distance(hand[b].truncate())
}

fn straight(hand: &Hand, mcp: usize, pip: usize, tip: usize) -> f32 {
    let a = (hand[pip] - hand[mcp]).truncate();
    let b = (hand[tip] - hand[pip]).truncate();
    a.dot(b) / (a.length() * b.length()).max(1e-6)
}

pub fn pointing(hand: &Hand) -> bool {
    let palm = distance(hand, 5, 17).max(0.01);
    straight(hand, 5, 6, 8) > 0.65
        && straight(hand, 9, 10, 12) < 0.45
        && straight(hand, 13, 14, 16) < 0.45
        && straight(hand, 17, 18, 20) < 0.45
        && distance(hand, 4, 5) < 1.45 * palm
        && distance(hand, 4, 8) > 0.55 * palm
}

pub fn scroll_pose(hand: &Hand) -> bool {
    let palm = distance(hand, 5, 17).max(0.01);
    straight(hand, 5, 6, 8) > 0.65
        && straight(hand, 9, 10, 12) > 0.65
        && straight(hand, 13, 14, 16) < 0.5
        && straight(hand, 17, 18, 20) < 0.5
        && distance(hand, 8, 12) > 0.35 * palm
        && distance(hand, 4, 8) > 0.65 * palm
        && distance(hand, 4, 12) > 0.65 * palm
}

#[derive(Clone, Copy, Debug)]
pub struct OneEuro {
    value: Vec3,
    derivative: Vec3,
    last: Option<Instant>,
    initialized: bool,
    min_cutoff: f32,
    beta: f32,
    derivative_cutoff: f32,
}

impl Default for OneEuro {
    fn default() -> Self {
        Self {
            value: Vec3::ZERO,
            derivative: Vec3::ZERO,
            last: None,
            initialized: false,
            min_cutoff: 1.5,
            beta: 4.0,
            derivative_cutoff: 1.0,
        }
    }
}

impl OneEuro {
    fn alpha(cutoff: f32, dt: f32) -> f32 {
        let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff.max(0.001));
        1.0 / (1.0 + tau / dt)
    }

    pub fn filter(&mut self, sample: Vec3, now: Instant) -> Vec3 {
        if !self.initialized {
            self.value = sample;
            self.last = Some(now);
            self.initialized = true;
            return sample;
        }
        let dt = now
            .duration_since(self.last.unwrap_or(now))
            .as_secs_f32()
            .clamp(0.001, 0.25);
        self.last = Some(now);
        let raw_derivative = (sample - self.value) / dt;
        self.derivative +=
            Self::alpha(self.derivative_cutoff, dt) * (raw_derivative - self.derivative);
        let cutoff = self.min_cutoff + self.beta * self.derivative.length();
        self.value += Self::alpha(cutoff, dt) * (sample - self.value);
        self.value
    }
}

#[derive(Default)]
pub struct Scroll {
    anchor: Option<(f32, f32)>,
    last: Option<Instant>,
    remainder: f32,
    pub velocity: f32,
}

impl Scroll {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn update(&mut self, hand: Option<&Hand>, now: Instant) -> i32 {
        let Some(hand) = hand else {
            self.reset();
            return 0;
        };
        let y = (hand[0].y + hand[9].y) / 2.0;
        let palm = distance(hand, 5, 17).max(0.01);
        let previous_time = self.last.replace(now);
        let Some((origin, anchor_palm)) = self.anchor else {
            self.anchor = Some((y, palm));
            return 0;
        };
        let displacement = (origin - y) / anchor_palm;
        let dt = now
            .duration_since(previous_time.unwrap_or(now))
            .as_secs_f32()
            .min(0.25);
        let speed = ((displacement.abs() - 0.18).max(0.0) * 12.0).min(24.0);
        let velocity = if speed > 0.0 {
            speed.copysign(displacement)
        } else {
            0.0
        };
        if velocity == 0.0 || self.velocity * velocity < 0.0 {
            self.remainder = 0.0;
        }
        self.velocity = velocity;
        self.remainder += velocity * dt;
        let steps = (self.remainder.trunc() as i32).clamp(-3, 3);
        self.remainder -= steps as f32;
        steps * 30
    }
}

#[derive(Default)]
pub struct Toggle {
    pub enabled: bool,
    armed: bool,
    pinched: u8,
    released: u8,
}

impl Toggle {
    pub fn new() -> Self {
        Self {
            enabled: true,
            armed: true,
            ..Self::default()
        }
    }

    pub fn update(&mut self, hand: Option<&Hand>) -> bool {
        let Some(hand) = hand else {
            self.pinched = 0;
            self.released = 0;
            return false;
        };
        let ratio = if pointing(hand) {
            1.0
        } else {
            distance(hand, 4, 20) / distance(hand, 5, 17).max(0.01)
        };
        if ratio < 0.42 {
            self.released = 0;
            self.pinched = self.pinched.saturating_add(1);
            if self.armed && self.pinched >= 2 {
                self.enabled = !self.enabled;
                self.armed = false;
                return true;
            }
        } else if ratio > 0.62 {
            self.pinched = 0;
            self.released = self.released.saturating_add(1);
            if self.released >= 2 {
                self.armed = true;
            }
        } else {
            self.pinched = 0;
            self.released = 0;
        }
        false
    }
}

pub fn triangle(left: &Hand, right: &Hand, aspect: f32) -> bool {
    let scaled = |v: Vec3| Vec3::new(v.x * aspect, v.y, 0.0);
    let li = scaled(left[8]);
    let ri = scaled(right[8]);
    let lt = scaled(left[4]);
    let rt = scaled(right[4]);
    let palm = (scaled(left[5]).distance(scaled(left[17]))
        + scaled(right[5]).distance(scaled(right[17])))
        / 2.0;
    if palm < 0.025 {
        return false;
    }
    let apex = (li + ri) / 2.0;
    let base = (lt + rt) / 2.0;
    let width = (lt.x - rt.x).abs();
    li.distance(ri) < 0.42 * palm
        && width > 0.85 * palm
        && (lt.y - rt.y).abs() < 0.4 * palm
        && base.y - apex.y > 0.55 * palm
        && (apex.x - base.x).abs() < 0.35 * width
        && apex.x > lt.x.min(rt.x)
        && apex.x < lt.x.max(rt.x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_euro_starts_at_first_sample_and_converges() {
        let mut filter = OneEuro::default();
        let now = Instant::now();
        assert_eq!(filter.filter(Vec3::ZERO, now), Vec3::ZERO);
        let result = filter.filter(Vec3::ONE, now + std::time::Duration::from_millis(50));
        assert!(result.x > 0.0 && result.x < 1.0);
    }

    #[test]
    fn one_euro_follows_a_deliberate_move_within_one_frame() {
        let mut filter = OneEuro::default();
        let now = Instant::now();
        filter.filter(Vec3::ZERO, now);
        let moved = filter.filter(
            Vec3::new(0.5, 0.0, 0.0),
            now + std::time::Duration::from_millis(50),
        );
        assert!(moved.x > 0.35 && moved.x < 0.5);
    }

    #[test]
    fn index_pinch_is_not_point_only() {
        let mut hand = [Vec3::ZERO; 21];
        for (index, x, y) in [
            (4, 0.6, 0.0),
            (5, 0.0, 0.0),
            (6, 0.0, -0.5),
            (8, 0.0, -1.0),
            (9, 0.3, 0.0),
            (10, 0.3, -0.3),
            (12, 0.3, 0.0),
            (13, 0.7, 0.0),
            (14, 0.7, -0.3),
            (16, 0.7, 0.0),
            (17, 1.0, 0.0),
            (18, 1.0, -0.3),
            (20, 1.0, 0.0),
        ] {
            hand[index] = Vec3::new(x, y, 0.0);
        }
        assert!(pointing(&hand));
        hand[4] = hand[8];
        assert!(!pointing(&hand));
    }

    #[test]
    fn toggle_requires_two_frames_and_release() {
        let mut toggle = Toggle::new();
        let mut hand = [Vec3::ZERO; 21];
        hand[17].x = 1.0;
        hand[5].x = 0.0;
        hand[4].x = 0.5;
        hand[20].x = 0.5;
        assert!(!toggle.update(Some(&hand)));
        assert!(toggle.update(Some(&hand)));
        assert!(!toggle.enabled);
        assert!(!toggle.update(Some(&hand)));
        hand[20].x = 2.0;
        toggle.update(Some(&hand));
        toggle.update(Some(&hand));
        hand[20].x = 0.5;
        toggle.update(Some(&hand));
        assert!(toggle.update(Some(&hand)));
        assert!(toggle.enabled);
    }

    #[test]
    fn scroll_continues_while_held_away_from_anchor() {
        let mut hand = [Vec3::ZERO; 21];
        for (index, x, y) in [
            (0, 0.5, 0.5),
            (4, 1.5, 0.0),
            (5, 0.0, 0.0),
            (6, 0.0, -1.0),
            (8, 0.0, -2.0),
            (9, 0.5, 0.0),
            (10, 0.5, -1.0),
            (12, 0.5, -2.0),
            (13, 0.75, 0.0),
            (14, 0.75, -0.4),
            (16, 0.75, -0.1),
            (17, 1.0, 0.0),
            (18, 1.0, -0.4),
            (20, 1.0, -0.1),
        ] {
            hand[index] = Vec3::new(x, y, 0.0);
        }
        assert!(scroll_pose(&hand));
        let mut scroll = Scroll::default();
        let now = Instant::now();
        assert_eq!(scroll.update(Some(&hand), now), 0);
        for point in &mut hand {
            point.y -= 1.0;
        }
        let first = scroll.update(Some(&hand), now + std::time::Duration::from_millis(250));
        assert!(scroll.velocity > 0.0);
        assert!(first > 0);
        let held = scroll.update(Some(&hand), now + std::time::Duration::from_millis(500));
        assert!(held > 0);
        for point in &mut hand {
            point.y += 1.0;
        }
        assert_eq!(
            scroll.update(Some(&hand), now + std::time::Duration::from_millis(750)),
            0
        );
        assert_eq!(scroll.velocity, 0.0);
    }

    #[test]
    fn pinch_thresholds_match_python_controller() {
        let mut hand = [Vec3::ZERO; 21];
        hand[17].x = 1.0;
        hand[4].x = 0.0;
        hand[8].x = 0.43;
        hand[12].x = 0.80;
        hand[16].x = 0.80;
        hand[20].x = 0.80;
        assert!(!pinches(&hand, true, false).left);
        assert!(pinches(&hand, true, true).left);
        hand[8].x = 0.37;
        assert!(pinches(&hand, true, false).left);
        hand[8].x = 0.80;
        hand[20].x = 0.43;
        assert!(!pinches(&hand, false, false).right);
        hand[20].x = 0.37;
        assert!(pinches(&hand, false, false).right);
        hand[20].x = 0.80;
        hand[12].x = 0.55;
        assert!(!pinches(&hand, false, false).double);
        hand[12].x = 0.51;
        assert!(pinches(&hand, false, false).double);
    }
}
