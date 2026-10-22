use anyhow::{bail, ensure, Context, Result};

pub const HELP: &str = "Hand Mouse (Windows)\n\nUsage: hand-mouse [OPTIONS]\n\n  -h, --help                 Print help and exit\n  --camera INDEX             Webcam index (default: 0)\n  --left                     Prefer the left pointer hand\n  --tracking-only            Preview without mouse or keyboard input\n  --cuda                     Request a compatible CUDA runtime\n  --benchmark-seconds N      Measure tracking for 0 < N <= 300 seconds (no input)\n  --benchmark-file PATH      Replay an HMR2 clip without camera or input\n\nPreview: H switch hand, Space pause, Q/Esc quit.\n";

#[derive(Debug, Default)]
pub struct Options {
    pub help: bool,
    pub camera: u32,
    pub left: bool,
    pub tracking_only: bool,
    pub cuda: bool,
    pub benchmark_seconds: Option<f32>,
    pub benchmark_file: Option<String>,
}

impl Options {
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut options = Self::default();
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => options.help = true,
                "--left" => options.left = true,
                "--tracking-only" => options.tracking_only = true,
                "--cuda" => options.cuda = true,
                "--camera" => {
                    options.camera = args
                        .next()
                        .context("--camera requires an index")?
                        .parse()
                        .context("--camera must be a nonnegative integer")?;
                }
                "--benchmark-seconds" => {
                    let seconds: f32 = args
                        .next()
                        .context("--benchmark-seconds requires a value")?
                        .parse()
                        .context("invalid --benchmark-seconds value")?;
                    ensure!(
                        seconds.is_finite() && seconds > 0.0 && seconds <= 300.0,
                        "--benchmark-seconds must be greater than 0 and at most 300"
                    );
                    options.benchmark_seconds = Some(seconds);
                    options.tracking_only = true;
                }
                "--benchmark-file" => {
                    let path = args.next().context("--benchmark-file requires a path")?;
                    ensure!(
                        !path.is_empty() && !path.starts_with("--"),
                        "--benchmark-file requires a path"
                    );
                    options.benchmark_file = Some(path);
                }
                _ => bail!("unknown argument {arg:?}; use --help"),
            }
        }
        ensure!(
            options.benchmark_seconds.is_none() || options.benchmark_file.is_none(),
            "choose either --benchmark-file or --benchmark-seconds"
        );
        Ok(options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Options> {
        Options::parse(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn help_does_not_request_camera_or_input_startup() {
        assert!(parse(&["--help"]).unwrap().help);
        assert!(parse(&["-h"]).unwrap().help);
    }

    #[test]
    fn malformed_options_fail_instead_of_starting_camera_zero() {
        for args in [
            vec!["--typo"],
            vec!["--camera"],
            vec!["--camera", "oops"],
            vec!["--camera", "-1"],
            vec!["--benchmark-file"],
            vec!["--benchmark-seconds"],
            vec!["--benchmark-seconds", "NaN"],
            vec!["--benchmark-seconds", "0"],
            vec!["--benchmark-seconds", "301"],
            vec!["--benchmark-file", "clip", "--benchmark-seconds", "1"],
        ] {
            assert!(parse(&args).is_err(), "accepted {args:?}");
        }
    }

    #[test]
    fn valid_options_and_benchmarks_have_explicit_behavior() {
        let options = parse(&["--camera", "2", "--left", "--tracking-only", "--cuda"]).unwrap();
        assert_eq!(options.camera, 2);
        assert!(options.left && options.tracking_only && options.cuda);
        let benchmark = parse(&["--benchmark-seconds", "2.5"]).unwrap();
        assert_eq!(benchmark.benchmark_seconds, Some(2.5));
        assert!(benchmark.tracking_only);
        assert_eq!(
            parse(&["--benchmark-file", "clip.hmr2"])
                .unwrap()
                .benchmark_file
                .as_deref(),
            Some("clip.hmr2")
        );
    }
}
