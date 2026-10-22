use anyhow::{bail, Context, Result};
use std::{
    process::{Child, ChildStdout},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[derive(Default)]
pub struct CaptureControl {
    stopping: AtomicBool,
    child: Mutex<Option<Child>>,
}

impl CaptureControl {
    pub fn cancelled(&self) -> bool {
        self.stopping.load(Ordering::SeqCst)
    }
    pub fn attach(&self, mut child: Child) -> Result<ChildStdout> {
        // Serialize registration with cancellation so a late camera open cannot
        // spawn a capture process after the main owner has already shut down.
        let mut slot = self.child.lock().unwrap_or_else(|error| error.into_inner());
        if self.cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            bail!("capture cancelled");
        }
        let Some(stdout) = child.stdout.take() else {
            let _ = child.kill();
            let _ = child.wait();
            bail!("capture stdout was not piped");
        };
        *slot = Some(child);
        Ok(stdout)
    }

    fn cancel(&self) {
        self.stopping.store(true, Ordering::SeqCst);
        if let Some(mut child) = self
            .child
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub struct CaptureWorker {
    control: Arc<CaptureControl>,
    handle: Option<JoinHandle<()>>,
}

impl CaptureWorker {
    pub fn spawn(work: impl FnOnce(Arc<CaptureControl>) + Send + 'static) -> Result<Self> {
        let control = Arc::new(CaptureControl::default());
        let worker_control = control.clone();
        let handle = thread::Builder::new()
            .name("camera".into())
            .spawn(move || {
                work(worker_control.clone());
                worker_control.cancel();
            })
            .context("start camera thread")?;
        Ok(Self {
            control,
            handle: Some(handle),
        })
    }
}

impl Drop for CaptureWorker {
    fn drop(&mut self) {
        self.control.cancel();
        if let Some(handle) = self.handle.take() {
            // Native SDK reads may not be cancellable. Never freeze the UI on
            // shutdown; the bridge is already killed by the main-thread owner.
            let deadline = Instant::now() + Duration::from_millis(500);
            while !handle.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(2));
            }
            if handle.is_finished() {
                let _ = handle.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        time::{Duration, Instant},
    };

    #[test]
    fn drop_requests_worker_cancellation() {
        let stopped = Arc::new(AtomicBool::new(false));
        let result = stopped.clone();
        let worker = CaptureWorker::spawn(move |control| {
            let start = Instant::now();
            while !control.cancelled() && start.elapsed() < Duration::from_secs(1) {
                thread::sleep(Duration::from_millis(1));
            }
            result.store(control.cancelled(), Ordering::SeqCst);
        })
        .unwrap();
        drop(worker);
        assert!(stopped.load(Ordering::SeqCst));
    }

    #[cfg(windows)]
    #[test]
    fn drop_terminates_child_even_when_worker_is_blocked_reading() {
        use std::{
            io::Read,
            process::{Command, Stdio},
        };
        let (ready, started) = mpsc::channel();
        let (done, finished) = mpsc::channel();
        let worker = CaptureWorker::spawn(move |control| {
            let child = Command::new("powershell.exe")
                .args([
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "Start-Sleep -Seconds 2",
                ])
                .stdout(Stdio::piped())
                .spawn()
                .unwrap();
            let mut stdout = control.attach(child).unwrap();
            ready.send(()).unwrap();
            let _ = stdout.read_to_end(&mut Vec::new());
            let _ = done.send(());
        })
        .unwrap();
        started.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(worker);
        assert!(finished.recv_timeout(Duration::from_millis(750)).is_ok());
    }
}
