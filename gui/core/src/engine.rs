//! Runs realesrgan-ncnn-vulkan, extracted on first use from the files embedded at build time.

use std::fmt;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use crate::domain::parse_progress;

mod embedded {
    include!(concat!(env!("OUT_DIR"), "/embedded.rs"));
}

/// Bump when the bundled engine or models change, so stale extractions are not reused.
const ENGINE_VERSION: &str = "v0.2.5.0";

#[cfg(windows)]
const BINARY_NAME: &str = "realesrgan-ncnn-vulkan.exe";
#[cfg(not(windows))]
const BINARY_NAME: &str = "realesrgan-ncnn-vulkan";

#[derive(Debug)]
pub enum EngineError {
    Missing(String),
    Io(String),
    Failed(String),
    Cancelled,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::Missing(m) | EngineError::Io(m) | EngineError::Failed(m) => f.write_str(m),
            EngineError::Cancelled => f.write_str("cancelled"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<std::io::Error> for EngineError {
    fn from(e: std::io::Error) -> Self {
        EngineError::Io(e.to_string())
    }
}

pub struct Engine {
    dir: PathBuf,
}

fn data_dir() -> PathBuf {
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
    };
    base.unwrap_or_else(std::env::temp_dir).join("Upscaler")
}

fn extract(dir: &Path) -> Result<(), EngineError> {
    for (rel, bytes) in embedded::FILES {
        let path = dir.join(rel);
        if fs::metadata(&path)
            .map(|m| m.len() == bytes.len() as u64)
            .unwrap_or(false)
        {
            continue;
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        // Write then rename so an interrupted first run never leaves a truncated model behind.
        let tmp = path.with_extension("partial");
        fs::write(&tmp, bytes)?;
        fs::rename(&tmp, &path)?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let bin = dir.join(BINARY_NAME);
        if bin.is_file() {
            fs::set_permissions(&bin, fs::Permissions::from_mode(0o755))?;
        }
    }
    Ok(())
}

impl Engine {
    /// `UPSCALER_ENGINE_DIR` points at an existing engine install (dev builds without embedding).
    pub fn prepare() -> Result<Engine, EngineError> {
        let dir = match std::env::var_os("UPSCALER_ENGINE_DIR") {
            Some(custom) => PathBuf::from(custom),
            None if embedded::FILES.is_empty() => {
                return Err(EngineError::Missing(
                    "Real-ESRGAN engine is not bundled in this build.".into(),
                ))
            }
            None => {
                let dir = data_dir().join(format!("engine-{ENGINE_VERSION}"));
                extract(&dir)?;
                dir
            }
        };
        if !dir.join(BINARY_NAME).is_file() {
            return Err(EngineError::Missing(format!(
                "Real-ESRGAN engine not found in {}",
                dir.display()
            )));
        }
        Ok(Engine { dir })
    }

    pub fn run(
        &self,
        src: &Path,
        dst: &Path,
        model: &str,
        scale: u32,
        cancel: &AtomicBool,
        on_progress: &mut dyn FnMut(f32),
    ) -> Result<(), EngineError> {
        let mut cmd = Command::new(self.dir.join(BINARY_NAME));
        cmd.arg("-i")
            .arg(src)
            .arg("-o")
            .arg(dst)
            .arg("-n")
            .arg(model)
            .arg("-s")
            .arg(scale.to_string())
            .arg("-m")
            .arg(self.dir.join("models"))
            .arg("-f")
            .arg("png")
            .current_dir(&self.dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| EngineError::Failed(format!("could not start Real-ESRGAN: {e}")))?;

        // Read stderr on a thread so the main loop can poll for cancel without blocking.
        let stderr = child.stderr.take().expect("stderr is piped");
        let (tx, rx) = mpsc::channel::<String>();
        let reader = thread::spawn(move || {
            for line in BufReader::new(stderr).split(b'\n').map_while(Result::ok) {
                if tx
                    .send(String::from_utf8_lossy(&line).into_owned())
                    .is_err()
                {
                    break;
                }
            }
        });

        let mut log_tail: Vec<String> = Vec::new();
        let mut handle_line = |line: String, log_tail: &mut Vec<String>| match parse_progress(&line)
        {
            Some(p) => on_progress(p),
            None => {
                log_tail.push(line.trim().to_string());
                if log_tail.len() > 8 {
                    log_tail.remove(0);
                }
            }
        };

        let status = loop {
            if cancel.load(Ordering::Relaxed) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(EngineError::Cancelled);
            }
            while let Ok(line) = rx.try_recv() {
                handle_line(line, &mut log_tail);
            }
            if let Some(status) = child.try_wait()? {
                break status;
            }
            thread::sleep(Duration::from_millis(100));
        };
        let _ = reader.join();
        while let Ok(line) = rx.try_recv() {
            handle_line(line, &mut log_tail);
        }

        // The binary can exit 0 without writing output (e.g. Vulkan init failure), so check the file.
        if !status.success() || !dst.is_file() {
            return Err(EngineError::Failed(format!(
                "Real-ESRGAN failed ({status}). Check that the GPU supports Vulkan.\n{}",
                log_tail.join("\n")
            )));
        }
        on_progress(100.0);
        Ok(())
    }
}
