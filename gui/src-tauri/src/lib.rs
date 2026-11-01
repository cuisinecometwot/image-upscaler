use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use serde::Serialize;
use tauri::{AppHandle, Emitter, State};
use upscaler_core::{expand_inputs as expand, upscale_file, Engine, EngineError, Mode, Scale};

#[derive(Default)]
struct JobState {
    running: AtomicBool,
    cancel: Arc<AtomicBool>,
}

#[derive(Serialize)]
struct FileInfo {
    path: String,
    name: String,
    size: u64,
}

#[derive(Serialize, Clone)]
struct FileStatus {
    index: usize,
    /// running | done | error | cancelled
    status: &'static str,
    progress: f32,
    message: Option<String>,
    output: Option<String>,
}

#[derive(Serialize, Clone)]
struct JobFinished {
    done: usize,
    failed: usize,
    cancelled: bool,
}

#[tauri::command]
fn expand_inputs(paths: Vec<String>) -> Vec<FileInfo> {
    let paths: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
    expand(&paths)
        .into_iter()
        .map(|p| FileInfo {
            name: p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            size: p.metadata().map(|m| m.len()).unwrap_or(0),
            path: p.to_string_lossy().into_owned(),
        })
        .collect()
}

#[tauri::command]
fn default_output_dir(first_input: String) -> String {
    let parent = Path::new(&first_input).parent().unwrap_or(Path::new("."));
    parent.join("upscaled").to_string_lossy().into_owned()
}

#[tauri::command]
fn open_folder(path: String) -> Result<(), String> {
    tauri_plugin_opener::open_path(path, None::<&str>).map_err(|e| e.to_string())
}

#[tauri::command]
fn cancel_job(state: State<'_, Arc<JobState>>) {
    state.cancel.store(true, Ordering::Relaxed);
}

#[tauri::command]
fn start_job(
    app: AppHandle,
    state: State<'_, Arc<JobState>>,
    files: Vec<String>,
    out_dir: String,
    mode: String,
    scale: String,
) -> Result<(), String> {
    let mode = Mode::parse(&mode).ok_or("invalid mode")?;
    let scale = Scale::parse(&scale).ok_or("invalid scale")?;
    if files.is_empty() {
        return Err("No images selected.".into());
    }
    if out_dir.trim().is_empty() {
        return Err("Choose an output folder.".into());
    }
    if state.running.swap(true, Ordering::SeqCst) {
        return Err("A job is already running.".into());
    }
    state.cancel.store(false, Ordering::SeqCst);

    let state = Arc::clone(&state);
    thread::spawn(move || {
        let summary = run_job(
            &app,
            &state.cancel,
            &files,
            Path::new(&out_dir),
            mode,
            scale,
        );
        state.running.store(false, Ordering::SeqCst);
        let _ = app.emit("job-finished", summary);
    });
    Ok(())
}

fn run_job(
    app: &AppHandle,
    cancel: &AtomicBool,
    files: &[String],
    out_dir: &Path,
    mode: Mode,
    scale: Scale,
) -> JobFinished {
    let emit = |index, status, progress, message: Option<String>, output: Option<String>| {
        let _ = app.emit(
            "file-status",
            FileStatus {
                index,
                status,
                progress,
                message,
                output,
            },
        );
    };

    let engine = match Engine::prepare() {
        Ok(e) => e,
        Err(e) => {
            for i in 0..files.len() {
                emit(i, "error", 0.0, Some(e.to_string()), None);
            }
            return JobFinished {
                done: 0,
                failed: files.len(),
                cancelled: false,
            };
        }
    };

    let (mut done, mut failed) = (0, 0);
    for (i, file) in files.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            emit(i, "cancelled", 0.0, None, None);
            continue;
        }
        emit(i, "running", 0.0, None, None);
        let mut on_progress = |p: f32| emit(i, "running", p, None, None);
        match upscale_file(
            &engine,
            Path::new(file),
            out_dir,
            mode,
            scale,
            cancel,
            &mut on_progress,
        ) {
            Ok(dst) => {
                done += 1;
                emit(
                    i,
                    "done",
                    100.0,
                    None,
                    Some(dst.to_string_lossy().into_owned()),
                );
            }
            Err(EngineError::Cancelled) => emit(i, "cancelled", 0.0, None, None),
            Err(e) => {
                failed += 1;
                emit(i, "error", 0.0, Some(e.to_string()), None);
            }
        }
    }
    JobFinished {
        done,
        failed,
        cancelled: cancel.load(Ordering::Relaxed),
    }
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(Arc::new(JobState::default()))
        .invoke_handler(tauri::generate_handler![
            expand_inputs,
            default_output_dir,
            open_folder,
            start_job,
            cancel_job
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
