//! Image side of the pipeline: decode, alpha split, AI pass, exact resize, save.
//! Mirrors the Python `upscaler/imaging.py`.

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use image::codecs::jpeg::JpegEncoder;
use image::imageops::{self, FilterType};
use image::{DynamicImage, GrayImage, ImageDecoder, ImageReader, Rgb, RgbImage, RgbaImage};

use crate::domain::{is_supported_image, make_plan, output_path, Mode, Scale};
use crate::engine::{Engine, EngineError};

/// Files as given, plus images directly inside given folders (not recursive, like the CLI).
/// Sorted, de-duplicated.
pub fn expand_inputs(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_dir() {
            if let Ok(entries) = fs::read_dir(p) {
                out.extend(
                    entries
                        .flatten()
                        .map(|e| e.path())
                        .filter(|f| f.is_file() && is_supported_image(f)),
                );
            }
        } else if p.is_file() && is_supported_image(p) {
            out.push(p.clone());
        }
    }
    out.sort();
    out.dedup();
    out
}

fn load(path: &Path) -> Result<DynamicImage, EngineError> {
    let err = |e: image::ImageError| EngineError::Io(format!("cannot read image: {e}"));
    let mut decoder = ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()
        .map_err(err)?;
    let orientation = decoder.orientation().map_err(err)?;
    let mut img = DynamicImage::from_decoder(decoder).map_err(err)?;
    img.apply_orientation(orientation);
    Ok(img)
}

fn split_alpha(img: &DynamicImage) -> (RgbImage, Option<GrayImage>) {
    if img.color().has_alpha() {
        let rgba = img.to_rgba8();
        // A fully opaque alpha would only cost time to carry through.
        if rgba.pixels().any(|p| p[3] != 255) {
            let alpha = GrayImage::from_fn(rgba.width(), rgba.height(), |x, y| {
                image::Luma([rgba.get_pixel(x, y)[3]])
            });
            return (img.to_rgb8(), Some(alpha));
        }
    }
    (img.to_rgb8(), None)
}

fn save(img: DynamicImage, path: &Path) -> Result<(), EngineError> {
    let err = |e: image::ImageError| EngineError::Io(format!("cannot save image: {e}"));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "jpg" | "jpeg" => {
            let rgb = match img {
                DynamicImage::ImageRgba8(rgba) => flatten_on_white(&rgba),
                other => other.to_rgb8(),
            };
            let mut w = BufWriter::new(File::create(path)?);
            JpegEncoder::new_with_quality(&mut w, 95)
                .encode_image(&rgb)
                .map_err(err)
        }
        _ => img.save(path).map_err(err),
    }
}

fn flatten_on_white(rgba: &RgbaImage) -> RgbImage {
    RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        let p = rgba.get_pixel(x, y);
        let a = p[3] as u32;
        let mix = |c: u8| ((c as u32 * a + 255 * (255 - a)) / 255) as u8;
        Rgb([mix(p[0]), mix(p[1]), mix(p[2])])
    })
}

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> std::io::Result<TempDir> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("upscaler-{}-{n}", std::process::id()));
        fs::create_dir_all(&dir)?;
        Ok(TempDir(dir))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Upscales one file into `out_dir` and returns the written path (never overwrites).
pub fn upscale_file(
    engine: &Engine,
    src: &Path,
    out_dir: &Path,
    mode: Mode,
    scale: Scale,
    cancel: &AtomicBool,
    on_progress: &mut dyn FnMut(f32),
) -> Result<PathBuf, EngineError> {
    let img = load(src)?;
    let plan = make_plan(img.width(), img.height(), mode, scale);
    let (rgb, alpha) = split_alpha(&img);
    drop(img);

    let tmp = TempDir::new()?;
    let tmp_in = tmp.0.join("in.png");
    let tmp_out = tmp.0.join("out.png");
    rgb.save(&tmp_in)
        .map_err(|e| EngineError::Io(format!("cannot write temp file: {e}")))?;
    drop(rgb);
    engine.run(
        &tmp_in,
        &tmp_out,
        plan.engine_model,
        plan.engine_scale,
        cancel,
        on_progress,
    )?;

    let (w, h) = plan.output_size;
    let mut rgb = load(&tmp_out)?.to_rgb8();
    if rgb.dimensions() != (w, h) {
        rgb = imageops::resize(&rgb, w, h, FilterType::Lanczos3);
    }
    let result = match alpha {
        None => DynamicImage::ImageRgb8(rgb),
        Some(alpha) => {
            let alpha = imageops::resize(&alpha, w, h, FilterType::Lanczos3);
            let rgba = RgbaImage::from_fn(w, h, |x, y| {
                let [r, g, b] = rgb.get_pixel(x, y).0;
                image::Rgba([r, g, b, alpha.get_pixel(x, y)[0]])
            });
            DynamicImage::ImageRgba8(rgba)
        }
    };

    let dst = output_path(src, out_dir, scale, |p| p.exists());
    save(result, &dst)?;
    Ok(dst)
}
