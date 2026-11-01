//! Upscale rules: modes, scales, engine plans, output naming. Pure standard library.

use std::path::{Path, PathBuf};

/// Same list the Python CLI accepts (`imaging.SAVE_FORMATS`).
pub const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Normal,
    Anime,
}

impl Mode {
    pub fn parse(text: &str) -> Option<Mode> {
        match text.trim().to_ascii_lowercase().as_str() {
            "normal" | "photo" => Some(Mode::Normal),
            "anime" => Some(Mode::Anime),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scale {
    X2,
    X4,
}

impl Scale {
    pub fn parse(text: &str) -> Option<Scale> {
        match text.trim().to_ascii_lowercase().as_str() {
            "x2" | "2" => Some(Scale::X2),
            "x4" | "4" => Some(Scale::X4),
            _ => None,
        }
    }

    pub fn factor(self) -> u32 {
        match self {
            Scale::X2 => 2,
            Scale::X4 => 4,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Scale::X2 => "x2",
            Scale::X4 => "x4",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub engine_model: &'static str,
    pub engine_scale: u32,
    pub output_size: (u32, u32),
}

pub fn make_plan(width: u32, height: u32, mode: Mode, scale: Scale) -> Plan {
    let factor = scale.factor();
    let (engine_model, engine_scale) = match mode {
        Mode::Anime => ("realesr-animevideov3", factor),
        // x4plus only exists as a x4 model; x2 is a x4 pass Lanczos-downscaled, which stays crisp.
        Mode::Normal => ("realesrgan-x4plus", 4),
    };
    Plan {
        engine_model,
        engine_scale,
        output_size: (width * factor, height * factor),
    }
}

pub fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// `out_dir/<stem>_x2.<ext>`, or `<stem>_x2 (2).<ext>` etc. when taken, so nothing is overwritten.
pub fn output_path(
    src: &Path,
    out_dir: &Path,
    scale: Scale,
    exists: impl Fn(&Path) -> bool,
) -> PathBuf {
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let ext = src
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_else(|| "png".into());
    let base = format!("{stem}_{}", scale.label());
    let mut candidate = out_dir.join(format!("{base}.{ext}"));
    let mut n = 2;
    while exists(&candidate) {
        candidate = out_dir.join(format!("{base} ({n}).{ext}"));
        n += 1;
    }
    candidate
}

/// The engine prints progress as lines like `37.50%`.
pub fn parse_progress(line: &str) -> Option<f32> {
    let value: f32 = line.trim().strip_suffix('%')?.parse().ok()?;
    (0.0..=100.0).contains(&value).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anime_uses_native_scale() {
        let p = make_plan(100, 50, Mode::Anime, Scale::X2);
        assert_eq!(p.engine_model, "realesr-animevideov3");
        assert_eq!(p.engine_scale, 2);
        assert_eq!(p.output_size, (200, 100));
        assert_eq!(make_plan(100, 50, Mode::Anime, Scale::X4).engine_scale, 4);
    }

    #[test]
    fn normal_always_runs_x4() {
        let p = make_plan(100, 50, Mode::Normal, Scale::X2);
        assert_eq!(p.engine_model, "realesrgan-x4plus");
        assert_eq!(p.engine_scale, 4);
        assert_eq!(p.output_size, (200, 100));
        assert_eq!(
            make_plan(10, 10, Mode::Normal, Scale::X4).output_size,
            (40, 40)
        );
    }

    #[test]
    fn parses_mode_and_scale() {
        assert_eq!(Mode::parse("Anime"), Some(Mode::Anime));
        assert_eq!(Mode::parse("normal"), Some(Mode::Normal));
        assert_eq!(Mode::parse("x"), None);
        assert_eq!(Scale::parse("x4"), Some(Scale::X4));
        assert_eq!(Scale::parse("x3"), None);
    }

    #[test]
    fn filters_extensions() {
        assert!(is_supported_image(Path::new("a/b.PNG")));
        assert!(is_supported_image(Path::new("c.jpeg")));
        assert!(!is_supported_image(Path::new("c.gif")));
        assert!(!is_supported_image(Path::new("noext")));
    }

    #[test]
    fn output_names_avoid_collisions() {
        let out = Path::new("out");
        let free = output_path(Path::new("in/cat.JPG"), out, Scale::X2, |_| false);
        assert_eq!(free, out.join("cat_x2.jpg"));

        let taken = [out.join("cat_x4.png"), out.join("cat_x4 (2).png")];
        let p = output_path(Path::new("cat.png"), out, Scale::X4, |c| {
            taken.iter().any(|t| t == c)
        });
        assert_eq!(p, out.join("cat_x4 (3).png"));
    }

    #[test]
    fn parses_progress_lines() {
        assert_eq!(parse_progress("37.50%"), Some(37.5));
        assert_eq!(parse_progress("  0.00%\r"), Some(0.0));
        assert_eq!(parse_progress("[0 Intel HD]  queueC=0"), None);
        assert_eq!(parse_progress("150%"), None);
    }
}
