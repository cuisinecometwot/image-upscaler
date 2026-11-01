//! Smoke test without the GUI: cargo run -p upscaler-core --example upscale -- <in> <out_dir> <normal|anime> <x2|x4>

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use upscaler_core::{expand_inputs, upscale_file, Engine, Mode, Scale};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [input, out_dir, mode, scale] = args.as_slice() else {
        eprintln!("usage: upscale <file|folder> <out_dir> <normal|anime> <x2|x4>");
        std::process::exit(2);
    };
    let mode = Mode::parse(mode).expect("mode");
    let scale = Scale::parse(scale).expect("scale");
    let engine = Engine::prepare().unwrap_or_else(|e| panic!("{e}"));
    let cancel = AtomicBool::new(false);
    for src in expand_inputs(&[PathBuf::from(input)]) {
        let started = std::time::Instant::now();
        let result = upscale_file(
            &engine,
            &src,
            out_dir.as_ref(),
            mode,
            scale,
            &cancel,
            &mut |_| {},
        );
        match result {
            Ok(dst) => println!(
                "{} -> {} ({:.1?})",
                src.display(),
                dst.display(),
                started.elapsed()
            ),
            Err(e) => println!("{}: error: {e}", src.display()),
        }
    }
}
