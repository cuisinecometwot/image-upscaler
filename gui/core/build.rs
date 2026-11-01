//! Embeds every file under `gui/engine/` (fetched by scripts/fetch-engine.sh) into the binary,
//! so the GUI ships as one portable exe. A missing folder yields an empty list.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn collect(dir: &Path, root: &Path, out: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, root, out);
        } else {
            let rel = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            out.push((rel, path));
        }
    }
}

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../engine");
    println!("cargo:rerun-if-changed={}", root.display());
    let mut files = Vec::new();
    collect(&root, &root, &mut files);
    files.sort();
    for (_, path) in &files {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let mut code = String::from("pub static FILES: &[(&str, &[u8])] = &[\n");
    for (rel, path) in &files {
        let abs = fs::canonicalize(path).unwrap();
        code += &format!(
            "    ({rel:?}, include_bytes!({:?})),\n",
            abs.to_string_lossy()
        );
    }
    code += "];\n";
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("embedded.rs");
    fs::write(out, code).unwrap();
}
