use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    let deepgemm_root = discover_deepgemm_root();
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );

    println!("cargo:rerun-if-env-changed=DEEPGEMM_ROOT");
    println!("cargo:rerun-if-changed=csrc/deepgemm_c_api.h");
    println!("cargo:rerun-if-changed=csrc/deepgemm_c_api.cc");
    println!(
        "cargo:rerun-if-changed={}",
        deepgemm_root.join("deep_gemm").join("include").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        deepgemm_root.join("csrc").display()
    );
    println!(
        "cargo:rustc-env=DEEPGEMM_SOURCE_ROOT={}",
        deepgemm_root.display()
    );
    println!("cargo:metadata=source_root={}", deepgemm_root.display());

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(manifest_dir.join("csrc"))
        .file(manifest_dir.join("csrc").join("deepgemm_c_api.cc"))
        .compile("deepgemm_c_api");
}

fn discover_deepgemm_root() -> PathBuf {
    if let Some(root) = env::var_os("DEEPGEMM_ROOT") {
        if root.is_empty() {
            panic!("DEEPGEMM_ROOT is set but empty");
        }
        return validate_deepgemm_root(PathBuf::from(root), "DEEPGEMM_ROOT");
    }

    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );
    validate_deepgemm_root(
        manifest_dir.join("vendor").join("DeepGEMM"),
        "vendored DeepGEMM submodule",
    )
}

fn validate_deepgemm_root(path: PathBuf, source: &str) -> PathBuf {
    let root = path.canonicalize().unwrap_or_else(|error| {
        panic!(
            "{source} does not point to a readable DeepGEMM checkout at {}: {error}",
            path.display()
        )
    });

    require_path(&root, "CMakeLists.txt", source);
    require_path(&root, "csrc/python_api.cpp", source);
    require_path(
        &root,
        "deep_gemm/include/deep_gemm/common/types.cuh",
        source,
    );
    require_path(
        &root,
        "deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh",
        source,
    );
    require_path(
        &root,
        "third-party/cutlass/include/cutlass/cutlass.h",
        source,
    );
    require_path(&root, "third-party/fmt/include/fmt/core.h", source);

    root
}

fn require_path(root: &Path, relative: &str, source: &str) {
    let path = root.join(relative);
    if path.exists() {
        return;
    }

    if relative.starts_with("third-party/") {
        panic!(
            "{source} DeepGEMM checkout is missing {relative}. Initialize nested submodules with `git submodule update --init --recursive crates/deepgemm-sys/vendor/DeepGEMM`."
        );
    }

    panic!(
        "{source} does not look like a DeepGEMM checkout: missing {}",
        path.display()
    );
}
