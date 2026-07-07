use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    let deepgemm_root = discover_deepgemm_root();
    let cuda_include = discover_cuda_include();
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );

    println!("cargo:rerun-if-env-changed=DEEPGEMM_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-changed=csrc/deepgemm_c_api.h");
    println!("cargo:rerun-if-changed=csrc/deepgemm_c_api.cc");
    println!("cargo:rerun-if-changed=csrc/deepgemm_raw_runtime.h");
    println!("cargo:rerun-if-changed=csrc/deepgemm_raw_runtime.cc");
    println!("cargo:rerun-if-changed=csrc/deepgemm_raw_mqa.h");
    println!("cargo:rerun-if-changed=csrc/deepgemm_raw_mqa.cc");
    println!("cargo:rerun-if-changed=csrc/deepgemm_raw_gemm.h");
    println!("cargo:rerun-if-changed=csrc/deepgemm_raw_gemm.cc");
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
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=dl");
    }

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(manifest_dir.join("csrc"))
        .include(cuda_include)
        .file(manifest_dir.join("csrc").join("deepgemm_c_api.cc"))
        .file(manifest_dir.join("csrc").join("deepgemm_raw_runtime.cc"))
        .file(manifest_dir.join("csrc").join("deepgemm_raw_mqa.cc"))
        .file(manifest_dir.join("csrc").join("deepgemm_raw_gemm.cc"))
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

fn discover_cuda_include() -> PathBuf {
    let mut candidates = Vec::new();
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = env::var_os(var) {
            add_cuda_include_candidates(&mut candidates, PathBuf::from(root));
        }
    }
    add_cuda_include_candidates(&mut candidates, PathBuf::from("/usr/local/cuda"));

    for candidate in candidates {
        if candidate.join("cuda.h").exists() {
            return candidate;
        }
    }

    panic!(
        "could not find CUDA headers. Set CUDA_HOME to a CUDA toolkit containing include/cuda.h or targets/*/include/cuda.h"
    );
}

fn add_cuda_include_candidates(candidates: &mut Vec<PathBuf>, root: PathBuf) {
    candidates.push(root.join("include"));

    let targets = root.join("targets");
    if let Ok(entries) = std::fs::read_dir(targets) {
        for entry in entries.flatten() {
            candidates.push(entry.path().join("include"));
        }
    }
}

fn validate_deepgemm_root(path: PathBuf, source: &str) -> PathBuf {
    let root = path.canonicalize().unwrap_or_else(|error| {
        panic!(
            "{source} does not point to a readable DeepGEMM source tree at {}: {error}",
            path.display()
        )
    });

    require_path(
        &root,
        "deep_gemm/include/deep_gemm/common/types.cuh",
        source,
    );
    require_path(
        &root,
        "deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh",
        source,
    );
    require_path(
        &root,
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh",
        source,
    );
    require_path(
        &root,
        "deep_gemm/include/deep_gemm/impls/smxx_layout.cuh",
        source,
    );
    require_path(
        &root,
        "deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh",
        source,
    );
    require_path(
        &root,
        "deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh",
        source,
    );
    require_path(
        &root,
        "deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh",
        source,
    );
    require_path(
        &root,
        "deep_gemm/include/deep_gemm/impls/smxx_clean_logits.cuh",
        source,
    );
    require_path(
        &root,
        "deep_gemm/include/deep_gemm/scheduler/sm90_paged_mqa_logits.cuh",
        source,
    );
    require_path(
        &root,
        "deep_gemm/include/deep_gemm/scheduler/sm100_paged_mqa_logits.cuh",
        source,
    );
    require_path(
        &root,
        "third-party/cutlass/include/cutlass/cutlass.h",
        source,
    );
    require_path(
        &root,
        "third-party/cutlass/tools/util/include/cutlass/util/host_tensor.h",
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
            "{source} DeepGEMM source tree is missing {relative}. Initialize nested submodules with `git submodule update --init --recursive crates/deepgemm-sys/vendor/DeepGEMM`, or include the required runtime headers when packaging."
        );
    }

    panic!(
        "{source} does not look like a DeepGEMM runtime source tree: missing {}",
        path.display()
    );
}
