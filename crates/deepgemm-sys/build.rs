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
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set by Cargo"));
    let (patch_include, patch_id) = prepare_sm120_patch(&deepgemm_root, &out_dir);

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
    println!("cargo:rerun-if-changed=csrc/deepgemm_raw_grouped_gemm.h");
    println!("cargo:rerun-if-changed=csrc/deepgemm_raw_grouped_gemm.cc");
    println!("cargo:rerun-if-changed=csrc/deepgemm_raw_mega_moe.h");
    println!("cargo:rerun-if-changed=csrc/deepgemm_raw_mega_moe.cc");
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

    let patch_include = patch_include
        .to_str()
        .expect("DeepGEMM patch include path must contain valid Unicode");
    let patch_include_define = format!(
        "\"{}\"",
        patch_include.replace('\\', "\\\\").replace('\"', "\\\"")
    );
    let patch_id = format!("{patch_id}ULL");
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .define(
            "DEEPGEMM_PATCH_INCLUDE",
            Some(patch_include_define.as_str()),
        )
        .define("DEEPGEMM_SM120_PATCH_ID", Some(patch_id.as_str()))
        .include(manifest_dir.join("csrc"))
        .include(cuda_include)
        .file(manifest_dir.join("csrc").join("deepgemm_c_api.cc"))
        .file(manifest_dir.join("csrc").join("deepgemm_raw_runtime.cc"))
        .file(manifest_dir.join("csrc").join("deepgemm_raw_mqa.cc"))
        .file(manifest_dir.join("csrc").join("deepgemm_raw_gemm.cc"))
        .file(
            manifest_dir
                .join("csrc")
                .join("deepgemm_raw_grouped_gemm.cc"),
        )
        .file(manifest_dir.join("csrc").join("deepgemm_raw_mega_moe.cc"));
    build.compile("deepgemm_c_api");
}

fn prepare_sm120_patch(deepgemm_root: &Path, out_dir: &Path) -> (PathBuf, u64) {
    let relative = Path::new("deep_gemm/impls/sm120_bf16_gemm.cuh");
    let source_path = deepgemm_root.join("deep_gemm/include").join(relative);
    let source = std::fs::read_to_string(&source_path).unwrap_or_else(|error| {
        panic!(
            "failed to read DeepGEMM SM120 grouped GEMM header at {}: {error}",
            source_path.display()
        )
    });
    let loop_start = "while (scheduler.get_next_block(m_block_idx, n_block_idx)) {\n";
    let valid_block_guard = concat!(
        "while (scheduler.get_next_block(m_block_idx, n_block_idx)) {\n",
        "#if DEEPGEMM_SM120_SKIP_INVALID_PADDING\n",
        "            // M-grouped capacity is a worst-case allocation; skip -1 padding blocks.\n",
        "            if constexpr (kGemmType == GemmType::MGroupedContiguous) {\n",
        "                if (not scheduler.is_computation_valid(m_block_idx, 0)) continue;\n",
        "            }\n",
        "#endif\n",
    );
    let occurrences = source.matches(loop_start).count();
    assert_eq!(
        occurrences,
        2,
        "expected producer and consumer scheduler loops in {}",
        source_path.display()
    );
    let scheduler_end = "        shape_m, shape_n, shape_k, grouped_layout);\n";
    let trim_invalid_tail = concat!(
        "        shape_m, shape_n, shape_k, grouped_layout);\n",
        "#if DEEPGEMM_SM120_TRIM_INVALID_TAIL\n",
        "    // M-grouped rows are block-aligned and valid block starts form a prefix.\n",
        "    __shared__ uint32_t effective_m_blocks;\n",
        "    if (threadIdx.x == 0) {\n",
        "        uint32_t low = 0, high = scheduler.num_m_blocks;\n",
        "        while (low < high) {\n",
        "            const uint32_t mid = low + (high - low) / 2;\n",
        "            if (grouped_layout[mid * BLOCK_M] >= 0) low = mid + 1;\n",
        "            else high = mid;\n",
        "        }\n",
        "        effective_m_blocks = low;\n",
        "    }\n",
        "    __syncthreads();\n",
        "    if constexpr (kGemmType == GemmType::MGroupedContiguous) {\n",
        "        scheduler.num_m_blocks = effective_m_blocks;\n",
        "        scheduler.num_mn_blocks = effective_m_blocks * scheduler.num_n_blocks;\n",
        "        scheduler.num_blocks = scheduler.num_mn_blocks;\n",
        "    }\n",
        "#endif\n",
    );
    assert_eq!(
        source.matches(scheduler_end).count(),
        1,
        "expected one SM120 scheduler construction in {}",
        source_path.display()
    );
    let patched = source
        .replace(loop_start, valid_block_guard)
        .replace(scheduler_end, trim_invalid_tail);
    let patch_root = out_dir.join("deepgemm-patches");
    let output_path = patch_root.join(relative);
    std::fs::create_dir_all(output_path.parent().expect("patched header has a parent"))
        .expect("failed to create DeepGEMM patch include directory");
    std::fs::write(&output_path, &patched).expect("failed to write patched DeepGEMM SM120 header");

    let patch_id = patched.bytes().fold(1469598103934665603u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(1099511628211)
    });
    (patch_root, patch_id)
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
        "deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh",
        source,
    );
    require_path(
        &root,
        "deep_gemm/include/deep_gemm/impls/sm120_bf16_gemm.cuh",
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
