use std::path::{Path, PathBuf};

/// Important paths in the selected DeepGEMM checkout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLayout {
    /// DeepGEMM repository root.
    pub root: PathBuf,
    /// DeepGEMM public CUDA include directory.
    pub deep_gemm_include: PathBuf,
    /// Vendored CUTLASS include directory.
    pub cutlass_include: PathBuf,
    /// Vendored fmt include directory.
    pub fmt_include: PathBuf,
    /// DeepGEMM C++ API implementation directory.
    pub csrc: PathBuf,
}

/// Returns the DeepGEMM source root selected by `deepgemm-sys`.
pub fn source_root() -> &'static Path {
    Path::new(deepgemm_sys::DEEPGEMM_SOURCE_ROOT)
}

/// Returns important include and source paths in the selected DeepGEMM checkout.
pub fn source_layout() -> SourceLayout {
    let root = source_root();
    SourceLayout {
        root: root.to_path_buf(),
        deep_gemm_include: root.join("deep_gemm").join("include"),
        cutlass_include: root.join("third-party").join("cutlass").join("include"),
        fmt_include: root.join("third-party").join("fmt").join("include"),
        csrc: root.join("csrc"),
    }
}
