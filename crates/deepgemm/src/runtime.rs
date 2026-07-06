use std::ffi::CString;

use crate::{Error, Result};

/// Initializes the DeepGEMM native runtime.
///
/// This currently validates and forwards the configured paths to the native shim. The next native
/// launch slice will use the same entry point to initialize DeepGEMM's JIT compiler/cache state.
pub fn init(deepgemm_root: &str, cuda_home: &str) -> Result<()> {
    let deepgemm_root = CString::new(deepgemm_root)
        .map_err(|_| Error::InvalidArgument("deepgemm_root contains an interior NUL".into()))?;
    let cuda_home = CString::new(cuda_home)
        .map_err(|_| Error::InvalidArgument("cuda_home contains an interior NUL".into()))?;

    // SAFETY: both C strings are NUL-terminated and live for the duration of the call.
    let status = unsafe { deepgemm_sys::deepgemm_init(deepgemm_root.as_ptr(), cuda_home.as_ptr()) };
    Error::check_raw_status(status)
}
