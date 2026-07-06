use std::ffi::CString;

use crate::{Arch, Error, Result};

/// Current CUDA device information reported by the native DeepGEMM runtime.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// CUDA device ordinal.
    pub device: i32,
    /// Compute capability major version.
    pub compute_capability_major: i32,
    /// Compute capability minor version.
    pub compute_capability_minor: i32,
    /// Physical SM count for the current device.
    pub num_sms: i32,
}

impl DeviceInfo {
    /// Returns the DeepGEMM architecture family for this device.
    pub fn arch(self) -> Result<Arch> {
        Arch::from_compute_capability(self.compute_capability_major, self.compute_capability_minor)
            .ok_or_else(|| {
                Error::UnsupportedArch(format!(
                    "unsupported compute capability {}.{}",
                    self.compute_capability_major, self.compute_capability_minor
                ))
            })
    }
}

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

/// Returns information for the current CUDA device.
pub fn device_info() -> Result<DeviceInfo> {
    let mut raw = deepgemm_sys::deepgemm_device_info_t {
        device: 0,
        compute_capability_major: 0,
        compute_capability_minor: 0,
        num_sms: 0,
    };
    // SAFETY: `raw` is a valid output pointer for the duration of the call.
    let status = unsafe { deepgemm_sys::deepgemm_get_device_info(&mut raw) };
    Error::check_raw_status(status)?;
    Ok(DeviceInfo {
        device: raw.device,
        compute_capability_major: raw.compute_capability_major,
        compute_capability_minor: raw.compute_capability_minor,
        num_sms: raw.num_sms,
    })
}

/// Returns the active SM count override, or the physical SM count if no override is set.
pub fn num_sms() -> Result<i32> {
    let mut value = 0;
    // SAFETY: `value` is a valid output pointer for the duration of the call.
    let status = unsafe { deepgemm_sys::deepgemm_get_num_sms(&mut value) };
    Error::check_raw_status(status)?;
    Ok(value)
}

/// Overrides the SM count used by future launches. Pass `0` to clear the override.
pub fn set_num_sms(num_sms: i32) -> Result<()> {
    // SAFETY: forwards a plain integer to the native runtime.
    let status = unsafe { deepgemm_sys::deepgemm_set_num_sms(num_sms) };
    Error::check_raw_status(status)
}

/// Enables or disables CUDA programmatic dependent launch attributes for future launches.
pub fn set_pdl(enabled: bool) -> Result<()> {
    // SAFETY: forwards a plain boolean to the native runtime.
    let status = unsafe { deepgemm_sys::deepgemm_set_pdl(enabled) };
    Error::check_raw_status(status)
}
