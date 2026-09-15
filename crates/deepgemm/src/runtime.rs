use std::{
    ffi::{CStr, CString},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use crate::{Arch, Error, Result};

/// Process-global callback used to observe DeepGEMM kernel materialization events.
pub type KernelMaterializationHook = dyn Fn(KernelMaterializationEvent) + Send + Sync + 'static;

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

/// Source that produced a launchable DeepGEMM kernel for the current process.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum KernelMaterializationSource {
    /// The current process already had a live kernel runtime cached in memory.
    ProcessCache,
    /// The current process loaded a precompiled cubin from the on-disk JIT cache.
    DiskCubin,
    /// The current process compiled a new cubin through `nvcc`.
    JitCompile,
}

impl KernelMaterializationSource {
    fn from_raw(raw: deepgemm_sys::deepgemm_kernel_materialization_source_t) -> Option<Self> {
        match raw {
            deepgemm_sys::DEEPGEMM_KERNEL_MATERIALIZATION_SOURCE_PROCESS_CACHE => {
                Some(Self::ProcessCache)
            }
            deepgemm_sys::DEEPGEMM_KERNEL_MATERIALIZATION_SOURCE_DISK_CUBIN => {
                Some(Self::DiskCubin)
            }
            deepgemm_sys::DEEPGEMM_KERNEL_MATERIALIZATION_SOURCE_JIT_COMPILE => {
                Some(Self::JitCompile)
            }
            _ => None,
        }
    }
}

/// Phase of a DeepGEMM kernel materialization event.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum KernelMaterializationPhase {
    /// The runtime has started loading or compiling a kernel.
    Start,
    /// The runtime finished loading or compiling a kernel.
    Finish,
}

impl KernelMaterializationPhase {
    fn from_raw(raw: deepgemm_sys::deepgemm_kernel_materialization_phase_t) -> Option<Self> {
        match raw {
            deepgemm_sys::DEEPGEMM_KERNEL_MATERIALIZATION_PHASE_START => Some(Self::Start),
            deepgemm_sys::DEEPGEMM_KERNEL_MATERIALIZATION_PHASE_FINISH => Some(Self::Finish),
            _ => None,
        }
    }
}

/// Kernel materialization event emitted by the DeepGEMM runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelMaterializationEvent {
    /// Low-cardinality kernel family name from the native runtime.
    pub kernel: String,
    /// Source that satisfied the kernel materialization request.
    pub source: KernelMaterializationSource,
    /// Whether the event marks the start or finish of the materialization.
    pub phase: KernelMaterializationPhase,
    /// Whether the materialization path succeeded.
    pub success: bool,
    /// Wall-clock duration for finished materializations.
    pub duration: Option<Duration>,
}

fn kernel_materialization_hook() -> &'static Mutex<Option<Arc<KernelMaterializationHook>>> {
    static HOOK: OnceLock<Mutex<Option<Arc<KernelMaterializationHook>>>> = OnceLock::new();
    HOOK.get_or_init(|| Mutex::new(None))
}

fn kernel_materialization_event_from_raw(
    kernel_name: &CStr,
    source: deepgemm_sys::deepgemm_kernel_materialization_source_t,
    phase: deepgemm_sys::deepgemm_kernel_materialization_phase_t,
    success: bool,
    has_duration: bool,
    duration_ns: u64,
) -> Option<KernelMaterializationEvent> {
    Some(KernelMaterializationEvent {
        kernel: kernel_name.to_str().ok()?.to_owned(),
        source: KernelMaterializationSource::from_raw(source)?,
        phase: KernelMaterializationPhase::from_raw(phase)?,
        success,
        duration: has_duration.then(|| Duration::from_nanos(duration_ns)),
    })
}

unsafe extern "C" fn kernel_materialization_callback(
    kernel_name: *const std::ffi::c_char,
    source: deepgemm_sys::deepgemm_kernel_materialization_source_t,
    phase: deepgemm_sys::deepgemm_kernel_materialization_phase_t,
    success: bool,
    has_duration: bool,
    duration_ns: u64,
) {
    if kernel_name.is_null() {
        return;
    }
    let hook = match kernel_materialization_hook().lock() {
        Ok(guard) => guard.clone(),
        Err(_) => None,
    };
    let Some(hook) = hook else {
        return;
    };
    let Some(event) = ({
        // SAFETY: the native runtime only invokes the hook with a valid NUL-terminated kernel name.
        let kernel_name = unsafe { CStr::from_ptr(kernel_name) };
        kernel_materialization_event_from_raw(
            kernel_name,
            source,
            phase,
            success,
            has_duration,
            duration_ns,
        )
    }) else {
        return;
    };
    let _ = catch_unwind(AssertUnwindSafe(|| hook(event)));
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

/// Registers or clears a process-global hook for kernel materialization events.
pub fn set_kernel_materialization_hook(hook: Option<Arc<KernelMaterializationHook>>) -> Result<()> {
    let callback = match kernel_materialization_hook().lock() {
        Ok(mut guard) => {
            *guard = hook;
            guard.as_ref().map(|_| kernel_materialization_callback as _)
        }
        Err(_) => None,
    };
    // SAFETY: registers either a static callback or null with the native runtime.
    let status = unsafe { deepgemm_sys::deepgemm_set_kernel_materialization_hook(callback) };
    Error::check_raw_status(status)
}

#[cfg(test)]
mod tests {
    use super::{
        KernelMaterializationPhase, KernelMaterializationSource,
        kernel_materialization_event_from_raw,
    };
    use std::{ffi::CString, time::Duration};

    #[test]
    fn raw_kernel_materialization_event_maps_to_rust_event() {
        let event = kernel_materialization_event_from_raw(
            &CString::new("sm100_paged_mqa_logits").unwrap(),
            deepgemm_sys::DEEPGEMM_KERNEL_MATERIALIZATION_SOURCE_JIT_COMPILE,
            deepgemm_sys::DEEPGEMM_KERNEL_MATERIALIZATION_PHASE_FINISH,
            true,
            true,
            123,
        )
        .unwrap();

        assert_eq!(event.kernel, "sm100_paged_mqa_logits");
        assert_eq!(event.source, KernelMaterializationSource::JitCompile);
        assert_eq!(event.phase, KernelMaterializationPhase::Finish);
        assert!(event.success);
        assert_eq!(event.duration, Some(Duration::from_nanos(123)));
    }
}
