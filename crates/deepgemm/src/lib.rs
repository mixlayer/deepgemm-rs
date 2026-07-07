#![deny(unsafe_op_in_unsafe_fn)]
//! Candle-independent safe wrappers for DeepGEMM raw FFI.

/// Supported DeepGEMM GPU architecture families.
pub mod arch;
/// Tensor dtype metadata shared by wrapper APIs.
pub mod dtype;
/// Error types returned by the safe wrapper layer.
pub mod error;
/// Dense FP8 GEMM shape validation, scale transforms, and launches.
pub mod gemm;
/// MQA logits shape validation and layout helpers.
pub mod mqa;
/// Runtime initialization helpers.
pub mod runtime;
/// Build-time DeepGEMM source checkout metadata.
pub mod source;
/// Tensor shape, stride, dtype, and pointer descriptors.
pub mod tensor;

pub use arch::Arch;
pub use dtype::DType;
pub use error::{Error, Result};
pub use gemm::{
    Fp8GemmNtLaunch, Fp8GemmNtSpec, Fp8GemmScaleTransformLaunch, Fp8GemmScaleTransformSpec,
    fp8_gemm_nt, fp8_gemm_nt_output_layout, fp8_gemm_scale_layout, fp8_gemm_transform_scale,
};
pub use mqa::{
    MqaLogitsLaunch, MqaLogitsSpec, PagedMqaLogitsLaunch, PagedMqaLogitsMetadataLaunch,
    PagedMqaLogitsMetadataSpec, PagedMqaLogitsSpec, fp8_fp4_mqa_logits, fp8_fp4_paged_mqa_logits,
    logits_layout_from_sys, mqa_logits_layout, paged_mqa_logits_layout, paged_mqa_logits_metadata,
    paged_mqa_logits_metadata_layout,
};
pub use runtime::{DeviceInfo, device_info, init, num_sms, set_num_sms, set_pdl};
pub use source::{SourceLayout, source_layout, source_root};
pub use tensor::{TensorArg, TensorLayout2D, TensorOut, TensorSpec};
