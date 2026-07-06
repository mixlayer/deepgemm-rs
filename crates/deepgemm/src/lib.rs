#![deny(unsafe_op_in_unsafe_fn)]
//! Candle-independent safe wrappers for DeepGEMM raw FFI.

/// Supported DeepGEMM GPU architecture families.
pub mod arch;
/// Tensor dtype metadata shared by wrapper APIs.
pub mod dtype;
/// Error types returned by the safe wrapper layer.
pub mod error;
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
pub use mqa::{
    MqaLogitsLaunch, MqaLogitsSpec, PagedMqaLogitsLaunch, PagedMqaLogitsMetadataLaunch,
    PagedMqaLogitsMetadataSpec, PagedMqaLogitsSpec, fp8_fp4_mqa_logits, fp8_fp4_paged_mqa_logits,
    logits_layout_from_sys, mqa_logits_layout, paged_mqa_logits_layout, paged_mqa_logits_metadata,
    paged_mqa_logits_metadata_layout,
};
pub use runtime::init;
pub use source::{SourceLayout, source_layout, source_root};
pub use tensor::{TensorArg, TensorLayout2D, TensorOut, TensorSpec};
