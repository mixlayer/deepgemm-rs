#![deny(unsafe_op_in_unsafe_fn)]
//! Candle integration layer for DeepGEMM.

/// Candle integration error types.
pub mod error;
/// Candle dense FP8 GEMM integration.
pub mod gemm;
/// Candle MQA logits integration.
pub mod mqa;
/// Candle tensor validation and CUDA pointer helpers.
pub mod tensor;

pub use deepgemm;

pub use error::{Error, Result};
pub use gemm::fp8_gemm_nt;
pub use mqa::{
    MqaLogitsConfig, PagedMqaLogitsConfig, PagedMqaLogitsPlan, fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits, paged_mqa_logits_plan,
};
