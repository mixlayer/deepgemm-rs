#![deny(unsafe_op_in_unsafe_fn)]
//! Candle integration layer for DeepGEMM.

/// Candle integration error types.
pub mod error;
/// Candle dense FP8 GEMM integration.
pub mod gemm;
/// Candle BF16 Mega MoE integration.
pub mod mega_moe;
/// Candle MQA logits integration.
pub mod mqa;
/// Candle tensor validation and CUDA pointer helpers.
pub mod tensor;

pub use deepgemm;

pub use error::{Error, Result};
pub use gemm::{
    bf16_m_grouped_gemm_nt_contiguous, bf16_m_grouped_gemm_nt_contiguous_into, fp8_gemm_nt,
    fp8_gemm_nt_prepared_scales, prepare_fp8_gemm_scale,
};
pub use mega_moe::{Bf16MegaMoeWorkspace, bf16_mega_moe, interleave_bf16_mega_moe_l1_weights};
pub use mqa::{
    MqaLogitsConfig, PagedMqaLogitsConfig, PagedMqaLogitsPlan, fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits, paged_mqa_logits_plan,
};
