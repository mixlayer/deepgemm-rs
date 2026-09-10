#![allow(non_camel_case_types)]
//! Raw DeepGEMM C ABI bindings.

use core::ffi::{c_char, c_int, c_void};

/// Opaque CUDA stream handle used by the DeepGEMM C ABI.
pub type deepgemm_cuda_stream_t = *mut c_void;

/// Status code returned by DeepGEMM C ABI functions.
pub type deepgemm_status_t = c_int;

/// The operation completed successfully.
pub const DEEPGEMM_STATUS_SUCCESS: deepgemm_status_t = 0;
/// A caller-provided argument was invalid.
pub const DEEPGEMM_STATUS_INVALID_ARGUMENT: deepgemm_status_t = 1;
/// The selected GPU architecture is not supported.
pub const DEEPGEMM_STATUS_UNSUPPORTED_ARCH: deepgemm_status_t = 2;
/// CUDA returned an error.
pub const DEEPGEMM_STATUS_CUDA_ERROR: deepgemm_status_t = 3;
/// DeepGEMM hit an unexpected internal error.
pub const DEEPGEMM_STATUS_INTERNAL_ERROR: deepgemm_status_t = 4;

/// Tensor element dtypes understood by the DeepGEMM C ABI.
pub type deepgemm_dtype_t = c_int;

pub const DEEPGEMM_DTYPE_INVALID: deepgemm_dtype_t = 0;
pub const DEEPGEMM_DTYPE_FP8_E4M3: deepgemm_dtype_t = 1;
pub const DEEPGEMM_DTYPE_PACKED_FP4_E2M1: deepgemm_dtype_t = 2;
pub const DEEPGEMM_DTYPE_PACKED_UE8M0: deepgemm_dtype_t = 3;
pub const DEEPGEMM_DTYPE_F32: deepgemm_dtype_t = 4;
pub const DEEPGEMM_DTYPE_BF16: deepgemm_dtype_t = 5;
pub const DEEPGEMM_DTYPE_I32: deepgemm_dtype_t = 6;
pub const DEEPGEMM_DTYPE_U8: deepgemm_dtype_t = 7;
pub const DEEPGEMM_DTYPE_I64: deepgemm_dtype_t = 8;

/// Matrix-multiply implementation used by Mega MoE.
pub type deepgemm_mega_moe_mma_kind_t = c_int;

pub const DEEPGEMM_MEGA_MOE_MMA_INVALID: deepgemm_mega_moe_mma_kind_t = 0;
pub const DEEPGEMM_MEGA_MOE_MMA_FP8_FP4: deepgemm_mega_moe_mma_kind_t = 1;
pub const DEEPGEMM_MEGA_MOE_MMA_BF16: deepgemm_mega_moe_mma_kind_t = 2;

/// DeepGEMM source root selected by the build script.
pub const DEEPGEMM_SOURCE_ROOT: &str = env!("DEEPGEMM_SOURCE_ROOT");

/// Immutable tensor argument descriptor.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct deepgemm_tensor_t {
    pub data: *const c_void,
    pub dtype: deepgemm_dtype_t,
    pub rank: u32,
    pub shape: [i64; 4],
    pub stride: [i64; 4],
}

/// Mutable tensor argument descriptor.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct deepgemm_tensor_mut_t {
    pub data: *mut c_void,
    pub dtype: deepgemm_dtype_t,
    pub rank: u32,
    pub shape: [i64; 4],
    pub stride: [i64; 4],
}

/// 2D tensor layout returned by C ABI layout helpers.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct deepgemm_tensor_layout_2d_t {
    pub dtype: deepgemm_dtype_t,
    pub logical_shape: [i64; 2],
    pub allocation_shape: [i64; 2],
    pub stride: [i64; 2],
    pub element_count: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct deepgemm_device_info_t {
    pub device: i32,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub num_sms: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct deepgemm_mqa_logits_layout_params_t {
    pub seq_len: i64,
    pub seq_len_kv: i64,
    pub num_heads: i64,
    pub max_seqlen_k: i64,
    pub logits_dtype: deepgemm_dtype_t,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct deepgemm_paged_mqa_logits_layout_params_t {
    pub batch_size: i64,
    pub next_n: i64,
    pub max_context_len: i64,
    pub logits_dtype: deepgemm_dtype_t,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct deepgemm_paged_mqa_logits_metadata_layout_params_t {
    pub num_sms: i64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct deepgemm_fp8_gemm_nt_output_layout_params_t {
    pub m: i64,
    pub n: i64,
    pub output_dtype: deepgemm_dtype_t,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct deepgemm_fp8_gemm_scale_layout_params_t {
    pub mn: i64,
    pub k: i64,
    pub gran_k: i64,
    pub scale_dtype: deepgemm_dtype_t,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct deepgemm_mega_moe_ring_limits_params_t {
    pub num_ranks: i64,
    pub num_experts: i64,
    pub num_max_tokens_per_rank: i64,
    pub num_topk: i64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct deepgemm_mega_moe_ring_limits_t {
    pub min_tokens: i64,
    pub max_tokens: i64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct deepgemm_mega_moe_buffer_params_t {
    pub num_ranks: i64,
    pub num_experts: i64,
    pub num_max_tokens_per_rank: i64,
    pub num_topk: i64,
    pub hidden: i64,
    pub intermediate_hidden: i64,
    pub num_ring_tokens: i64,
    pub mma_kind: deepgemm_mega_moe_mma_kind_t,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct deepgemm_mega_moe_buffer_view_t {
    pub offset_bytes: u64,
    pub dtype: deepgemm_dtype_t,
    pub rank: u32,
    pub shape: [i64; 2],
    pub stride: [i64; 2],
    pub element_count: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct deepgemm_mega_moe_buffer_layout_t {
    pub total_bytes: u64,
    pub workspace_bytes: u64,
    pub x: deepgemm_mega_moe_buffer_view_t,
    pub x_scale: deepgemm_mega_moe_buffer_view_t,
    pub topk_indices: deepgemm_mega_moe_buffer_view_t,
    pub topk_weights: deepgemm_mega_moe_buffer_view_t,
    pub l1_acts: deepgemm_mega_moe_buffer_view_t,
    pub l1_acts_scale: deepgemm_mega_moe_buffer_view_t,
    pub l2_acts: deepgemm_mega_moe_buffer_view_t,
    pub l2_acts_scale: deepgemm_mega_moe_buffer_view_t,
    pub combine: deepgemm_mega_moe_buffer_view_t,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct deepgemm_mqa_logits_params_t {
    pub q: deepgemm_tensor_t,
    pub has_q_scale: bool,
    pub q_scale: deepgemm_tensor_t,
    pub kv: deepgemm_tensor_t,
    pub kv_scale: deepgemm_tensor_t,
    pub weights: deepgemm_tensor_t,
    pub cu_seq_len_k_start: deepgemm_tensor_t,
    pub cu_seq_len_k_end: deepgemm_tensor_t,
    pub logits: deepgemm_tensor_mut_t,
    pub clean_logits: bool,
    pub max_seqlen_k: i64,
    pub stream: deepgemm_cuda_stream_t,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct deepgemm_paged_mqa_logits_metadata_params_t {
    pub context_lens: deepgemm_tensor_t,
    pub has_indices: bool,
    pub indices: deepgemm_tensor_t,
    pub schedule_meta: deepgemm_tensor_mut_t,
    pub block_kv: i64,
    pub num_sms: i64,
    pub stream: deepgemm_cuda_stream_t,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct deepgemm_paged_mqa_logits_params_t {
    pub q: deepgemm_tensor_t,
    pub has_q_scale: bool,
    pub q_scale: deepgemm_tensor_t,
    pub fused_kv_cache: deepgemm_tensor_t,
    pub weights: deepgemm_tensor_t,
    pub context_lens: deepgemm_tensor_t,
    pub block_table: deepgemm_tensor_t,
    pub schedule_meta: deepgemm_tensor_t,
    pub has_indices: bool,
    pub indices: deepgemm_tensor_t,
    pub logits: deepgemm_tensor_mut_t,
    pub max_context_len: i64,
    pub num_sms: i64,
    pub clean_logits: bool,
    pub stream: deepgemm_cuda_stream_t,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct deepgemm_fp8_gemm_scale_transform_params_t {
    pub scale: deepgemm_tensor_t,
    pub transformed: deepgemm_tensor_mut_t,
    pub mn: i64,
    pub k: i64,
    pub gran_mn: i64,
    pub gran_k: i64,
    pub stream: deepgemm_cuda_stream_t,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct deepgemm_fp8_gemm_nt_params_t {
    pub a: deepgemm_tensor_t,
    pub a_scale: deepgemm_tensor_t,
    pub b: deepgemm_tensor_t,
    pub b_scale: deepgemm_tensor_t,
    pub d: deepgemm_tensor_mut_t,
    pub stream: deepgemm_cuda_stream_t,
}

/// SM120-family BF16 M-grouped contiguous GEMM launch parameters.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct deepgemm_bf16_m_grouped_gemm_nt_contiguous_params_t {
    pub a: deepgemm_tensor_t,
    pub b: deepgemm_tensor_t,
    pub d: deepgemm_tensor_mut_t,
    pub grouped_layout: deepgemm_tensor_t,
    pub row_alignment: i64,
    pub stream: deepgemm_cuda_stream_t,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct deepgemm_bf16_mega_moe_params_t {
    pub x: deepgemm_tensor_t,
    pub topk_indices: deepgemm_tensor_t,
    pub topk_weights: deepgemm_tensor_t,
    pub l1_weights: deepgemm_tensor_t,
    pub l2_weights: deepgemm_tensor_t,
    pub y: deepgemm_tensor_mut_t,
    pub sym_buffer: deepgemm_tensor_mut_t,
    pub sym_buffer_ptrs: *const u64,
    pub num_ranks: i32,
    pub rank_idx: i32,
    pub num_max_tokens_per_rank: i32,
    pub num_ring_tokens: i32,
    pub num_experts: i32,
    pub num_topk: i32,
    pub x_offset_bytes: u64,
    pub topk_indices_offset_bytes: u64,
    pub topk_weights_offset_bytes: u64,
    pub l1_acts_offset_bytes: u64,
    pub l2_acts_offset_bytes: u64,
    pub activation_clamp: f32,
    pub fast_math: bool,
    pub stream: deepgemm_cuda_stream_t,
}

unsafe extern "C" {
    /// Returns the most recent error message recorded by the C ABI layer.
    pub fn deepgemm_last_error() -> *const c_char;

    pub fn deepgemm_init(
        deepgemm_root: *const c_char,
        cuda_home: *const c_char,
    ) -> deepgemm_status_t;

    pub fn deepgemm_get_device_info(out: *mut deepgemm_device_info_t) -> deepgemm_status_t;

    pub fn deepgemm_get_num_sms(out: *mut i32) -> deepgemm_status_t;

    pub fn deepgemm_set_num_sms(num_sms: i32) -> deepgemm_status_t;

    pub fn deepgemm_set_pdl(enabled: bool) -> deepgemm_status_t;

    pub fn deepgemm_mqa_logits_layout(
        params: *const deepgemm_mqa_logits_layout_params_t,
        out: *mut deepgemm_tensor_layout_2d_t,
    ) -> deepgemm_status_t;

    pub fn deepgemm_paged_mqa_logits_layout(
        params: *const deepgemm_paged_mqa_logits_layout_params_t,
        out: *mut deepgemm_tensor_layout_2d_t,
    ) -> deepgemm_status_t;

    pub fn deepgemm_paged_mqa_logits_metadata_layout(
        params: *const deepgemm_paged_mqa_logits_metadata_layout_params_t,
        out: *mut deepgemm_tensor_layout_2d_t,
    ) -> deepgemm_status_t;

    pub fn deepgemm_fp8_gemm_nt_output_layout(
        params: *const deepgemm_fp8_gemm_nt_output_layout_params_t,
        out: *mut deepgemm_tensor_layout_2d_t,
    ) -> deepgemm_status_t;

    pub fn deepgemm_fp8_gemm_scale_layout(
        params: *const deepgemm_fp8_gemm_scale_layout_params_t,
        out: *mut deepgemm_tensor_layout_2d_t,
    ) -> deepgemm_status_t;

    pub fn deepgemm_mega_moe_token_alignment(out: *mut i64) -> deepgemm_status_t;

    pub fn deepgemm_mega_moe_ring_limits(
        params: *const deepgemm_mega_moe_ring_limits_params_t,
        out: *mut deepgemm_mega_moe_ring_limits_t,
    ) -> deepgemm_status_t;

    pub fn deepgemm_mega_moe_buffer_layout(
        params: *const deepgemm_mega_moe_buffer_params_t,
        out: *mut deepgemm_mega_moe_buffer_layout_t,
    ) -> deepgemm_status_t;

    pub fn deepgemm_fp8_fp4_mqa_logits(
        params: *const deepgemm_mqa_logits_params_t,
    ) -> deepgemm_status_t;

    pub fn deepgemm_paged_mqa_logits_metadata(
        params: *const deepgemm_paged_mqa_logits_metadata_params_t,
    ) -> deepgemm_status_t;

    pub fn deepgemm_fp8_fp4_paged_mqa_logits(
        params: *const deepgemm_paged_mqa_logits_params_t,
    ) -> deepgemm_status_t;

    pub fn deepgemm_fp8_gemm_transform_scale(
        params: *const deepgemm_fp8_gemm_scale_transform_params_t,
    ) -> deepgemm_status_t;

    pub fn deepgemm_fp8_gemm_nt(params: *const deepgemm_fp8_gemm_nt_params_t) -> deepgemm_status_t;

    pub fn deepgemm_bf16_m_grouped_gemm_nt_contiguous(
        params: *const deepgemm_bf16_m_grouped_gemm_nt_contiguous_params_t,
    ) -> deepgemm_status_t;

    pub fn deepgemm_bf16_mega_moe(
        params: *const deepgemm_bf16_mega_moe_params_t,
    ) -> deepgemm_status_t;
}
