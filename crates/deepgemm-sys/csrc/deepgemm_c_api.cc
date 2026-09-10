#include "deepgemm_c_api.h"

#include "deepgemm_raw_gemm.h"
#include "deepgemm_raw_grouped_gemm.h"
#include "deepgemm_raw_mqa.h"
#include "deepgemm_raw_mega_moe.h"
#include "deepgemm_raw_runtime.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <string>
#include <utility>

namespace {

thread_local std::string g_last_error;

deepgemm_status_t clear_error() {
  g_last_error.clear();
  return DEEPGEMM_STATUS_SUCCESS;
}

deepgemm_status_t set_error(deepgemm_status_t status, const char* message) {
  g_last_error = message;
  return status;
}

template <typename Fn>
deepgemm_status_t ffi_call(Fn&& fn) {
  try {
    std::forward<Fn>(fn)();
    return clear_error();
  } catch (const deepgemm_rs::StatusError& error) {
    return set_error(error.status(), error.what());
  } catch (const std::exception& error) {
    return set_error(DEEPGEMM_STATUS_INTERNAL_ERROR, error.what());
  } catch (...) {
    return set_error(DEEPGEMM_STATUS_INTERNAL_ERROR, "unknown native DeepGEMM error");
  }
}

int64_t dtype_size(deepgemm_dtype_t dtype) {
  switch (dtype) {
    case DEEPGEMM_DTYPE_FP8_E4M3:
    case DEEPGEMM_DTYPE_PACKED_FP4_E2M1:
    case DEEPGEMM_DTYPE_U8:
      return 1;
    case DEEPGEMM_DTYPE_BF16:
      return 2;
    case DEEPGEMM_DTYPE_PACKED_UE8M0:
    case DEEPGEMM_DTYPE_F32:
    case DEEPGEMM_DTYPE_I32:
      return 4;
    case DEEPGEMM_DTYPE_I64:
      return 8;
    default:
      return 0;
  }
}

bool checked_mul_u64(uint64_t left, uint64_t right, uint64_t* out) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    return false;
  }
  *out = left * right;
  return true;
}

bool checked_add_u64(uint64_t left, uint64_t right, uint64_t* out) {
  if (right > std::numeric_limits<uint64_t>::max() - left) {
    return false;
  }
  *out = left + right;
  return true;
}

bool align_u64(uint64_t value, uint64_t alignment, uint64_t* out) {
  if (alignment == 0) {
    return false;
  }
  const uint64_t remainder = value % alignment;
  if (remainder == 0) {
    *out = value;
    return true;
  }
  return checked_add_u64(value, alignment - remainder, out);
}

bool align_i64(int64_t value, int64_t alignment, int64_t* out) {
  if (value < 0 || alignment <= 0) {
    return false;
  }
  const int64_t remainder = value % alignment;
  if (remainder == 0) {
    *out = value;
    return true;
  }
  const int64_t delta = alignment - remainder;
  if (value > std::numeric_limits<int64_t>::max() - delta) {
    return false;
  }
  *out = value + delta;
  return true;
}

bool ceil_div_i64(int64_t value, int64_t divisor, int64_t* out) {
  if (value < 0 || divisor <= 0) {
    return false;
  }
  if (value > std::numeric_limits<int64_t>::max() - divisor + 1) {
    return false;
  }
  *out = (value + divisor - 1) / divisor;
  return true;
}

deepgemm_status_t fill_2d_layout(
    deepgemm_dtype_t dtype,
    int64_t logical_rows,
    int64_t logical_cols,
    int64_t allocation_rows,
    int64_t allocation_cols,
    deepgemm_tensor_layout_2d_t* out) {
  if (out == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "layout output must not be null");
  }
  if (logical_rows < 0 || logical_cols < 0 || allocation_rows < logical_rows ||
      allocation_cols < logical_cols) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "invalid 2D layout dimensions");
  }

  uint64_t element_count = 0;
  if (!checked_mul_u64(
          static_cast<uint64_t>(allocation_rows),
          static_cast<uint64_t>(allocation_cols),
          &element_count)) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "layout element count overflowed");
  }

  out->dtype = dtype;
  out->logical_shape[0] = logical_rows;
  out->logical_shape[1] = logical_cols;
  out->allocation_shape[0] = allocation_rows;
  out->allocation_shape[1] = allocation_cols;
  out->stride[0] = allocation_cols;
  out->stride[1] = 1;
  out->element_count = element_count;
  return clear_error();
}

deepgemm_status_t fill_2d_layout_explicit(
    deepgemm_dtype_t dtype,
    int64_t logical_rows,
    int64_t logical_cols,
    int64_t allocation_rows,
    int64_t allocation_cols,
    int64_t stride_rows,
    int64_t stride_cols,
    deepgemm_tensor_layout_2d_t* out) {
  if (out == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "layout output must not be null");
  }
  if (logical_rows < 0 || logical_cols < 0 || allocation_rows < 0 || allocation_cols < 0 ||
      stride_rows < 0 || stride_cols < 0) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "invalid 2D layout dimensions");
  }

  uint64_t element_count = 0;
  if (!checked_mul_u64(
          static_cast<uint64_t>(allocation_rows),
          static_cast<uint64_t>(allocation_cols),
          &element_count)) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "layout element count overflowed");
  }

  out->dtype = dtype;
  out->logical_shape[0] = logical_rows;
  out->logical_shape[1] = logical_cols;
  out->allocation_shape[0] = allocation_rows;
  out->allocation_shape[1] = allocation_cols;
  out->stride[0] = stride_rows;
  out->stride[1] = stride_cols;
  out->element_count = element_count;
  return clear_error();
}

deepgemm_status_t validate_logits_dtype(deepgemm_dtype_t dtype, int64_t* elem_size) {
  *elem_size = dtype_size(dtype);
  if (dtype != DEEPGEMM_DTYPE_F32 && dtype != DEEPGEMM_DTYPE_BF16) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "logits dtype must be f32 or bf16");
  }
  return DEEPGEMM_STATUS_SUCCESS;
}

constexpr int64_t kMegaMoeTokenAlignment = 384;
constexpr int64_t kMegaMoeMinBlockM = 8;
constexpr int64_t kMegaMoeMaxBlockM = 192;
constexpr int64_t kMegaMoeCandidateBlockMs[] = {8, 16, 32, 64, 96, 128, 192};

deepgemm_status_t validate_mega_moe_dimensions(
    int64_t num_ranks,
    int64_t num_experts,
    int64_t num_max_tokens_per_rank,
    int64_t num_topk) {
  if (num_ranks <= 0 || num_experts <= 0 || num_max_tokens_per_rank <= 0 ||
      num_topk <= 0) {
    return set_error(
        DEEPGEMM_STATUS_INVALID_ARGUMENT,
        "Mega MoE rank, expert, token, and top-k counts must be positive");
  }
  if (num_ranks > std::numeric_limits<int32_t>::max() ||
      num_experts > std::numeric_limits<int32_t>::max() ||
      num_max_tokens_per_rank > std::numeric_limits<int32_t>::max() ||
      num_topk > std::numeric_limits<int32_t>::max()) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE dimension exceeds i32");
  }
  if (num_ranks > 72) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE supports at most 72 ranks");
  }
  if (num_experts % num_ranks != 0) {
    return set_error(
        DEEPGEMM_STATUS_INVALID_ARGUMENT,
        "num_experts must be divisible by num_ranks");
  }
  if (num_topk > num_experts) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "num_topk must not exceed num_experts");
  }
  if (num_max_tokens_per_rank % kMegaMoeTokenAlignment != 0) {
    return set_error(
        DEEPGEMM_STATUS_INVALID_ARGUMENT,
        "num_max_tokens_per_rank must be a multiple of 384");
  }
  return DEEPGEMM_STATUS_SUCCESS;
}

deepgemm_status_t mega_moe_ring_limits_impl(
    int64_t num_ranks,
    int64_t num_experts,
    int64_t num_max_tokens_per_rank,
    int64_t num_topk,
    deepgemm_mega_moe_ring_limits_t* out) {
  if (out == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE ring limits output must not be null");
  }
  const auto status = validate_mega_moe_dimensions(
      num_ranks, num_experts, num_max_tokens_per_rank, num_topk);
  if (status != DEEPGEMM_STATUS_SUCCESS) {
    return status;
  }

  const uint64_t ranks = static_cast<uint64_t>(num_ranks);
  const uint64_t experts_per_rank = static_cast<uint64_t>(num_experts / num_ranks);
  const uint64_t max_tokens = static_cast<uint64_t>(num_max_tokens_per_rank);
  const uint64_t topk = static_cast<uint64_t>(num_topk);
  uint64_t all_rank_tokens = 0;
  if (!checked_mul_u64(ranks, max_tokens, &all_rank_tokens)) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE ring limit overflowed");
  }

  uint64_t max_tokens_by_experts = all_rank_tokens;
  if (experts_per_rank > 1 &&
      !checked_mul_u64(all_rank_tokens, experts_per_rank, &max_tokens_by_experts)) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE ring limit overflowed");
  }

  uint64_t routed_tokens = 0;
  uint64_t expert_padding = 0;
  uint64_t routed_with_padding = 0;
  uint64_t aligned_routed_tokens = 0;
  if (!checked_mul_u64(all_rank_tokens, topk, &routed_tokens) ||
      !checked_mul_u64(experts_per_rank, kMegaMoeTokenAlignment - 1, &expert_padding) ||
      !checked_add_u64(routed_tokens, expert_padding, &routed_with_padding) ||
      !align_u64(routed_with_padding, kMegaMoeTokenAlignment, &aligned_routed_tokens)) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE ring limit overflowed");
  }

  const uint64_t max_ring_tokens = experts_per_rank == 1
      ? all_rank_tokens
      : std::min(max_tokens_by_experts, aligned_routed_tokens);
  if (all_rank_tokens > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
      max_ring_tokens > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE ring limit exceeds upstream i32 capacity");
  }
  out->min_tokens = static_cast<int64_t>(all_rank_tokens);
  out->max_tokens = static_cast<int64_t>(max_ring_tokens);
  return DEEPGEMM_STATUS_SUCCESS;
}

deepgemm_mega_moe_buffer_view_t empty_mega_moe_view() {
  return {0, DEEPGEMM_DTYPE_INVALID, 0, {0, 0}, {0, 0}, 0};
}

bool append_mega_moe_view(
    uint64_t* offset,
    deepgemm_dtype_t dtype,
    uint64_t rows,
    uint64_t cols,
    uint64_t stride_rows,
    uint64_t stride_cols,
    deepgemm_mega_moe_buffer_view_t* out) {
  const int64_t element_size = dtype_size(dtype);
  uint64_t element_count = 0;
  uint64_t byte_count = 0;
  uint64_t next_offset = 0;
  if (element_size <= 0 ||
      !checked_mul_u64(rows, cols, &element_count) ||
      !checked_mul_u64(element_count, static_cast<uint64_t>(element_size), &byte_count) ||
      !checked_add_u64(*offset, byte_count, &next_offset) ||
      rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      cols > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      stride_rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      stride_cols > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return false;
  }
  *out = {
      *offset,
      dtype,
      2,
      {static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
      {static_cast<int64_t>(stride_rows), static_cast<int64_t>(stride_cols)},
      element_count};
  *offset = next_offset;
  return true;
}

}  // namespace

extern "C" const char* deepgemm_last_error(void) {
  return g_last_error.c_str();
}

extern "C" deepgemm_status_t deepgemm_init(
    const char* deepgemm_root,
    const char* cuda_home) {
  if (deepgemm_root == nullptr || deepgemm_root[0] == '\0') {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "deepgemm_root must not be empty");
  }
  if (cuda_home == nullptr || cuda_home[0] == '\0') {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "cuda_home must not be empty");
  }
  return ffi_call([&]() {
    deepgemm_rs::runtime_init(deepgemm_root, cuda_home);
  });
}

extern "C" deepgemm_status_t deepgemm_get_device_info(
    deepgemm_device_info_t* out) {
  if (out == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "device info output must not be null");
  }
  return ffi_call([&]() {
    const auto info = deepgemm_rs::current_device_info();
    out->device = info.device;
    out->compute_capability_major = info.major;
    out->compute_capability_minor = info.minor;
    out->num_sms = info.num_sms;
  });
}

extern "C" deepgemm_status_t deepgemm_get_num_sms(int32_t* out) {
  if (out == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "num_sms output must not be null");
  }
  return ffi_call([&]() {
    *out = deepgemm_rs::effective_num_sms();
  });
}

extern "C" deepgemm_status_t deepgemm_set_num_sms(int32_t num_sms) {
  return ffi_call([&]() {
    deepgemm_rs::set_num_sms_override(num_sms);
  });
}

extern "C" deepgemm_status_t deepgemm_set_pdl(bool enabled) {
  return ffi_call([&]() {
    deepgemm_rs::set_pdl(enabled);
  });
}

extern "C" deepgemm_status_t deepgemm_mqa_logits_layout(
    const deepgemm_mqa_logits_layout_params_t* params,
    deepgemm_tensor_layout_2d_t* out) {
  if (params == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "layout params must not be null");
  }
  if (params->seq_len <= 0 || params->seq_len_kv <= 0 || params->num_heads <= 0) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "seq_len, seq_len_kv, and num_heads must be positive");
  }
  if (params->max_seqlen_k < 0) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "max_seqlen_k must be non-negative");
  }
  if (128 % params->num_heads != 0) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "num_heads must divide 128");
  }

  int64_t elem_size = 0;
  const auto dtype_status = validate_logits_dtype(params->logits_dtype, &elem_size);
  if (dtype_status != DEEPGEMM_STATUS_SUCCESS) {
    return dtype_status;
  }

  constexpr int64_t block_kv = 256;
  const int64_t block_q = 128 / params->num_heads;
  int64_t aligned_seq_len = 0;
  if (!align_i64(params->seq_len, block_q, &aligned_seq_len)) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "aligned seq_len overflowed");
  }

  const int64_t stride_alignment = 1024 / elem_size;
  int64_t logical_cols = 0;
  int64_t stride_cols = 0;
  if (params->max_seqlen_k == 0) {
    logical_cols = params->seq_len_kv;
    if (!align_i64(params->seq_len_kv + block_kv, stride_alignment, &stride_cols)) {
      return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "full logits stride overflowed");
    }
  } else {
    logical_cols = params->max_seqlen_k;
    int64_t aligned_max_seqlen_k = 0;
    if (!align_i64(params->max_seqlen_k, block_kv, &aligned_max_seqlen_k) ||
        !align_i64(aligned_max_seqlen_k, stride_alignment, &stride_cols)) {
      return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "compressed logits stride overflowed");
    }
  }

  return fill_2d_layout(
      params->logits_dtype,
      params->seq_len,
      logical_cols,
      aligned_seq_len,
      stride_cols,
      out);
}

extern "C" deepgemm_status_t deepgemm_paged_mqa_logits_layout(
    const deepgemm_paged_mqa_logits_layout_params_t* params,
    deepgemm_tensor_layout_2d_t* out) {
  if (params == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "layout params must not be null");
  }
  if (params->batch_size <= 0 || params->next_n <= 0 || params->max_context_len <= 0) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "batch_size, next_n, and max_context_len must be positive");
  }

  int64_t elem_size = 0;
  const auto dtype_status = validate_logits_dtype(params->logits_dtype, &elem_size);
  if (dtype_status != DEEPGEMM_STATUS_SUCCESS) {
    return dtype_status;
  }

  constexpr int64_t split_kv = 256;
  const int64_t stride_alignment = 1024 / elem_size;
  int64_t split_aligned = 0;
  int64_t stride_cols = 0;
  if (!align_i64(params->max_context_len, split_kv, &split_aligned) ||
      !align_i64(split_aligned, stride_alignment, &stride_cols)) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "paged logits stride overflowed");
  }

  uint64_t rows = 0;
  if (!checked_mul_u64(
          static_cast<uint64_t>(params->batch_size),
          static_cast<uint64_t>(params->next_n),
          &rows) ||
      rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "paged logits row count overflowed");
  }

  return fill_2d_layout(
      params->logits_dtype,
      static_cast<int64_t>(rows),
      params->max_context_len,
      static_cast<int64_t>(rows),
      stride_cols,
      out);
}

extern "C" deepgemm_status_t deepgemm_paged_mqa_logits_metadata_layout(
    const deepgemm_paged_mqa_logits_metadata_layout_params_t* params,
    deepgemm_tensor_layout_2d_t* out) {
  if (params == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "metadata layout params must not be null");
  }
  if (params->num_sms <= 0) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "num_sms must be positive");
  }
  if (params->num_sms == std::numeric_limits<int64_t>::max()) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "num_sms overflowed metadata rows");
  }
  return fill_2d_layout(
      DEEPGEMM_DTYPE_I32,
      params->num_sms + 1,
      2,
      params->num_sms + 1,
      2,
      out);
}

extern "C" deepgemm_status_t deepgemm_fp8_gemm_nt_output_layout(
    const deepgemm_fp8_gemm_nt_output_layout_params_t* params,
    deepgemm_tensor_layout_2d_t* out) {
  if (params == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "GEMM output layout params must not be null");
  }
  if (params->m <= 0 || params->n <= 0) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "m and n must be positive");
  }
  if (params->output_dtype != DEEPGEMM_DTYPE_BF16) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "FP8 GEMM output dtype must be bf16");
  }
  return fill_2d_layout(
      params->output_dtype,
      params->m,
      params->n,
      params->m,
      params->n,
      out);
}

extern "C" deepgemm_status_t deepgemm_fp8_gemm_scale_layout(
    const deepgemm_fp8_gemm_scale_layout_params_t* params,
    deepgemm_tensor_layout_2d_t* out) {
  if (params == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "GEMM scale layout params must not be null");
  }
  if (params->mn <= 0 || params->k <= 0 || params->gran_k <= 0) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "mn, k, and gran_k must be positive");
  }
  if (params->scale_dtype != DEEPGEMM_DTYPE_F32 &&
      params->scale_dtype != DEEPGEMM_DTYPE_PACKED_UE8M0) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "scale dtype must be f32 or packed UE8M0");
  }
  const int64_t elem_size = dtype_size(params->scale_dtype);
  if (elem_size <= 0 || 16 % elem_size != 0) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "scale dtype has invalid element size");
  }

  int64_t scale_cols = 0;
  const int64_t packed_factor = params->scale_dtype == DEEPGEMM_DTYPE_F32 ? 1 : 4;
  if (params->gran_k > std::numeric_limits<int64_t>::max() / packed_factor ||
      !ceil_div_i64(params->k, params->gran_k * packed_factor, &scale_cols)) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "scale K dimension overflowed");
  }
  int64_t aligned_mn = 0;
  if (!align_i64(params->mn, 16 / elem_size, &aligned_mn)) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "scale MN alignment overflowed");
  }

  return fill_2d_layout_explicit(
      params->scale_dtype,
      params->mn,
      scale_cols,
      scale_cols,
      aligned_mn,
      1,
      aligned_mn,
      out);
}

extern "C" deepgemm_status_t deepgemm_mega_moe_token_alignment(int64_t* out) {
  if (out == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE token alignment output must not be null");
  }
  *out = kMegaMoeTokenAlignment;
  return clear_error();
}

extern "C" deepgemm_status_t deepgemm_mega_moe_ring_limits(
    const deepgemm_mega_moe_ring_limits_params_t* params,
    deepgemm_mega_moe_ring_limits_t* out) {
  if (params == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE ring limit params must not be null");
  }
  const auto status = mega_moe_ring_limits_impl(
      params->num_ranks,
      params->num_experts,
      params->num_max_tokens_per_rank,
      params->num_topk,
      out);
  return status == DEEPGEMM_STATUS_SUCCESS ? clear_error() : status;
}

extern "C" deepgemm_status_t deepgemm_mega_moe_buffer_layout(
    const deepgemm_mega_moe_buffer_params_t* params,
    deepgemm_mega_moe_buffer_layout_t* out) {
  if (params == nullptr || out == nullptr) {
    return set_error(
        DEEPGEMM_STATUS_INVALID_ARGUMENT,
        "Mega MoE buffer params and output must not be null");
  }

  deepgemm_mega_moe_ring_limits_t ring_limits{};
  const auto limits_status = mega_moe_ring_limits_impl(
      params->num_ranks,
      params->num_experts,
      params->num_max_tokens_per_rank,
      params->num_topk,
      &ring_limits);
  if (limits_status != DEEPGEMM_STATUS_SUCCESS) {
    return limits_status;
  }
  if (params->hidden <= 0 || params->intermediate_hidden <= 0 ||
      params->hidden > std::numeric_limits<int32_t>::max() ||
      params->intermediate_hidden > std::numeric_limits<int32_t>::max()) {
    return set_error(
        DEEPGEMM_STATUS_INVALID_ARGUMENT,
        "Mega MoE hidden dimensions must be positive i32 values");
  }
  if (params->num_ring_tokens < ring_limits.min_tokens ||
      params->num_ring_tokens > ring_limits.max_tokens ||
      params->num_ring_tokens % kMegaMoeTokenAlignment != 0) {
    return set_error(
        DEEPGEMM_STATUS_INVALID_ARGUMENT,
        "num_ring_tokens must be 384-aligned and within the derived ring limits");
  }
  if (params->mma_kind != DEEPGEMM_MEGA_MOE_MMA_FP8_FP4 &&
      params->mma_kind != DEEPGEMM_MEGA_MOE_MMA_BF16) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "invalid Mega MoE MMA kind");
  }

  const bool with_scales = params->mma_kind == DEEPGEMM_MEGA_MOE_MMA_FP8_FP4;
  const uint64_t hidden = static_cast<uint64_t>(params->hidden);
  const uint64_t intermediate_hidden = static_cast<uint64_t>(params->intermediate_hidden);
  if ((with_scales && (hidden % 128 != 0 || intermediate_hidden % 128 != 0)) ||
      (!with_scales && (hidden % 8 != 0 || intermediate_hidden % 8 != 0))) {
    return set_error(
        DEEPGEMM_STATUS_INVALID_ARGUMENT,
        with_scales
            ? "FP8/FP4 Mega MoE hidden dimensions must be multiples of 128"
            : "BF16 Mega MoE hidden dimensions must be multiples of 8");
  }

  const uint64_t ranks = static_cast<uint64_t>(params->num_ranks);
  const uint64_t experts = static_cast<uint64_t>(params->num_experts);
  const uint64_t experts_per_rank = experts / ranks;
  const uint64_t max_tokens = static_cast<uint64_t>(params->num_max_tokens_per_rank);
  const uint64_t topk = static_cast<uint64_t>(params->num_topk);
  const uint64_t ring_tokens = static_cast<uint64_t>(params->num_ring_tokens);

  uint64_t max_recv_tokens = 0;
  uint64_t routed_pool_tokens = 0;
  uint64_t pool_padding = 0;
  uint64_t unaligned_pool_tokens = 0;
  uint64_t max_pool_tokens = 0;
  if (!checked_mul_u64(ranks, max_tokens, &max_recv_tokens) ||
      !checked_mul_u64(max_recv_tokens, std::min(topk, experts_per_rank), &routed_pool_tokens) ||
      !checked_mul_u64(experts_per_rank, kMegaMoeMaxBlockM - 1, &pool_padding) ||
      !checked_add_u64(routed_pool_tokens, pool_padding, &unaligned_pool_tokens) ||
      !align_u64(unaligned_pool_tokens, kMegaMoeTokenAlignment, &max_pool_tokens)) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE pool size overflowed");
  }

  uint64_t workspace_bytes = 32;
  auto append_workspace_term = [&](uint64_t left, uint64_t right) {
    uint64_t term = 0;
    return checked_mul_u64(left, right, &term) &&
        checked_add_u64(workspace_bytes, term, &workspace_bytes);
  };
  uint64_t dispatch_entries = 0;
  if (!append_workspace_term(experts, 16) ||
      !append_workspace_term(experts_per_rank, 8) ||
      !append_workspace_term(ring_tokens / kMegaMoeMinBlockM, 16) ||
      !checked_mul_u64(experts_per_rank, ranks, &dispatch_entries) ||
      !checked_mul_u64(dispatch_entries, max_recv_tokens, &dispatch_entries) ||
      !append_workspace_term(dispatch_entries, 4) ||
      !append_workspace_term(max_pool_tokens, 12) ||
      !align_u64(workspace_bytes, 16, &workspace_bytes)) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE workspace size overflowed");
  }

  uint64_t num_sf_ring_tokens = 0;
  if (with_scales) {
    for (const int64_t block_m : kMegaMoeCandidateBlockMs) {
      uint64_t aligned_block_m = 0;
      if (!align_u64(static_cast<uint64_t>(block_m), 128, &aligned_block_m)) {
        return set_error(DEEPGEMM_STATUS_INTERNAL_ERROR, "Mega MoE scale layout overflowed");
      }
      uint64_t candidate_sf_tokens = 0;
      if (!checked_mul_u64(
              ring_tokens / static_cast<uint64_t>(block_m),
              aligned_block_m,
              &candidate_sf_tokens)) {
        return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE scale layout overflowed");
      }
      num_sf_ring_tokens = std::max(num_sf_ring_tokens, candidate_sf_tokens);
    }
  }

  *out = {};
  out->workspace_bytes = workspace_bytes;
  uint64_t offset = workspace_bytes;
  const deepgemm_dtype_t activation_dtype = with_scales
      ? DEEPGEMM_DTYPE_FP8_E4M3
      : DEEPGEMM_DTYPE_BF16;
  const auto append = [&](deepgemm_dtype_t dtype,
                          uint64_t rows,
                          uint64_t cols,
                          uint64_t stride_rows,
                          uint64_t stride_cols,
                          deepgemm_mega_moe_buffer_view_t* view) {
    return append_mega_moe_view(
        &offset, dtype, rows, cols, stride_rows, stride_cols, view);
  };

  if (!append(activation_dtype, max_tokens, hidden, hidden, 1, &out->x)) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE x layout overflowed");
  }
  if (with_scales) {
    if (!append(
            DEEPGEMM_DTYPE_PACKED_UE8M0,
            max_tokens,
            hidden / 128,
            hidden / 128,
            1,
            &out->x_scale)) {
      return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE x scale layout overflowed");
    }
  } else {
    out->x_scale = empty_mega_moe_view();
  }
  if (!append(DEEPGEMM_DTYPE_I64, max_tokens, topk, topk, 1, &out->topk_indices) ||
      !append(DEEPGEMM_DTYPE_F32, max_tokens, topk, topk, 1, &out->topk_weights) ||
      !append(activation_dtype, ring_tokens, hidden, hidden, 1, &out->l1_acts)) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE input layout overflowed");
  }
  if (with_scales) {
    if (!append(
            DEEPGEMM_DTYPE_PACKED_UE8M0,
            num_sf_ring_tokens,
            hidden / 128,
            1,
            num_sf_ring_tokens,
            &out->l1_acts_scale)) {
      return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE L1 scale layout overflowed");
    }
  } else {
    out->l1_acts_scale = empty_mega_moe_view();
  }

  uint64_t l1_topk_weight_bytes = 0;
  if (!checked_mul_u64(ring_tokens, 4, &l1_topk_weight_bytes) ||
      !checked_add_u64(offset, l1_topk_weight_bytes, &offset) ||
      !append(
          activation_dtype,
          ring_tokens,
          intermediate_hidden,
          intermediate_hidden,
          1,
          &out->l2_acts)) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE intermediate layout overflowed");
  }
  if (with_scales) {
    if (!append(
            DEEPGEMM_DTYPE_PACKED_UE8M0,
            num_sf_ring_tokens,
            intermediate_hidden / 128,
            1,
            num_sf_ring_tokens,
            &out->l2_acts_scale)) {
      return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE L2 scale layout overflowed");
    }
  } else {
    out->l2_acts_scale = empty_mega_moe_view();
  }

  uint64_t combine_rows = 0;
  if (!checked_mul_u64(topk, max_tokens, &combine_rows) ||
      !append(DEEPGEMM_DTYPE_BF16, combine_rows, hidden, hidden, 1, &out->combine)) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE combine layout overflowed");
  }
  out->total_bytes = offset;
  return clear_error();
}

extern "C" deepgemm_status_t deepgemm_fp8_fp4_mqa_logits(
    const deepgemm_mqa_logits_params_t* params) {
  if (params == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "MQA logits params must not be null");
  }
  return ffi_call([&]() {
    deepgemm_rs::launch_fp8_fp4_mqa_logits(*params);
  });
}

extern "C" deepgemm_status_t deepgemm_paged_mqa_logits_metadata(
    const deepgemm_paged_mqa_logits_metadata_params_t* params) {
  if (params == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "paged metadata params must not be null");
  }
  return ffi_call([&]() {
    deepgemm_rs::launch_paged_mqa_logits_metadata(*params);
  });
}

extern "C" deepgemm_status_t deepgemm_fp8_fp4_paged_mqa_logits(
    const deepgemm_paged_mqa_logits_params_t* params) {
  if (params == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "paged MQA logits params must not be null");
  }
  return ffi_call([&]() {
    deepgemm_rs::launch_fp8_fp4_paged_mqa_logits(*params);
  });
}

extern "C" deepgemm_status_t deepgemm_fp8_gemm_transform_scale(
    const deepgemm_fp8_gemm_scale_transform_params_t* params) {
  if (params == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "FP8 GEMM scale transform params must not be null");
  }
  return ffi_call([&]() {
    deepgemm_rs::launch_fp8_gemm_transform_scale(*params);
  });
}

extern "C" deepgemm_status_t deepgemm_fp8_gemm_nt(
    const deepgemm_fp8_gemm_nt_params_t* params) {
  if (params == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "FP8 GEMM params must not be null");
  }
  return ffi_call([&]() {
    deepgemm_rs::launch_fp8_gemm_nt(*params);
  });
}

extern "C" deepgemm_status_t deepgemm_bf16_m_grouped_gemm_nt_contiguous(
    const deepgemm_bf16_m_grouped_gemm_nt_contiguous_params_t* params) {
  if (params == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "params must not be null");
  }
  return ffi_call([&]() {
    deepgemm_rs::launch_bf16_m_grouped_gemm_nt_contiguous(*params);
  });
}

extern "C" deepgemm_status_t deepgemm_bf16_mega_moe(
    const deepgemm_bf16_mega_moe_params_t* params) {
  if (params == nullptr) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "params must not be null");
  }
  return ffi_call([&]() {
    deepgemm_rs::launch_bf16_mega_moe(*params);
  });
}
