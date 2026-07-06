#include "deepgemm_c_api.h"

#include "deepgemm_raw_mqa.h"
#include "deepgemm_raw_runtime.h"

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

deepgemm_status_t validate_logits_dtype(deepgemm_dtype_t dtype, int64_t* elem_size) {
  *elem_size = dtype_size(dtype);
  if (dtype != DEEPGEMM_DTYPE_F32 && dtype != DEEPGEMM_DTYPE_BF16) {
    return set_error(DEEPGEMM_STATUS_INVALID_ARGUMENT, "logits dtype must be f32 or bf16");
  }
  return DEEPGEMM_STATUS_SUCCESS;
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
  return set_error(
      DEEPGEMM_STATUS_INTERNAL_ERROR,
      "DeepGEMM paged MQA logits launch is not wired in the native shim yet");
}
