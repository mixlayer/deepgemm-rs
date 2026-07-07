#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum deepgemm_status_t {
  DEEPGEMM_STATUS_SUCCESS = 0,
  DEEPGEMM_STATUS_INVALID_ARGUMENT = 1,
  DEEPGEMM_STATUS_UNSUPPORTED_ARCH = 2,
  DEEPGEMM_STATUS_CUDA_ERROR = 3,
  DEEPGEMM_STATUS_INTERNAL_ERROR = 4,
} deepgemm_status_t;

typedef enum deepgemm_dtype_t {
  DEEPGEMM_DTYPE_INVALID = 0,
  DEEPGEMM_DTYPE_FP8_E4M3 = 1,
  DEEPGEMM_DTYPE_PACKED_FP4_E2M1 = 2,
  DEEPGEMM_DTYPE_PACKED_UE8M0 = 3,
  DEEPGEMM_DTYPE_F32 = 4,
  DEEPGEMM_DTYPE_BF16 = 5,
  DEEPGEMM_DTYPE_I32 = 6,
  DEEPGEMM_DTYPE_U8 = 7,
} deepgemm_dtype_t;

typedef void* deepgemm_cuda_stream_t;

typedef struct deepgemm_tensor_t {
  const void* data;
  deepgemm_dtype_t dtype;
  uint32_t rank;
  int64_t shape[4];
  int64_t stride[4];
} deepgemm_tensor_t;

typedef struct deepgemm_tensor_mut_t {
  void* data;
  deepgemm_dtype_t dtype;
  uint32_t rank;
  int64_t shape[4];
  int64_t stride[4];
} deepgemm_tensor_mut_t;

typedef struct deepgemm_tensor_layout_2d_t {
  deepgemm_dtype_t dtype;
  int64_t logical_shape[2];
  int64_t allocation_shape[2];
  int64_t stride[2];
  uint64_t element_count;
} deepgemm_tensor_layout_2d_t;

typedef struct deepgemm_device_info_t {
  int32_t device;
  int32_t compute_capability_major;
  int32_t compute_capability_minor;
  int32_t num_sms;
} deepgemm_device_info_t;

typedef struct deepgemm_mqa_logits_layout_params_t {
  int64_t seq_len;
  int64_t seq_len_kv;
  int64_t num_heads;
  int64_t max_seqlen_k;
  deepgemm_dtype_t logits_dtype;
} deepgemm_mqa_logits_layout_params_t;

typedef struct deepgemm_paged_mqa_logits_layout_params_t {
  int64_t batch_size;
  int64_t next_n;
  int64_t max_context_len;
  deepgemm_dtype_t logits_dtype;
} deepgemm_paged_mqa_logits_layout_params_t;

typedef struct deepgemm_paged_mqa_logits_metadata_layout_params_t {
  int64_t num_sms;
} deepgemm_paged_mqa_logits_metadata_layout_params_t;

typedef struct deepgemm_fp8_gemm_nt_output_layout_params_t {
  int64_t m;
  int64_t n;
  deepgemm_dtype_t output_dtype;
} deepgemm_fp8_gemm_nt_output_layout_params_t;

typedef struct deepgemm_fp8_gemm_scale_layout_params_t {
  int64_t mn;
  int64_t k;
  int64_t gran_k;
  deepgemm_dtype_t scale_dtype;
} deepgemm_fp8_gemm_scale_layout_params_t;

typedef struct deepgemm_mqa_logits_params_t {
  deepgemm_tensor_t q;
  bool has_q_scale;
  deepgemm_tensor_t q_scale;
  deepgemm_tensor_t kv;
  deepgemm_tensor_t kv_scale;
  deepgemm_tensor_t weights;
  deepgemm_tensor_t cu_seq_len_k_start;
  deepgemm_tensor_t cu_seq_len_k_end;
  deepgemm_tensor_mut_t logits;
  bool clean_logits;
  int64_t max_seqlen_k;
  deepgemm_cuda_stream_t stream;
} deepgemm_mqa_logits_params_t;

typedef struct deepgemm_paged_mqa_logits_metadata_params_t {
  deepgemm_tensor_t context_lens;
  bool has_indices;
  deepgemm_tensor_t indices;
  deepgemm_tensor_mut_t schedule_meta;
  int64_t block_kv;
  int64_t num_sms;
  deepgemm_cuda_stream_t stream;
} deepgemm_paged_mqa_logits_metadata_params_t;

typedef struct deepgemm_paged_mqa_logits_params_t {
  deepgemm_tensor_t q;
  bool has_q_scale;
  deepgemm_tensor_t q_scale;
  deepgemm_tensor_t fused_kv_cache;
  deepgemm_tensor_t weights;
  deepgemm_tensor_t context_lens;
  deepgemm_tensor_t block_table;
  deepgemm_tensor_t schedule_meta;
  bool has_indices;
  deepgemm_tensor_t indices;
  deepgemm_tensor_mut_t logits;
  int64_t max_context_len;
  int64_t num_sms;
  bool clean_logits;
  deepgemm_cuda_stream_t stream;
} deepgemm_paged_mqa_logits_params_t;

typedef struct deepgemm_fp8_gemm_scale_transform_params_t {
  deepgemm_tensor_t scale;
  deepgemm_tensor_mut_t transformed;
  int64_t mn;
  int64_t k;
  int64_t gran_k;
  deepgemm_cuda_stream_t stream;
} deepgemm_fp8_gemm_scale_transform_params_t;

typedef struct deepgemm_fp8_gemm_nt_params_t {
  deepgemm_tensor_t a;
  deepgemm_tensor_t a_scale;
  deepgemm_tensor_t b;
  deepgemm_tensor_t b_scale;
  deepgemm_tensor_mut_t d;
  deepgemm_cuda_stream_t stream;
} deepgemm_fp8_gemm_nt_params_t;

const char* deepgemm_last_error(void);

deepgemm_status_t deepgemm_init(
  const char* deepgemm_root,
  const char* cuda_home
);

deepgemm_status_t deepgemm_get_device_info(
  deepgemm_device_info_t* out
);

deepgemm_status_t deepgemm_get_num_sms(
  int32_t* out
);

deepgemm_status_t deepgemm_set_num_sms(
  int32_t num_sms
);

deepgemm_status_t deepgemm_set_pdl(
  bool enabled
);

deepgemm_status_t deepgemm_mqa_logits_layout(
  const deepgemm_mqa_logits_layout_params_t* params,
  deepgemm_tensor_layout_2d_t* out
);

deepgemm_status_t deepgemm_paged_mqa_logits_layout(
  const deepgemm_paged_mqa_logits_layout_params_t* params,
  deepgemm_tensor_layout_2d_t* out
);

deepgemm_status_t deepgemm_paged_mqa_logits_metadata_layout(
  const deepgemm_paged_mqa_logits_metadata_layout_params_t* params,
  deepgemm_tensor_layout_2d_t* out
);

deepgemm_status_t deepgemm_fp8_gemm_nt_output_layout(
  const deepgemm_fp8_gemm_nt_output_layout_params_t* params,
  deepgemm_tensor_layout_2d_t* out
);

deepgemm_status_t deepgemm_fp8_gemm_scale_layout(
  const deepgemm_fp8_gemm_scale_layout_params_t* params,
  deepgemm_tensor_layout_2d_t* out
);

deepgemm_status_t deepgemm_fp8_fp4_mqa_logits(
  const deepgemm_mqa_logits_params_t* params
);

deepgemm_status_t deepgemm_paged_mqa_logits_metadata(
  const deepgemm_paged_mqa_logits_metadata_params_t* params
);

deepgemm_status_t deepgemm_fp8_fp4_paged_mqa_logits(
  const deepgemm_paged_mqa_logits_params_t* params
);

deepgemm_status_t deepgemm_fp8_gemm_transform_scale(
  const deepgemm_fp8_gemm_scale_transform_params_t* params
);

deepgemm_status_t deepgemm_fp8_gemm_nt(
  const deepgemm_fp8_gemm_nt_params_t* params
);

#ifdef __cplusplus
}
#endif
