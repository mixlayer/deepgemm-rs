#include "deepgemm_raw_mqa.h"

#include "deepgemm_raw_runtime.h"

#include <cstdint>
#include <sstream>
#include <string>

namespace deepgemm_rs {
namespace {

int64_t align_i64_or_throw(int64_t value, int64_t alignment, const char* name) {
  if (value < 0 || alignment <= 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, std::string(name) + " cannot be aligned");
  }
  const int64_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  return value + (alignment - remainder);
}

int as_i32(int64_t value, const char* name) {
  if (value < 0 || value > INT32_MAX) {
    throw_status(
        DEEPGEMM_STATUS_INVALID_ARGUMENT,
        std::string(name) + " must fit in a positive i32");
  }
  return static_cast<int>(value);
}

void require_tensor_rank(
    const deepgemm_tensor_t& tensor,
    uint32_t rank,
    const char* name) {
  if (tensor.rank != rank) {
    std::ostringstream message;
    message << name << " rank must be " << rank << ", got " << tensor.rank;
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, message.str());
  }
  if (tensor.data == nullptr) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, std::string(name) + " data must not be null");
  }
}

void require_tensor_mut_rank(
    const deepgemm_tensor_mut_t& tensor,
    uint32_t rank,
    const char* name) {
  if (tensor.rank != rank) {
    std::ostringstream message;
    message << name << " rank must be " << rank << ", got " << tensor.rank;
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, message.str());
  }
  if (tensor.data == nullptr) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, std::string(name) + " data must not be null");
  }
}

void require_dtype(
    deepgemm_dtype_t actual,
    deepgemm_dtype_t expected,
    const char* name) {
  if (actual != expected) {
    std::ostringstream message;
    message << name << " dtype must be " << expected << ", got " << actual;
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, message.str());
  }
}

void require_shape_positive(
    const int64_t* shape,
    uint32_t rank,
    const char* name) {
  for (uint32_t i = 0; i < rank; ++i) {
    if (shape[i] <= 0) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, std::string(name) + " dimensions must be positive");
    }
  }
}

void require_contiguous(
    const deepgemm_tensor_t& tensor,
    uint32_t rank,
    const char* name) {
  require_tensor_rank(tensor, rank, name);
  require_shape_positive(tensor.shape, rank, name);

  int64_t expected_stride = 1;
  for (int index = static_cast<int>(rank) - 1; index >= 0; --index) {
    if (tensor.stride[index] != expected_stride) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, std::string(name) + " must be contiguous");
    }
    expected_stride *= tensor.shape[index];
  }
}

void require_contiguous_2d_i32(
    const deepgemm_tensor_t& tensor,
    const char* name) {
  require_tensor_rank(tensor, 2, name);
  require_dtype(tensor.dtype, DEEPGEMM_DTYPE_I32, name);
  if (tensor.shape[0] <= 0 || tensor.shape[1] <= 0 ||
      tensor.stride[0] != tensor.shape[1] || tensor.stride[1] != 1) {
    throw_status(
        DEEPGEMM_STATUS_INVALID_ARGUMENT,
        std::string(name) + " must be contiguous i32 [rows, cols]");
  }
}

void require_contiguous_1d_i32(
    const deepgemm_tensor_t& tensor,
    int64_t expected_len,
    const char* name) {
  require_tensor_rank(tensor, 1, name);
  require_dtype(tensor.dtype, DEEPGEMM_DTYPE_I32, name);
  if (tensor.shape[0] != expected_len || tensor.stride[0] != 1) {
    std::ostringstream message;
    message << name << " must be contiguous i32 [" << expected_len << "]";
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, message.str());
  }
}

void require_i32_vector(
    const deepgemm_tensor_t& tensor,
    int64_t expected_len,
    const char* name) {
  require_contiguous_1d_i32(tensor, expected_len, name);
}

void require_logits_dtype(deepgemm_dtype_t dtype) {
  if (dtype != DEEPGEMM_DTYPE_F32 && dtype != DEEPGEMM_DTYPE_BF16) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "logits dtype must be f32 or bf16");
  }
}

void require_logits_output(
    const deepgemm_tensor_mut_t& logits,
    int64_t rows,
    int64_t cols,
    int64_t stride,
    const char* name) {
  require_tensor_mut_rank(logits, 2, name);
  require_logits_dtype(logits.dtype);
  if (logits.shape[0] != rows || logits.shape[1] != cols ||
      logits.stride[0] != stride || logits.stride[1] != 1) {
    std::ostringstream message;
    message << name << " must have shape [" << rows << ", " << cols
            << "] and strides [" << stride << ", 1]";
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, message.str());
  }
}

void require_schedule_meta(
    const deepgemm_tensor_mut_t& tensor,
    int64_t num_sms) {
  require_tensor_mut_rank(tensor, 2, "schedule_meta");
  require_dtype(tensor.dtype, DEEPGEMM_DTYPE_I32, "schedule_meta");
  if (tensor.shape[0] != num_sms + 1 || tensor.shape[1] != 2 ||
      tensor.stride[0] != 2 || tensor.stride[1] != 1) {
    std::ostringstream message;
    message << "schedule_meta must be contiguous i32 [" << (num_sms + 1) << ", 2]";
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, message.str());
  }
}

std::string bool_literal(bool value) {
  return value ? "true" : "false";
}

int64_t align_i64(int64_t value, int64_t alignment) {
  return align_i64_or_throw(value, alignment, "value");
}

int64_t mqa_logits_stride(
    int64_t seq_len_kv,
    int64_t max_seqlen_k,
    int64_t block_kv,
    deepgemm_dtype_t dtype) {
  const int64_t elem_size = dtype_element_size(dtype);
  if (elem_size <= 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "invalid logits dtype");
  }
  const int64_t stride_alignment = 1024 / elem_size;
  if (max_seqlen_k == 0) {
    return align_i64(seq_len_kv + block_kv, stride_alignment);
  }
  return align_i64(align_i64(max_seqlen_k, block_kv), stride_alignment);
}

std::string sm90_mqa_logits_code(
    int num_heads,
    int head_dim,
    bool is_compressed_logits,
    int block_q,
    int block_kv,
    int num_q_stages,
    int num_kv_stages,
    int num_sms,
    int num_specialized_threads,
    int num_math_threads,
    deepgemm_dtype_t logits_dtype) {
  std::ostringstream code;
  code
      << "#include <deep_gemm/impls/sm90_fp8_mqa_logits.cuh>\n\n"
      << "using namespace deep_gemm;\n\n"
      << "static void __instantiate_kernel() {\n"
      << "    auto ptr = reinterpret_cast<void*>(&sm90_fp8_mqa_logits<\n"
      << "        " << num_heads << ", " << head_dim << ",\n"
      << "        " << bool_literal(is_compressed_logits) << ",\n"
      << "        " << block_q << ", " << block_kv << ",\n"
      << "        " << num_q_stages << ", " << num_kv_stages << ",\n"
      << "        " << num_sms << ",\n"
      << "        " << num_specialized_threads << ", " << num_math_threads << ",\n"
      << "        " << kernel_dtype_name(logits_dtype) << "\n"
      << "    >);\n"
      << "};\n";
  return code.str();
}

std::string sm100_mqa_logits_code(
    bool is_fp4,
    int num_heads,
    int head_dim,
    bool is_compressed_logits,
    int block_q,
    int split_kv,
    int num_q_stages,
    int num_kv_stages,
    int num_sms,
    int num_specialized_threads,
    int num_math_threads,
    deepgemm_dtype_t logits_dtype,
    deepgemm_dtype_t weights_dtype) {
  std::ostringstream code;
  code
      << "#include <deep_gemm/impls/sm100_mqa_logits.cuh>\n\n"
      << "using namespace deep_gemm;\n\n"
      << "static void __instantiate_kernel() {\n"
      << "    auto ptr = reinterpret_cast<void*>(&sm100_mqa_logits<\n"
      << "        " << bool_literal(is_fp4) << ",\n"
      << "        " << num_heads << ", " << head_dim << ",\n"
      << "        " << bool_literal(is_compressed_logits) << ",\n"
      << "        " << block_q << ", " << split_kv << ",\n"
      << "        " << num_q_stages << ", " << num_kv_stages << ",\n"
      << "        " << num_sms << ",\n"
      << "        " << num_specialized_threads << ", " << num_math_threads << ",\n"
      << "        " << kernel_dtype_name(logits_dtype) << ",\n"
      << "        " << (weights_dtype == DEEPGEMM_DTYPE_BF16 ? "cutlass::bfloat16_t" : "float") << "\n"
      << "    >);\n"
      << "};\n";
  return code.str();
}

std::string clean_logits_code(
    int next_n,
    int block_kv,
    int num_warps,
    deepgemm_dtype_t logits_dtype) {
  std::ostringstream code;
  code
      << "#include <deep_gemm/impls/smxx_clean_logits.cuh>\n\n"
      << "using namespace deep_gemm;\n\n"
      << "static void __instantiate_kernel() {\n"
      << "    auto ptr = reinterpret_cast<void*>(&smxx_clean_logits<\n"
      << "        " << next_n << ", " << block_kv << ", " << num_warps << ", "
      << kernel_dtype_name(logits_dtype) << "\n"
      << "    >);\n"
      << "};\n";
  return code.str();
}

std::string sm90_metadata_code(
    int aligned_batch_size,
    int split_kv,
    int num_sms,
    bool is_varlen) {
  std::ostringstream code;
  code
      << "#include <deep_gemm/scheduler/sm90_paged_mqa_logits.cuh>\n\n"
      << "using namespace deep_gemm;\n\n"
      << "static void __instantiate_kernel() {\n"
      << "    auto ptr = reinterpret_cast<void*>(&sched::sm90_paged_mqa_logits_metadata<\n"
      << "        " << aligned_batch_size << ", " << split_kv << ", " << num_sms << ", "
      << bool_literal(is_varlen) << "\n"
      << "    >);\n"
      << "};\n";
  return code.str();
}

std::string sm100_metadata_code(
    int next_n,
    bool is_context_lens_2d,
    bool is_varlen,
    int split_kv,
    int num_sms) {
  std::ostringstream code;
  code
      << "#include <deep_gemm/scheduler/sm100_paged_mqa_logits.cuh>\n\n"
      << "using namespace deep_gemm;\n\n"
      << "static void __instantiate_kernel() {\n"
      << "    auto ptr = reinterpret_cast<void*>(&sched::sm100_paged_mqa_logits_metadata<\n"
      << "        " << next_n << ", " << bool_literal(is_context_lens_2d) << ", "
      << bool_literal(is_varlen) << ", 1, " << split_kv << ", " << num_sms << "\n"
      << "    >);\n"
      << "};\n";
  return code.str();
}

std::string sm90_paged_mqa_logits_code(
    int next_n,
    int num_heads,
    int head_dim,
    int block_kv,
    bool is_context_lens_2d,
    bool is_varlen,
    int num_q_stages,
    int num_kv_stages,
    int split_kv,
    int num_specialized_threads,
    int num_math_threads,
    deepgemm_dtype_t logits_dtype) {
  std::ostringstream code;
  code
      << "#include <deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh>\n\n"
      << "using namespace deep_gemm;\n\n"
      << "static void __instantiate_kernel() {\n"
      << "    auto ptr = reinterpret_cast<void*>(&sm90_fp8_paged_mqa_logits<\n"
      << "        " << next_n << ", " << num_heads << ",\n"
      << "        " << head_dim << ", " << block_kv << ",\n"
      << "        " << bool_literal(is_context_lens_2d) << ", "
      << bool_literal(is_varlen) << ",\n"
      << "        " << num_q_stages << ", " << num_kv_stages << ",\n"
      << "        " << split_kv << ",\n"
      << "        " << num_specialized_threads << ", " << num_math_threads << ",\n"
      << "        " << kernel_dtype_name(logits_dtype) << "\n"
      << "    >);\n"
      << "};\n";
  return code.str();
}

std::string sm100_paged_mqa_logits_code(
    bool is_fp4,
    int tokens_per_request,
    int num_heads,
    int head_dim,
    int page_kv,
    bool is_context_lens_2d,
    bool is_varlen,
    int num_q_stages,
    int num_kv_stages,
    int split_kv,
    int splits_per_chunk,
    int num_specialized_threads,
    int num_math_threads,
    deepgemm_dtype_t logits_dtype,
    deepgemm_dtype_t weights_dtype) {
  std::ostringstream code;
  code
      << "#include <deep_gemm/impls/sm100_mqa_logits.cuh>\n\n"
      << "using namespace deep_gemm;\n\n"
      << "static void __instantiate_kernel() {\n"
      << "    auto ptr = reinterpret_cast<void*>(&sm100_paged_mqa_logits<\n"
      << "        " << bool_literal(is_fp4) << ",\n"
      << "        " << tokens_per_request << ", " << num_heads << ",\n"
      << "        " << head_dim << ", " << page_kv << ",\n"
      << "        " << bool_literal(is_context_lens_2d) << ", "
      << bool_literal(is_varlen) << ",\n"
      << "        " << num_q_stages << ", " << num_kv_stages << ",\n"
      << "        " << split_kv << ", " << splits_per_chunk << ",\n"
      << "        " << num_specialized_threads << ", " << num_math_threads << ",\n"
      << "        " << kernel_dtype_name(logits_dtype) << ",\n"
      << "        " << (weights_dtype == DEEPGEMM_DTYPE_BF16 ? "cutlass::bfloat16_t" : "float") << "\n"
      << "    >);\n"
      << "};\n";
  return code.str();
}

int64_t paged_logits_stride(int64_t max_context_len, deepgemm_dtype_t dtype) {
  const int64_t elem_size = dtype_element_size(dtype);
  if (elem_size <= 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "invalid logits dtype");
  }
  constexpr int64_t split_kv = 256;
  const int64_t stride_alignment = 1024 / elem_size;
  return align_i64(align_i64(max_context_len, split_kv), stride_alignment);
}

const std::uint8_t* byte_ptr(const void* ptr) {
  return reinterpret_cast<const std::uint8_t*>(ptr);
}

void launch_clean_logits(
    const deepgemm_mqa_logits_params_t& params,
    int seq_len,
    int seq_len_kv,
    int64_t stride_logits,
    int num_sms) {
  constexpr int next_n = 1;
  constexpr int block_kv = 8192;
  constexpr int num_warps = 8;

  const auto runtime = build_kernel(
      "smxx_clean_logits",
      clean_logits_code(next_n, block_kv, num_warps, params.logits.dtype));

  const uint32_t seq_len_arg = static_cast<uint32_t>(seq_len);
  const uint32_t seq_len_kv_arg = static_cast<uint32_t>(seq_len_kv);
  const uint64_t stride_logits_arg = static_cast<uint64_t>(stride_logits);
  const auto* cu_seq_len_k_start = reinterpret_cast<const uint32_t*>(params.cu_seq_len_k_start.data);
  const auto* cu_seq_len_k_end = reinterpret_cast<const uint32_t*>(params.cu_seq_len_k_end.data);
  auto* logits = params.logits.data;

  LaunchArgs launch_args;
  launch_args.grid_x = num_sms;
  launch_args.num_threads = num_warps * 32;
  launch_args.smem_size = block_kv * static_cast<int>(sizeof(float));
  launch_args.enable_pdl = pdl_enabled();

  launch_kernel(
      runtime,
      reinterpret_cast<CUstream>(params.stream),
      launch_args,
      seq_len_arg,
      seq_len_kv_arg,
      stride_logits_arg,
      cu_seq_len_k_start,
      cu_seq_len_k_end,
      logits);
}

void launch_sm90_fp8_mqa_logits(
    const deepgemm_mqa_logits_params_t& params,
    int seq_len,
    int seq_len_kv,
    int num_heads,
    int head_dim,
    int block_q,
    int block_kv,
    int64_t stride_logits,
    int num_sms) {
  constexpr int num_specialized_threads = 128;
  constexpr int num_q_stages = 3;
  constexpr int num_kv_stages = 3;
  constexpr int num_math_threads = 512;
  const bool is_compressed_logits = params.max_seqlen_k > 0;

  const auto tensor_map_q = make_tma_2d_desc(
      params.q.data,
      params.q.dtype,
      head_dim,
      seq_len * num_heads,
      head_dim,
      block_q * num_heads,
      head_dim,
      head_dim);
  const auto tensor_map_kv = make_tma_2d_desc(
      params.kv.data,
      params.kv.dtype,
      head_dim,
      seq_len_kv,
      head_dim,
      block_kv,
      head_dim,
      head_dim);
  const auto tensor_map_kv_scales = make_tma_2d_desc(
      params.kv_scale.data,
      params.kv_scale.dtype,
      as_i32(get_tma_aligned_size(seq_len_kv, dtype_element_size(params.kv_scale.dtype)), "kv_scale aligned size"),
      1,
      block_kv,
      1,
      0,
      0);
  const auto tensor_map_weights = make_tma_2d_desc(
      params.weights.data,
      params.weights.dtype,
      num_heads,
      seq_len,
      num_heads,
      block_q,
      as_i32(params.weights.stride[0], "weights stride"),
      0);

  const int smem_q_size_per_stage = block_q * num_heads * head_dim * static_cast<int>(dtype_element_size(params.q.dtype));
  const int smem_weight_size_per_stage = block_q * num_heads * static_cast<int>(dtype_element_size(params.weights.dtype));
  const int smem_kv_size_per_stage = block_kv * head_dim * static_cast<int>(dtype_element_size(params.kv.dtype));
  const int kv_scale_size_per_stage = block_kv * static_cast<int>(dtype_element_size(params.kv_scale.dtype));
  int smem_size = 0;
  smem_size += num_q_stages * smem_q_size_per_stage;
  smem_size += num_kv_stages * smem_kv_size_per_stage;
  smem_size += num_q_stages * smem_weight_size_per_stage;
  smem_size += num_kv_stages * kv_scale_size_per_stage;
  smem_size += (num_q_stages * 2 + num_kv_stages * 2 + (num_math_threads / 128) * 2) * 8;
  smem_size += 4;

  const auto runtime = build_kernel(
      "sm90_fp8_mqa_logits",
      sm90_mqa_logits_code(
          num_heads,
          head_dim,
          is_compressed_logits,
          block_q,
          block_kv,
          num_q_stages,
          num_kv_stages,
          num_sms,
          num_specialized_threads,
          num_math_threads,
          params.logits.dtype));

  const uint32_t seq_len_arg = static_cast<uint32_t>(seq_len);
  const uint32_t seq_len_kv_arg = static_cast<uint32_t>(seq_len_kv);
  const uint32_t max_seqlen_k_arg = static_cast<uint32_t>(params.max_seqlen_k);
  const uint32_t stride_logits_arg = static_cast<uint32_t>(stride_logits);
  auto* cu_seq_len_k_start = const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(params.cu_seq_len_k_start.data));
  auto* cu_seq_len_k_end = const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(params.cu_seq_len_k_end.data));
  auto* logits = params.logits.data;

  LaunchArgs launch_args;
  launch_args.grid_x = num_sms;
  launch_args.num_threads = num_specialized_threads + num_math_threads;
  launch_args.smem_size = smem_size;
  launch_args.enable_pdl = pdl_enabled();

  launch_kernel(
      runtime,
      reinterpret_cast<CUstream>(params.stream),
      launch_args,
      seq_len_arg,
      seq_len_kv_arg,
      max_seqlen_k_arg,
      stride_logits_arg,
      cu_seq_len_k_start,
      cu_seq_len_k_end,
      logits,
      tensor_map_q,
      tensor_map_kv,
      tensor_map_kv_scales,
      tensor_map_weights);
}

void launch_sm100_mqa_logits(
    const deepgemm_mqa_logits_params_t& params,
    bool is_fp4,
    int seq_len,
    int seq_len_kv,
    int num_heads,
    int head_dim,
    int block_q,
    int split_kv,
    int64_t stride_logits,
    int num_sms) {
  constexpr int num_specialized_threads = 128;
  constexpr int num_math_threads = 256;
  constexpr int num_q_stages = 3;
  const int num_kv_stages = is_fp4 ? 10 : 5;
  const bool is_compressed_logits = params.max_seqlen_k > 0;

  CUtensorMap tensor_map_q{};
  CUtensorMap tensor_map_sf_q{};
  CUtensorMap tensor_map_kv{};
  CUtensorMap tensor_map_sf_kv{};

  if (is_fp4) {
    tensor_map_q = make_tma_2d_desc(
        params.q.data,
        params.q.dtype,
        head_dim,
        seq_len * num_heads,
        head_dim,
        block_q * num_heads,
        as_i32(params.q.stride[1], "q stride(1)"),
        head_dim / 2,
        0,
        false,
        false);
    tensor_map_sf_q = make_tma_2d_desc(
        params.q_scale.data,
        params.q_scale.dtype,
        num_heads,
        seq_len,
        num_heads,
        block_q,
        as_i32(params.q_scale.stride[0], "q_scale stride(0)"),
        0);
    tensor_map_kv = make_tma_2d_desc(
        params.kv.data,
        params.kv.dtype,
        head_dim,
        seq_len_kv,
        head_dim,
        split_kv,
        as_i32(params.kv.stride[0], "kv stride(0)"),
        head_dim / 2,
        0,
        false,
        false);
    tensor_map_sf_kv = make_tma_2d_desc(
        params.kv_scale.data,
        params.kv_scale.dtype,
        as_i32(get_tma_aligned_size(seq_len_kv, dtype_element_size(params.kv_scale.dtype)), "kv_scale aligned size"),
        1,
        split_kv,
        1,
        0,
        0);
  } else {
    tensor_map_q = make_tma_2d_desc(
        params.q.data,
        params.q.dtype,
        head_dim,
        seq_len * num_heads,
        head_dim,
        block_q * num_heads,
        as_i32(params.q.stride[1], "q stride(1)"),
        head_dim);
    tensor_map_kv = make_tma_2d_desc(
        params.kv.data,
        params.kv.dtype,
        head_dim,
        seq_len_kv,
        head_dim,
        split_kv,
        as_i32(params.kv.stride[0], "kv stride(0)"),
        head_dim);
    tensor_map_sf_kv = make_tma_2d_desc(
        params.kv_scale.data,
        params.kv_scale.dtype,
        as_i32(get_tma_aligned_size(seq_len_kv, dtype_element_size(params.kv_scale.dtype)), "kv_scale aligned size"),
        1,
        split_kv,
        1,
        0,
        0);
    tensor_map_sf_q = tensor_map_sf_kv;
  }

  const auto tensor_map_weights = make_tma_2d_desc(
      params.weights.data,
      params.weights.dtype,
      num_heads,
      seq_len,
      num_heads,
      block_q,
      as_i32(params.weights.stride[0], "weights stride"),
      0);

  const auto runtime = build_kernel(
      "sm100_mqa_logits",
      sm100_mqa_logits_code(
          is_fp4,
          num_heads,
          head_dim,
          is_compressed_logits,
          block_q,
          split_kv,
          num_q_stages,
          num_kv_stages,
          num_sms,
          num_specialized_threads,
          num_math_threads,
          params.logits.dtype,
          params.weights.dtype));

  const uint32_t seq_len_arg = static_cast<uint32_t>(seq_len);
  const uint32_t seq_len_kv_arg = static_cast<uint32_t>(seq_len_kv);
  const uint32_t stride_logits_arg = static_cast<uint32_t>(stride_logits);
  auto* cu_seq_len_k_start = const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(params.cu_seq_len_k_start.data));
  auto* cu_seq_len_k_end = const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(params.cu_seq_len_k_end.data));
  auto* logits = params.logits.data;

  LaunchArgs launch_args;
  launch_args.grid_x = num_sms;
  launch_args.num_threads = num_specialized_threads + num_math_threads;
  launch_args.smem_size = 232448;
  launch_args.enable_pdl = pdl_enabled();

  launch_kernel(
      runtime,
      reinterpret_cast<CUstream>(params.stream),
      launch_args,
      seq_len_arg,
      seq_len_kv_arg,
      stride_logits_arg,
      cu_seq_len_k_start,
      cu_seq_len_k_end,
      logits,
      tensor_map_q,
      tensor_map_sf_q,
      tensor_map_kv,
      tensor_map_sf_kv,
      tensor_map_weights);
}

void launch_sm90_metadata(
    const deepgemm_paged_mqa_logits_metadata_params_t& params,
    int batch_size,
    int next_n,
    int num_sms,
    bool is_varlen) {
  constexpr int split_kv = 256;
  constexpr int num_threads = 32;
  const int aligned_batch_size = as_i32(align_i64_or_throw(batch_size, 32, "batch_size"), "aligned_batch_size");
  const int num_smem_ints = is_varlen ? 3 * aligned_batch_size + 1 : aligned_batch_size;

  const auto runtime = build_kernel(
      "sm90_paged_mqa_logits_metadata",
      sm90_metadata_code(aligned_batch_size, split_kv, num_sms, is_varlen));

  const uint32_t batch_size_arg = static_cast<uint32_t>(batch_size);
  const uint32_t next_n_arg = static_cast<uint32_t>(next_n);
  const bool is_context_lens_2d = true;
  const auto* context_lens = reinterpret_cast<const uint32_t*>(params.context_lens.data);
  const auto* indices = reinterpret_cast<const uint32_t*>(params.indices.data);
  auto* schedule_meta = reinterpret_cast<uint32_t*>(params.schedule_meta.data);
  LaunchArgs launch_args;
  launch_args.grid_x = 1;
  launch_args.num_threads = num_threads;
  launch_args.smem_size = num_smem_ints * static_cast<int>(sizeof(uint32_t));
  launch_args.enable_pdl = pdl_enabled();

  launch_kernel(
      runtime,
      reinterpret_cast<CUstream>(params.stream),
      launch_args,
      batch_size_arg,
      next_n_arg,
      is_context_lens_2d,
      context_lens,
      indices,
      schedule_meta);
}

void launch_sm100_metadata(
    const deepgemm_paged_mqa_logits_metadata_params_t& params,
    int batch_size,
    int next_n,
    int num_sms,
    bool is_varlen) {
  constexpr int split_kv = 256;
  constexpr int num_threads = 256;

  const auto runtime = build_kernel(
      "sm100_paged_mqa_logits_metadata",
      sm100_metadata_code(next_n, true, is_varlen, split_kv, num_sms));

  const uint32_t num_requests = static_cast<uint32_t>(batch_size);
  const uint32_t num_q_tokens_total = static_cast<uint32_t>(batch_size * next_n);
  const auto* context_lens = reinterpret_cast<const uint32_t*>(params.context_lens.data);
  const auto* indices = reinterpret_cast<const uint32_t*>(params.indices.data);
  auto* schedule_meta = reinterpret_cast<uint32_t*>(params.schedule_meta.data);
  LaunchArgs launch_args;
  launch_args.grid_x = 1;
  launch_args.num_threads = num_threads;
  launch_args.smem_size = 2 * batch_size * static_cast<int>(sizeof(uint32_t));
  launch_args.enable_pdl = pdl_enabled();

  launch_kernel(
      runtime,
      reinterpret_cast<CUstream>(params.stream),
      launch_args,
      num_requests,
      num_q_tokens_total,
      context_lens,
      indices,
      schedule_meta);
}

void launch_sm90_fp8_paged_mqa_logits(
    const deepgemm_paged_mqa_logits_params_t& params,
    int batch_size,
    int next_n,
    int num_heads,
    int head_dim,
    int num_kv_blocks,
    int block_kv,
    int64_t stride_logits,
    int block_table_stride,
    int num_sms) {
  constexpr int num_specialized_threads = 128;
  constexpr int mma_m = 64;
  constexpr int split_kv = 256;
  constexpr int num_q_stages = 3;
  constexpr int num_kv_stages = 3;
  const int num_math_warp_groups = split_kv / mma_m;
  const int num_math_threads = num_math_warp_groups * 128;
  const bool is_context_lens_2d = true;
  const bool is_varlen = false;
  const int next_n_atom = next_n >= 2 ? 2 : 1;

  const int kv_cache_stride_bytes = as_i32(params.fused_kv_cache.stride[0], "fused_kv_cache stride(0)");
  const auto* kv_scale_data = byte_ptr(params.fused_kv_cache.data) + block_kv * head_dim;

  const auto tensor_map_q = make_tma_2d_desc(
      params.q.data,
      params.q.dtype,
      head_dim,
      batch_size * next_n * num_heads,
      head_dim,
      next_n_atom * num_heads,
      as_i32(params.q.stride[2], "q stride(2)"),
      head_dim);
  const auto tensor_map_kv = make_tma_3d_desc(
      params.fused_kv_cache.data,
      DEEPGEMM_DTYPE_FP8_E4M3,
      head_dim,
      block_kv,
      num_kv_blocks,
      head_dim,
      block_kv,
      1,
      head_dim,
      kv_cache_stride_bytes,
      head_dim);
  const auto tensor_map_kv_scales = make_tma_2d_desc(
      kv_scale_data,
      DEEPGEMM_DTYPE_F32,
      block_kv,
      num_kv_blocks,
      block_kv,
      1,
      kv_cache_stride_bytes / static_cast<int>(sizeof(float)),
      0);
  const auto tensor_map_weights = make_tma_2d_desc(
      params.weights.data,
      params.weights.dtype,
      num_heads,
      batch_size * next_n,
      num_heads,
      next_n_atom,
      as_i32(params.weights.stride[0], "weights stride(0)"),
      0);

  const int swizzle_alignment = head_dim * 8;
  const int smem_q_size_per_stage =
      next_n * num_heads * head_dim * static_cast<int>(dtype_element_size(params.q.dtype));
  const int aligned_smem_weight_size_per_stage = as_i32(
      align_i64(
          next_n * num_heads * static_cast<int>(dtype_element_size(params.weights.dtype)),
          swizzle_alignment),
      "aligned SM90 paged weights smem size");
  const int smem_q_pipe_size =
      num_q_stages * (smem_q_size_per_stage + aligned_smem_weight_size_per_stage) +
      as_i32(align_i64(num_q_stages * 8 * 2, swizzle_alignment), "SM90 paged q barrier smem size");
  const int smem_kv_size_per_stage =
      block_kv * head_dim * static_cast<int>(dtype_element_size(DEEPGEMM_DTYPE_FP8_E4M3));
  const int aligned_smem_kv_scale_size_per_stage = as_i32(
      align_i64(block_kv * static_cast<int>(sizeof(float)), swizzle_alignment),
      "aligned SM90 paged kv scale smem size");
  const int smem_kv_pipe_size =
      num_kv_stages * (smem_kv_size_per_stage + aligned_smem_kv_scale_size_per_stage) +
      as_i32(align_i64(num_kv_stages * 8 * 2, swizzle_alignment), "SM90 paged kv barrier smem size");
  const int smem_umma_barriers = num_math_warp_groups * 2 * 8;
  const int smem_tmem_ptr = 4;
  const int smem_size =
      smem_q_pipe_size + num_math_warp_groups * smem_kv_pipe_size + smem_umma_barriers + smem_tmem_ptr;

  const auto runtime = build_kernel(
      "sm90_fp8_paged_mqa_logits",
      sm90_paged_mqa_logits_code(
          next_n,
          num_heads,
          head_dim,
          block_kv,
          is_context_lens_2d,
          is_varlen,
          num_q_stages,
          num_kv_stages,
          split_kv,
          num_specialized_threads,
          num_math_threads,
          params.logits.dtype));

  const uint32_t batch_size_arg = static_cast<uint32_t>(batch_size);
  const uint32_t stride_logits_arg = static_cast<uint32_t>(stride_logits);
  const uint32_t block_table_stride_arg = static_cast<uint32_t>(block_table_stride);
  const auto* context_lens = reinterpret_cast<const uint32_t*>(params.context_lens.data);
  auto* logits = params.logits.data;
  const auto* block_table = reinterpret_cast<const uint32_t*>(params.block_table.data);
  const uint32_t* indices = nullptr;
  const auto* schedule_meta = reinterpret_cast<const uint32_t*>(params.schedule_meta.data);

  LaunchArgs launch_args;
  launch_args.grid_x = num_sms;
  launch_args.num_threads = num_specialized_threads + num_math_threads;
  launch_args.smem_size = smem_size;
  launch_args.enable_pdl = pdl_enabled();

  launch_kernel(
      runtime,
      reinterpret_cast<CUstream>(params.stream),
      launch_args,
      batch_size_arg,
      stride_logits_arg,
      block_table_stride_arg,
      context_lens,
      logits,
      block_table,
      indices,
      schedule_meta,
      tensor_map_q,
      tensor_map_kv,
      tensor_map_kv_scales,
      tensor_map_weights);
}

void launch_sm100_paged_mqa_logits(
    const deepgemm_paged_mqa_logits_params_t& params,
    bool is_fp4,
    int batch_size,
    int next_n,
    int num_heads,
    int head_dim,
    int num_kv_blocks,
    int page_kv,
    int64_t stride_logits,
    int block_table_stride,
    int num_sms) {
  constexpr int num_specialized_threads = 128;
  constexpr int num_math_threads = 256;
  constexpr int num_q_stages = 3;
  constexpr int split_kv = 256;
  constexpr int splits_per_chunk = 16;
  const int num_kv_stages = is_fp4 ? 10 : 5;
  const bool is_context_lens_2d = true;
  const bool is_varlen = params.has_indices;
  const int block_q = 128 / num_heads;
  const int num_q_tokens_total = batch_size * next_n;
  const int kv_bytes_per_token = is_fp4 ? head_dim / 2 : head_dim;
  const int kv_cache_stride_bytes = as_i32(params.fused_kv_cache.stride[0], "fused_kv_cache stride(0)");
  const auto* kv_scale_data = byte_ptr(params.fused_kv_cache.data) + page_kv * kv_bytes_per_token;
  const deepgemm_dtype_t kv_dtype =
      is_fp4 ? DEEPGEMM_DTYPE_PACKED_FP4_E2M1 : DEEPGEMM_DTYPE_FP8_E4M3;
  const deepgemm_dtype_t kv_scale_dtype =
      is_fp4 ? DEEPGEMM_DTYPE_PACKED_UE8M0 : DEEPGEMM_DTYPE_F32;

  CUtensorMap tensor_map_q{};
  CUtensorMap tensor_map_sf_q{};
  CUtensorMap tensor_map_kv{};
  CUtensorMap tensor_map_sf_kv{};

  if (is_fp4) {
    tensor_map_q = make_tma_2d_desc(
        params.q.data,
        params.q.dtype,
        head_dim,
        num_q_tokens_total * num_heads,
        head_dim,
        block_q * num_heads,
        as_i32(params.q.stride[2], "q stride(2)"),
        head_dim / 2,
        0,
        false,
        false);
    tensor_map_sf_q = make_tma_2d_desc(
        params.q_scale.data,
        params.q_scale.dtype,
        num_heads,
        num_q_tokens_total,
        num_heads,
        block_q,
        as_i32(params.q_scale.stride[1], "q_scale stride(1)"),
        0);
    tensor_map_kv = make_tma_3d_desc(
        params.fused_kv_cache.data,
        kv_dtype,
        head_dim,
        page_kv,
        num_kv_blocks,
        head_dim,
        page_kv,
        1,
        kv_bytes_per_token,
        kv_cache_stride_bytes,
        head_dim / 2,
        0,
        false,
        false);
  } else {
    tensor_map_q = make_tma_2d_desc(
        params.q.data,
        params.q.dtype,
        head_dim,
        num_q_tokens_total * num_heads,
        head_dim,
        block_q * num_heads,
        as_i32(params.q.stride[2], "q stride(2)"),
        head_dim);
    tensor_map_kv = make_tma_3d_desc(
        params.fused_kv_cache.data,
        kv_dtype,
        head_dim,
        page_kv,
        num_kv_blocks,
        head_dim,
        page_kv,
        1,
        kv_bytes_per_token,
        kv_cache_stride_bytes,
        head_dim);
  }

  tensor_map_sf_kv = make_tma_2d_desc(
      kv_scale_data,
      kv_scale_dtype,
      page_kv,
      num_kv_blocks,
      page_kv,
      1,
      kv_cache_stride_bytes / static_cast<int>(sizeof(int)),
      0);
  if (!is_fp4) {
    tensor_map_sf_q = tensor_map_sf_kv;
  }
  const auto tensor_map_weights = make_tma_2d_desc(
      params.weights.data,
      params.weights.dtype,
      num_heads,
      num_q_tokens_total,
      num_heads,
      block_q,
      as_i32(params.weights.stride[0], "weights stride(0)"),
      0);

  const auto runtime = build_kernel(
      "sm100_paged_mqa_logits",
      sm100_paged_mqa_logits_code(
          is_fp4,
          next_n,
          num_heads,
          head_dim,
          page_kv,
          is_context_lens_2d,
          is_varlen,
          num_q_stages,
          num_kv_stages,
          split_kv,
          splits_per_chunk,
          num_specialized_threads,
          num_math_threads,
          params.logits.dtype,
          params.weights.dtype));

  const uint32_t num_q_tokens_total_arg = static_cast<uint32_t>(num_q_tokens_total);
  const uint32_t stride_logits_arg = static_cast<uint32_t>(stride_logits);
  const uint32_t block_table_stride_arg = static_cast<uint32_t>(block_table_stride);
  const auto* context_lens = reinterpret_cast<const uint32_t*>(params.context_lens.data);
  auto* logits = params.logits.data;
  const auto* block_table = reinterpret_cast<const uint32_t*>(params.block_table.data);
  const auto* indices = is_varlen ? reinterpret_cast<const uint32_t*>(params.indices.data) : nullptr;
  const auto* schedule_meta = reinterpret_cast<const uint32_t*>(params.schedule_meta.data);

  LaunchArgs launch_args;
  launch_args.grid_x = num_sms;
  launch_args.num_threads = num_specialized_threads + num_math_threads;
  launch_args.smem_size = 232448;
  launch_args.enable_pdl = pdl_enabled();

  launch_kernel(
      runtime,
      reinterpret_cast<CUstream>(params.stream),
      launch_args,
      num_q_tokens_total_arg,
      stride_logits_arg,
      block_table_stride_arg,
      context_lens,
      logits,
      block_table,
      indices,
      schedule_meta,
      tensor_map_q,
      tensor_map_sf_q,
      tensor_map_kv,
      tensor_map_sf_kv,
      tensor_map_weights);
}

}  // namespace

void launch_fp8_fp4_mqa_logits(
    const deepgemm_mqa_logits_params_t& params) {
  const auto device = current_device_info();
  const bool is_fp4 = params.has_q_scale;

  require_contiguous(params.q, 3, "q");
  require_contiguous(params.kv, 2, "kv");
  const int64_t seq_len_i64 = params.q.shape[0];
  const int64_t num_heads_i64 = params.q.shape[1];
  const int64_t physical_head_dim_i64 = params.q.shape[2];
  const int64_t head_dim_i64 = is_fp4 ? physical_head_dim_i64 * 2 : physical_head_dim_i64;
  const int64_t seq_len_kv_i64 = params.kv.shape[0];
  if (params.kv.shape[1] != (is_fp4 ? head_dim_i64 / 2 : head_dim_i64)) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "kv head dimension does not match q");
  }

  if (is_fp4) {
    require_dtype(params.q.dtype, DEEPGEMM_DTYPE_PACKED_FP4_E2M1, "q");
    require_dtype(params.kv.dtype, DEEPGEMM_DTYPE_PACKED_FP4_E2M1, "kv");
    require_contiguous(params.q_scale, 2, "q_scale");
    require_dtype(params.q_scale.dtype, DEEPGEMM_DTYPE_PACKED_UE8M0, "q_scale");
    if (params.q_scale.shape[0] != seq_len_i64 || params.q_scale.shape[1] != num_heads_i64) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "q_scale shape must be [seq_len, num_heads]");
    }
    require_dtype(params.kv_scale.dtype, DEEPGEMM_DTYPE_PACKED_UE8M0, "kv_scale");
  } else {
    require_dtype(params.q.dtype, DEEPGEMM_DTYPE_FP8_E4M3, "q");
    require_dtype(params.kv.dtype, DEEPGEMM_DTYPE_FP8_E4M3, "kv");
    require_dtype(params.kv_scale.dtype, DEEPGEMM_DTYPE_F32, "kv_scale");
  }

  require_contiguous(params.kv_scale, 1, "kv_scale");
  if (params.kv_scale.shape[0] != seq_len_kv_i64) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "kv_scale shape must be [seq_len_kv]");
  }

  require_tensor_rank(params.weights, 2, "weights");
  require_shape_positive(params.weights.shape, 2, "weights");
  if (params.weights.shape[0] != seq_len_i64 || params.weights.shape[1] != num_heads_i64 ||
      params.weights.stride[1] != 1) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "weights must be [seq_len, num_heads] with stride(1) == 1");
  }
  if (params.weights.dtype != DEEPGEMM_DTYPE_F32 && params.weights.dtype != DEEPGEMM_DTYPE_BF16) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "weights dtype must be f32 or bf16");
  }

  require_i32_vector(params.cu_seq_len_k_start, seq_len_i64, "cu_seq_len_k_start");
  require_i32_vector(params.cu_seq_len_k_end, seq_len_i64, "cu_seq_len_k_end");

  require_logits_dtype(params.logits.dtype);
  if (params.clean_logits && params.max_seqlen_k > 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "clean_logits is not supported with compressed logits");
  }

  const int seq_len = as_i32(seq_len_i64, "seq_len");
  const int seq_len_kv = as_i32(seq_len_kv_i64, "seq_len_kv");
  const int num_heads = as_i32(num_heads_i64, "num_heads");
  const int head_dim = as_i32(head_dim_i64, "head_dim");
  const int max_seqlen_k = as_i32(params.max_seqlen_k, "max_seqlen_k");
  constexpr int block_qh = 128;
  constexpr int block_kv = 256;
  if (block_qh % num_heads != 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "num_heads must divide 128");
  }
  const int block_q = block_qh / num_heads;
  const int64_t stride_logits = mqa_logits_stride(seq_len_kv, max_seqlen_k, block_kv, params.logits.dtype);
  const int64_t logits_cols = max_seqlen_k == 0 ? seq_len_kv_i64 : params.max_seqlen_k;
  require_logits_output(params.logits, seq_len_i64, logits_cols, stride_logits, "logits");

  if (device.major == 9) {
    if (is_fp4) {
      throw_status(DEEPGEMM_STATUS_UNSUPPORTED_ARCH, "FP4 MQA logits require SM100");
    }
    if (params.weights.dtype != DEEPGEMM_DTYPE_F32) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM90 MQA logits require f32 weights");
    }
    if ((num_heads != 32 && num_heads != 64) ||
        (head_dim != 32 && head_dim != 64 && head_dim != 128)) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "unsupported SM90 MQA logits shape");
    }
    const int num_sms = effective_num_sms();
    launch_sm90_fp8_mqa_logits(
        params,
        seq_len,
        seq_len_kv,
        num_heads,
        head_dim,
        block_q,
        block_kv,
        stride_logits,
        num_sms);
    if (params.clean_logits) {
      launch_clean_logits(params, seq_len, seq_len_kv, stride_logits, num_sms);
    }
    return;
  }

  if (device.major == 10) {
    if (params.weights.dtype == DEEPGEMM_DTYPE_BF16 && params.logits.dtype != DEEPGEMM_DTYPE_BF16) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "bf16 weights require bf16 logits");
    }
    if (num_heads != 8 && num_heads != 16 && num_heads != 32 && num_heads != 64) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM100 MQA logits require 8, 16, 32, or 64 heads");
    }
    if (is_fp4) {
      if (head_dim != 64 && head_dim != 128) {
        throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM100 FP4 MQA logits require head_dim 64 or 128");
      }
    } else if (head_dim != 32 && head_dim != 64 && head_dim != 128) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM100 FP8 MQA logits require head_dim 32, 64, or 128");
    }
    const int num_sms = effective_num_sms();
    launch_sm100_mqa_logits(
        params,
        is_fp4,
        seq_len,
        seq_len_kv,
        num_heads,
        head_dim,
        block_q,
        block_kv,
        stride_logits,
        num_sms);
    if (params.clean_logits) {
      launch_clean_logits(params, seq_len, seq_len_kv, stride_logits, num_sms);
    }
    return;
  }

  std::ostringstream message;
  message << "MQA logits supports SM90 or SM100, got compute capability "
          << device.major << "." << device.minor;
  throw_status(DEEPGEMM_STATUS_UNSUPPORTED_ARCH, message.str());
}

void launch_paged_mqa_logits_metadata(
    const deepgemm_paged_mqa_logits_metadata_params_t& params) {
  require_contiguous_2d_i32(params.context_lens, "context_lens");
  const int64_t batch_size_i64 = params.context_lens.shape[0];
  const int64_t next_n_i64 = params.context_lens.shape[1];
  const int64_t num_sms_i64 = params.num_sms;
  if (num_sms_i64 <= 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "num_sms must be positive");
  }
  require_schedule_meta(params.schedule_meta, num_sms_i64);

  const bool is_varlen = params.has_indices;
  if (is_varlen) {
    require_contiguous_1d_i32(params.indices, batch_size_i64, "indices");
  }

  const int batch_size = as_i32(batch_size_i64, "batch_size");
  const int next_n = as_i32(next_n_i64, "next_n");
  const int num_sms = as_i32(num_sms_i64, "num_sms");
  const int block_kv = as_i32(params.block_kv, "block_kv");

  const auto device = current_device_info();
  if (num_sms > device.num_sms) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "num_sms exceeds current device SM count");
  }

  if (device.major == 9) {
    if (is_varlen) {
      throw_status(DEEPGEMM_STATUS_UNSUPPORTED_ARCH, "SM90 paged metadata does not support varlen indices");
    }
    if (block_kv != 64) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM90 paged metadata requires block_kv == 64");
    }
    launch_sm90_metadata(params, batch_size, next_n, num_sms, false);
    return;
  }

  if (device.major == 10) {
    if (block_kv != 32 && block_kv != 64) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM100 paged metadata requires block_kv == 32 or 64");
    }
    if (is_varlen && next_n != 1) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM100 varlen paged metadata requires next_n == 1");
    }
    launch_sm100_metadata(params, batch_size, next_n, num_sms, is_varlen);
    return;
  }

  std::ostringstream message;
  message << "paged MQA metadata supports SM90 or SM100, got compute capability "
          << device.major << "." << device.minor;
  throw_status(DEEPGEMM_STATUS_UNSUPPORTED_ARCH, message.str());
}

void launch_fp8_fp4_paged_mqa_logits(
    const deepgemm_paged_mqa_logits_params_t& params) {
  const auto device = current_device_info();
  const bool is_fp4 = params.has_q_scale;

  require_contiguous(params.q, 4, "q");
  const int64_t batch_size_i64 = params.q.shape[0];
  const int64_t next_n_i64 = params.q.shape[1];
  const int64_t num_heads_i64 = params.q.shape[2];
  const int64_t physical_head_dim_i64 = params.q.shape[3];
  const int64_t head_dim_i64 = is_fp4 ? physical_head_dim_i64 * 2 : physical_head_dim_i64;

  if (is_fp4) {
    require_dtype(params.q.dtype, DEEPGEMM_DTYPE_PACKED_FP4_E2M1, "q");
    require_contiguous(params.q_scale, 3, "q_scale");
    require_dtype(params.q_scale.dtype, DEEPGEMM_DTYPE_PACKED_UE8M0, "q_scale");
    if (params.q_scale.shape[0] != batch_size_i64 ||
        params.q_scale.shape[1] != next_n_i64 ||
        params.q_scale.shape[2] != num_heads_i64) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "q_scale shape must be [batch_size, next_n, num_heads]");
    }
  } else {
    require_dtype(params.q.dtype, DEEPGEMM_DTYPE_FP8_E4M3, "q");
  }

  require_tensor_rank(params.fused_kv_cache, 4, "fused_kv_cache");
  require_dtype(params.fused_kv_cache.dtype, DEEPGEMM_DTYPE_U8, "fused_kv_cache");
  require_shape_positive(params.fused_kv_cache.shape, 4, "fused_kv_cache");
  const int64_t num_kv_blocks_i64 = params.fused_kv_cache.shape[0];
  const int64_t block_kv_i64 = params.fused_kv_cache.shape[1];
  const int64_t kv_bytes_per_token_i64 = is_fp4 ? head_dim_i64 / 2 : head_dim_i64;
  const int64_t fused_bytes_per_token_i64 = kv_bytes_per_token_i64 + static_cast<int64_t>(sizeof(int));
  if (params.fused_kv_cache.shape[2] != 1 ||
      params.fused_kv_cache.shape[3] != fused_bytes_per_token_i64) {
    throw_status(
        DEEPGEMM_STATUS_INVALID_ARGUMENT,
        "fused_kv_cache shape must be [num_kv_blocks, block_kv, 1, packed_head_dim + 4]");
  }
  if (params.fused_kv_cache.stride[0] != block_kv_i64 * fused_bytes_per_token_i64 ||
      params.fused_kv_cache.stride[1] != fused_bytes_per_token_i64 ||
      params.fused_kv_cache.stride[2] != fused_bytes_per_token_i64 ||
      params.fused_kv_cache.stride[3] != 1) {
    throw_status(
        DEEPGEMM_STATUS_INVALID_ARGUMENT,
        "fused_kv_cache must be contiguous [num_kv_blocks, block_kv, 1, fused_bytes]");
  }
  if (params.fused_kv_cache.stride[0] % static_cast<int64_t>(sizeof(int)) != 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "fused_kv_cache block stride must be 4-byte aligned");
  }

  require_tensor_rank(params.weights, 2, "weights");
  require_shape_positive(params.weights.shape, 2, "weights");
  if (params.weights.shape[0] != batch_size_i64 * next_n_i64 ||
      params.weights.shape[1] != num_heads_i64 ||
      params.weights.stride[1] != 1) {
    throw_status(
        DEEPGEMM_STATUS_INVALID_ARGUMENT,
        "weights must be [batch_size * next_n, num_heads] with stride(1) == 1");
  }
  if (!is_fp4 && params.weights.stride[0] != num_heads_i64) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "FP8 paged MQA weights must be contiguous");
  }
  if (params.weights.dtype != DEEPGEMM_DTYPE_F32 && params.weights.dtype != DEEPGEMM_DTYPE_BF16) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "weights dtype must be f32 or bf16");
  }

  require_contiguous_2d_i32(params.context_lens, "context_lens");
  if (params.context_lens.shape[0] != batch_size_i64 ||
      params.context_lens.shape[1] != next_n_i64) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "context_lens shape must be [batch_size, next_n]");
  }

  require_tensor_rank(params.block_table, 2, "block_table");
  require_dtype(params.block_table.dtype, DEEPGEMM_DTYPE_I32, "block_table");
  require_shape_positive(params.block_table.shape, 2, "block_table");
  if (params.block_table.shape[0] != batch_size_i64 || params.block_table.stride[1] != 1) {
    throw_status(
        DEEPGEMM_STATUS_INVALID_ARGUMENT,
        "block_table must be [batch_size, max_block_len] with stride(1) == 1");
  }

  const int64_t num_sms_i64 = params.num_sms;
  if (num_sms_i64 <= 0 || num_sms_i64 > device.num_sms) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "num_sms must be positive and no larger than the device SM count");
  }
  require_contiguous_2d_i32(params.schedule_meta, "schedule_meta");
  if (params.schedule_meta.shape[0] != num_sms_i64 + 1 ||
      params.schedule_meta.shape[1] != 2) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "schedule_meta shape must be [num_sms + 1, 2]");
  }

  const bool is_varlen = params.has_indices;
  if (is_varlen) {
    require_contiguous_1d_i32(params.indices, batch_size_i64, "indices");
  }

  require_logits_dtype(params.logits.dtype);
  if (params.max_context_len <= 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "max_context_len must be positive");
  }
  if (params.clean_logits) {
    throw_status(
        DEEPGEMM_STATUS_INVALID_ARGUMENT,
        "clean_logits is not supported for 2D paged context lengths yet");
  }

  const int batch_size = as_i32(batch_size_i64, "batch_size");
  const int next_n = as_i32(next_n_i64, "next_n");
  const int num_heads = as_i32(num_heads_i64, "num_heads");
  const int head_dim = as_i32(head_dim_i64, "head_dim");
  const int num_kv_blocks = as_i32(num_kv_blocks_i64, "num_kv_blocks");
  const int block_kv = as_i32(block_kv_i64, "block_kv");
  const int max_context_len = as_i32(params.max_context_len, "max_context_len");
  const int num_sms = as_i32(num_sms_i64, "num_sms");
  const int block_table_stride = as_i32(params.block_table.stride[0], "block_table stride(0)");
  const int64_t stride_logits = paged_logits_stride(params.max_context_len, params.logits.dtype);
  require_logits_output(
      params.logits,
      batch_size_i64 * next_n_i64,
      params.max_context_len,
      stride_logits,
      "logits");
  (void)max_context_len;

  if (128 % num_heads != 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "num_heads must divide 128");
  }

  if (device.major == 9) {
    if (is_fp4) {
      throw_status(DEEPGEMM_STATUS_UNSUPPORTED_ARCH, "FP4 paged MQA logits require SM100");
    }
    if (is_varlen) {
      throw_status(DEEPGEMM_STATUS_UNSUPPORTED_ARCH, "SM90 paged MQA logits do not support varlen indices");
    }
    if (params.weights.dtype != DEEPGEMM_DTYPE_F32) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM90 paged MQA logits require f32 weights");
    }
    if (block_kv != 64) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM90 paged MQA logits require block_kv == 64");
    }
    if (next_n != 1 && next_n != 2) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM90 paged MQA logits require next_n == 1 or 2");
    }
    if ((num_heads != 32 && num_heads != 64) ||
        (head_dim != 32 && head_dim != 64 && head_dim != 128)) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "unsupported SM90 paged MQA logits shape");
    }
    launch_sm90_fp8_paged_mqa_logits(
        params,
        batch_size,
        next_n,
        num_heads,
        head_dim,
        num_kv_blocks,
        block_kv,
        stride_logits,
        block_table_stride,
        num_sms);
    return;
  }

  if (device.major == 10) {
    if (params.weights.dtype == DEEPGEMM_DTYPE_BF16 && params.logits.dtype != DEEPGEMM_DTYPE_BF16) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "bf16 weights require bf16 logits");
    }
    if (is_varlen && next_n != 1) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM100 varlen paged MQA logits require next_n == 1");
    }
    if (block_kv != 32 && block_kv != 64) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM100 paged MQA logits require block_kv == 32 or 64");
    }
    if (num_heads != 8 && num_heads != 16 && num_heads != 32 && num_heads != 64) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM100 paged MQA logits require 8, 16, 32, or 64 heads");
    }
    if (is_fp4) {
      if (head_dim != 64 && head_dim != 128) {
        throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM100 FP4 paged MQA logits require head_dim 64 or 128");
      }
    } else if (head_dim != 32 && head_dim != 64 && head_dim != 128) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM100 FP8 paged MQA logits require head_dim 32, 64, or 128");
    }
    launch_sm100_paged_mqa_logits(
        params,
        is_fp4,
        batch_size,
        next_n,
        num_heads,
        head_dim,
        num_kv_blocks,
        block_kv,
        stride_logits,
        block_table_stride,
        num_sms);
    return;
  }

  std::ostringstream message;
  message << "paged MQA logits supports SM90 or SM100, got compute capability "
          << device.major << "." << device.minor;
  throw_status(DEEPGEMM_STATUS_UNSUPPORTED_ARCH, message.str());
}

}  // namespace deepgemm_rs
