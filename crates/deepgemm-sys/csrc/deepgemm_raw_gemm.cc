#include "deepgemm_raw_gemm.h"

#include "deepgemm_raw_runtime.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>

namespace deepgemm_rs {
namespace {

constexpr int kBlockK = 128;
constexpr int kSmemCapacity = 232448;

int64_t ceil_div_i64(int64_t value, int64_t divisor) {
  if (value < 0 || divisor <= 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "invalid ceil_div input");
  }
  const int64_t addend = divisor - 1;
  if (value > INT64_MAX - addend) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "ceil_div input overflowed");
  }
  return (value + addend) / divisor;
}

int64_t align_i64(int64_t value, int64_t alignment) {
  if (value < 0 || alignment <= 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "invalid alignment input");
  }
  const int64_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const int64_t delta = alignment - remainder;
  if (value > INT64_MAX - delta) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "alignment input overflowed");
  }
  return value + delta;
}

int as_i32(int64_t value, const char* name) {
  if (value < 0 || value > INT32_MAX) {
    throw_status(
        DEEPGEMM_STATUS_INVALID_ARGUMENT,
        std::string(name) + " must fit in a positive i32");
  }
  return static_cast<int>(value);
}

int dtype_size_checked(deepgemm_dtype_t dtype, const char* name) {
  const auto size = dtype_element_size(dtype);
  if (size <= 0 || size > INT32_MAX) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, std::string(name) + " dtype is invalid");
  }
  return static_cast<int>(size);
}

int tma_aligned_mn(int64_t mn, deepgemm_dtype_t dtype, const char* name) {
  return as_i32(get_tma_aligned_size(mn, dtype_size_checked(dtype, name)), name);
}

int64_t scale_k_divisor(int gran_k, deepgemm_dtype_t dtype) {
  const int64_t factor = dtype == DEEPGEMM_DTYPE_F32 ? 1 : 4;
  if (gran_k <= 0 || gran_k > INT64_MAX / factor) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "scale gran_k overflowed");
  }
  return static_cast<int64_t>(gran_k) * factor;
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

void require_positive_2d_shape(const int64_t* shape, const char* name) {
  if (shape[0] <= 0 || shape[1] <= 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, std::string(name) + " dimensions must be positive");
  }
}

void require_contiguous_2d(
    const deepgemm_tensor_t& tensor,
    deepgemm_dtype_t dtype,
    const char* name) {
  require_tensor_rank(tensor, 2, name);
  require_dtype(tensor.dtype, dtype, name);
  require_positive_2d_shape(tensor.shape, name);
  if (tensor.stride[0] != tensor.shape[1] || tensor.stride[1] != 1) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, std::string(name) + " must be row-major contiguous");
  }
}

void require_contiguous_2d_mut(
    const deepgemm_tensor_mut_t& tensor,
    deepgemm_dtype_t dtype,
    const char* name) {
  require_tensor_mut_rank(tensor, 2, name);
  require_dtype(tensor.dtype, dtype, name);
  require_positive_2d_shape(tensor.shape, name);
  if (tensor.stride[0] != tensor.shape[1] || tensor.stride[1] != 1) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, std::string(name) + " must be row-major contiguous");
  }
}

void require_shape_2d(
    const deepgemm_tensor_t& tensor,
    int64_t rows,
    int64_t cols,
    const char* name) {
  if (tensor.shape[0] != rows || tensor.shape[1] != cols) {
    std::ostringstream message;
    message << name << " shape must be [" << rows << ", " << cols << "]";
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, message.str());
  }
}

void require_shape_2d_mut(
    const deepgemm_tensor_mut_t& tensor,
    int64_t rows,
    int64_t cols,
    const char* name) {
  if (tensor.shape[0] != rows || tensor.shape[1] != cols) {
    std::ostringstream message;
    message << name << " shape must be [" << rows << ", " << cols << "]";
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, message.str());
  }
}

void require_mn_major_scale(
    const deepgemm_tensor_t& tensor,
    int64_t mn,
    int64_t k,
    int gran_k,
    deepgemm_dtype_t dtype,
    const char* name) {
  require_tensor_rank(tensor, 2, name);
  require_dtype(tensor.dtype, dtype, name);
  const int64_t scale_cols = ceil_div_i64(k, scale_k_divisor(gran_k, dtype));
  const int64_t aligned_mn = get_tma_aligned_size(mn, dtype_element_size(dtype));
  if (tensor.shape[0] != mn || tensor.shape[1] != scale_cols ||
      tensor.stride[0] != 1 || tensor.stride[1] != aligned_mn) {
    std::ostringstream message;
    message << name << " must have shape [" << mn << ", " << scale_cols
            << "] and strides [1, " << aligned_mn << "]";
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, message.str());
  }
}

void require_mn_major_scale_out(
    const deepgemm_tensor_mut_t& tensor,
    int64_t mn,
    int64_t k,
    int gran_k,
    deepgemm_dtype_t dtype,
    const char* name) {
  require_tensor_mut_rank(tensor, 2, name);
  require_dtype(tensor.dtype, dtype, name);
  const int64_t scale_cols = ceil_div_i64(k, scale_k_divisor(gran_k, dtype));
  const int64_t aligned_mn = get_tma_aligned_size(mn, dtype_element_size(dtype));
  if (tensor.shape[0] != mn || tensor.shape[1] != scale_cols ||
      tensor.stride[0] != 1 || tensor.stride[1] != aligned_mn) {
    std::ostringstream message;
    message << name << " must have shape [" << mn << ", " << scale_cols
            << "] and strides [1, " << aligned_mn << "]";
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, message.str());
  }
}

std::string transpose_fp32_code(int num_threads, int block_mn, int sf_k) {
  std::ostringstream code;
  code
      << "#include <deep_gemm/impls/smxx_layout.cuh>\n\n"
      << "using namespace deep_gemm;\n\n"
      << "static void __instantiate_kernel() {\n"
      << "    auto ptr = reinterpret_cast<void*>(&transpose_fp32<\n"
      << "        " << num_threads << ", " << block_mn << ", " << sf_k << "\n"
      << "    >);\n"
      << "};\n";
  return code.str();
}

std::string transpose_and_pack_fp32_code(int num_threads, int block_mn, int sf_k) {
  std::ostringstream code;
  code
      << "#include <deep_gemm/impls/smxx_layout.cuh>\n\n"
      << "using namespace deep_gemm;\n\n"
      << "static void __instantiate_kernel() {\n"
      << "    auto ptr = reinterpret_cast<void*>(&transpose_and_pack_fp32_into_ue8m0<\n"
      << "        " << num_threads << ", " << block_mn << ", " << sf_k << ", 1, false\n"
      << "    >);\n"
      << "};\n";
  return code.str();
}

std::string sm90_fp8_gemm_1d2d_code(
    int m,
    int n,
    int k,
    int block_m,
    int block_n,
    int block_k,
    int num_stages,
    int num_tma_threads,
    int num_math_threads,
    int num_sms) {
  std::ostringstream code;
  code
      << "#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>\n\n"
      << "using namespace deep_gemm;\n\n"
      << "static void __instantiate_kernel() {\n"
      << "    auto ptr = reinterpret_cast<void*>(&sm90_fp8_gemm_1d2d_impl<\n"
      << "        cute::UMMA::Major::K,\n"
      << "        0, " << n << ", " << k << ",\n"
      << "        1,\n"
      << "        " << block_m << ", " << block_n << ", " << block_k << ",\n"
      << "        128, 128, 128,\n"
      << "        " << num_stages << ",\n"
      << "        " << num_tma_threads << ", " << num_math_threads << ",\n"
      << "        1, false,\n"
      << "        " << num_sms << ", GemmType::Normal,\n"
      << "        cutlass::bfloat16_t,\n"
      << "        epilogue::transform::EpilogueIdentity\n"
      << "    >);\n"
      << "};\n";
  (void)m;
  return code.str();
}

std::string sm100_fp8_gemm_1d1d_code(
    int m,
    int n,
    int k,
    int block_m,
    int block_n,
    int block_k,
    int gran_k_a,
    int gran_k_b,
    int num_stages,
    int num_non_epilogue_threads,
    int num_epilogue_threads,
    int num_sms) {
  std::ostringstream code;
  code
      << "#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>\n\n"
      << "using namespace deep_gemm;\n\n"
      << "static void __instantiate_kernel() {\n"
      << "    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_impl<\n"
      << "        cute::UMMA::Major::K, cute::UMMA::Major::K,\n"
      << "        " << gran_k_a << ", " << gran_k_b << ", " << gran_k_a << ",\n"
      << "        0, " << n << ", " << k << ",\n"
      << "        " << block_m << ", " << block_n << ", " << block_k << ",\n"
      << "        1,\n"
      << "        128, 128, 128,\n"
      << "        " << num_stages << ",\n"
      << "        " << num_non_epilogue_threads << ", " << num_epilogue_threads << ",\n"
      << "        1, false,\n"
      << "        " << num_sms << ",\n"
      << "        false, true,\n"
      << "        GemmType::Normal, false,\n"
      << "        cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t,\n"
      << "        epilogue::transform::EpilogueIdentity\n"
      << "    >);\n"
      << "};\n";
  (void)m;
  return code.str();
}

int sm90_smem_sfb_size(int k, int block_n, int block_k) {
  const bool use_uniform_sfb = block_k % block_n == 0;
  return as_i32(
      align_i64(ceil_div_i64(k, block_k) * (use_uniform_sfb ? 1 : 2) * 4, 8),
      "SM90 smem_sfb");
}

int sm90_num_stages(int k, int block_m, int block_n, int block_k) {
  constexpr int kNumMaxStages = 16;
  const int smem_d = as_i32(align_i64(block_m * block_n * 2, 1024), "SM90 smem_d");
  const int smem_a_per_stage = block_m * block_k;
  const int smem_b_per_stage = block_n * block_k;
  const int smem_sfa_per_stage = as_i32(align_i64(block_m * 4, 128), "SM90 smem_sfa");
  const int smem_extra = smem_d + sm90_smem_sfb_size(k, block_n, block_k) + kNumMaxStages * 8 * 2;
  const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage;
  return std::min((kSmemCapacity - smem_extra) / smem_per_stage, kNumMaxStages);
}

int sm90_smem_size(int k, int block_m, int block_n, int block_k, int num_stages) {
  const int smem_d = as_i32(align_i64(block_m * block_n * 2, 1024), "SM90 smem_d");
  const int smem_a_per_stage = block_m * block_k;
  const int smem_b_per_stage = block_n * block_k;
  const int smem_sfa_per_stage = as_i32(align_i64(block_m * 4, 128), "SM90 smem_sfa");
  const int smem_sfb = sm90_smem_sfb_size(k, block_n, block_k);
  const int smem_barriers = num_stages * 8 * 2;
  return smem_d +
      num_stages * (smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage) +
      smem_sfb +
      smem_barriers;
}

int sm100_smem_size(
    int block_m,
    int block_n,
    int block_k,
    int num_stages,
    int elem_a,
    int elem_b,
    int elem_cd) {
  constexpr int kNumEpilogueStages = 2;
  constexpr int kNumTmaStoreStages = 2;
  const int store_block_m = std::min(block_m, 128);
  const int store_block_n = 128 / elem_cd;
  const int sf_block_m = as_i32(align_i64(block_m, 128), "SM100 sf_block_m");
  const int sf_block_n = as_i32(align_i64(block_n, 128), "SM100 sf_block_n");
  const int smem_cd = store_block_m * store_block_n * elem_cd * kNumTmaStoreStages;
  const int smem_a_per_stage = block_m * block_k * elem_a;
  const int smem_b_per_stage = block_n * block_k * elem_b;
  const int smem_sfa_per_stage = sf_block_m * 4;
  const int smem_sfb_per_stage = sf_block_n * 4;
  const int smem_barriers = (num_stages * 3 + kNumEpilogueStages * 2) * 8;
  const int smem_tmem_ptr = 4;
  return smem_cd +
      num_stages * (smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage) +
      smem_barriers +
      smem_tmem_ptr;
}

int sm100_num_stages(int block_m, int block_n, int block_k, int elem_a, int elem_b, int elem_cd) {
  constexpr int kNumMaxStages = 32;
  constexpr int kNumTmaStoreStages = 2;
  const int store_block_m = std::min(block_m, 128);
  const int store_block_n = 128 / elem_cd;
  const int sf_block_m = as_i32(align_i64(block_m, 128), "SM100 sf_block_m");
  const int sf_block_n = as_i32(align_i64(block_n, 128), "SM100 sf_block_n");
  const int smem_cd = store_block_m * store_block_n * elem_cd * kNumTmaStoreStages;
  const int smem_barriers = kNumMaxStages * 8 * 3 + 2 * 8 * 2 + 8;
  const int smem_tmem_ptr = 4;
  const int smem_extra = smem_cd + smem_barriers + smem_tmem_ptr;
  const int smem_per_stage =
      block_m * block_k * elem_a +
      block_n * block_k * elem_b +
      sf_block_m * 4 +
      sf_block_n * 4;
  return std::min((kSmemCapacity - smem_extra) / smem_per_stage, kNumMaxStages);
}

void validate_common_gemm_nt(
    const deepgemm_fp8_gemm_nt_params_t& params,
    int64_t* m_out,
    int64_t* n_out,
    int64_t* k_out) {
  require_contiguous_2d(params.a, DEEPGEMM_DTYPE_FP8_E4M3, "a");
  require_contiguous_2d(params.b, DEEPGEMM_DTYPE_FP8_E4M3, "b");
  require_contiguous_2d_mut(params.d, DEEPGEMM_DTYPE_BF16, "d");

  const int64_t m = params.a.shape[0];
  const int64_t k = params.a.shape[1];
  const int64_t n = params.b.shape[0];
  if (params.b.shape[1] != k) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "b shape must be [n, k] with the same k as a");
  }
  require_shape_2d_mut(params.d, m, n, "d");
  if (n % 8 != 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "n must be a multiple of 8");
  }

  *m_out = m;
  *n_out = n;
  *k_out = k;
}

void launch_sm90_fp8_gemm_nt(
    const deepgemm_fp8_gemm_nt_params_t& params,
    int m,
    int n,
    int k,
    int num_sms) {
  constexpr int block_m = 64;
  constexpr int block_n = 128;
  constexpr int block_k = kBlockK;
  constexpr int num_tma_threads = 128;
  constexpr int num_math_threads = 128;
  const int num_stages = sm90_num_stages(k, block_m, block_n, block_k);
  if (num_stages <= 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM90 GEMM config exceeds shared memory capacity");
  }

  require_mn_major_scale(params.a_scale, m, k, block_k, DEEPGEMM_DTYPE_F32, "a_scale");
  require_contiguous_2d(params.b_scale, DEEPGEMM_DTYPE_F32, "b_scale");
  require_shape_2d(params.b_scale, ceil_div_i64(n, block_k), ceil_div_i64(k, block_k), "b_scale");

  const auto tensor_map_a = make_tma_2d_desc(
      params.a.data,
      params.a.dtype,
      k,
      m,
      block_k,
      block_m,
      as_i32(params.a.stride[0], "a stride(0)"),
      128);
  const auto tensor_map_b = make_tma_2d_desc(
      params.b.data,
      params.b.dtype,
      k,
      n,
      block_k,
      block_n,
      as_i32(params.b.stride[0], "b stride(0)"),
      128);
  const auto tensor_map_d = make_tma_2d_desc(
      params.d.data,
      params.d.dtype,
      n,
      m,
      block_n,
      block_m,
      as_i32(params.d.stride[0], "d stride(0)"),
      128);
  const auto tensor_map_sfa = make_tma_2d_desc(
      params.a_scale.data,
      params.a_scale.dtype,
      tma_aligned_mn(m, params.a_scale.dtype, "a_scale aligned m"),
      as_i32(ceil_div_i64(k, block_k), "a_scale k blocks"),
      block_m,
      1,
      tma_aligned_mn(m, params.a_scale.dtype, "a_scale stride"),
      0);

  const auto runtime = build_kernel(
      "sm90_fp8_gemm_1d2d",
      sm90_fp8_gemm_1d2d_code(
          m,
          n,
          k,
          block_m,
          block_n,
          block_k,
          num_stages,
          num_tma_threads,
          num_math_threads,
          num_sms));

  LaunchArgs launch_args;
  launch_args.grid_x = num_sms;
  launch_args.num_threads = num_tma_threads + num_math_threads;
  launch_args.smem_size = sm90_smem_size(k, block_m, block_n, block_k, num_stages);
  launch_args.enable_pdl = pdl_enabled();

  auto* sfb = const_cast<float*>(reinterpret_cast<const float*>(params.b_scale.data));
  int* grouped_layout = nullptr;
  const auto shape_m_arg = static_cast<uint32_t>(m);
  const auto shape_n_arg = static_cast<uint32_t>(n);
  const auto shape_k_arg = static_cast<uint32_t>(k);

  launch_kernel(
      runtime,
      reinterpret_cast<CUstream>(params.stream),
      launch_args,
      sfb,
      grouped_layout,
      shape_m_arg,
      shape_n_arg,
      shape_k_arg,
      tensor_map_a,
      tensor_map_b,
      tensor_map_d,
      tensor_map_sfa);
}

void launch_sm100_fp8_gemm_nt(
    const deepgemm_fp8_gemm_nt_params_t& params,
    int m,
    int n,
    int k,
    int num_sms) {
  constexpr int block_m = 128;
  constexpr int block_n = 128;
  constexpr int block_k = kBlockK;
  constexpr int gran_k_a = 128;
  constexpr int gran_k_b = 128;
  constexpr int num_non_epilogue_threads = 128;
  constexpr int num_epilogue_threads = 128;

  require_mn_major_scale(params.a_scale, m, k, gran_k_a, DEEPGEMM_DTYPE_PACKED_UE8M0, "a_scale");
  require_mn_major_scale(params.b_scale, n, k, gran_k_b, DEEPGEMM_DTYPE_PACKED_UE8M0, "b_scale");

  const int num_stages = sm100_num_stages(block_m, block_n, block_k, 1, 1, 2);
  if (num_stages <= 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM100 GEMM config exceeds shared memory capacity");
  }

  const auto tensor_map_a = make_tma_2d_desc(
      params.a.data,
      params.a.dtype,
      k,
      m,
      block_k,
      block_m,
      as_i32(params.a.stride[0], "a stride(0)"),
      128);
  const auto tensor_map_b = make_tma_2d_desc(
      params.b.data,
      params.b.dtype,
      k,
      n,
      block_k,
      block_n,
      as_i32(params.b.stride[0], "b stride(0)"),
      128);
  const auto tensor_map_sfa = make_tma_2d_desc(
      params.a_scale.data,
      params.a_scale.dtype,
      tma_aligned_mn(m, params.a_scale.dtype, "a_scale aligned m"),
      as_i32(ceil_div_i64(k, gran_k_a * 4), "a_scale packed k blocks"),
      block_m,
      1,
      tma_aligned_mn(m, params.a_scale.dtype, "a_scale stride"),
      0);
  const auto tensor_map_sfb = make_tma_2d_desc(
      params.b_scale.data,
      params.b_scale.dtype,
      tma_aligned_mn(n, params.b_scale.dtype, "b_scale aligned n"),
      as_i32(ceil_div_i64(k, gran_k_b * 4), "b_scale packed k blocks"),
      block_n,
      1,
      tma_aligned_mn(n, params.b_scale.dtype, "b_scale stride"),
      0);
  const auto tensor_map_cd = make_tma_2d_desc(
      params.d.data,
      params.d.dtype,
      n,
      m,
      block_n,
      block_m,
      as_i32(params.d.stride[0], "d stride(0)"),
      128);

  const auto runtime = build_kernel(
      "sm100_fp8_gemm_1d1d",
      sm100_fp8_gemm_1d1d_code(
          m,
          n,
          k,
          block_m,
          block_n,
          block_k,
          gran_k_a,
          gran_k_b,
          num_stages,
          num_non_epilogue_threads,
          num_epilogue_threads,
          num_sms));

  LaunchArgs launch_args;
  launch_args.grid_x = num_sms;
  launch_args.num_threads = num_non_epilogue_threads + num_epilogue_threads;
  launch_args.smem_size = sm100_smem_size(block_m, block_n, block_k, num_stages, 1, 1, 2);
  launch_args.enable_pdl = pdl_enabled();

  int* grouped_layout = nullptr;
  const auto shape_m_arg = static_cast<uint32_t>(m);
  const auto shape_n_arg = static_cast<uint32_t>(n);
  const auto shape_k_arg = static_cast<uint32_t>(k);

  launch_kernel(
      runtime,
      reinterpret_cast<CUstream>(params.stream),
      launch_args,
      grouped_layout,
      shape_m_arg,
      shape_n_arg,
      shape_k_arg,
      tensor_map_a,
      tensor_map_b,
      tensor_map_sfa,
      tensor_map_sfb,
      tensor_map_cd);
}

}  // namespace

void launch_fp8_gemm_transform_scale(
    const deepgemm_fp8_gemm_scale_transform_params_t& params) {
  require_contiguous_2d(params.scale, DEEPGEMM_DTYPE_F32, "scale");
  const int64_t mn = params.mn;
  const int64_t k = params.k;
  const int64_t gran_k_i64 = params.gran_k;
  if (mn <= 0 || k <= 0 || gran_k_i64 <= 0 || gran_k_i64 > INT32_MAX) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "mn, k, and gran_k must be positive");
  }
  const int gran_k = static_cast<int>(gran_k_i64);
  const int64_t sf_k = ceil_div_i64(k, gran_k);
  require_shape_2d(params.scale, mn, sf_k, "scale");

  if (params.transformed.dtype != DEEPGEMM_DTYPE_F32 &&
      params.transformed.dtype != DEEPGEMM_DTYPE_PACKED_UE8M0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "transformed dtype must be f32 or packed UE8M0");
  }
  require_mn_major_scale_out(
      params.transformed,
      mn,
      k,
      gran_k,
      params.transformed.dtype,
      "transformed");

  if (sf_k <= 0 || sf_k > 512) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "scale K blocks must be between 1 and 512");
  }

  if (params.transformed.dtype == DEEPGEMM_DTYPE_F32) {
    constexpr int block_mn = 64;
    constexpr int num_threads = 512;
    const int smem_size = block_mn *
        as_i32(sf_k + (1 - (sf_k % 2)), "transpose padded sf_k") *
        static_cast<int>(sizeof(float));
    const auto runtime = build_kernel(
        "transpose_fp32",
        transpose_fp32_code(num_threads, block_mn, as_i32(sf_k, "sf_k")));
    LaunchArgs launch_args;
    launch_args.grid_x = as_i32(ceil_div_i64(mn, block_mn), "transpose grid x");
    launch_args.grid_y = 1;
    launch_args.num_threads = num_threads;
    launch_args.smem_size = smem_size;
    launch_args.enable_pdl = pdl_enabled();
    auto* scale = const_cast<float*>(reinterpret_cast<const float*>(params.scale.data));
    auto* transformed = reinterpret_cast<float*>(params.transformed.data);
    const auto mn_arg = static_cast<uint32_t>(mn);
    launch_kernel(
        runtime,
        reinterpret_cast<CUstream>(params.stream),
        launch_args,
        scale,
        transformed,
        mn_arg);
    return;
  }

  constexpr int block_mn = 48;
  constexpr int num_threads = 512;
  const int smem_size = block_mn * as_i32(sf_k, "pack sf_k") * static_cast<int>(sizeof(float));
  const auto runtime = build_kernel(
      "transpose_and_pack_fp32_into_ue8m0",
      transpose_and_pack_fp32_code(num_threads, block_mn, as_i32(sf_k, "sf_k")));
  LaunchArgs launch_args;
  launch_args.grid_x = as_i32(ceil_div_i64(mn, block_mn), "pack grid x");
  launch_args.grid_y = 1;
  launch_args.num_threads = num_threads;
  launch_args.smem_size = smem_size;
  launch_args.enable_pdl = pdl_enabled();
  auto* scale = const_cast<float*>(reinterpret_cast<const float*>(params.scale.data));
  auto* transformed = reinterpret_cast<uint32_t*>(params.transformed.data);
  uint32_t* grouped_layout = nullptr;
  const auto mn_arg = static_cast<uint32_t>(mn);
  const uint32_t m_alignment = 0;
  launch_kernel(
      runtime,
      reinterpret_cast<CUstream>(params.stream),
      launch_args,
      scale,
      transformed,
      mn_arg,
      grouped_layout,
      m_alignment);
}

void launch_fp8_gemm_nt(
    const deepgemm_fp8_gemm_nt_params_t& params) {
  int64_t m_i64 = 0;
  int64_t n_i64 = 0;
  int64_t k_i64 = 0;
  validate_common_gemm_nt(params, &m_i64, &n_i64, &k_i64);
  const int m = as_i32(m_i64, "m");
  const int n = as_i32(n_i64, "n");
  const int k = as_i32(k_i64, "k");
  const int num_sms = effective_num_sms();

  const auto device = current_device_info();
  if (device.major == 9) {
    launch_sm90_fp8_gemm_nt(params, m, n, k, num_sms);
    return;
  }
  if (device.major == 10) {
    launch_sm100_fp8_gemm_nt(params, m, n, k, num_sms);
    return;
  }

  std::ostringstream message;
  message << "FP8 GEMM supports SM90 or SM100, got compute capability "
          << device.major << "." << device.minor;
  throw_status(DEEPGEMM_STATUS_UNSUPPORTED_ARCH, message.str());
}

}  // namespace deepgemm_rs
