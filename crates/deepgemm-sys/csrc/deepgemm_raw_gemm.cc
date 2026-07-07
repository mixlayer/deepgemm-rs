#include "deepgemm_raw_gemm.h"

#include "deepgemm_raw_runtime.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace deepgemm_rs {
namespace {

constexpr int kBlockK = 128;
constexpr int kSmemCapacity = 232448;
constexpr int kBlockMMultipleOf = 1;
constexpr int kBlockNMultipleOf = 1;

struct GemmLayout {
  bool swap_ab = false;
  int block_m = 1;
  int block_n = 1;
  int block_k = kBlockK;
  int cluster_m = 1;
  int cluster_n = 1;

  int cluster_size() const {
    return cluster_m * cluster_n;
  }
};

struct StorageConfig {
  int load_block_m = 1;
  int load_block_n = 1;
  int store_block_m = 1;
  int store_block_n = 1;
  int swizzle_a_mode = 0;
  int swizzle_b_mode = 0;
  int swizzle_cd_mode = 0;
};

struct PipelineConfig {
  int smem_size = 0;
  int num_stages = 0;
};

struct LaunchConfig {
  int num_threads = 1;
  int num_tma_threads = 0;
  int num_math_threads = 0;
  int num_non_epilogue_threads = 0;
  int num_epilogue_threads = 0;
};

struct GemmConfig {
  GemmLayout layout;
  StorageConfig storage;
  PipelineConfig pipeline;
  LaunchConfig launch;
};

struct LayoutInfo {
  int num_waves = 0;
  int last_wave_util = 0;
  double num_cycles = 0.0;
  GemmLayout layout;
};

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

int ceil_div_int(int value, int divisor) {
  return as_i32(ceil_div_i64(value, divisor), "ceil_div");
}

int align_int(int value, int alignment) {
  return as_i32(align_i64(value, alignment), "alignment");
}

int get_swizzle_mode(int block_size, int elem_size) {
  for (int mode : {128, 64, 32, 16}) {
    if ((block_size * elem_size) % mode == 0) {
      return mode;
    }
  }
  throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "unsupported swizzle configuration");
  return 0;
}

bool env_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::atoi(value) != 0;
}

void print_gemm_config_once(
    const char* arch,
    int m,
    int n,
    int k,
    int num_sms,
    const GemmConfig& config,
    const LayoutInfo& info) {
  if (!env_enabled("DG_JIT_DEBUG") && !env_enabled("DG_PRINT_CONFIGS")) {
    return;
  }

  std::ostringstream key;
  key << arch << ':' << m << 'x' << n << 'x' << k << ':' << num_sms;
  static std::unordered_set<std::string> printed;
  if (!printed.insert(key.str()).second) {
    return;
  }

  std::cout
      << "DeepGEMM raw " << arch << " fp8_gemm_nt"
      << "(m=" << m << ", n=" << n << ", k=" << k << ", num_sms=" << num_sms << "): "
      << "layout(swap_ab=" << config.layout.swap_ab
      << ", block_m=" << config.layout.block_m
      << ", block_n=" << config.layout.block_n
      << ", block_k=" << config.layout.block_k
      << ", cluster_m=" << config.layout.cluster_m
      << ", cluster_n=" << config.layout.cluster_n << "), "
      << "storage(load_m=" << config.storage.load_block_m
      << ", load_n=" << config.storage.load_block_n
      << ", store_m=" << config.storage.store_block_m
      << ", store_n=" << config.storage.store_block_n
      << ", swizzle_a=" << config.storage.swizzle_a_mode
      << ", swizzle_b=" << config.storage.swizzle_b_mode
      << ", swizzle_cd=" << config.storage.swizzle_cd_mode << "), "
      << "pipeline(stages=" << config.pipeline.num_stages
      << ", smem=" << config.pipeline.smem_size << "), "
      << "launch(threads=" << config.launch.num_threads
      << ", cluster=" << config.layout.cluster_size() << "), "
      << "layout_info(waves=" << info.num_waves
      << ", last_wave_util=" << info.last_wave_util
      << ", cycles=" << info.num_cycles << ")"
      << std::endl;
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

StorageConfig sm90_storage_config(const GemmLayout& layout) {
  StorageConfig storage;
  storage.load_block_m = layout.block_m;
  storage.load_block_n = layout.block_n;
  storage.store_block_m = layout.block_m;
  storage.store_block_n = layout.block_n;
  storage.swizzle_a_mode = get_swizzle_mode(layout.block_k, 1);
  storage.swizzle_b_mode = get_swizzle_mode(layout.block_k, 1);
  storage.swizzle_cd_mode = get_swizzle_mode(storage.store_block_n, 2);
  return storage;
}

PipelineConfig sm90_pipeline_config(int k, const GemmLayout& layout, const StorageConfig& storage) {
  constexpr int kNumMaxStages = 16;
  const int smem_cd = align_int(layout.block_m * layout.block_n * 2, 1024);
  const int smem_barriers = kNumMaxStages * 8 * 2;
  const int smem_a_per_stage = storage.load_block_m * layout.block_k;
  const int smem_b_per_stage = storage.load_block_n * layout.block_k;
  const int smem_sfa_per_stage = align_int(layout.block_m * 4, 128);
  const int use_uniform_sfb = layout.block_k % layout.block_n == 0 ? 1 : 2;
  const int smem_extra_sfb = align_int(ceil_div_int(k, layout.block_k) * 4 * use_uniform_sfb, 8);
  const int smem_extra = smem_cd + smem_barriers + smem_extra_sfb;
  const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage;
  const int num_stages = std::min((kSmemCapacity - smem_extra) / smem_per_stage, kNumMaxStages);
  return {smem_extra + num_stages * smem_per_stage, num_stages};
}

LaunchConfig sm90_launch_config(const GemmLayout& layout) {
  LaunchConfig launch;
  launch.num_tma_threads = 128;
  launch.num_math_threads = layout.block_m <= 64 ? 128 : 256;
  launch.num_threads = launch.num_tma_threads + launch.num_math_threads;
  return launch;
}

LayoutInfo sm90_layout_info(int m, int n, int k, int num_sms, const GemmLayout& layout) {
  const int64_t num_blocks =
      static_cast<int64_t>(ceil_div_int(m, layout.block_m)) *
      static_cast<int64_t>(ceil_div_int(n, layout.block_n));
  const int num_waves = as_i32(ceil_div_i64(num_blocks, num_sms), "SM90 waves");
  const int num_last_blocks = static_cast<int>(num_blocks % num_sms);
  const int last_wave_util = num_last_blocks == 0 ? num_sms : num_last_blocks;

  const double l2_bandwidth_per_cycle = std::min(64.0 * num_sms, 8e6 / 1.3e3);
  const double l1_bandwidth_per_cycle = 128.0 * num_sms;
  constexpr int wgmma_m = 64;
  constexpr int elem_size_ab = 1;
  constexpr int elem_size_cd = 2;

  const int64_t num_bytes_l2_ab =
      static_cast<int64_t>(k) *
      (layout.block_m / layout.cluster_n + layout.block_n / layout.cluster_m) *
      elem_size_ab;
  const int64_t num_bytes_l1_ab =
      static_cast<int64_t>(k) * (layout.block_m + layout.block_n) * elem_size_ab;
  const int64_t num_bytes_l1_tc =
      static_cast<int64_t>(k) * (std::max(wgmma_m, layout.block_m) + layout.block_n) * elem_size_ab +
      static_cast<int64_t>(layout.block_m) * layout.block_n * elem_size_cd;
  const int64_t num_bytes_l1_l2_cd =
      static_cast<int64_t>(layout.block_m) * layout.block_n * elem_size_cd;
  const double num_l2_cycles =
      (num_bytes_l2_ab + num_bytes_l1_l2_cd) * static_cast<double>(num_blocks) /
      l2_bandwidth_per_cycle;
  const double num_l1_cycles =
      (num_bytes_l1_ab + num_bytes_l1_tc + num_bytes_l1_l2_cd) *
      static_cast<double>(num_blocks) /
      l1_bandwidth_per_cycle;
  const double wave_efficiency = static_cast<double>(num_blocks) / (num_waves * num_sms);
  double num_cycles = std::max(num_l1_cycles, num_l2_cycles) / wave_efficiency;
  if (layout.cluster_size() > 1 && num_waves <= 1) {
    num_cycles = std::numeric_limits<double>::infinity();
  }
  return {num_waves, last_wave_util, num_cycles, layout};
}

std::vector<GemmLayout> sm90_layout_candidates(int m, int, int k, int num_sms) {
  std::vector<int> block_m_candidates = {64, 128};
  if (m <= 16) {
    block_m_candidates.push_back(16);
  }
  if (m <= 32) {
    block_m_candidates.push_back(32);
  }
  block_m_candidates.push_back(256);

  std::vector<int> block_n_candidates;
  const int step = std::lcm(16, kBlockNMultipleOf);
  for (int block_n = step; block_n <= 192; block_n += step) {
    block_n_candidates.push_back(block_n);
  }

  std::vector<GemmLayout> candidates;
  for (int cluster_m = 1; cluster_m <= 2; ++cluster_m) {
    for (int cluster_n = 1; cluster_n <= 2; ++cluster_n) {
      if (cluster_m * cluster_n > 2) {
        continue;
      }
      if (num_sms % (cluster_m * cluster_n) != 0) {
        continue;
      }
      for (int block_m : block_m_candidates) {
        for (int block_n : block_n_candidates) {
          if (block_n > kBlockK &&
              (block_n % (block_n - kBlockK) != 0 && kBlockK % (block_n - kBlockK) != 0)) {
            continue;
          }
          if (block_m > 128 && block_n > 128) {
            continue;
          }
          GemmLayout layout{false, block_m, block_n, kBlockK, cluster_m, cluster_n};
          const auto storage = sm90_storage_config(layout);
          if (storage.swizzle_a_mode % 64 != 0 || storage.swizzle_b_mode % 64 != 0) {
            continue;
          }
          const auto pipeline = sm90_pipeline_config(k, layout, storage);
          if (pipeline.num_stages < 3 ||
              (block_m * block_n < 128 * 192 && pipeline.num_stages < 4)) {
            continue;
          }
          candidates.push_back(layout);
        }
      }
    }
  }
  return candidates;
}

GemmConfig sm90_best_config(int m, int n, int k, int num_sms, LayoutInfo* selected_info) {
  const auto candidates = sm90_layout_candidates(m, n, k, num_sms);
  if (candidates.empty()) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "no SM90 FP8 GEMM heuristic candidates");
  }

  GemmLayout best_layout = candidates.front();
  LayoutInfo best_info = sm90_layout_info(m, n, k, num_sms, best_layout);
  for (size_t i = 1; i < candidates.size(); ++i) {
    const auto info = sm90_layout_info(m, n, k, num_sms, candidates[i]);
    if (info.num_cycles < best_info.num_cycles) {
      best_layout = candidates[i];
      best_info = info;
    }
  }

  GemmConfig config;
  config.layout = best_layout;
  config.storage = sm90_storage_config(best_layout);
  config.pipeline = sm90_pipeline_config(k, best_layout, config.storage);
  config.launch = sm90_launch_config(best_layout);
  *selected_info = best_info;
  return config;
}

std::pair<int, int> sm100_sf_aligned_blocks(int block_m, int block_n) {
  return {align_int(block_m, 128), align_int(block_n, 128)};
}

StorageConfig sm100_storage_config(const GemmLayout& layout) {
  StorageConfig storage;
  storage.load_block_m = layout.block_m / layout.cluster_n;
  storage.load_block_n = layout.block_n / layout.cluster_m;
  storage.store_block_m = layout.swap_ab ? 16 : std::min(128, layout.block_m);
  storage.store_block_n = layout.block_n;
  storage.swizzle_a_mode = get_swizzle_mode(layout.block_k, 1);
  storage.swizzle_b_mode = get_swizzle_mode(layout.block_k, 1);
  storage.swizzle_cd_mode = get_swizzle_mode(storage.store_block_n, 2);
  return storage;
}

PipelineConfig sm100_pipeline_config(const GemmLayout& layout, const StorageConfig& storage) {
  constexpr int kNumMaxStages = 32;
  const int smem_cd = layout.swap_ab
      ? storage.store_block_m * storage.store_block_n * 2 * 2
      : storage.store_block_m * storage.swizzle_cd_mode * 2;
  const int smem_barriers = kNumMaxStages * 8 * 3 + 2 * 8 * 2 + 8;
  const int smem_tmem_ptr = 4;
  const int smem_a_per_stage = storage.load_block_m * layout.block_k;
  const int smem_b_per_stage = storage.load_block_n * layout.block_k;
  const auto [sf_block_m, sf_block_n] =
      sm100_sf_aligned_blocks(layout.block_m, layout.block_n);
  const int smem_sfa_per_stage = sf_block_m * 4;
  const int smem_sfb_per_stage = sf_block_n * 4;
  const int smem_extra = smem_cd + smem_barriers + smem_tmem_ptr;
  const int smem_per_stage =
      smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
  const int num_stages = std::min((kSmemCapacity - smem_extra) / smem_per_stage, kNumMaxStages);
  return {smem_extra + num_stages * smem_per_stage, num_stages};
}

LaunchConfig sm100_launch_config() {
  LaunchConfig launch;
  launch.num_threads = 256;
  launch.num_tma_threads = 32;
  launch.num_math_threads = 128;
  launch.num_non_epilogue_threads = 128;
  launch.num_epilogue_threads = 128;
  return launch;
}

LayoutInfo sm100_layout_info(int m, int n, int num_sms, const GemmLayout& layout) {
  const int64_t num_blocks =
      static_cast<int64_t>(ceil_div_int(m, layout.block_m)) *
      static_cast<int64_t>(ceil_div_int(n, layout.block_n));
  const int num_waves = as_i32(ceil_div_i64(num_blocks, num_sms), "SM100 waves");
  const int num_last_blocks = static_cast<int>(num_blocks % num_sms);
  const int last_wave_util = num_last_blocks == 0 ? num_sms : num_last_blocks;
  return {num_waves, last_wave_util, 0.0, layout};
}

bool sm100_layout_is_better(const LayoutInfo& a, const LayoutInfo& b) {
  if ((a.num_waves == 1 || b.num_waves == 1) && a.num_waves != b.num_waves) {
    return a.num_waves < b.num_waves;
  }
  if (a.layout.cluster_size() != b.layout.cluster_size()) {
    return a.layout.cluster_size() > b.layout.cluster_size();
  }
  if (a.num_waves != b.num_waves) {
    return a.num_waves < b.num_waves;
  }
  if (a.last_wave_util != b.last_wave_util) {
    return a.last_wave_util > b.last_wave_util;
  }
  const int a_extent = a.layout.block_m + a.layout.block_n;
  const int b_extent = b.layout.block_m + b.layout.block_n;
  if (a_extent != b_extent) {
    return a_extent < b_extent;
  }
  return a.layout.block_m * a.layout.block_n < b.layout.block_m * b.layout.block_n;
}

std::vector<GemmLayout> sm100_layout_candidates(int m, int n, int k, int num_sms) {
  std::vector<GemmLayout> candidates;
  for (int swap_ab = 0; swap_ab < 2; ++swap_ab) {
    std::vector<int> block_m_candidates;
    std::vector<int> block_n_candidates;
    if (swap_ab != 0) {
      const int step = std::lcm(16, kBlockMMultipleOf);
      for (int block_m = step; block_m <= 256; block_m += step) {
        block_m_candidates.push_back(block_m);
      }
      block_n_candidates = {128};
    } else {
      if (m <= 32) {
        block_m_candidates = {32};
      } else if (m <= 64) {
        block_m_candidates = {64};
      } else {
        block_m_candidates = {128};
      }
      if (16 % kBlockNMultipleOf == 0) {
        block_n_candidates.push_back(16);
      }
      const int step = std::lcm(32, kBlockNMultipleOf);
      const int end = k <= 256 ? 128 : 256;
      for (int block_n = step; block_n <= end; block_n += step) {
        block_n_candidates.push_back(block_n);
      }
    }

    for (int cluster_m = 1; cluster_m <= 2; ++cluster_m) {
      if (swap_ab != 0 && cluster_m > 1) {
        continue;
      }
      for (int cluster_n = 1; cluster_n <= 2; ++cluster_n) {
        if (cluster_m * cluster_n > 2) {
          continue;
        }
        if (swap_ab == 0 && cluster_n > 1) {
          continue;
        }
        if (num_sms % (cluster_m * cluster_n) != 0) {
          continue;
        }
        for (int block_m : block_m_candidates) {
          if ((block_m / cluster_n) % 8 != 0) {
            continue;
          }
          if (ceil_div_int(m, block_m) % cluster_m != 0) {
            continue;
          }
          for (int block_n : block_n_candidates) {
            if ((block_n / cluster_m) % 8 != 0) {
              continue;
            }
            if (ceil_div_int(n, block_n) % cluster_n != 0) {
              continue;
            }
            if (swap_ab != 0 && block_n != 128) {
              continue;
            }
            const auto [sf_block_m, sf_block_n] = sm100_sf_aligned_blocks(block_m, block_n);
            const int tmem_sf_cols = sf_block_m / 32 + sf_block_n / 32;
            const int umma_n = swap_ab != 0 ? block_m : block_n;
            if (2 * umma_n + tmem_sf_cols > 512) {
              continue;
            }
            GemmLayout layout{swap_ab != 0, block_m, block_n, kBlockK, cluster_m, cluster_n};
            const auto storage = sm100_storage_config(layout);
            if (storage.swizzle_a_mode != 128 || storage.swizzle_b_mode != 128) {
              continue;
            }
            candidates.push_back(layout);
          }
        }
      }
    }
  }
  return candidates;
}

GemmConfig sm100_best_config(int m, int n, int k, int num_sms, LayoutInfo* selected_info) {
  const auto candidates = sm100_layout_candidates(m, n, k, num_sms);
  if (candidates.empty()) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "no SM100 FP8 GEMM heuristic candidates");
  }

  GemmLayout best_layout = candidates.front();
  LayoutInfo best_info = sm100_layout_info(m, n, num_sms, best_layout);
  for (size_t i = 1; i < candidates.size(); ++i) {
    const auto info = sm100_layout_info(m, n, num_sms, candidates[i]);
    if (sm100_layout_is_better(info, best_info)) {
      best_layout = candidates[i];
      best_info = info;
    }
  }

  GemmConfig config;
  config.layout = best_layout;
  config.storage = sm100_storage_config(best_layout);
  config.pipeline = sm100_pipeline_config(best_layout, config.storage);
  config.launch = sm100_launch_config();
  *selected_info = best_info;
  return config;
}

std::string sm90_fp8_gemm_1d2d_code(
    int m,
    int n,
    int k,
    const GemmConfig& config,
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
      << "        " << config.layout.block_m << ", " << config.layout.block_n << ", " << config.layout.block_k << ",\n"
      << "        " << config.storage.swizzle_a_mode << ", "
      << config.storage.swizzle_b_mode << ", " << config.storage.swizzle_cd_mode << ",\n"
      << "        " << config.pipeline.num_stages << ",\n"
      << "        " << config.launch.num_tma_threads << ", " << config.launch.num_math_threads << ",\n"
      << "        " << config.layout.cluster_size() << ", "
      << (config.layout.cluster_n > 1 ? "true" : "false") << ",\n"
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
    int gran_k_a,
    int gran_k_b,
    const GemmConfig& config,
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
      << "        " << config.layout.block_m << ", " << config.layout.block_n << ", " << config.layout.block_k << ",\n"
      << "        1,\n"
      << "        " << config.storage.swizzle_a_mode << ", "
      << config.storage.swizzle_b_mode << ", " << config.storage.swizzle_cd_mode << ",\n"
      << "        " << config.pipeline.num_stages << ",\n"
      << "        " << config.launch.num_non_epilogue_threads << ", "
      << config.launch.num_epilogue_threads << ",\n"
      << "        " << config.layout.cluster_size() << ", "
      << (config.layout.cluster_n > 1 ? "true" : "false") << ",\n"
      << "        " << num_sms << ",\n"
      << "        " << (config.layout.swap_ab ? "true" : "false") << ", true,\n"
      << "        GemmType::Normal, false,\n"
      << "        cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t,\n"
      << "        epilogue::transform::EpilogueIdentity\n"
      << "    >);\n"
      << "};\n";
  (void)m;
  return code.str();
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
  LayoutInfo layout_info;
  const auto config = sm90_best_config(m, n, k, num_sms, &layout_info);
  print_gemm_config_once("SM90", m, n, k, num_sms, config, layout_info);

  require_mn_major_scale(params.a_scale, m, k, config.layout.block_k, DEEPGEMM_DTYPE_F32, "a_scale");
  require_contiguous_2d(params.b_scale, DEEPGEMM_DTYPE_F32, "b_scale");
  require_shape_2d(
      params.b_scale,
      ceil_div_i64(n, config.layout.block_k),
      ceil_div_i64(k, config.layout.block_k),
      "b_scale");

  const auto tensor_map_a = make_tma_2d_desc(
      params.a.data,
      params.a.dtype,
      k,
      m,
      config.layout.block_k,
      config.storage.load_block_m,
      as_i32(params.a.stride[0], "a stride(0)"),
      config.storage.swizzle_a_mode);
  const auto tensor_map_b = make_tma_2d_desc(
      params.b.data,
      params.b.dtype,
      k,
      n,
      config.layout.block_k,
      config.storage.load_block_n,
      as_i32(params.b.stride[0], "b stride(0)"),
      config.storage.swizzle_b_mode);
  const auto tensor_map_d = make_tma_2d_desc(
      params.d.data,
      params.d.dtype,
      n,
      m,
      config.storage.store_block_n,
      config.storage.store_block_m,
      as_i32(params.d.stride[0], "d stride(0)"),
      config.storage.swizzle_cd_mode);
  const auto tensor_map_sfa = make_tma_2d_desc(
      params.a_scale.data,
      params.a_scale.dtype,
      tma_aligned_mn(m, params.a_scale.dtype, "a_scale aligned m"),
      as_i32(ceil_div_i64(k, config.layout.block_k), "a_scale k blocks"),
      config.layout.block_m,
      1,
      tma_aligned_mn(m, params.a_scale.dtype, "a_scale stride"),
      0);

  const auto runtime = build_kernel(
      "sm90_fp8_gemm_1d2d",
      sm90_fp8_gemm_1d2d_code(
          m,
          n,
          k,
          config,
          num_sms));

  LaunchArgs launch_args;
  launch_args.grid_x = num_sms;
  launch_args.num_threads = config.launch.num_threads;
  launch_args.smem_size = config.pipeline.smem_size;
  launch_args.cluster_dim = config.layout.cluster_size();
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
  constexpr int gran_k_a = 128;
  constexpr int gran_k_b = 128;
  LayoutInfo layout_info;
  const auto config = sm100_best_config(m, n, k, num_sms, &layout_info);
  print_gemm_config_once("SM100", m, n, k, num_sms, config, layout_info);

  require_mn_major_scale(params.a_scale, m, k, gran_k_a, DEEPGEMM_DTYPE_PACKED_UE8M0, "a_scale");
  require_mn_major_scale(params.b_scale, n, k, gran_k_b, DEEPGEMM_DTYPE_PACKED_UE8M0, "b_scale");

  const auto tensor_map_a = make_tma_2d_desc(
      params.a.data,
      params.a.dtype,
      k,
      m,
      config.layout.block_k,
      config.storage.load_block_m,
      as_i32(params.a.stride[0], "a stride(0)"),
      config.storage.swizzle_a_mode);
  const auto tensor_map_b = make_tma_2d_desc(
      params.b.data,
      params.b.dtype,
      k,
      n,
      config.layout.block_k,
      config.storage.load_block_n,
      as_i32(params.b.stride[0], "b stride(0)"),
      config.storage.swizzle_b_mode);
  const auto tensor_map_sfa = make_tma_2d_desc(
      params.a_scale.data,
      params.a_scale.dtype,
      tma_aligned_mn(m, params.a_scale.dtype, "a_scale aligned m"),
      as_i32(ceil_div_i64(k, gran_k_a * 4), "a_scale packed k blocks"),
      config.layout.block_m,
      1,
      tma_aligned_mn(m, params.a_scale.dtype, "a_scale stride"),
      0);
  const auto tensor_map_sfb = make_tma_2d_desc(
      params.b_scale.data,
      params.b_scale.dtype,
      tma_aligned_mn(n, params.b_scale.dtype, "b_scale aligned n"),
      as_i32(ceil_div_i64(k, gran_k_b * 4), "b_scale packed k blocks"),
      config.layout.block_n,
      1,
      tma_aligned_mn(n, params.b_scale.dtype, "b_scale stride"),
      0);
  const auto tensor_map_cd = make_tma_2d_desc(
      params.d.data,
      params.d.dtype,
      n,
      m,
      config.storage.store_block_n,
      config.storage.store_block_m,
      as_i32(params.d.stride[0], "d stride(0)"),
      config.storage.swizzle_cd_mode);

  const auto runtime = build_kernel(
      "sm100_fp8_gemm_1d1d",
      sm100_fp8_gemm_1d1d_code(
          m,
          n,
          k,
          gran_k_a,
          gran_k_b,
          config,
          num_sms));

  LaunchArgs launch_args;
  launch_args.grid_x = num_sms;
  launch_args.num_threads = config.launch.num_threads;
  launch_args.smem_size = config.pipeline.smem_size;
  launch_args.cluster_dim = config.layout.cluster_size();
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
