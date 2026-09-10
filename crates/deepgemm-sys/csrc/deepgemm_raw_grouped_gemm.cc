#include "deepgemm_raw_grouped_gemm.h"

#include "deepgemm_raw_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>

namespace deepgemm_rs {
namespace {

constexpr int kSm120SmemCapacity = 101376;
int ceil_div(int value, int divisor) { return (value + divisor - 1) / divisor; }

int swizzle_mode(int row_elements, int element_bytes) {
  for (const int mode : {128, 64, 32, 16}) {
    if ((row_elements * element_bytes) % mode == 0) return mode;
  }
  throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "unsupported SM120 swizzle configuration");
  return 0;
}

void require_rank(const deepgemm_tensor_t& tensor, uint32_t rank, const char* name) {
  if (tensor.data == nullptr || tensor.rank != rank) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT,
                 std::string(name) + " must be a non-null rank-" + std::to_string(rank) + " tensor");
  }
}

void require_rank(const deepgemm_tensor_mut_t& tensor, uint32_t rank, const char* name) {
  if (tensor.data == nullptr || tensor.rank != rank) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT,
                 std::string(name) + " must be a non-null rank-" + std::to_string(rank) + " tensor");
  }
}

template <typename Tensor>
void require_contiguous(const Tensor& tensor, const char* name) {
  int64_t stride = 1;
  for (int axis = static_cast<int>(tensor.rank) - 1; axis >= 0; --axis) {
    if (tensor.shape[axis] <= 0 || tensor.stride[axis] != stride) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, std::string(name) + " must be positive and contiguous");
    }
    stride *= tensor.shape[axis];
  }
}

int checked_i32(int64_t value, const char* name) {
  if (value <= 0 || value > std::numeric_limits<int32_t>::max()) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, std::string(name) + " must fit a positive i32");
  }
  return static_cast<int>(value);
}

struct Config {
  int block_m;
  int block_n;
  int block_k;
  int swizzle_a;
  int swizzle_b;
  int swizzle_d;
  int stages;
  int smem;
  int64_t cycles;
};

Config make_config(int m, int n, int k, int num_sms, int block_m, int block_n, int block_k) {
  const int swizzle_a = swizzle_mode(block_k, 2);
  const int swizzle_b = swizzle_mode(block_k, 2);
  const int swizzle_d = block_n * 2 >= 128 ? 128 : 0;
  const int smem_d = swizzle_d > 0 && (block_n * 2) % swizzle_d == 0
      ? block_m * block_n * 2
      : 0;
  constexpr int barrier_bytes = 16 * 8 * 2;
  const int per_stage = (block_m + block_n) * block_k * 2;
  const int stages = std::min((kSm120SmemCapacity - barrier_bytes - smem_d) / per_stage, 16);
  if (stages < 2) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM120 grouped GEMM has fewer than two pipeline stages");
  }
  const int smem = barrier_bytes + smem_d + stages * per_stage;
  const int blocks = ceil_div(m, block_m) * ceil_div(n, block_n);
  const int waves = ceil_div(blocks, num_sms);
  const int k_blocks = ceil_div(k, block_k);
  const int64_t tma_bytes = static_cast<int64_t>(block_m + block_n) * block_k * 2;
  const double per_k_block = tma_bytes * 0.07 + 120.0 / std::sqrt(static_cast<double>(stages));
  const int64_t cycles = static_cast<int64_t>(waves * (k_blocks * per_k_block + 2000.0));
  return {block_m, block_n, block_k, swizzle_a, swizzle_b, swizzle_d, stages, smem, cycles};
}

int config_override(const char* name, int first, int second) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return 0;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (*end != '\0' || (parsed != first && parsed != second)) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT,
                 std::string(name) + " must be " + std::to_string(first) +
                     " or " + std::to_string(second));
  }
  return static_cast<int>(parsed);
}

bool better(const Config& candidate, const Config& current) {
  const double ratio = current.cycles > 0
      ? static_cast<double>(candidate.cycles) / static_cast<double>(current.cycles)
      : 1.0;
  if (ratio < 0.95) return true;
  if (ratio > 1.05) return false;
  if (candidate.block_n != current.block_n) return candidate.block_n > current.block_n;
  if (candidate.block_k != current.block_k) return candidate.block_k < current.block_k;
  return candidate.cycles < current.cycles;
}

Config choose_config(int m, int n, int k, int num_sms, int block_m, bool prefer_block_k_64) {
  const int forced_block_n = config_override("DG_SM120_GROUPED_BLOCK_N", 64, 128);
  int forced_block_k = config_override("DG_SM120_GROUPED_BLOCK_K", 32, 64);
  if (forced_block_k == 0 && prefer_block_k_64) forced_block_k = 64;
  Config best{};
  bool found = false;
  for (const int block_k : {32, 64}) {
    if (block_k == 32 && m < 2048) continue;
    if (forced_block_k != 0 && block_k != forced_block_k) continue;
    for (const int block_n : {64, 128}) {
      if (forced_block_n != 0 && block_n != forced_block_n) continue;
      const Config candidate = make_config(m, n, k, num_sms, block_m, block_n, block_k);
      if (!found || better(candidate, best)) {
        best = candidate;
        found = true;
      }
    }
  }
  if (!found) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT,
                 "no valid SM120 grouped GEMM configuration for the requested overrides");
  }
  return best;
}

std::string kernel_code(int groups, int n, int k, int num_sms, const Config& config) {
  std::ostringstream code;
  code << "#include <deep_gemm/impls/sm120_bf16_gemm.cuh>\n\n"
       << "using namespace deep_gemm;\n\n"
       << "static void __instantiate_kernel() {\n"
       << "  auto ptr = reinterpret_cast<void*>(&sm120_bf16_gemm_impl<\n"
       << "    0, " << n << ", " << k << ",\n"
       << "    " << groups << ",\n"
       << "    " << config.block_m << ", " << config.block_n << ", " << config.block_k << ",\n"
       << "    " << config.swizzle_a << ", " << config.swizzle_b << ",\n"
       << "    " << config.swizzle_d << ",\n"
       << "    " << config.stages << ",\n"
       << "    128, 256,\n"
       << "    " << num_sms << ",\n"
       << "    GemmType::MGroupedContiguous, false,\n"
       << "    cutlass::bfloat16_t,\n"
       << "    epilogue::transform::EpilogueIdentity,\n"
       << "    true, 2\n"
       << "  >);\n"
       << "};\n";
  return code.str();
}

void print_config_once(int m, int n, int k, int groups, int num_sms, const Config& config) {
  if (std::getenv("DG_PRINT_CONFIGS") == nullptr && std::getenv("DG_JIT_DEBUG") == nullptr) return;
  const std::string key = std::to_string(m) + ":" + std::to_string(n) + ":" + std::to_string(k);
  static std::unordered_set<std::string> printed;
  if (!printed.insert(key).second) return;
  std::cout << "DeepGEMM raw SM120 BF16 grouped GEMM(m=" << m << ", n=" << n
            << ", k=" << k << ", groups=" << groups << ", sms=" << num_sms
            << "): BM=" << config.block_m << " BN=" << config.block_n << " BK=" << config.block_k
            << " stages=" << config.stages << " smem=" << config.smem << std::endl;
}

}  // namespace

void launch_bf16_m_grouped_gemm_nt_contiguous(
    const deepgemm_bf16_m_grouped_gemm_nt_contiguous_params_t& params) {
  require_rank(params.a, 2, "a");
  require_rank(params.b, 3, "b");
  require_rank(params.d, 2, "d");
  require_rank(params.grouped_layout, 1, "grouped_layout");
  require_contiguous(params.a, "a");
  require_contiguous(params.b, "b");
  require_contiguous(params.d, "d");
  require_contiguous(params.grouped_layout, "grouped_layout");
  if (params.a.dtype != DEEPGEMM_DTYPE_BF16 || params.b.dtype != DEEPGEMM_DTYPE_BF16 ||
      params.d.dtype != DEEPGEMM_DTYPE_BF16 || params.grouped_layout.dtype != DEEPGEMM_DTYPE_I32) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT,
                 "SM120 grouped GEMM requires BF16 a/b/d and I32 grouped_layout");
  }

  const int groups = checked_i32(params.b.shape[0], "groups");
  const int m = checked_i32(params.a.shape[0], "m");
  const int k = checked_i32(params.a.shape[1], "k");
  const int n = checked_i32(params.b.shape[1], "n");
  const int block_m = checked_i32(params.row_alignment, "row_alignment");
  if (params.b.shape[2] != k || params.d.shape[0] != m || params.d.shape[1] != n ||
      params.grouped_layout.shape[0] != m || (block_m != 64 && block_m != 128) ||
      m % block_m != 0 || n % 8 != 0 || k % 16 != 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT,
                 "invalid SM120 grouped GEMM dimensions or 64-row alignment");
  }

  const auto device = current_device_info();
  if (device.major != 12 || (device.minor != 0 && device.minor != 1)) {
    std::ostringstream message;
    message << "SM120 grouped GEMM requires SM120/SM121, got compute capability "
            << device.major << '.' << device.minor;
    throw_status(DEEPGEMM_STATUS_UNSUPPORTED_ARCH, message.str());
  }

  const int num_sms = effective_num_sms();
  // GB10 measurements at Qwen MoE shapes consistently favor BK=64. Retain
  // the analytical selector for SM120 and small-M SM121 workloads.
  const bool prefer_block_k_64 = device.minor == 1 && m >= 2048;
  const Config config = choose_config(m, n, k, num_sms, block_m, prefer_block_k_64);
  print_config_once(m, n, k, groups, num_sms, config);
  const auto tma_a = make_tma_2d_desc(params.a.data, params.a.dtype, k, m,
                                      config.block_k, config.block_m, k, config.swizzle_a);
  const auto tma_b = make_tma_2d_desc(params.b.data, params.b.dtype, k, n * groups,
                                      config.block_k, config.block_n, k, config.swizzle_b);
  const auto tma_d = make_tma_2d_desc(params.d.data, params.d.dtype, n, m,
                                      config.block_n, config.block_m, n, config.swizzle_d);
  const auto runtime = build_kernel("sm120_bf16_m_grouped_gemm_contiguous",
                                    kernel_code(groups, n, k, num_sms, config));

  LaunchArgs launch;
  launch.grid_x = num_sms;
  launch.num_threads = 384;
  launch.smem_size = config.smem;
  launch.enable_pdl = pdl_enabled();
  auto* d = params.d.data;
  void* c = nullptr;
  void* a_ptr = nullptr;
  void* b_ptr = nullptr;
  auto* grouped_layout = const_cast<void*>(params.grouped_layout.data);
  void* tensor_map_buffer = nullptr;
  const uint32_t shape_m = static_cast<uint32_t>(m);
  const uint32_t shape_n = static_cast<uint32_t>(n);
  const uint32_t shape_k = static_cast<uint32_t>(k);
  launch_kernel(runtime, reinterpret_cast<CUstream>(params.stream), launch,
                d, c, a_ptr, b_ptr, grouped_layout, tensor_map_buffer,
                shape_m, shape_n, shape_k, tma_a, tma_b, tma_d);
}

}  // namespace deepgemm_rs
