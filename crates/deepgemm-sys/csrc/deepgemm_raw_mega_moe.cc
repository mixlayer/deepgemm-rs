#include "deepgemm_raw_mega_moe.h"

#include "deepgemm_raw_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>

namespace deepgemm_rs {
namespace {

constexpr int kSmemCapacity = 232448;
constexpr int kNumMaxRanks = 72;

struct MegaMoeConfig {
  int block_m;
  int block_n;
  int block_k;
  int load_block_m;
  int load_block_n;
  int store_block_m;
  int num_ring_tokens;
  int num_stages;
  int smem_size;
  int num_dispatch_threads;
  int num_non_epilogue_threads;
  int num_epilogue_threads;
  int num_bytes_per_pull;
};

// This must remain ABI-identical to deep_gemm::layout::SymBuffer<>.
struct SymBuffer {
  int64_t base;
  int64_t offsets[kNumMaxRanks];
  uint32_t rank_idx;
};

static_assert(sizeof(SymBuffer) == 592, "unexpected DeepGEMM SymBuffer ABI");

int64_t ceil_div(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

int align_int(int value, int alignment) {
  return static_cast<int>(ceil_div(value, alignment) * alignment);
}

bool env_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::atoi(value) != 0;
}

int as_i32(int64_t value, const char* name) {
  if (value < 0 || value > INT32_MAX) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, std::string(name) + " does not fit i32");
  }
  return static_cast<int>(value);
}

uint64_t checked_bytes(int64_t rows, int64_t columns, uint64_t element_size, const char* name) {
  if (rows < 0 || columns < 0 ||
      static_cast<uint64_t>(rows) > UINT64_MAX / static_cast<uint64_t>(columns) ||
      static_cast<uint64_t>(rows) * static_cast<uint64_t>(columns) > UINT64_MAX / element_size) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, std::string(name) + " byte size overflowed");
  }
  return static_cast<uint64_t>(rows) * static_cast<uint64_t>(columns) * element_size;
}

void require_tensor(
    const deepgemm_tensor_t& tensor,
    deepgemm_dtype_t dtype,
    uint32_t rank,
    const char* name) {
  if (tensor.data == nullptr || tensor.dtype != dtype || tensor.rank != rank) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, std::string(name) + " has an invalid pointer, dtype, or rank");
  }
  int64_t expected_stride = 1;
  for (int index = static_cast<int>(rank) - 1; index >= 0; --index) {
    if (tensor.shape[index] <= 0 || tensor.stride[index] != expected_stride) {
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, std::string(name) + " must be positive and contiguous");
    }
    expected_stride *= tensor.shape[index];
  }
}

void require_tensor_mut(
    const deepgemm_tensor_mut_t& tensor,
    deepgemm_dtype_t dtype,
    uint32_t rank,
    const char* name) {
  deepgemm_tensor_t immutable{
      tensor.data, tensor.dtype, tensor.rank,
      {tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]},
      {tensor.stride[0], tensor.stride[1], tensor.stride[2], tensor.stride[3]}};
  require_tensor(immutable, dtype, rank, name);
}

void require_range(uint64_t offset, uint64_t bytes, uint64_t total, const char* name) {
  if (offset > total || bytes > total - offset) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, std::string(name) + " exceeds symmetric buffer");
  }
}

MegaMoeConfig choose_config(
    int num_ranks,
    int num_experts,
    int num_tokens,
    int num_topk,
    int hidden,
    int num_ring_tokens) {
  const float expected = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
  int block_m;
  int store_block_m;
  int block_k_bytes;
  int num_epilogue_threads;
  if (expected <= 8.5f) {
    block_m = 16; store_block_m = 8; block_k_bytes = 256; num_epilogue_threads = 256;
  } else if (expected <= 16.5f) {
    block_m = 32; store_block_m = 16; block_k_bytes = 128; num_epilogue_threads = 256;
  } else if (expected <= 32.5f) {
    block_m = 64; store_block_m = 32; block_k_bytes = 128; num_epilogue_threads = 128;
  } else if (expected <= 64.5f) {
    block_m = 96; store_block_m = 16; block_k_bytes = 128; num_epilogue_threads = 256;
  } else if (expected <= 96.5f) {
    block_m = 128; store_block_m = 32; block_k_bytes = 128; num_epilogue_threads = 256;
  } else {
    block_m = 192; store_block_m = 32; block_k_bytes = 128; num_epilogue_threads = 256;
  }

  const int block_n = 128;
  const int block_k = block_k_bytes / 2;
  const int load_block_m = block_m / 2;
  const int num_dispatch_threads = 128;
  const int num_non_epilogue_threads = 128;
  int num_bytes_per_pull = hidden * 2;
  while (num_bytes_per_pull > 4096) {
    num_bytes_per_pull /= 2;
  }

  const int num_dispatch_warps = num_dispatch_threads / 32;
  const int num_epilogue_warps = num_epilogue_threads / 32;
  const int expert_counts = align_int(num_experts * static_cast<int>(sizeof(uint32_t)), 1024);
  // layout::Buffer(Data(bytes), warps, 1) is 1024-byte aligned between warp rows.
  const int send_buffers = align_int(num_bytes_per_pull * num_dispatch_warps, 1024);
  const int dispatch_size = expert_counts + send_buffers;
  const int epilogue_warpgroups = num_epilogue_warps / 4;
  const int cd_l1 = epilogue_warpgroups * store_block_m * (block_n / 2) * 2 * 2;
  const int cd_l2 = epilogue_warpgroups * store_block_m * block_n * 2;
  const int cd_size = align_int(std::max(cd_l1, cd_l2), 1024);
  constexpr int num_schedule_stages = 2;
  // Upstream sched::TaskInfo has eight uint32_t fields and 16-byte alignment.
  constexpr int task_info_bytes = 32;
  const int schedule_tasks = num_schedule_stages * task_info_bytes;
  const int barriers =
      (num_dispatch_warps + 4 + num_epilogue_warps * 2 + num_schedule_stages * 2) * 8;
  const int fixed = dispatch_size + cd_size + schedule_tasks + barriers + 4;
  const int stage_size = load_block_m * block_k * 2 + block_n * block_k * 2 + 16;
  const int num_stages = (kSmemCapacity - fixed) / stage_size;
  if (num_stages < 2) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE configuration has fewer than two pipeline stages");
  }

  return MegaMoeConfig{
      block_m,
      block_n,
      block_k,
      load_block_m,
      block_n,
      store_block_m,
      num_ring_tokens,
      num_stages,
      fixed + num_stages * stage_size,
      num_dispatch_threads,
      num_non_epilogue_threads,
      num_epilogue_threads,
      num_bytes_per_pull};
}

std::string float_literal(float value) {
  if (std::isinf(value)) {
    return value > 0 ? "3.402823466e+38F" : "-3.402823466e+38F";
  }
  std::ostringstream out;
  out << std::setprecision(9) << value << 'F';
  return out.str();
}

std::string generate_code(
    const deepgemm_bf16_mega_moe_params_t& params,
    int hidden,
    int intermediate_hidden,
    int num_experts,
    const MegaMoeConfig& config,
    int num_sms) {
  std::ostringstream code;
  code << "#include <deep_gemm/impls/sm100_bf16_mega_moe.cuh>\n\n"
       << "using namespace deep_gemm;\n\n"
       << "static void __instantiate_kernel() {\n"
       << "  auto ptr = reinterpret_cast<void*>(&sm100_bf16_mega_moe_impl<\n"
       << "    " << params.num_max_tokens_per_rank << ",\n"
       << "    " << hidden << ", " << intermediate_hidden << ",\n"
       << "    " << num_experts << ", 0,\n"
       << "    " << params.num_topk << ",\n"
       << "    " << config.block_m << ", " << config.block_n << ", " << config.block_k << ",\n"
       << "    " << config.store_block_m << ",\n"
       << "    " << config.num_ring_tokens << ",\n"
       << "    " << config.num_stages << ",\n"
       << "    " << config.num_bytes_per_pull << ",\n"
       << "    " << config.num_dispatch_threads << ", " << config.num_non_epilogue_threads << ", " << config.num_epilogue_threads << ",\n"
       << "    " << num_sms << ", " << params.num_ranks << ",\n"
       << "    " << float_literal(params.activation_clamp) << ",\n"
       << "    " << (params.fast_math ? "true" : "false") << ">);\n"
       << "}\n";
  return code.str();
}

void print_config_once(
    int num_tokens,
    int hidden,
    int intermediate_hidden,
    int num_experts,
    const MegaMoeConfig& config) {
  if (!env_enabled("DG_JIT_DEBUG") && !env_enabled("DG_PRINT_CONFIGS")) {
    return;
  }
  std::ostringstream key;
  key << num_tokens << ':' << hidden << ':' << intermediate_hidden << ':' << num_experts
      << ':' << config.block_m;
  static std::unordered_set<std::string> printed;
  if (!printed.insert(key.str()).second) {
    return;
  }
  std::cout << "DeepGEMM raw sm100 bf16_mega_moe(tokens=" << num_tokens
            << ", hidden=" << hidden << ", intermediate=" << intermediate_hidden
            << ", experts=" << num_experts << "): block_m=" << config.block_m
            << ", block_k=" << config.block_k
            << ", stages=" << config.num_stages << ", smem=" << config.smem_size << std::endl;
}

}  // namespace

void launch_bf16_mega_moe(const deepgemm_bf16_mega_moe_params_t& params) {
  const auto info = current_device_info();
  if (info.major != 10) {
    std::ostringstream message;
    message << "BF16 Mega MoE requires the SM10x family, got compute capability " << info.major << '.' << info.minor;
    throw_status(DEEPGEMM_STATUS_UNSUPPORTED_ARCH, message.str());
  }
  if (params.num_ranks <= 0 || params.num_ranks > kNumMaxRanks ||
      params.rank_idx < 0 || params.rank_idx >= params.num_ranks ||
      params.sym_buffer_ptrs == nullptr) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "invalid Mega MoE rank or pointer table");
  }
  if (params.num_max_tokens_per_rank <= 0 || params.num_ring_tokens <= 0 ||
      params.num_experts <= 0 || params.num_topk <= 0 ||
      (!std::isfinite(params.activation_clamp) && !std::isinf(params.activation_clamp)) ||
      params.activation_clamp < 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "invalid Mega MoE scalar configuration");
  }

  require_tensor(params.x, DEEPGEMM_DTYPE_BF16, 2, "x");
  require_tensor(params.topk_indices, DEEPGEMM_DTYPE_I64, 2, "topk_indices");
  require_tensor(params.topk_weights, DEEPGEMM_DTYPE_F32, 2, "topk_weights");
  require_tensor(params.l1_weights, DEEPGEMM_DTYPE_BF16, 3, "l1_weights");
  require_tensor(params.l2_weights, DEEPGEMM_DTYPE_BF16, 3, "l2_weights");
  require_tensor_mut(params.y, DEEPGEMM_DTYPE_BF16, 2, "y");
  require_tensor_mut(params.sym_buffer, DEEPGEMM_DTYPE_U8, 1, "sym_buffer");

  const int num_tokens = as_i32(params.x.shape[0], "num_tokens");
  const int hidden = as_i32(params.x.shape[1], "hidden");
  const int num_experts_per_rank = as_i32(params.l1_weights.shape[0], "num_experts_per_rank");
  const int intermediate_hidden_2 = as_i32(params.l1_weights.shape[1], "twice intermediate_hidden");
  const int intermediate_hidden = as_i32(params.l2_weights.shape[2], "intermediate_hidden");
  if (num_tokens > params.num_max_tokens_per_rank ||
      static_cast<int64_t>(intermediate_hidden_2) != 2ll * intermediate_hidden ||
      params.l1_weights.shape[2] != hidden || params.l2_weights.shape[0] != num_experts_per_rank ||
      params.l2_weights.shape[1] != hidden ||
      static_cast<int64_t>(params.num_experts) != static_cast<int64_t>(num_experts_per_rank) * params.num_ranks ||
      params.topk_indices.shape[0] != num_tokens || params.topk_indices.shape[1] != params.num_topk ||
      params.topk_weights.shape[0] != num_tokens || params.topk_weights.shape[1] != params.num_topk ||
      params.y.shape[0] != num_tokens || params.y.shape[1] != hidden) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "Mega MoE tensor shapes do not match the launch configuration");
  }

  const uint64_t buffer_bytes = static_cast<uint64_t>(params.sym_buffer.shape[0]);
  const uint64_t x_bytes = checked_bytes(num_tokens, hidden, 2, "x");
  const uint64_t indices_bytes = checked_bytes(num_tokens, params.num_topk, 8, "topk_indices");
  const uint64_t weights_bytes = checked_bytes(num_tokens, params.num_topk, 4, "topk_weights");
  require_range(params.x_offset_bytes, x_bytes, buffer_bytes, "x");
  require_range(params.topk_indices_offset_bytes, indices_bytes, buffer_bytes, "topk_indices");
  require_range(params.topk_weights_offset_bytes, weights_bytes, buffer_bytes, "topk_weights");
  require_range(params.l1_acts_offset_bytes, checked_bytes(params.num_ring_tokens, hidden, 2, "l1_acts"), buffer_bytes, "l1_acts");
  require_range(params.l2_acts_offset_bytes, checked_bytes(params.num_ring_tokens, intermediate_hidden, 2, "l2_acts"), buffer_bytes, "l2_acts");

  auto* buffer = static_cast<uint8_t*>(params.sym_buffer.data);
  const auto stream = reinterpret_cast<CUstream>(params.stream);
  copy_device_to_device_async(buffer + params.x_offset_bytes, params.x.data, x_bytes, stream);
  copy_device_to_device_async(buffer + params.topk_indices_offset_bytes, params.topk_indices.data, indices_bytes, stream);
  copy_device_to_device_async(buffer + params.topk_weights_offset_bytes, params.topk_weights.data, weights_bytes, stream);

  const auto config = choose_config(
      params.num_ranks, params.num_experts, num_tokens, params.num_topk,
      hidden, params.num_ring_tokens);
  const int num_sms = effective_num_sms();
  print_config_once(num_tokens, hidden, intermediate_hidden, params.num_experts, config);

  const auto l1_acts = buffer + params.l1_acts_offset_bytes;
  const auto l2_acts = buffer + params.l2_acts_offset_bytes;
  const auto tensor_map_l1_acts = make_tma_2d_desc(
      l1_acts, DEEPGEMM_DTYPE_BF16, hidden, params.num_ring_tokens,
      config.block_k, config.load_block_m, hidden, 128);
  const auto tensor_map_l1_weights = make_tma_2d_desc(
      params.l1_weights.data, DEEPGEMM_DTYPE_BF16,
      hidden, num_experts_per_rank * intermediate_hidden * 2,
      config.block_k, config.load_block_n, hidden, 128);
  const auto tensor_map_l1_output = make_tma_2d_desc(
      l2_acts, DEEPGEMM_DTYPE_BF16, intermediate_hidden, params.num_ring_tokens,
      config.block_n / 2, config.store_block_m, intermediate_hidden, 128);
  const auto tensor_map_l2_acts = make_tma_2d_desc(
      l2_acts, DEEPGEMM_DTYPE_BF16, intermediate_hidden, params.num_ring_tokens,
      config.block_k, config.load_block_m, intermediate_hidden, 128);
  const auto tensor_map_l2_weights = make_tma_2d_desc(
      params.l2_weights.data, DEEPGEMM_DTYPE_BF16,
      intermediate_hidden, num_experts_per_rank * hidden,
      config.block_k, config.load_block_n, intermediate_hidden, 128);

  SymBuffer sym_buffer{};
  sym_buffer.base = static_cast<int64_t>(params.sym_buffer_ptrs[params.rank_idx]);
  sym_buffer.rank_idx = static_cast<uint32_t>(params.rank_idx);
  for (int index = 0; index < kNumMaxRanks; ++index) {
    sym_buffer.offsets[index] = index < params.num_ranks
        ? static_cast<int64_t>(params.sym_buffer_ptrs[index]) - sym_buffer.base
        : 0;
  }

  const auto code = generate_code(params, hidden, intermediate_hidden, params.num_experts, config, num_sms);
  const auto runtime = build_kernel("sm100_bf16_mega_moe", code);
  const LaunchArgs launch_args{
      num_sms,
      1,
      config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
      config.smem_size,
      2,
      false};
  int* stats = nullptr;
  launch_kernel(
      runtime, stream, launch_args,
      params.y.data, stats, num_tokens, sym_buffer,
      tensor_map_l1_acts, tensor_map_l1_weights, tensor_map_l1_output,
      tensor_map_l2_acts, tensor_map_l2_weights,
      tensor_map_l1_acts, tensor_map_l1_weights, tensor_map_l1_output,
      tensor_map_l2_acts, tensor_map_l2_weights);
}

}  // namespace deepgemm_rs
