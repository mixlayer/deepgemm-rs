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

struct Sm90Config {
  int block_m;
  int block_n;
  int block_k;
  int num_max_pool_tokens;
  int num_padded_sf_pool_tokens;
  int sf_pool_stride_tokens;
  int num_experts_per_wave;
  int num_sms;
  int num_stages;
  int smem_size;
  int num_dispatch_threads;
  int num_non_epilogue_threads;
  int num_epilogue_threads;
  bool direct_l2_scatter;
  bool nmajor_schedule;
  bool one_warp_cleanup;
  bool swap_ab;
};

// This must remain ABI-identical to deep_gemm::layout::SymBuffer<>.
struct SymBuffer {
  int64_t base;
  int64_t offsets[kNumMaxRanks];
  uint32_t rank_idx;
};

static_assert(sizeof(SymBuffer) == 592, "unexpected DeepGEMM SymBuffer ABI");

int ceil_div(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

int align_int(int value, int alignment) {
  return ceil_div(value, alignment) * alignment;
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
      (rows != 0 && static_cast<uint64_t>(columns) > UINT64_MAX / static_cast<uint64_t>(rows)) ||
      static_cast<uint64_t>(rows) * static_cast<uint64_t>(columns) > UINT64_MAX / element_size) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, std::string(name) + " byte size overflowed");
  }
  return static_cast<uint64_t>(rows) * static_cast<uint64_t>(columns) * element_size;
}

void require_tensor(const deepgemm_tensor_t& tensor, deepgemm_dtype_t dtype, uint32_t rank, const char* name) {
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

void require_tensor_mut(const deepgemm_tensor_mut_t& tensor, deepgemm_dtype_t dtype, uint32_t rank, const char* name) {
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

int generic_experts_per_wave(int experts, int tokens, int topk, int intermediate, int block_n, int num_sms) {
  const float expected = static_cast<float>(tokens) * topk / experts;
  if (expected < 1.0f) {
    return experts;
  }
  const int m_blocks = ceil_div(static_cast<int>(std::ceil(expected)), 64);
  const int n_blocks = (2 * intermediate) / block_n;
  const int blocks_per_expert = m_blocks * n_blocks;
  const int minimum = blocks_per_expert > 0 ? ceil_div(2 * num_sms, blocks_per_expert) : 1;
  if (minimum >= experts) {
    return experts;
  }
  if (blocks_per_expert >= num_sms) {
    return minimum;
  }
  const int maximum = std::min(experts, minimum * 2);
  int best = minimum;
  float best_tail = -1.0f;
  for (int candidate = minimum; candidate <= maximum; ++candidate) {
    const int remainder = experts % candidate;
    const float tail = remainder == 0 ? 1.0f : static_cast<float>(remainder) / candidate;
    if (tail > best_tail) {
      best_tail = tail;
      best = candidate;
    }
  }
  return best;
}

int normalize_experts_per_wave(int experts, int requested) {
  if (experts % requested == 0) {
    return requested;
  }
  for (int candidate = requested + 1; candidate < experts; ++candidate) {
    if (experts % candidate == 0) {
      return candidate;
    }
  }
  return experts;
}

Sm90Config choose_sm90_config(
    int num_ranks,
    int num_experts,
    int max_tokens,
    int tokens,
    int topk,
    int hidden,
    int intermediate,
    int num_sms) {
  const int experts_per_rank = num_experts / num_ranks;
  const int routed_tokens = tokens * topk;
  const bool swap_ab = routed_tokens > 0 && routed_tokens <= 16 * experts_per_rank;
  const int block_n = swap_ab ? 128 : 256;
  const int num_epilogue_threads = 256;
  const int num_dispatch_threads = 64;
  const int num_non_epilogue_threads = 64;
  const int max_pool = align_int(
      num_ranks * max_tokens * std::min(topk, experts_per_rank) + experts_per_rank * 191,
      384);
  const int padded_sf_pool = (max_pool / 64) * 128;

  int requested_epw;
  const float load = static_cast<float>(routed_tokens) / experts_per_rank;
  if (load < 1.0f || load > 4.0f) {
    requested_epw = experts_per_rank;
  } else {
    requested_epw = generic_experts_per_wave(
        experts_per_rank, tokens, topk, intermediate, block_n, num_sms);
  }
  const int experts_per_wave = normalize_experts_per_wave(experts_per_rank, requested_epw);

  const int dispatch_warps = num_dispatch_threads / 32;
  const int epilogue_warps = num_epilogue_threads / 32;
  const int expert_counts = align_int(num_experts * static_cast<int>(sizeof(uint32_t)), 1024);
  const int send_buffers = align_int(hidden * dispatch_warps, 1024);
  const int dispatch_size = expert_counts + send_buffers;
  const int epilogue_warpgroups = num_epilogue_threads / 128;
  const int wg_block_n = block_n / epilogue_warpgroups;
  const int cd_l1 = epilogue_warpgroups * 64 * (wg_block_n / 2);
  const int cd_l2 = epilogue_warpgroups * 64 * wg_block_n * 2;
  const int cd_swap_l1 = swap_ab ? 64 * (block_n / 2) * 5 : 0;
  const int cd_size = align_int(std::max({cd_l1, cd_l2, cd_swap_l1}), 1024);
  const int sfa_per_stage = (128 / 64) * align_int(64 * 4, 128);
  const int stage_size = 64 * 128 + block_n * 128 + sfa_per_stage;
  const int fixed_barriers = (dispatch_warps + 2 * epilogue_warps) * 8;
  const int fixed = dispatch_size + cd_size + fixed_barriers;
  const int stage_with_barriers = stage_size + 16;
  const int stages = (kSmemCapacity - fixed) / stage_with_barriers;
  if (stages < 2) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM90 FP8 Mega MoE configuration has fewer than two pipeline stages");
  }
  return Sm90Config{
      64, block_n, 128,
      max_pool, padded_sf_pool, padded_sf_pool,
      experts_per_wave, num_sms,
      stages, fixed + stages * stage_with_barriers,
      num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
      false, false, false, swap_ab};
}

std::string float_literal(float value) {
  if (std::isinf(value)) {
    return value > 0 ? "3.402823466e+38F" : "-3.402823466e+38F";
  }
  std::ostringstream out;
  out << std::setprecision(9) << value << 'F';
  return out.str();
}

std::string generate_sm90_code(
    const deepgemm_sm90_fp8_mega_moe_params_t& params,
    int hidden,
    int intermediate,
    const Sm90Config& config,
    bool linear2) {
  const char* symbol = linear2 ? "sm90_fp8_mega_moe_l2_impl" : "sm90_fp8_mega_moe_l1_impl";
  std::ostringstream code;
  code << "#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>\n\n"
       << "using namespace deep_gemm;\n\n"
       << "static void __instantiate_kernel() {\n"
       << "  auto ptr = reinterpret_cast<void*>(&" << symbol << "<\n"
       << "    " << params.num_max_tokens_per_rank << ",\n"
       << "    " << hidden << ", " << intermediate << ",\n"
       << "    " << params.num_experts << ", " << params.num_topk << ",\n"
       << "    " << config.num_experts_per_wave << ",\n"
       << "    " << config.block_m << ", " << config.block_n << ", " << config.block_k << ",\n"
       << "    " << config.num_max_pool_tokens << ",\n"
       << "    " << config.num_padded_sf_pool_tokens << ", " << config.sf_pool_stride_tokens << ",\n"
       << "    " << config.num_stages << ",\n"
       << "    " << config.num_dispatch_threads << ", " << config.num_non_epilogue_threads << ", " << config.num_epilogue_threads << ",\n"
       << "    " << config.num_sms << ", " << params.num_ranks << ",\n"
       << "    " << float_literal(params.activation_clamp) << ",\n"
       << "    " << (params.fast_math ? "true" : "false") << ",\n"
       << "    " << (config.swap_ab ? "true" : "false") << ",\n"
       << "    false";
  if (linear2) {
    code << ",\n    " << (config.direct_l2_scatter ? "true" : "false")
         << ", " << (config.nmajor_schedule ? "true" : "false")
         << ", " << (config.one_warp_cleanup ? "true" : "false");
  } else {
    code << ",\n    " << (config.nmajor_schedule ? "true" : "false");
  }
  code << ">);\n}\n";
  return code.str();
}

void print_config_once(int tokens, int hidden, int intermediate, int experts, const Sm90Config& config) {
  if (!env_enabled("DG_JIT_DEBUG") && !env_enabled("DG_PRINT_CONFIGS")) {
    return;
  }
  std::ostringstream key;
  key << tokens << ':' << hidden << ':' << intermediate << ':' << experts << ':' << config.block_n;
  static std::unordered_set<std::string> printed;
  if (!printed.insert(key.str()).second) {
    return;
  }
  std::cout << "DeepGEMM raw sm90 fp8_mega_moe(tokens=" << tokens
            << ", hidden=" << hidden << ", intermediate=" << intermediate
            << ", experts=" << experts << "): block_n=" << config.block_n
            << ", swap_ab=" << config.swap_ab << ", epw=" << config.num_experts_per_wave
            << ", stages=" << config.num_stages << ", smem=" << config.smem_size << std::endl;
}

}  // namespace

void launch_sm90_fp8_mega_moe(const deepgemm_sm90_fp8_mega_moe_params_t& params) {
  const auto info = current_device_info();
  if (info.major != 9) {
    std::ostringstream message;
    message << "FP8 Mega MoE requires the SM9x family, got compute capability " << info.major << '.' << info.minor;
    throw_status(DEEPGEMM_STATUS_UNSUPPORTED_ARCH, message.str());
  }
  if (params.num_ranks <= 0 || params.num_ranks > kNumMaxRanks ||
      params.rank_idx < 0 || params.rank_idx >= params.num_ranks || params.sym_buffer_ptrs == nullptr ||
      params.num_max_tokens_per_rank <= 0 || params.num_experts <= 0 || params.num_topk <= 0 ||
      params.num_experts % params.num_ranks != 0 || params.num_max_tokens_per_rank % 128 != 0 ||
      params.activation_clamp < 0 || std::isnan(params.activation_clamp)) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "invalid SM90 FP8 Mega MoE scalar configuration");
  }

  require_tensor(params.x, DEEPGEMM_DTYPE_FP8_E4M3, 2, "x");
  require_tensor(params.x_scale, DEEPGEMM_DTYPE_F32, 2, "x_scale");
  require_tensor(params.topk_indices, DEEPGEMM_DTYPE_I64, 2, "topk_indices");
  require_tensor(params.topk_weights, DEEPGEMM_DTYPE_F32, 2, "topk_weights");
  require_tensor(params.l1_weights, DEEPGEMM_DTYPE_FP8_E4M3, 3, "l1_weights");
  require_tensor(params.l1_weights_scale, DEEPGEMM_DTYPE_F32, 3, "l1_weights_scale");
  require_tensor(params.l2_weights, DEEPGEMM_DTYPE_FP8_E4M3, 3, "l2_weights");
  require_tensor(params.l2_weights_scale, DEEPGEMM_DTYPE_F32, 3, "l2_weights_scale");
  require_tensor_mut(params.y, DEEPGEMM_DTYPE_BF16, 2, "y");
  require_tensor_mut(params.sym_buffer, DEEPGEMM_DTYPE_U8, 1, "sym_buffer");

  const int tokens = as_i32(params.x.shape[0], "tokens");
  const int hidden = as_i32(params.x.shape[1], "hidden");
  const int local_experts = as_i32(params.l1_weights.shape[0], "local_experts");
  const int twice_intermediate = as_i32(params.l1_weights.shape[1], "twice_intermediate");
  const int intermediate = as_i32(params.l2_weights.shape[2], "intermediate");
  if (tokens > params.num_max_tokens_per_rank || hidden % 256 != 0 || intermediate % 128 != 0 ||
      twice_intermediate != 2 * intermediate || params.l1_weights.shape[2] != hidden ||
      params.l2_weights.shape[0] != local_experts || params.l2_weights.shape[1] != hidden ||
      local_experts * params.num_ranks != params.num_experts ||
      params.x_scale.shape[0] != tokens || params.x_scale.shape[1] != hidden / 128 ||
      params.topk_indices.shape[0] != tokens || params.topk_indices.shape[1] != params.num_topk ||
      params.topk_weights.shape[0] != tokens || params.topk_weights.shape[1] != params.num_topk ||
      params.l1_weights_scale.shape[0] != local_experts ||
      params.l1_weights_scale.shape[1] != twice_intermediate / 128 ||
      params.l1_weights_scale.shape[2] != hidden / 128 ||
      params.l2_weights_scale.shape[0] != local_experts ||
      params.l2_weights_scale.shape[1] != hidden / 128 ||
      params.l2_weights_scale.shape[2] != intermediate / 128 ||
      params.y.shape[0] != tokens || params.y.shape[1] != hidden) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "SM90 FP8 Mega MoE tensor shapes do not match");
  }

  const int num_sms = effective_num_sms();
  const auto config = choose_sm90_config(
      params.num_ranks, params.num_experts, params.num_max_tokens_per_rank,
      tokens, params.num_topk, hidden, intermediate, num_sms);
  print_config_once(tokens, hidden, intermediate, params.num_experts, config);

  const uint64_t buffer_bytes = static_cast<uint64_t>(params.sym_buffer.shape[0]);
  require_range(params.x_offset_bytes, checked_bytes(tokens, hidden, 1, "x"), buffer_bytes, "x");
  require_range(params.x_scale_offset_bytes, checked_bytes(tokens, hidden / 128, 4, "x_scale"), buffer_bytes, "x_scale");
  require_range(params.topk_indices_offset_bytes, checked_bytes(tokens, params.num_topk, 8, "topk_indices"), buffer_bytes, "topk_indices");
  require_range(params.topk_weights_offset_bytes, checked_bytes(tokens, params.num_topk, 4, "topk_weights"), buffer_bytes, "topk_weights");
  require_range(params.l1_acts_offset_bytes, checked_bytes(config.num_max_pool_tokens, hidden, 1, "l1_acts"), buffer_bytes, "l1_acts");
  require_range(params.l1_acts_scale_offset_bytes, checked_bytes(config.num_padded_sf_pool_tokens, hidden / 128, 4, "l1_acts_scale"), buffer_bytes, "l1_acts_scale");
  require_range(params.l2_acts_offset_bytes, checked_bytes(config.num_max_pool_tokens, intermediate, 1, "l2_acts"), buffer_bytes, "l2_acts");
  require_range(params.l2_acts_scale_offset_bytes, checked_bytes(config.num_padded_sf_pool_tokens, intermediate / 64, 4, "l2_acts_scale"), buffer_bytes, "l2_acts_scale");

  auto* buffer = static_cast<uint8_t*>(params.sym_buffer.data);
  const auto stream = reinterpret_cast<CUstream>(params.stream);
  copy_device_to_device_async(buffer + params.x_offset_bytes, params.x.data,
                              checked_bytes(tokens, hidden, 1, "x"), stream);
  copy_device_to_device_async(buffer + params.x_scale_offset_bytes, params.x_scale.data,
                              checked_bytes(tokens, hidden / 128, 4, "x_scale"), stream);
  copy_device_to_device_async(buffer + params.topk_indices_offset_bytes, params.topk_indices.data,
                              checked_bytes(tokens, params.num_topk, 8, "topk_indices"), stream);
  copy_device_to_device_async(buffer + params.topk_weights_offset_bytes, params.topk_weights.data,
                              checked_bytes(tokens, params.num_topk, 4, "topk_weights"), stream);

  auto* l1_acts = buffer + params.l1_acts_offset_bytes;
  auto* l1_acts_scale = buffer + params.l1_acts_scale_offset_bytes;
  auto* l2_acts = buffer + params.l2_acts_offset_bytes;
  auto* l2_acts_scale = buffer + params.l2_acts_scale_offset_bytes;
  const int tma_block_n = std::min(config.block_n, 256);
  const auto map_l1_acts = make_tma_2d_desc(
      l1_acts, DEEPGEMM_DTYPE_FP8_E4M3, hidden, config.num_max_pool_tokens,
      128, config.block_m, hidden, 128);
  const auto map_l1_acts_scale = make_tma_sf_desc(
      l1_acts_scale, DEEPGEMM_DTYPE_F32, config.sf_pool_stride_tokens, hidden,
      config.block_m, 128);
  const auto map_l1_weights = make_tma_2d_desc(
      params.l1_weights.data, DEEPGEMM_DTYPE_FP8_E4M3,
      hidden, local_experts * twice_intermediate,
      128, tma_block_n, hidden, 128);
  const int l1_output_box_n = config.block_n / 2;
  const auto map_l1_output = make_tma_2d_desc(
      l2_acts, DEEPGEMM_DTYPE_FP8_E4M3,
      intermediate, config.num_max_pool_tokens,
      l1_output_box_n, config.block_m, intermediate, 0);
  const auto map_l2_acts = make_tma_2d_desc(
      l2_acts, DEEPGEMM_DTYPE_FP8_E4M3,
      intermediate, config.num_max_pool_tokens,
      128, config.block_m, intermediate, 128);
  const auto map_l2_acts_scale = make_tma_sf_desc(
      l2_acts_scale, DEEPGEMM_DTYPE_F32, config.sf_pool_stride_tokens, intermediate,
      config.block_m, 64);
  const auto map_l2_weights = make_tma_2d_desc(
      params.l2_weights.data, DEEPGEMM_DTYPE_FP8_E4M3,
      intermediate, local_experts * hidden,
      128, tma_block_n, intermediate, 128);

  SymBuffer sym_buffer{};
  sym_buffer.base = static_cast<int64_t>(params.sym_buffer_ptrs[params.rank_idx]);
  sym_buffer.rank_idx = static_cast<uint32_t>(params.rank_idx);
  for (int index = 0; index < kNumMaxRanks; ++index) {
    sym_buffer.offsets[index] = index < params.num_ranks
        ? static_cast<int64_t>(params.sym_buffer_ptrs[index]) - sym_buffer.base
        : 0;
  }

  int* stats = nullptr;
  for (const bool linear2 : {false, true}) {
    const auto code = generate_sm90_code(params, hidden, intermediate, config, linear2);
    const auto runtime = build_kernel(
        linear2 ? "sm90_fp8_mega_moe_l2_impl" : "sm90_fp8_mega_moe_l1_impl",
        code);
    const LaunchArgs launch_args{
        config.num_sms,
        1,
        config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
        config.smem_size,
        1,
        false};
    launch_kernel(
        runtime, stream, launch_args,
        params.y.data, stats, tokens, sym_buffer,
        map_l1_acts, map_l1_acts_scale, map_l1_weights,
        params.l1_weights_scale.data, map_l1_output,
        map_l2_acts, map_l2_acts_scale, map_l2_weights,
        params.l2_weights_scale.data);
  }
}

void launch_fp8_mega_moe_interleave_l1_weights(
    const deepgemm_fp8_mega_moe_interleave_params_t& params) {
  require_tensor(params.weights, DEEPGEMM_DTYPE_FP8_E4M3, 3, "weights");
  require_tensor_mut(params.output, DEEPGEMM_DTYPE_FP8_E4M3, 3, "output");
  if (params.weights.shape[0] != params.output.shape[0] ||
      params.weights.shape[1] != params.output.shape[1] ||
      params.weights.shape[2] != params.output.shape[2] ||
      params.weights.shape[1] % 16 != 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "FP8 Mega MoE interleave shapes are invalid");
  }
  const int64_t num_elements =
      params.weights.shape[0] * params.weights.shape[1] * params.weights.shape[2];
  const int64_t intermediate = params.weights.shape[1] / 2;
  const int64_t hidden = params.weights.shape[2];
  const std::string code = R"(
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void fp8_mega_moe_interleave_l1(
    const uint8_t* input,
    uint8_t* output,
    int64_t num_elements,
    int64_t intermediate,
    int64_t hidden,
    bool input_up_gate) {
  for (int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       output_index < num_elements;
       output_index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t column = output_index % hidden;
    const int64_t output_row = output_index / hidden;
    const int64_t expert = output_row / (2 * intermediate);
    const int64_t row = output_row % (2 * intermediate);
    const int64_t block = row / 16;
    const int64_t within = row % 16;
    const bool gate = within < 8;
    const int64_t logical_row = block * 8 + (gate ? within : within - 8);
    const int64_t half_offset = gate == input_up_gate ? intermediate : 0;
    const int64_t input_row = expert * (2 * intermediate) + half_offset + logical_row;
    output[output_index] = input[input_row * hidden + column];
  }
}
)";
  const auto runtime = build_kernel("fp8_mega_moe_interleave_l1", code);
  const LaunchArgs launch_args{
      effective_num_sms() * 4,
      1,
      256,
      0,
      1,
      false};
  launch_kernel(
      runtime,
      reinterpret_cast<CUstream>(params.stream),
      launch_args,
      params.weights.data,
      params.output.data,
      num_elements,
      intermediate,
      hidden,
      params.input_up_gate);
}

}  // namespace deepgemm_rs
