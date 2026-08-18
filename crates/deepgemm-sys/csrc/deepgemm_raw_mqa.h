#pragma once

#include "deepgemm_c_api.h"

namespace deepgemm_rs {

constexpr bool sm90_native_next_n(int64_t next_n) {
  return next_n == 1 || next_n == 2 || next_n == 4;
}

constexpr int64_t sm90_num_kv_multicast(int64_t next_n) {
  return next_n == 4 ? 2 : 1;
}

void launch_fp8_fp4_mqa_logits(
    const deepgemm_mqa_logits_params_t& params);

void launch_paged_mqa_logits_metadata(
    const deepgemm_paged_mqa_logits_metadata_params_t& params);

void launch_fp8_fp4_paged_mqa_logits(
    const deepgemm_paged_mqa_logits_params_t& params);

}  // namespace deepgemm_rs
