#pragma once

#include "deepgemm_c_api.h"

namespace deepgemm_rs {

void launch_bf16_mega_moe(const deepgemm_bf16_mega_moe_params_t& params);
void launch_sm90_fp8_mega_moe(const deepgemm_sm90_fp8_mega_moe_params_t& params);
void launch_fp8_mega_moe_interleave_l1_weights(
    const deepgemm_fp8_mega_moe_interleave_params_t& params);

}  // namespace deepgemm_rs
