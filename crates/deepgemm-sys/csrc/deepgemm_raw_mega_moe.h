#pragma once

#include "deepgemm_c_api.h"

namespace deepgemm_rs {

void launch_bf16_mega_moe(const deepgemm_bf16_mega_moe_params_t& params);

}  // namespace deepgemm_rs
