#pragma once

#include "deepgemm_c_api.h"

namespace deepgemm_rs {

void launch_fp8_gemm_transform_scale(
    const deepgemm_fp8_gemm_scale_transform_params_t& params);

void launch_fp8_gemm_nt(
    const deepgemm_fp8_gemm_nt_params_t& params);

}  // namespace deepgemm_rs
