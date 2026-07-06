#pragma once

#include "deepgemm_c_api.h"

namespace deepgemm_rs {

void launch_fp8_fp4_mqa_logits(
    const deepgemm_mqa_logits_params_t& params);

void launch_paged_mqa_logits_metadata(
    const deepgemm_paged_mqa_logits_metadata_params_t& params);

void launch_fp8_fp4_paged_mqa_logits(
    const deepgemm_paged_mqa_logits_params_t& params);

}  // namespace deepgemm_rs
