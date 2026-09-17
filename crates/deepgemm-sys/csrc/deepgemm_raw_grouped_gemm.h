#pragma once

#include "deepgemm_c_api.h"

namespace deepgemm_rs {

/// Launches an SM120-family BF16 M-grouped contiguous NT GEMM.
///
/// Tensor dimensions are `a[M,K]`, `b[G,N,K]`, `d[M,N]`, and
/// `grouped_layout[M]`. The layout contains the group index for valid rows and
/// `-1` for padding rows; each group segment must be 64-row aligned.
void launch_bf16_m_grouped_gemm_nt_contiguous(
    const deepgemm_bf16_m_grouped_gemm_nt_contiguous_params_t& params);

}  // namespace deepgemm_rs
