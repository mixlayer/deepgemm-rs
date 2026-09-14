//! Mega MoE symmetric-buffer planning.
//!
//! Upstream calls this kernel family "Mega MoE". Its input activations and
//! routing tensors live inside one symmetric allocation on every rank. These
//! helpers expose the exact allocation contract without requiring CUDA or a
//! distributed runtime.

use std::mem::MaybeUninit;

use core::ffi::c_void;

use crate::{
    DType, Error, Result, TensorArg, TensorOut,
    tensor::{require_contiguous, require_dtype, usize_to_i64},
};

/// Matrix-multiply implementation used by a Mega MoE launch.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MegaMoeMmaKind {
    /// FP8 activations with packed FP4 expert weights and UE8M0 scales.
    Fp8Fp4,
    /// BF16 activations and expert weights.
    Bf16,
}

impl MegaMoeMmaKind {
    const fn to_sys(self) -> deepgemm_sys::deepgemm_mega_moe_mma_kind_t {
        match self {
            Self::Fp8Fp4 => deepgemm_sys::DEEPGEMM_MEGA_MOE_MMA_FP8_FP4,
            Self::Bf16 => deepgemm_sys::DEEPGEMM_MEGA_MOE_MMA_BF16,
        }
    }
}

/// Topology values used to derive valid Mega MoE ring-buffer capacities.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MegaMoeRingConfig {
    pub num_ranks: usize,
    pub num_experts: usize,
    pub num_max_tokens_per_rank: usize,
    pub num_topk: usize,
}

/// Inclusive range of valid ring-buffer capacities, in tokens.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MegaMoeRingLimits {
    pub min_tokens: usize,
    pub max_tokens: usize,
}

/// Parameters needed to derive the per-rank symmetric-buffer allocation.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MegaMoeBufferConfig {
    pub num_ranks: usize,
    pub num_experts: usize,
    pub num_max_tokens_per_rank: usize,
    pub num_topk: usize,
    pub hidden: usize,
    pub intermediate_hidden: usize,
    pub num_ring_tokens: usize,
    pub mma_kind: MegaMoeMmaKind,
}

impl MegaMoeBufferConfig {
    /// Returns the topology subset used for ring-limit derivation.
    pub const fn ring_config(self) -> MegaMoeRingConfig {
        MegaMoeRingConfig {
            num_ranks: self.num_ranks,
            num_experts: self.num_experts,
            num_max_tokens_per_rank: self.num_max_tokens_per_rank,
            num_topk: self.num_topk,
        }
    }
}

/// A typed 2D view into a rank's symmetric allocation.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MegaMoeBufferView {
    /// Byte offset from the beginning of the symmetric allocation.
    pub offset_bytes: usize,
    pub dtype: DType,
    pub shape: [usize; 2],
    pub strides: [isize; 2],
    pub element_count: usize,
}

/// Complete per-rank symmetric-buffer layout consumed by Mega MoE.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MegaMoeBufferLayout {
    /// Total allocation size required on every rank.
    pub total_bytes: usize,
    /// Opaque workspace at the beginning of the allocation.
    pub workspace_bytes: usize,
    pub x: MegaMoeBufferView,
    /// Present only for `Fp8Fp4`.
    pub x_scale: Option<MegaMoeBufferView>,
    pub topk_indices: MegaMoeBufferView,
    pub topk_weights: MegaMoeBufferView,
    pub l1_acts: MegaMoeBufferView,
    /// Present only for `Fp8Fp4`.
    pub l1_acts_scale: Option<MegaMoeBufferView>,
    pub l2_acts: MegaMoeBufferView,
    /// Present only for `Fp8Fp4`.
    pub l2_acts_scale: Option<MegaMoeBufferView>,
    /// Internal BF16 cross-rank combine region.
    pub combine: MegaMoeBufferView,
}

/// Static topology and symmetric-buffer layout for BF16 Mega MoE launches.
#[derive(Debug, Copy, Clone)]
pub struct Bf16MegaMoeSpec {
    /// Global number of experts across all expert-parallel ranks.
    pub num_experts: usize,
    /// Number of experts selected for every token.
    pub num_topk: usize,
    /// Maximum input tokens accepted on each rank.
    pub num_max_tokens_per_rank: usize,
    /// Token capacity of the persistent expert ring.
    pub num_ring_tokens: usize,
    /// Exact symmetric-buffer allocation layout.
    pub buffer_layout: MegaMoeBufferLayout,
}

/// Device pointers and runtime values for one BF16 Mega MoE launch.
///
/// All tensors and `sym_buffer` must reside on the same current CUDA device.
/// `sym_buffer_ptrs` contains one peer-accessible device address per rank; for
/// a single rank this is just `sym_buffer.data`.
#[derive(Debug)]
pub struct Bf16MegaMoeLaunch<'a> {
    /// Contiguous BF16 `[tokens, hidden]` input activations.
    pub x: TensorArg<2>,
    /// Contiguous I64 `[tokens, topk]` global expert indices.
    pub topk_indices: TensorArg<2>,
    /// Contiguous F32 `[tokens, topk]` normalized routing weights.
    pub topk_weights: TensorArg<2>,
    /// Contiguous, gate/up-interleaved BF16 `[local_experts, 2 * intermediate, hidden]` weights.
    pub l1_weights: TensorArg<3>,
    /// Contiguous BF16 `[local_experts, hidden, intermediate]` down weights.
    pub l2_weights: TensorArg<3>,
    /// Contiguous BF16 `[tokens, hidden]` output.
    pub y: TensorOut<2>,
    /// Contiguous U8 allocation described by `spec.buffer_layout`.
    pub sym_buffer: TensorOut<1>,
    /// Peer-visible symmetric allocation addresses for all ranks.
    pub sym_buffer_ptrs: &'a [u64],
    /// This process's expert-parallel rank.
    pub rank_idx: usize,
    /// Optional finite nonnegative SwiGLU clamp; infinity disables clamping.
    pub activation_clamp: f32,
    /// Enables the upstream fast-math implementation.
    pub fast_math: bool,
    /// CUDA stream handle used for staging copies and the kernel launch; null selects the default stream.
    pub stream: *mut c_void,
}

/// Launches DeepGEMM's fused BF16 Mega MoE persistent kernel.
///
/// The operation stages `x`, `topk_indices`, and `topk_weights` into the
/// provided symmetric allocation, then performs dispatch, two expert GEMMs,
/// SwiGLU, routing-weight application, and cross-rank combine. The current
/// upstream kernel supports SM100 only.
pub fn bf16_mega_moe(spec: &Bf16MegaMoeSpec, launch: Bf16MegaMoeLaunch<'_>) -> Result<()> {
    validate_bf16_launch(spec, &launch)?;
    let raw = deepgemm_sys::deepgemm_bf16_mega_moe_params_t {
        x: launch.x.to_raw()?,
        topk_indices: launch.topk_indices.to_raw()?,
        topk_weights: launch.topk_weights.to_raw()?,
        l1_weights: launch.l1_weights.to_raw()?,
        l2_weights: launch.l2_weights.to_raw()?,
        y: launch.y.to_raw()?,
        sym_buffer: launch.sym_buffer.to_raw()?,
        sym_buffer_ptrs: launch.sym_buffer_ptrs.as_ptr(),
        num_ranks: i32::try_from(launch.sym_buffer_ptrs.len())
            .map_err(|_| Error::InvalidArgument("num_ranks does not fit i32".into()))?,
        rank_idx: i32::try_from(launch.rank_idx)
            .map_err(|_| Error::InvalidArgument("rank_idx does not fit i32".into()))?,
        num_max_tokens_per_rank: i32::try_from(spec.num_max_tokens_per_rank).map_err(|_| {
            Error::InvalidArgument("num_max_tokens_per_rank does not fit i32".into())
        })?,
        num_ring_tokens: i32::try_from(spec.num_ring_tokens)
            .map_err(|_| Error::InvalidArgument("num_ring_tokens does not fit i32".into()))?,
        num_experts: i32::try_from(spec.num_experts)
            .map_err(|_| Error::InvalidArgument("num_experts does not fit i32".into()))?,
        num_topk: i32::try_from(spec.num_topk)
            .map_err(|_| Error::InvalidArgument("num_topk does not fit i32".into()))?,
        x_offset_bytes: u64::try_from(spec.buffer_layout.x.offset_bytes)
            .map_err(|_| Error::InvalidArgument("x offset does not fit u64".into()))?,
        topk_indices_offset_bytes: u64::try_from(spec.buffer_layout.topk_indices.offset_bytes)
            .map_err(|_| Error::InvalidArgument("topk indices offset does not fit u64".into()))?,
        topk_weights_offset_bytes: u64::try_from(spec.buffer_layout.topk_weights.offset_bytes)
            .map_err(|_| Error::InvalidArgument("topk weights offset does not fit u64".into()))?,
        l1_acts_offset_bytes: u64::try_from(spec.buffer_layout.l1_acts.offset_bytes)
            .map_err(|_| Error::InvalidArgument("l1 acts offset does not fit u64".into()))?,
        l2_acts_offset_bytes: u64::try_from(spec.buffer_layout.l2_acts.offset_bytes)
            .map_err(|_| Error::InvalidArgument("l2 acts offset does not fit u64".into()))?,
        activation_clamp: launch.activation_clamp,
        fast_math: launch.fast_math,
        stream: launch.stream,
    };
    // SAFETY: all tensor descriptors and the host pointer table remain alive
    // for this synchronous FFI call; CUDA work is enqueued on `launch.stream`.
    let status = unsafe { deepgemm_sys::deepgemm_bf16_mega_moe(&raw) };
    Error::check_raw_status(status)
}

fn validate_bf16_launch(spec: &Bf16MegaMoeSpec, launch: &Bf16MegaMoeLaunch<'_>) -> Result<()> {
    for (tensor, dtype, name) in [
        (&launch.x.spec, DType::BF16, "x"),
        (&launch.topk_indices.spec, DType::I64, "topk_indices"),
        (&launch.topk_weights.spec, DType::F32, "topk_weights"),
    ] {
        require_dtype(tensor, dtype, name)?;
        require_contiguous(tensor, name)?;
    }
    require_dtype(&launch.l1_weights.spec, DType::BF16, "l1_weights")?;
    require_dtype(&launch.l2_weights.spec, DType::BF16, "l2_weights")?;
    require_dtype(&launch.y.spec, DType::BF16, "y")?;
    require_dtype(&launch.sym_buffer.spec, DType::U8, "sym_buffer")?;
    require_contiguous(&launch.l1_weights.spec, "l1_weights")?;
    require_contiguous(&launch.l2_weights.spec, "l2_weights")?;
    require_contiguous(&launch.y.spec, "y")?;
    require_contiguous(&launch.sym_buffer.spec, "sym_buffer")?;

    let [tokens, hidden] = launch.x.spec.shape;
    let [local_experts, intermediate_twice, l1_hidden] = launch.l1_weights.spec.shape;
    let [l2_experts, l2_hidden, intermediate] = launch.l2_weights.spec.shape;
    if tokens == 0
        || tokens > spec.num_max_tokens_per_rank
        || hidden == 0
        || intermediate == 0
        || intermediate.checked_mul(2) != Some(intermediate_twice)
        || l1_hidden != hidden
        || l2_hidden != hidden
        || l2_experts != local_experts
        || local_experts.checked_mul(launch.sym_buffer_ptrs.len()) != Some(spec.num_experts)
        || launch.topk_indices.spec.shape != [tokens, spec.num_topk]
        || launch.topk_weights.spec.shape != [tokens, spec.num_topk]
        || launch.y.spec.shape != [tokens, hidden]
    {
        return Err(Error::InvalidArgument(
            "BF16 Mega MoE tensor shapes are inconsistent".into(),
        ));
    }
    if spec.num_topk == 0
        || launch.sym_buffer_ptrs.is_empty()
        || launch.rank_idx >= launch.sym_buffer_ptrs.len()
        || launch.sym_buffer.spec.shape != [spec.buffer_layout.total_bytes]
    {
        return Err(Error::InvalidArgument(
            "BF16 Mega MoE launch topology or buffer is invalid".into(),
        ));
    }
    if !launch.activation_clamp.is_finite() && !launch.activation_clamp.is_infinite()
        || launch.activation_clamp.is_sign_negative()
    {
        return Err(Error::InvalidArgument(
            "activation_clamp must be nonnegative".into(),
        ));
    }
    Ok(())
}

/// Returns the token alignment shared by Mega MoE routing and ring buffers.
pub fn mega_moe_token_alignment() -> Result<usize> {
    let mut alignment = 0i64;
    // SAFETY: `alignment` is a valid output pointer for the duration of this call.
    let status = unsafe { deepgemm_sys::deepgemm_mega_moe_token_alignment(&mut alignment) };
    Error::check_raw_status(status)?;
    usize::try_from(alignment)
        .map_err(|_| Error::InvalidArgument("Mega MoE token alignment does not fit usize".into()))
}

/// Derives the inclusive range of ring-buffer capacities accepted upstream.
pub fn mega_moe_ring_limits(config: MegaMoeRingConfig) -> Result<MegaMoeRingLimits> {
    let params = deepgemm_sys::deepgemm_mega_moe_ring_limits_params_t {
        num_ranks: usize_to_i64(config.num_ranks, "num_ranks")?,
        num_experts: usize_to_i64(config.num_experts, "num_experts")?,
        num_max_tokens_per_rank: usize_to_i64(
            config.num_max_tokens_per_rank,
            "num_max_tokens_per_rank",
        )?,
        num_topk: usize_to_i64(config.num_topk, "num_topk")?,
    };
    let mut raw = MaybeUninit::<deepgemm_sys::deepgemm_mega_moe_ring_limits_t>::uninit();
    // SAFETY: `params` is initialized and `raw` points to writable output storage.
    let status = unsafe { deepgemm_sys::deepgemm_mega_moe_ring_limits(&params, raw.as_mut_ptr()) };
    Error::check_raw_status(status)?;
    // SAFETY: the C ABI initializes the complete output on success.
    let raw = unsafe { raw.assume_init() };
    Ok(MegaMoeRingLimits {
        min_tokens: usize_from_i64(raw.min_tokens, "minimum ring tokens")?,
        max_tokens: usize_from_i64(raw.max_tokens, "maximum ring tokens")?,
    })
}

/// Derives the exact typed views and total bytes for one symmetric allocation.
pub fn mega_moe_buffer_layout(config: MegaMoeBufferConfig) -> Result<MegaMoeBufferLayout> {
    let params = deepgemm_sys::deepgemm_mega_moe_buffer_params_t {
        num_ranks: usize_to_i64(config.num_ranks, "num_ranks")?,
        num_experts: usize_to_i64(config.num_experts, "num_experts")?,
        num_max_tokens_per_rank: usize_to_i64(
            config.num_max_tokens_per_rank,
            "num_max_tokens_per_rank",
        )?,
        num_topk: usize_to_i64(config.num_topk, "num_topk")?,
        hidden: usize_to_i64(config.hidden, "hidden")?,
        intermediate_hidden: usize_to_i64(config.intermediate_hidden, "intermediate_hidden")?,
        num_ring_tokens: usize_to_i64(config.num_ring_tokens, "num_ring_tokens")?,
        mma_kind: config.mma_kind.to_sys(),
    };
    let mut raw = MaybeUninit::<deepgemm_sys::deepgemm_mega_moe_buffer_layout_t>::uninit();
    // SAFETY: `params` is initialized and `raw` points to writable output storage.
    let status =
        unsafe { deepgemm_sys::deepgemm_mega_moe_buffer_layout(&params, raw.as_mut_ptr()) };
    Error::check_raw_status(status)?;
    // SAFETY: the C ABI initializes the complete output on success.
    let raw = unsafe { raw.assume_init() };

    Ok(MegaMoeBufferLayout {
        total_bytes: usize_from_u64(raw.total_bytes, "total buffer bytes")?,
        workspace_bytes: usize_from_u64(raw.workspace_bytes, "workspace bytes")?,
        x: view_from_sys(raw.x, "x")?,
        x_scale: optional_view_from_sys(raw.x_scale, "x_scale")?,
        topk_indices: view_from_sys(raw.topk_indices, "topk_indices")?,
        topk_weights: view_from_sys(raw.topk_weights, "topk_weights")?,
        l1_acts: view_from_sys(raw.l1_acts, "l1_acts")?,
        l1_acts_scale: optional_view_from_sys(raw.l1_acts_scale, "l1_acts_scale")?,
        l2_acts: view_from_sys(raw.l2_acts, "l2_acts")?,
        l2_acts_scale: optional_view_from_sys(raw.l2_acts_scale, "l2_acts_scale")?,
        combine: view_from_sys(raw.combine, "combine")?,
    })
}

fn optional_view_from_sys(
    raw: deepgemm_sys::deepgemm_mega_moe_buffer_view_t,
    name: &str,
) -> Result<Option<MegaMoeBufferView>> {
    if raw.rank == 0 && raw.dtype == deepgemm_sys::DEEPGEMM_DTYPE_INVALID {
        return Ok(None);
    }
    view_from_sys(raw, name).map(Some)
}

fn view_from_sys(
    raw: deepgemm_sys::deepgemm_mega_moe_buffer_view_t,
    name: &str,
) -> Result<MegaMoeBufferView> {
    if raw.rank != 2 {
        return Err(Error::Internal(format!(
            "Mega MoE {name} view returned rank {}, expected 2",
            raw.rank
        )));
    }
    Ok(MegaMoeBufferView {
        offset_bytes: usize_from_u64(raw.offset_bytes, "view offset")?,
        dtype: DType::from_sys(raw.dtype).ok_or_else(|| {
            Error::Internal(format!(
                "Mega MoE {name} returned unknown dtype {}",
                raw.dtype
            ))
        })?,
        shape: [
            usize_from_i64(raw.shape[0], "view rows")?,
            usize_from_i64(raw.shape[1], "view columns")?,
        ],
        strides: [
            isize::try_from(raw.stride[0])
                .map_err(|_| Error::Internal("Mega MoE row stride does not fit isize".into()))?,
            isize::try_from(raw.stride[1])
                .map_err(|_| Error::Internal("Mega MoE column stride does not fit isize".into()))?,
        ],
        element_count: usize_from_u64(raw.element_count, "view element count")?,
    })
}

fn usize_from_i64(value: i64, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::Internal(format!("{name} does not fit usize")))
}

fn usize_from_u64(value: u64, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::Internal(format!("{name} does not fit usize")))
}

#[cfg(test)]
mod tests {
    use crate::TensorSpec;

    use super::*;

    fn single_rank_config(mma_kind: MegaMoeMmaKind) -> MegaMoeBufferConfig {
        MegaMoeBufferConfig {
            num_ranks: 1,
            num_experts: 8,
            num_max_tokens_per_rank: 384,
            num_topk: 2,
            hidden: 128,
            intermediate_hidden: 128,
            num_ring_tokens: 384,
            mma_kind,
        }
    }

    #[test]
    fn exposes_upstream_token_alignment_and_ring_limits() {
        assert_eq!(mega_moe_token_alignment().unwrap(), 384);
        assert_eq!(
            mega_moe_ring_limits(MegaMoeRingConfig {
                num_ranks: 8,
                num_experts: 384,
                num_max_tokens_per_rank: 768,
                num_topk: 6,
            })
            .unwrap(),
            MegaMoeRingLimits {
                min_tokens: 6144,
                max_tokens: 55296,
            }
        );
    }

    #[test]
    fn derives_fp8_fp4_symmetric_buffer_views() {
        let layout = mega_moe_buffer_layout(single_rank_config(MegaMoeMmaKind::Fp8Fp4)).unwrap();
        assert_eq!(layout.workspace_bytes, 40928);
        assert_eq!(layout.total_bytes, 446432);
        assert_eq!(layout.x.offset_bytes, 40928);
        assert_eq!(layout.x.dtype, DType::Fp8E4M3);
        assert_eq!(layout.x.shape, [384, 128]);
        assert_eq!(layout.x_scale.unwrap().shape, [384, 1]);
        assert_eq!(layout.topk_indices.dtype, DType::I64);
        assert_eq!(layout.topk_indices.shape, [384, 2]);
        assert_eq!(layout.l1_acts_scale.unwrap().shape, [6144, 1]);
        assert_eq!(layout.l1_acts_scale.unwrap().strides, [1, 6144]);
        assert_eq!(layout.l2_acts.offset_bytes, 176096);
        assert_eq!(layout.combine.shape, [768, 128]);
    }

    #[test]
    fn bf16_layout_omits_scale_views() {
        let layout = mega_moe_buffer_layout(single_rank_config(MegaMoeMmaKind::Bf16)).unwrap();
        assert_eq!(layout.x.dtype, DType::BF16);
        assert_eq!(layout.x_scale, None);
        assert_eq!(layout.l1_acts_scale, None);
        assert_eq!(layout.l2_acts_scale, None);
        assert_eq!(layout.l2_acts.shape, [384, 128]);
        assert_eq!(layout.total_bytes, 543200);
    }

    #[test]
    fn bf16_launch_accepts_the_cuda_default_stream() {
        let config = single_rank_config(MegaMoeMmaKind::Bf16);
        let buffer_layout = mega_moe_buffer_layout(config).unwrap();
        let spec = Bf16MegaMoeSpec {
            num_experts: config.num_experts,
            num_topk: config.num_topk,
            num_max_tokens_per_rank: config.num_max_tokens_per_rank,
            num_ring_tokens: config.num_ring_tokens,
            buffer_layout,
        };
        let const_ptr = std::ptr::NonNull::<u8>::dangling().as_ptr().cast();
        let mut_ptr = std::ptr::NonNull::<u8>::dangling().as_ptr().cast();
        let sym_buffer_ptrs = [const_ptr as usize as u64];
        let launch = Bf16MegaMoeLaunch {
            x: TensorArg {
                data: const_ptr,
                spec: TensorSpec::contiguous(DType::BF16, [1, config.hidden]),
            },
            topk_indices: TensorArg {
                data: const_ptr,
                spec: TensorSpec::contiguous(DType::I64, [1, config.num_topk]),
            },
            topk_weights: TensorArg {
                data: const_ptr,
                spec: TensorSpec::contiguous(DType::F32, [1, config.num_topk]),
            },
            l1_weights: TensorArg {
                data: const_ptr,
                spec: TensorSpec::contiguous(
                    DType::BF16,
                    [
                        config.num_experts,
                        2 * config.intermediate_hidden,
                        config.hidden,
                    ],
                ),
            },
            l2_weights: TensorArg {
                data: const_ptr,
                spec: TensorSpec::contiguous(
                    DType::BF16,
                    [
                        config.num_experts,
                        config.hidden,
                        config.intermediate_hidden,
                    ],
                ),
            },
            y: TensorOut {
                data: mut_ptr,
                spec: TensorSpec::contiguous(DType::BF16, [1, config.hidden]),
            },
            sym_buffer: TensorOut {
                data: mut_ptr,
                spec: TensorSpec::contiguous(DType::U8, [buffer_layout.total_bytes]),
            },
            sym_buffer_ptrs: &sym_buffer_ptrs,
            rank_idx: 0,
            activation_clamp: f32::INFINITY,
            fast_math: true,
            stream: std::ptr::null_mut(),
        };

        validate_bf16_launch(&spec, &launch).unwrap();
    }

    #[test]
    fn rejects_misaligned_token_capacity_and_out_of_range_ring() {
        let error = mega_moe_ring_limits(MegaMoeRingConfig {
            num_max_tokens_per_rank: 383,
            ..single_rank_config(MegaMoeMmaKind::Fp8Fp4).ring_config()
        })
        .unwrap_err();
        assert!(error.to_string().contains("multiple of 384"));

        let error = mega_moe_buffer_layout(MegaMoeBufferConfig {
            num_ring_tokens: 3456,
            ..single_rank_config(MegaMoeMmaKind::Fp8Fp4)
        })
        .unwrap_err();
        assert!(error.to_string().contains("derived ring limits"));
    }
}
