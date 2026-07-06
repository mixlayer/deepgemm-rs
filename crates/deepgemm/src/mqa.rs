use crate::{
    Arch, DType, Error, Result, TensorArg, TensorLayout2D, TensorOut, TensorSpec,
    tensor::{i64_to_isize, i64_to_usize, require_contiguous, require_dtype, usize_to_i64},
};

/// Shape and dtype contract for `fp8_fp4_mqa_logits`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MqaLogitsSpec {
    /// Q tensor: FP8 `[seq_len, num_heads, head_dim]`, or packed FP4
    /// `[seq_len, num_heads, head_dim / 2]`.
    pub q: TensorSpec<3>,
    /// Q scale tensor for FP4 only: `[seq_len, num_heads]`, packed UE8M0 in `i32` lanes.
    pub q_scale: Option<TensorSpec<2>>,
    /// KV tensor: FP8 `[seq_len_kv, head_dim]`, or packed FP4 `[seq_len_kv, head_dim / 2]`.
    pub kv: TensorSpec<2>,
    /// KV scale tensor: FP8 uses `[seq_len_kv]` `f32`; FP4 uses `[seq_len_kv]` packed UE8M0.
    pub kv_scale: TensorSpec<1>,
    /// Reduction weights tensor: `[seq_len, num_heads]`, `f32` or SM100 `bf16`.
    pub weights: TensorSpec<2>,
    /// Inclusive start offsets for K: `[seq_len]`, `i32`.
    pub cu_seq_len_k_start: TensorSpec<1>,
    /// Exclusive end offsets for K: `[seq_len]`, `i32`.
    pub cu_seq_len_k_end: TensorSpec<1>,
    /// Whether to clean unfilled full-width logits to `-inf`.
    pub clean_logits: bool,
    /// Zero for full logits, otherwise compressed logits width.
    pub max_seqlen_k: usize,
    /// Output logits dtype, `f32` or `bf16`.
    pub logits_dtype: DType,
}

/// Shape and dtype contract for `get_paged_mqa_logits_metadata`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PagedMqaLogitsMetadataSpec {
    /// Context lengths: `[batch_size, next_n]`, `i32`.
    pub context_lens: TensorSpec<2>,
    /// Optional SM100 varlen indices: `[batch_size]`, `i32`.
    pub indices: Option<TensorSpec<1>>,
    /// KV page size. SM90 supports `64`; SM100 supports `32` or `64`.
    pub block_kv: usize,
    /// Number of SMs used by the launch.
    pub num_sms: usize,
}

/// Shape and dtype contract for `fp8_fp4_paged_mqa_logits`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PagedMqaLogitsSpec {
    /// Q tensor: FP8 `[batch_size, next_n, num_heads, head_dim]`, or packed FP4
    /// `[batch_size, next_n, num_heads, head_dim / 2]`.
    pub q: TensorSpec<4>,
    /// Q scale tensor for FP4 only: `[batch_size, next_n, num_heads]`, packed UE8M0.
    pub q_scale: Option<TensorSpec<3>>,
    /// Fused byte KV cache:
    /// `[num_kv_blocks, block_kv, 1, head_dim + sizeof(f32)]` for FP8, or
    /// `[num_kv_blocks, block_kv, 1, head_dim / 2 + sizeof(i32)]` for FP4.
    pub fused_kv_cache: TensorSpec<4>,
    /// Reduction weights tensor: `[batch_size * next_n, num_heads]`, `f32` or SM100 `bf16`.
    pub weights: TensorSpec<2>,
    /// Context lengths: `[batch_size, next_n]`, `i32`.
    pub context_lens: TensorSpec<2>,
    /// Block table: `[batch_size, max_block_len]`, `i32`.
    pub block_table: TensorSpec<2>,
    /// Schedule metadata: `[num_sms + 1, 2]`, `i32`.
    pub schedule_meta: TensorSpec<2>,
    /// Optional SM100 varlen indices: `[batch_size]`, `i32`.
    pub indices: Option<TensorSpec<1>>,
    /// Logical output width.
    pub max_context_len: usize,
    /// Unsupported initially while upstream only accepts 2D context lengths.
    pub clean_logits: bool,
    /// Output logits dtype, `f32` or `bf16`.
    pub logits_dtype: DType,
    /// Number of SMs used by the launch.
    pub num_sms: usize,
}

/// Raw launch arguments for `fp8_fp4_mqa_logits`.
#[derive(Debug, Copy, Clone)]
pub struct MqaLogitsLaunch {
    pub q: TensorArg<3>,
    pub q_scale: Option<TensorArg<2>>,
    pub kv: TensorArg<2>,
    pub kv_scale: TensorArg<1>,
    pub weights: TensorArg<2>,
    pub cu_seq_len_k_start: TensorArg<1>,
    pub cu_seq_len_k_end: TensorArg<1>,
    pub logits: TensorOut<2>,
    pub clean_logits: bool,
    pub max_seqlen_k: usize,
    pub stream: deepgemm_sys::deepgemm_cuda_stream_t,
}

impl MqaLogitsLaunch {
    /// Returns the shape and dtype contract for this launch.
    pub fn spec(&self) -> MqaLogitsSpec {
        MqaLogitsSpec {
            q: self.q.spec,
            q_scale: self.q_scale.map(|arg| arg.spec),
            kv: self.kv.spec,
            kv_scale: self.kv_scale.spec,
            weights: self.weights.spec,
            cu_seq_len_k_start: self.cu_seq_len_k_start.spec,
            cu_seq_len_k_end: self.cu_seq_len_k_end.spec,
            clean_logits: self.clean_logits,
            max_seqlen_k: self.max_seqlen_k,
            logits_dtype: self.logits.spec.dtype,
        }
    }
}

/// Raw launch arguments for `get_paged_mqa_logits_metadata`.
#[derive(Debug, Copy, Clone)]
pub struct PagedMqaLogitsMetadataLaunch {
    pub context_lens: TensorArg<2>,
    pub indices: Option<TensorArg<1>>,
    pub schedule_meta: TensorOut<2>,
    pub block_kv: usize,
    pub num_sms: usize,
    pub stream: deepgemm_sys::deepgemm_cuda_stream_t,
}

impl PagedMqaLogitsMetadataLaunch {
    /// Returns the shape and dtype contract for this launch.
    pub fn spec(&self) -> PagedMqaLogitsMetadataSpec {
        PagedMqaLogitsMetadataSpec {
            context_lens: self.context_lens.spec,
            indices: self.indices.map(|arg| arg.spec),
            block_kv: self.block_kv,
            num_sms: self.num_sms,
        }
    }
}

/// Raw launch arguments for `fp8_fp4_paged_mqa_logits`.
#[derive(Debug, Copy, Clone)]
pub struct PagedMqaLogitsLaunch {
    pub q: TensorArg<4>,
    pub q_scale: Option<TensorArg<3>>,
    pub fused_kv_cache: TensorArg<4>,
    pub weights: TensorArg<2>,
    pub context_lens: TensorArg<2>,
    pub block_table: TensorArg<2>,
    pub schedule_meta: TensorArg<2>,
    pub indices: Option<TensorArg<1>>,
    pub logits: TensorOut<2>,
    pub max_context_len: usize,
    pub clean_logits: bool,
    pub stream: deepgemm_sys::deepgemm_cuda_stream_t,
    pub num_sms: usize,
}

impl PagedMqaLogitsLaunch {
    /// Returns the shape and dtype contract for this launch.
    pub fn spec(&self) -> PagedMqaLogitsSpec {
        PagedMqaLogitsSpec {
            q: self.q.spec,
            q_scale: self.q_scale.map(|arg| arg.spec),
            fused_kv_cache: self.fused_kv_cache.spec,
            weights: self.weights.spec,
            context_lens: self.context_lens.spec,
            block_table: self.block_table.spec,
            schedule_meta: self.schedule_meta.spec,
            indices: self.indices.map(|arg| arg.spec),
            max_context_len: self.max_context_len,
            clean_logits: self.clean_logits,
            logits_dtype: self.logits.spec.dtype,
            num_sms: self.num_sms,
        }
    }
}

/// Converts a raw C ABI 2D layout into a Rust layout.
pub fn logits_layout_from_sys(
    raw: deepgemm_sys::deepgemm_tensor_layout_2d_t,
) -> Result<TensorLayout2D> {
    Ok(TensorLayout2D {
        dtype: DType::from_sys(raw.dtype)
            .ok_or_else(|| Error::InvalidArgument(format!("unknown dtype {}", raw.dtype)))?,
        logical_shape: [
            i64_to_usize(raw.logical_shape[0], "logical rows")?,
            i64_to_usize(raw.logical_shape[1], "logical cols")?,
        ],
        allocation_shape: [
            i64_to_usize(raw.allocation_shape[0], "allocation rows")?,
            i64_to_usize(raw.allocation_shape[1], "allocation cols")?,
        ],
        strides: [
            i64_to_isize(raw.stride[0], "row stride")?,
            i64_to_isize(raw.stride[1], "col stride")?,
        ],
        element_count: usize::try_from(raw.element_count).map_err(|_| {
            Error::InvalidArgument("layout element count does not fit usize".into())
        })?,
    })
}

/// Computes the required output layout for `fp8_fp4_mqa_logits`.
pub fn mqa_logits_layout(spec: &MqaLogitsSpec, arch: Arch) -> Result<TensorLayout2D> {
    let dims = validate_mqa_logits_spec(spec, arch)?;
    let params = deepgemm_sys::deepgemm_mqa_logits_layout_params_t {
        seq_len: usize_to_i64(dims.seq_len, "seq_len")?,
        seq_len_kv: usize_to_i64(dims.seq_len_kv, "seq_len_kv")?,
        num_heads: usize_to_i64(dims.num_heads, "num_heads")?,
        max_seqlen_k: usize_to_i64(spec.max_seqlen_k, "max_seqlen_k")?,
        logits_dtype: spec.logits_dtype.to_sys(),
    };
    call_layout(|out| {
        // SAFETY: `params` and `out` are valid for this call.
        unsafe { deepgemm_sys::deepgemm_mqa_logits_layout(&params, out) }
    })
}

/// Computes the required schedule metadata layout for paged MQA logits.
pub fn paged_mqa_logits_metadata_layout(
    spec: &PagedMqaLogitsMetadataSpec,
    arch: Arch,
) -> Result<TensorLayout2D> {
    validate_paged_metadata_spec(spec, arch)?;
    let params = deepgemm_sys::deepgemm_paged_mqa_logits_metadata_layout_params_t {
        num_sms: usize_to_i64(spec.num_sms, "num_sms")?,
    };
    call_layout(|out| {
        // SAFETY: `params` and `out` are valid for this call.
        unsafe { deepgemm_sys::deepgemm_paged_mqa_logits_metadata_layout(&params, out) }
    })
}

/// Computes the required output layout for `fp8_fp4_paged_mqa_logits`.
pub fn paged_mqa_logits_layout(spec: &PagedMqaLogitsSpec, arch: Arch) -> Result<TensorLayout2D> {
    let dims = validate_paged_mqa_logits_spec(spec, arch)?;
    let params = deepgemm_sys::deepgemm_paged_mqa_logits_layout_params_t {
        batch_size: usize_to_i64(dims.batch_size, "batch_size")?,
        next_n: usize_to_i64(dims.next_n, "next_n")?,
        max_context_len: usize_to_i64(spec.max_context_len, "max_context_len")?,
        logits_dtype: spec.logits_dtype.to_sys(),
    };
    call_layout(|out| {
        // SAFETY: `params` and `out` are valid for this call.
        unsafe { deepgemm_sys::deepgemm_paged_mqa_logits_layout(&params, out) }
    })
}

/// Launches `fp8_fp4_mqa_logits` through the raw DeepGEMM C ABI.
///
/// # Safety
///
/// All pointers must refer to valid CUDA device buffers matching the attached shape and dtype
/// metadata. Buffers must remain live until work enqueued on `stream` has completed.
pub unsafe fn fp8_fp4_mqa_logits(params: &MqaLogitsLaunch) -> Result<()> {
    let arch = crate::runtime::device_info()?.arch()?;
    let expected_logits = mqa_logits_layout(&params.spec(), arch)?.logical_spec();
    require_spec(params.logits.spec, expected_logits, "logits")?;

    let raw = deepgemm_sys::deepgemm_mqa_logits_params_t {
        q: params.q.to_raw()?,
        has_q_scale: params.q_scale.is_some(),
        q_scale: optional_tensor_arg(params.q_scale)?,
        kv: params.kv.to_raw()?,
        kv_scale: params.kv_scale.to_raw()?,
        weights: params.weights.to_raw()?,
        cu_seq_len_k_start: params.cu_seq_len_k_start.to_raw()?,
        cu_seq_len_k_end: params.cu_seq_len_k_end.to_raw()?,
        logits: params.logits.to_raw()?,
        clean_logits: params.clean_logits,
        max_seqlen_k: usize_to_i64(params.max_seqlen_k, "max_seqlen_k")?,
        stream: params.stream,
    };
    // SAFETY: the caller upholds pointer and stream validity; `raw` is valid for this call.
    let status = unsafe { deepgemm_sys::deepgemm_fp8_fp4_mqa_logits(&raw) };
    Error::check_raw_status(status)
}

/// Launches paged MQA metadata generation through the raw DeepGEMM C ABI.
///
/// # Safety
///
/// All pointers must refer to valid CUDA device buffers matching the attached shape and dtype
/// metadata. Buffers must remain live until work enqueued on `stream` has completed.
pub unsafe fn paged_mqa_logits_metadata(params: &PagedMqaLogitsMetadataLaunch) -> Result<()> {
    let arch = crate::runtime::device_info()?.arch()?;
    let expected_schedule_meta =
        paged_mqa_logits_metadata_layout(&params.spec(), arch)?.logical_spec();
    require_spec(
        params.schedule_meta.spec,
        expected_schedule_meta,
        "schedule_meta",
    )?;

    let raw = deepgemm_sys::deepgemm_paged_mqa_logits_metadata_params_t {
        context_lens: params.context_lens.to_raw()?,
        has_indices: params.indices.is_some(),
        indices: optional_tensor_arg(params.indices)?,
        schedule_meta: params.schedule_meta.to_raw()?,
        block_kv: usize_to_i64(params.block_kv, "block_kv")?,
        num_sms: usize_to_i64(params.num_sms, "num_sms")?,
        stream: params.stream,
    };
    // SAFETY: the caller upholds pointer and stream validity; `raw` is valid for this call.
    let status = unsafe { deepgemm_sys::deepgemm_paged_mqa_logits_metadata(&raw) };
    Error::check_raw_status(status)
}

/// Launches `fp8_fp4_paged_mqa_logits` through the raw DeepGEMM C ABI.
///
/// # Safety
///
/// All pointers must refer to valid CUDA device buffers matching the attached shape and dtype
/// metadata. Buffers must remain live until work enqueued on `stream` has completed.
pub unsafe fn fp8_fp4_paged_mqa_logits(params: &PagedMqaLogitsLaunch) -> Result<()> {
    let arch = crate::runtime::device_info()?.arch()?;
    let expected_logits = paged_mqa_logits_layout(&params.spec(), arch)?.logical_spec();
    require_spec(params.logits.spec, expected_logits, "logits")?;

    let raw = deepgemm_sys::deepgemm_paged_mqa_logits_params_t {
        q: params.q.to_raw()?,
        has_q_scale: params.q_scale.is_some(),
        q_scale: optional_tensor_arg(params.q_scale)?,
        fused_kv_cache: params.fused_kv_cache.to_raw()?,
        weights: params.weights.to_raw()?,
        context_lens: params.context_lens.to_raw()?,
        block_table: params.block_table.to_raw()?,
        schedule_meta: params.schedule_meta.to_raw()?,
        has_indices: params.indices.is_some(),
        indices: optional_tensor_arg(params.indices)?,
        logits: params.logits.to_raw()?,
        max_context_len: usize_to_i64(params.max_context_len, "max_context_len")?,
        num_sms: usize_to_i64(params.num_sms, "num_sms")?,
        clean_logits: params.clean_logits,
        stream: params.stream,
    };
    // SAFETY: the caller upholds pointer and stream validity; `raw` is valid for this call.
    let status = unsafe { deepgemm_sys::deepgemm_fp8_fp4_paged_mqa_logits(&raw) };
    Error::check_raw_status(status)
}

fn optional_tensor_arg<const RANK: usize>(
    arg: Option<TensorArg<RANK>>,
) -> Result<deepgemm_sys::deepgemm_tensor_t> {
    match arg {
        Some(arg) => arg.to_raw(),
        None => Ok(empty_tensor()),
    }
}

fn empty_tensor() -> deepgemm_sys::deepgemm_tensor_t {
    deepgemm_sys::deepgemm_tensor_t {
        data: core::ptr::null(),
        dtype: deepgemm_sys::DEEPGEMM_DTYPE_INVALID,
        rank: 0,
        shape: [0; 4],
        stride: [0; 4],
    }
}

fn call_layout(
    call: impl FnOnce(*mut deepgemm_sys::deepgemm_tensor_layout_2d_t) -> deepgemm_sys::deepgemm_status_t,
) -> Result<TensorLayout2D> {
    let mut raw = deepgemm_sys::deepgemm_tensor_layout_2d_t {
        dtype: deepgemm_sys::DEEPGEMM_DTYPE_INVALID,
        logical_shape: [0; 2],
        allocation_shape: [0; 2],
        stride: [0; 2],
        element_count: 0,
    };
    let status = call(&mut raw);
    Error::check_raw_status(status)?;
    logits_layout_from_sys(raw)
}

#[derive(Debug, Copy, Clone)]
struct MqaDims {
    seq_len: usize,
    seq_len_kv: usize,
    num_heads: usize,
}

fn validate_mqa_logits_spec(spec: &MqaLogitsSpec, arch: Arch) -> Result<MqaDims> {
    require_logits_dtype(spec.logits_dtype)?;
    let is_fp4 = spec.q_scale.is_some();
    if spec.clean_logits && spec.max_seqlen_k > 0 {
        return Err(Error::InvalidArgument(
            "clean_logits is not supported with compressed logits".into(),
        ));
    }

    let seq_len = spec.q.shape[0];
    let num_heads = spec.q.shape[1];
    let head_dim = logical_head_dim(spec.q.shape[2], is_fp4)?;
    let seq_len_kv = spec.kv.shape[0];
    if spec.kv.shape[1] != physical_head_dim(head_dim, is_fp4) {
        return Err(Error::InvalidArgument(
            "kv head dimension does not match q".into(),
        ));
    }

    require_contiguous(&spec.q, "q")?;
    require_contiguous(&spec.kv, "kv")?;
    require_dtype(
        &spec.q,
        if is_fp4 {
            DType::PackedFp4E2M1
        } else {
            DType::Fp8E4M3
        },
        "q",
    )?;
    require_dtype(
        &spec.kv,
        if is_fp4 {
            DType::PackedFp4E2M1
        } else {
            DType::Fp8E4M3
        },
        "kv",
    )?;

    if is_fp4 {
        let q_scale = spec.q_scale.expect("checked is_fp4");
        require_contiguous(&q_scale, "q_scale")?;
        require_dtype(&q_scale, DType::PackedUe8M0, "q_scale")?;
        require_shape(q_scale.shape, [seq_len, num_heads], "q_scale")?;
        require_dtype(&spec.kv_scale, DType::PackedUe8M0, "kv_scale")?;
    } else {
        require_dtype(&spec.kv_scale, DType::F32, "kv_scale")?;
    }
    require_contiguous(&spec.kv_scale, "kv_scale")?;
    require_shape(spec.kv_scale.shape, [seq_len_kv], "kv_scale")?;

    require_weights(spec.weights, seq_len, num_heads, arch, spec.logits_dtype)?;
    require_i32_vector(spec.cu_seq_len_k_start, seq_len, "cu_seq_len_k_start")?;
    require_i32_vector(spec.cu_seq_len_k_end, seq_len, "cu_seq_len_k_end")?;
    require_arch_support(arch, is_fp4, num_heads, head_dim, false)?;

    Ok(MqaDims {
        seq_len,
        seq_len_kv,
        num_heads,
    })
}

#[derive(Debug, Copy, Clone)]
struct PagedDims {
    batch_size: usize,
    next_n: usize,
}

fn validate_paged_metadata_spec(spec: &PagedMqaLogitsMetadataSpec, arch: Arch) -> Result<()> {
    require_contiguous(&spec.context_lens, "context_lens")?;
    require_dtype(&spec.context_lens, DType::I32, "context_lens")?;
    let batch_size = spec.context_lens.shape[0];
    let next_n = spec.context_lens.shape[1];
    validate_block_kv(arch, spec.block_kv)?;
    require_positive(spec.num_sms, "num_sms")?;

    if let Some(indices) = spec.indices {
        if arch != Arch::Sm100 || next_n != 1 {
            return Err(Error::UnsupportedArch(
                "varlen paged MQA indices require SM100 and next_n == 1".into(),
            ));
        }
        require_i32_vector(indices, batch_size, "indices")?;
    }
    Ok(())
}

fn validate_paged_mqa_logits_spec(spec: &PagedMqaLogitsSpec, arch: Arch) -> Result<PagedDims> {
    require_logits_dtype(spec.logits_dtype)?;
    require_positive(spec.max_context_len, "max_context_len")?;
    require_positive(spec.num_sms, "num_sms")?;
    if spec.clean_logits {
        return Err(Error::InvalidArgument(
            "clean_logits is not supported for 2D paged context lengths yet".into(),
        ));
    }

    let is_fp4 = spec.q_scale.is_some();
    let batch_size = spec.q.shape[0];
    let next_n = spec.q.shape[1];
    let num_heads = spec.q.shape[2];
    let head_dim = logical_head_dim(spec.q.shape[3], is_fp4)?;
    let block_kv = spec.fused_kv_cache.shape[1];

    require_contiguous(&spec.q, "q")?;
    require_dtype(
        &spec.q,
        if is_fp4 {
            DType::PackedFp4E2M1
        } else {
            DType::Fp8E4M3
        },
        "q",
    )?;
    if is_fp4 {
        let q_scale = spec.q_scale.expect("checked is_fp4");
        require_contiguous(&q_scale, "q_scale")?;
        require_dtype(&q_scale, DType::PackedUe8M0, "q_scale")?;
        require_shape(q_scale.shape, [batch_size, next_n, num_heads], "q_scale")?;
    }

    require_dtype(&spec.fused_kv_cache, DType::U8, "fused_kv_cache")?;
    require_shape(
        [
            spec.fused_kv_cache.shape[1],
            spec.fused_kv_cache.shape[2],
            spec.fused_kv_cache.shape[3],
        ],
        [
            block_kv,
            1,
            if is_fp4 {
                head_dim / 2 + 4
            } else {
                head_dim + 4
            },
        ],
        "fused_kv_cache tail shape",
    )?;
    let fused_bytes = spec.fused_kv_cache.shape[3];
    if spec.fused_kv_cache.strides[0] != (block_kv * fused_bytes) as isize
        || spec.fused_kv_cache.strides[1] != fused_bytes as isize
        || spec.fused_kv_cache.strides[2] != fused_bytes as isize
        || spec.fused_kv_cache.strides[3] != 1
    {
        return Err(Error::InvalidArgument(
            "fused_kv_cache must be contiguous [num_kv_blocks, block_kv, 1, fused_bytes]".into(),
        ));
    }

    require_weights(
        spec.weights,
        batch_size * next_n,
        num_heads,
        arch,
        spec.logits_dtype,
    )?;
    require_2d_i32(spec.context_lens, [batch_size, next_n], "context_lens")?;
    if spec.block_table.shape[0] != batch_size || spec.block_table.strides[1] != 1 {
        return Err(Error::InvalidArgument(
            "block_table must be [batch_size, max_block_len] with stride(1) == 1".into(),
        ));
    }
    require_dtype(&spec.block_table, DType::I32, "block_table")?;
    require_2d_i32(spec.schedule_meta, [spec.num_sms + 1, 2], "schedule_meta")?;

    if let Some(indices) = spec.indices {
        if arch != Arch::Sm100 || next_n != 1 {
            return Err(Error::UnsupportedArch(
                "varlen paged MQA indices require SM100 and next_n == 1".into(),
            ));
        }
        require_i32_vector(indices, batch_size, "indices")?;
    }

    validate_block_kv(arch, block_kv)?;
    require_arch_support(arch, is_fp4, num_heads, head_dim, true)?;
    if arch == Arch::Sm90 && next_n != 1 && next_n != 2 {
        return Err(Error::InvalidArgument(
            "SM90 paged MQA requires next_n == 1 or 2".into(),
        ));
    }

    Ok(PagedDims { batch_size, next_n })
}

fn require_logits_dtype(dtype: DType) -> Result<()> {
    match dtype {
        DType::F32 | DType::BF16 => Ok(()),
        _ => Err(Error::InvalidArgument(
            "logits dtype must be f32 or bf16".into(),
        )),
    }
}

fn require_weights(
    weights: TensorSpec<2>,
    rows: usize,
    heads: usize,
    arch: Arch,
    logits_dtype: DType,
) -> Result<()> {
    if weights.shape != [rows, heads] || weights.strides[1] != 1 {
        return Err(Error::InvalidArgument(format!(
            "weights must be [{rows}, {heads}] with stride(1) == 1"
        )));
    }
    match weights.dtype {
        DType::F32 => Ok(()),
        DType::BF16 if arch == Arch::Sm100 && logits_dtype == DType::BF16 => Ok(()),
        DType::BF16 => Err(Error::UnsupportedArch(
            "bf16 weights require SM100 and bf16 logits".into(),
        )),
        _ => Err(Error::InvalidArgument(
            "weights dtype must be f32 or bf16".into(),
        )),
    }
}

fn require_i32_vector(tensor: TensorSpec<1>, len: usize, name: &str) -> Result<()> {
    require_contiguous(&tensor, name)?;
    require_dtype(&tensor, DType::I32, name)?;
    require_shape(tensor.shape, [len], name)
}

fn require_2d_i32(tensor: TensorSpec<2>, shape: [usize; 2], name: &str) -> Result<()> {
    require_contiguous(&tensor, name)?;
    require_dtype(&tensor, DType::I32, name)?;
    require_shape(tensor.shape, shape, name)
}

fn require_shape<const RANK: usize>(
    actual: [usize; RANK],
    expected: [usize; RANK],
    name: &str,
) -> Result<()> {
    if actual != expected {
        return Err(Error::InvalidArgument(format!(
            "{name} shape must be {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

fn require_spec<const RANK: usize>(
    actual: TensorSpec<RANK>,
    expected: TensorSpec<RANK>,
    name: &str,
) -> Result<()> {
    if actual != expected {
        return Err(Error::InvalidArgument(format!(
            "{name} spec must be {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

fn require_positive(value: usize, name: &str) -> Result<()> {
    if value == 0 {
        return Err(Error::InvalidArgument(format!("{name} must be positive")));
    }
    Ok(())
}

fn logical_head_dim(physical: usize, is_fp4: bool) -> Result<usize> {
    require_positive(physical, "head_dim")?;
    Ok(if is_fp4 { physical * 2 } else { physical })
}

fn physical_head_dim(logical: usize, is_fp4: bool) -> usize {
    if is_fp4 { logical / 2 } else { logical }
}

fn validate_block_kv(arch: Arch, block_kv: usize) -> Result<()> {
    match arch {
        Arch::Sm90 if block_kv == 64 => Ok(()),
        Arch::Sm90 => Err(Error::InvalidArgument(
            "SM90 paged MQA requires block_kv == 64".into(),
        )),
        Arch::Sm100 if block_kv == 32 || block_kv == 64 => Ok(()),
        Arch::Sm100 => Err(Error::InvalidArgument(
            "SM100 paged MQA requires block_kv == 32 or 64".into(),
        )),
    }
}

fn require_arch_support(
    arch: Arch,
    is_fp4: bool,
    num_heads: usize,
    head_dim: usize,
    _paged: bool,
) -> Result<()> {
    if is_fp4 && arch != Arch::Sm100 {
        return Err(Error::UnsupportedArch(
            "FP4 MQA logits require SM100".into(),
        ));
    }

    match arch {
        Arch::Sm90 => {
            if num_heads != 32 && num_heads != 64 {
                return Err(Error::InvalidArgument(
                    "SM90 MQA requires 32 or 64 heads".into(),
                ));
            }
            if !matches!(head_dim, 32 | 64 | 128) {
                return Err(Error::InvalidArgument(
                    "SM90 FP8 MQA requires head_dim 32, 64, or 128".into(),
                ));
            }
        }
        Arch::Sm100 => {
            if !matches!(num_heads, 8 | 16 | 32 | 64) {
                return Err(Error::InvalidArgument(
                    "SM100 MQA requires 8, 16, 32, or 64 heads".into(),
                ));
            }
            if is_fp4 {
                if !matches!(head_dim, 64 | 128) {
                    return Err(Error::InvalidArgument(
                        "SM100 FP4 MQA requires head_dim 64 or 128".into(),
                    ));
                }
            } else if !matches!(head_dim, 32 | 64 | 128) {
                return Err(Error::InvalidArgument(
                    "SM100 FP8 MQA requires head_dim 32, 64, or 128".into(),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp8_mqa_spec() -> MqaLogitsSpec {
        MqaLogitsSpec {
            q: TensorSpec::contiguous(DType::Fp8E4M3, [2048, 32, 64]),
            q_scale: None,
            kv: TensorSpec::contiguous(DType::Fp8E4M3, [8192, 64]),
            kv_scale: TensorSpec::contiguous(DType::F32, [8192]),
            weights: TensorSpec::contiguous(DType::F32, [2048, 32]),
            cu_seq_len_k_start: TensorSpec::contiguous(DType::I32, [2048]),
            cu_seq_len_k_end: TensorSpec::contiguous(DType::I32, [2048]),
            clean_logits: true,
            max_seqlen_k: 0,
            logits_dtype: DType::F32,
        }
    }

    #[test]
    fn nonpaged_fp8_sm90_layout_includes_padded_allocation() {
        let layout = mqa_logits_layout(&fp8_mqa_spec(), Arch::Sm90).unwrap();
        assert_eq!(layout.dtype, DType::F32);
        assert_eq!(layout.logical_shape, [2048, 8192]);
        assert_eq!(layout.allocation_shape, [2048, 8448]);
        assert_eq!(layout.strides, [8448, 1]);
    }

    #[test]
    fn nonpaged_fp4_is_rejected_on_sm90() {
        let mut spec = fp8_mqa_spec();
        spec.q = TensorSpec::contiguous(DType::PackedFp4E2M1, [2048, 32, 32]);
        spec.q_scale = Some(TensorSpec::contiguous(DType::PackedUe8M0, [2048, 32]));
        spec.kv = TensorSpec::contiguous(DType::PackedFp4E2M1, [8192, 32]);
        spec.kv_scale = TensorSpec::contiguous(DType::PackedUe8M0, [8192]);
        assert!(matches!(
            mqa_logits_layout(&spec, Arch::Sm90),
            Err(Error::UnsupportedArch(_))
        ));
    }

    #[test]
    fn paged_metadata_layout_is_explicit_i32() {
        let spec = PagedMqaLogitsMetadataSpec {
            context_lens: TensorSpec::contiguous(DType::I32, [16, 2]),
            indices: None,
            block_kv: 64,
            num_sms: 132,
        };
        let layout = paged_mqa_logits_metadata_layout(&spec, Arch::Sm90).unwrap();
        assert_eq!(layout.dtype, DType::I32);
        assert_eq!(layout.logical_shape, [133, 2]);
        assert_eq!(layout.allocation_shape, [133, 2]);
        assert_eq!(layout.strides, [2, 1]);
        assert_eq!(layout.element_count, 266);
    }

    #[test]
    fn paged_fp8_layout_uses_aligned_context_width() {
        let spec = PagedMqaLogitsSpec {
            q: TensorSpec::contiguous(DType::Fp8E4M3, [16, 2, 32, 64]),
            q_scale: None,
            fused_kv_cache: TensorSpec::contiguous(DType::U8, [128, 64, 1, 68]),
            weights: TensorSpec::contiguous(DType::F32, [32, 32]),
            context_lens: TensorSpec::contiguous(DType::I32, [16, 2]),
            block_table: TensorSpec::contiguous(DType::I32, [16, 128]),
            schedule_meta: TensorSpec::contiguous(DType::I32, [133, 2]),
            indices: None,
            max_context_len: 8193,
            clean_logits: false,
            logits_dtype: DType::BF16,
            num_sms: 132,
        };
        let layout = paged_mqa_logits_layout(&spec, Arch::Sm90).unwrap();
        assert_eq!(layout.dtype, DType::BF16);
        assert_eq!(layout.logical_shape, [32, 8193]);
        assert_eq!(layout.allocation_shape, [32, 8704]);
        assert_eq!(layout.strides, [8704, 1]);
    }
}
