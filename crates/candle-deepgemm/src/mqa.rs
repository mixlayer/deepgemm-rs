//! Candle CUDA bindings for DeepGEMM MQA logits.

use candle::{DType as CandleDType, Tensor};
use deepgemm::{
    DType as DeepGemmDType, MqaLogitsLaunch, MqaLogitsSpec, PagedMqaLogitsLaunch,
    PagedMqaLogitsMetadataLaunch, PagedMqaLogitsMetadataSpec, PagedMqaLogitsSpec, TensorArg,
    TensorOut, TensorSpec,
};

use crate::{
    Result,
    error::invalid_arg,
    tensor::cuda::{
        ensure_dtype, ensure_rank, ensure_same_device, stream_and_device_id, tensor_ptr_by_dtype,
    },
};

/// Runtime options for `fp8_fp4_mqa_logits`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MqaLogitsConfig {
    /// Whether to clean unfilled full-width logits to `-inf`.
    pub clean_logits: bool,
    /// Zero for full logits, otherwise compressed logits width.
    pub max_seqlen_k: usize,
    /// Output logits dtype. Must be `DType::F32` or `DType::BF16`.
    pub logits_dtype: CandleDType,
}

/// Caller-owned paged MQA scheduler metadata.
#[derive(Debug)]
pub struct PagedMqaLogitsPlan {
    /// I32 scheduler metadata shaped `[num_sms + 1, 2]`.
    pub schedule_meta: Tensor,
    /// KV page size used to generate `schedule_meta`.
    pub block_kv: usize,
    /// Number of SMs used to generate `schedule_meta`.
    pub num_sms: usize,
}

/// Runtime options for `fp8_fp4_paged_mqa_logits`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PagedMqaLogitsConfig {
    /// Logical output width.
    pub max_context_len: usize,
    /// Whether to clean unfilled logits to `-inf`.
    ///
    /// The native binding currently rejects this while DeepGEMM only supports
    /// 2D context lengths here.
    pub clean_logits: bool,
    /// Output logits dtype. Must be `DType::F32` or `DType::BF16`.
    pub logits_dtype: CandleDType,
}

impl PagedMqaLogitsConfig {
    /// Creates a paged MQA logits config with `F32` output logits.
    pub fn new(max_context_len: usize) -> Self {
        Self {
            max_context_len,
            clean_logits: false,
            logits_dtype: CandleDType::F32,
        }
    }
}

impl Default for MqaLogitsConfig {
    fn default() -> Self {
        Self {
            clean_logits: true,
            max_seqlen_k: 0,
            logits_dtype: CandleDType::F32,
        }
    }
}

/// Allocates and generates DeepGEMM paged MQA scheduler metadata.
///
/// Arguments:
/// - `context_lens`: `I32 [batch_size, next_n]`
/// - `indices`: optional SM100 varlen indices, `I32 [batch_size]`; requires `next_n == 1`
/// - `block_kv`: KV page size. SM90 supports `64`; SM100 supports `32` or `64`.
///
/// Returns a plan containing `schedule_meta`: `I32 [num_sms + 1, 2]`.
pub fn paged_mqa_logits_plan(
    context_lens: &Tensor,
    indices: Option<&Tensor>,
    block_kv: usize,
) -> Result<PagedMqaLogitsPlan> {
    if let Some(indices) = indices {
        ensure_same_device(context_lens, indices, "indices")?;
    }
    let spec = paged_metadata_spec(context_lens, indices, block_kv)?;

    let (stream, device_id) = stream_and_device_id(context_lens)?;
    let device_info = deepgemm::device_info()?;
    if device_info.device != device_id {
        return invalid_arg(format!(
            "context_lens is on CUDA device {device_id}, but DeepGEMM current device is {}",
            device_info.device
        ));
    }
    let layout = deepgemm::paged_mqa_logits_metadata_layout(&spec, device_info.arch()?)?;
    let schedule_meta = Tensor::zeros(
        (layout.allocation_shape[0], layout.allocation_shape[1]),
        CandleDType::I32,
        context_lens.device(),
    )?;

    {
        let (context_storage, context_layout) = context_lens.storage_and_layout();
        let context_ptr = tensor_ptr_by_dtype(
            &context_storage,
            CandleDType::I32,
            context_layout.start_offset(),
            &stream,
            "context_lens",
        )?;

        let indices_storage_and_layout = indices.map(Tensor::storage_and_layout);
        let indices_ptr = match &indices_storage_and_layout {
            Some((storage, layout)) => Some(tensor_ptr_by_dtype(
                storage,
                CandleDType::I32,
                layout.start_offset(),
                &stream,
                "indices",
            )?),
            None => None,
        };

        let (schedule_storage, schedule_layout) = schedule_meta.storage_and_layout();
        let schedule_ptr = tensor_ptr_by_dtype(
            &schedule_storage,
            CandleDType::I32,
            schedule_layout.start_offset(),
            &stream,
            "schedule_meta",
        )?;

        let launch = PagedMqaLogitsMetadataLaunch {
            context_lens: tensor_arg(context_ptr.as_const_void(), spec.context_lens),
            indices: match (indices_ptr.as_ref(), spec.indices) {
                (Some(ptr), Some(spec)) => Some(tensor_arg(ptr.as_const_void(), spec)),
                _ => None,
            },
            schedule_meta: TensorOut {
                data: schedule_ptr.as_mut_void(),
                spec: layout.logical_spec(),
            },
            block_kv,
            num_sms: spec.num_sms,
            stream: stream.cu_stream() as *mut std::ffi::c_void,
        };

        // SAFETY: all pointers come from live Candle CUDA tensors on the launch stream,
        // and specs were validated by the DeepGEMM layout path before launch.
        unsafe { deepgemm::paged_mqa_logits_metadata(&launch)? };
    }

    Ok(PagedMqaLogitsPlan {
        schedule_meta,
        block_kv,
        num_sms: spec.num_sms,
    })
}

/// Launches DeepGEMM `fp8_fp4_mqa_logits` on Candle CUDA tensors.
///
/// FP8 mode is selected when `q_scale` is `None`:
/// - `q`: `F8E4M3 [seq_len, num_heads, head_dim]`
/// - `kv`: `F8E4M3 [seq_len_kv, head_dim]`
/// - `kv_scale`: `F32 [seq_len_kv]`
///
/// FP4 mode is selected when `q_scale` is `Some`:
/// - `q`: `U8 [seq_len, num_heads, head_dim / 2]`, storing packed FP4 E2M1
/// - `q_scale`: `I32 [seq_len, num_heads]`, storing packed UE8M0 scales
/// - `kv`: `U8 [seq_len_kv, head_dim / 2]`, storing packed FP4 E2M1
/// - `kv_scale`: `I32 [seq_len_kv]`, storing packed UE8M0 scales
///
/// Shared arguments:
/// - `weights`: `F32 [seq_len, num_heads]`, or SM100 `BF16 [seq_len, num_heads]` when logits are BF16
/// - `cu_seq_len_k_start`: `I32 [seq_len]`
/// - `cu_seq_len_k_end`: `I32 [seq_len]`
///
/// Returns a padded-stride logits view shaped `[seq_len, seq_len_kv]`, or
/// `[seq_len, max_seqlen_k]` when `max_seqlen_k > 0`. The returned dtype is
/// `config.logits_dtype`.
pub fn fp8_fp4_mqa_logits(
    q: &Tensor,
    q_scale: Option<&Tensor>,
    kv: &Tensor,
    kv_scale: &Tensor,
    weights: &Tensor,
    cu_seq_len_k_start: &Tensor,
    cu_seq_len_k_end: &Tensor,
    config: MqaLogitsConfig,
) -> Result<Tensor> {
    validate_common_devices(
        q,
        q_scale,
        kv,
        kv_scale,
        weights,
        cu_seq_len_k_start,
        cu_seq_len_k_end,
    )?;
    let is_fp4 = q_scale.is_some();
    let spec = mqa_logits_spec(
        q,
        q_scale,
        kv,
        kv_scale,
        weights,
        cu_seq_len_k_start,
        cu_seq_len_k_end,
        config,
    )?;

    let (stream, device_id) = stream_and_device_id(q)?;
    let device_info = deepgemm::device_info()?;
    if device_info.device != device_id {
        return invalid_arg(format!(
            "q is on CUDA device {device_id}, but DeepGEMM current device is {}",
            device_info.device
        ));
    }
    let layout = deepgemm::mqa_logits_layout(&spec, device_info.arch()?)?;
    let allocation = Tensor::zeros(
        (layout.allocation_shape[0], layout.allocation_shape[1]),
        config.logits_dtype,
        q.device(),
    )?;
    let logits = allocation.narrow(1, 0, layout.logical_shape[1])?;

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let q_dtype = if is_fp4 {
            CandleDType::U8
        } else {
            CandleDType::F8E4M3
        };
        let q_ptr =
            tensor_ptr_by_dtype(&q_storage, q_dtype, q_layout.start_offset(), &stream, "q")?;

        let q_scale_storage_and_layout = q_scale.map(Tensor::storage_and_layout);
        let q_scale_ptr = match &q_scale_storage_and_layout {
            Some((storage, layout)) => Some(tensor_ptr_by_dtype(
                storage,
                CandleDType::I32,
                layout.start_offset(),
                &stream,
                "q_scale",
            )?),
            None => None,
        };

        let (kv_storage, kv_layout) = kv.storage_and_layout();
        let kv_dtype = if is_fp4 {
            CandleDType::U8
        } else {
            CandleDType::F8E4M3
        };
        let kv_ptr = tensor_ptr_by_dtype(
            &kv_storage,
            kv_dtype,
            kv_layout.start_offset(),
            &stream,
            "kv",
        )?;

        let (kv_scale_storage, kv_scale_layout) = kv_scale.storage_and_layout();
        let kv_scale_dtype = if is_fp4 {
            CandleDType::I32
        } else {
            CandleDType::F32
        };
        let kv_scale_ptr = tensor_ptr_by_dtype(
            &kv_scale_storage,
            kv_scale_dtype,
            kv_scale_layout.start_offset(),
            &stream,
            "kv_scale",
        )?;

        let (weights_storage, weights_layout) = weights.storage_and_layout();
        let weights_ptr = tensor_ptr_by_dtype(
            &weights_storage,
            weights.dtype(),
            weights_layout.start_offset(),
            &stream,
            "weights",
        )?;

        let (start_storage, start_layout) = cu_seq_len_k_start.storage_and_layout();
        let start_ptr = tensor_ptr_by_dtype(
            &start_storage,
            CandleDType::I32,
            start_layout.start_offset(),
            &stream,
            "cu_seq_len_k_start",
        )?;

        let (end_storage, end_layout) = cu_seq_len_k_end.storage_and_layout();
        let end_ptr = tensor_ptr_by_dtype(
            &end_storage,
            CandleDType::I32,
            end_layout.start_offset(),
            &stream,
            "cu_seq_len_k_end",
        )?;

        let (logits_storage, logits_layout) = logits.storage_and_layout();
        let logits_ptr = tensor_ptr_by_dtype(
            &logits_storage,
            config.logits_dtype,
            logits_layout.start_offset(),
            &stream,
            "logits",
        )?;

        let launch = MqaLogitsLaunch {
            q: tensor_arg(q_ptr.as_const_void(), spec.q),
            q_scale: match (q_scale_ptr.as_ref(), spec.q_scale) {
                (Some(ptr), Some(spec)) => Some(tensor_arg(ptr.as_const_void(), spec)),
                _ => None,
            },
            kv: tensor_arg(kv_ptr.as_const_void(), spec.kv),
            kv_scale: tensor_arg(kv_scale_ptr.as_const_void(), spec.kv_scale),
            weights: tensor_arg(weights_ptr.as_const_void(), spec.weights),
            cu_seq_len_k_start: tensor_arg(start_ptr.as_const_void(), spec.cu_seq_len_k_start),
            cu_seq_len_k_end: tensor_arg(end_ptr.as_const_void(), spec.cu_seq_len_k_end),
            logits: TensorOut {
                data: logits_ptr.as_mut_void(),
                spec: layout.logical_spec(),
            },
            clean_logits: config.clean_logits,
            max_seqlen_k: config.max_seqlen_k,
            stream: stream.cu_stream() as *mut std::ffi::c_void,
        };

        // SAFETY: all pointers come from live Candle CUDA tensors on the launch stream,
        // and specs were validated by the DeepGEMM layout path before launch.
        unsafe { deepgemm::fp8_fp4_mqa_logits(&launch)? };
    }

    Ok(logits)
}

/// Launches DeepGEMM `fp8_fp4_paged_mqa_logits` on Candle CUDA tensors.
///
/// FP8 mode is selected when `q_scale` is `None`:
/// - `q`: `F8E4M3 [batch_size, next_n, num_heads, head_dim]`
/// - `fused_kv_cache`: `U8 [num_kv_blocks, block_kv, 1, head_dim + 4]`
///
/// FP4 mode is selected when `q_scale` is `Some`:
/// - `q`: `U8 [batch_size, next_n, num_heads, head_dim / 2]`, storing packed FP4 E2M1
/// - `q_scale`: `I32 [batch_size, next_n, num_heads]`, storing packed UE8M0 scales
/// - `fused_kv_cache`: `U8 [num_kv_blocks, block_kv, 1, head_dim / 2 + 4]`
///
/// Shared arguments:
/// - `weights`: `F32 [batch_size * next_n, num_heads]`, or SM100 `BF16` when logits are BF16
/// - `context_lens`: `I32 [batch_size, next_n]`
/// - `block_table`: `I32 [batch_size, max_block_len]`
/// - `plan.schedule_meta`: `I32 [num_sms + 1, 2]`, generated with `paged_mqa_logits_plan`
/// - `indices`: optional SM100 varlen indices, `I32 [batch_size]`; requires `next_n == 1`
///
/// Returns a padded-stride logits view shaped `[batch_size * next_n, max_context_len]`.
pub fn fp8_fp4_paged_mqa_logits(
    q: &Tensor,
    q_scale: Option<&Tensor>,
    fused_kv_cache: &Tensor,
    weights: &Tensor,
    context_lens: &Tensor,
    block_table: &Tensor,
    plan: &PagedMqaLogitsPlan,
    indices: Option<&Tensor>,
    config: PagedMqaLogitsConfig,
) -> Result<Tensor> {
    validate_paged_devices(
        q,
        q_scale,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        &plan.schedule_meta,
        indices,
    )?;
    let block_kv = *fused_kv_cache
        .dims()
        .get(1)
        .ok_or_else(|| crate::Error::Tensor("fused_kv_cache must have rank 4".to_string()))?;
    if plan.block_kv != block_kv {
        return invalid_arg(format!(
            "plan block_kv {} does not match fused_kv_cache block_kv {block_kv}",
            plan.block_kv
        ));
    }
    let spec = paged_mqa_logits_spec(
        q,
        q_scale,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        &plan.schedule_meta,
        indices,
        config,
        plan.num_sms,
    )?;

    let (stream, device_id) = stream_and_device_id(q)?;
    let device_info = deepgemm::device_info()?;
    if device_info.device != device_id {
        return invalid_arg(format!(
            "q is on CUDA device {device_id}, but DeepGEMM current device is {}",
            device_info.device
        ));
    }
    let layout = deepgemm::paged_mqa_logits_layout(&spec, device_info.arch()?)?;
    let allocation = Tensor::zeros(
        (layout.allocation_shape[0], layout.allocation_shape[1]),
        config.logits_dtype,
        q.device(),
    )?;
    let logits = allocation.narrow(1, 0, layout.logical_shape[1])?;

    {
        let is_fp4 = q_scale.is_some();
        let (q_storage, q_layout) = q.storage_and_layout();
        let q_dtype = if is_fp4 {
            CandleDType::U8
        } else {
            CandleDType::F8E4M3
        };
        let q_ptr =
            tensor_ptr_by_dtype(&q_storage, q_dtype, q_layout.start_offset(), &stream, "q")?;

        let q_scale_storage_and_layout = q_scale.map(Tensor::storage_and_layout);
        let q_scale_ptr = match &q_scale_storage_and_layout {
            Some((storage, layout)) => Some(tensor_ptr_by_dtype(
                storage,
                CandleDType::I32,
                layout.start_offset(),
                &stream,
                "q_scale",
            )?),
            None => None,
        };

        let (fused_storage, fused_layout) = fused_kv_cache.storage_and_layout();
        let fused_ptr = tensor_ptr_by_dtype(
            &fused_storage,
            CandleDType::U8,
            fused_layout.start_offset(),
            &stream,
            "fused_kv_cache",
        )?;

        let (weights_storage, weights_layout) = weights.storage_and_layout();
        let weights_ptr = tensor_ptr_by_dtype(
            &weights_storage,
            weights.dtype(),
            weights_layout.start_offset(),
            &stream,
            "weights",
        )?;

        let (context_storage, context_layout) = context_lens.storage_and_layout();
        let context_ptr = tensor_ptr_by_dtype(
            &context_storage,
            CandleDType::I32,
            context_layout.start_offset(),
            &stream,
            "context_lens",
        )?;

        let (block_storage, block_layout) = block_table.storage_and_layout();
        let block_ptr = tensor_ptr_by_dtype(
            &block_storage,
            CandleDType::I32,
            block_layout.start_offset(),
            &stream,
            "block_table",
        )?;

        let (schedule_storage, schedule_layout) = plan.schedule_meta.storage_and_layout();
        let schedule_ptr = tensor_ptr_by_dtype(
            &schedule_storage,
            CandleDType::I32,
            schedule_layout.start_offset(),
            &stream,
            "schedule_meta",
        )?;

        let indices_storage_and_layout = indices.map(Tensor::storage_and_layout);
        let indices_ptr = match &indices_storage_and_layout {
            Some((storage, layout)) => Some(tensor_ptr_by_dtype(
                storage,
                CandleDType::I32,
                layout.start_offset(),
                &stream,
                "indices",
            )?),
            None => None,
        };

        let (logits_storage, logits_layout) = logits.storage_and_layout();
        let logits_ptr = tensor_ptr_by_dtype(
            &logits_storage,
            config.logits_dtype,
            logits_layout.start_offset(),
            &stream,
            "logits",
        )?;

        let launch = PagedMqaLogitsLaunch {
            q: tensor_arg(q_ptr.as_const_void(), spec.q),
            q_scale: match (q_scale_ptr.as_ref(), spec.q_scale) {
                (Some(ptr), Some(spec)) => Some(tensor_arg(ptr.as_const_void(), spec)),
                _ => None,
            },
            fused_kv_cache: tensor_arg(fused_ptr.as_const_void(), spec.fused_kv_cache),
            weights: tensor_arg(weights_ptr.as_const_void(), spec.weights),
            context_lens: tensor_arg(context_ptr.as_const_void(), spec.context_lens),
            block_table: tensor_arg(block_ptr.as_const_void(), spec.block_table),
            schedule_meta: tensor_arg(schedule_ptr.as_const_void(), spec.schedule_meta),
            indices: match (indices_ptr.as_ref(), spec.indices) {
                (Some(ptr), Some(spec)) => Some(tensor_arg(ptr.as_const_void(), spec)),
                _ => None,
            },
            logits: TensorOut {
                data: logits_ptr.as_mut_void(),
                spec: layout.logical_spec(),
            },
            max_context_len: config.max_context_len,
            clean_logits: config.clean_logits,
            stream: stream.cu_stream() as *mut std::ffi::c_void,
            num_sms: plan.num_sms,
        };

        // SAFETY: all pointers come from live Candle CUDA tensors on the launch stream,
        // and specs were validated by the DeepGEMM layout path before launch.
        unsafe { deepgemm::fp8_fp4_paged_mqa_logits(&launch)? };
    }

    Ok(logits)
}

fn validate_common_devices(
    q: &Tensor,
    q_scale: Option<&Tensor>,
    kv: &Tensor,
    kv_scale: &Tensor,
    weights: &Tensor,
    cu_seq_len_k_start: &Tensor,
    cu_seq_len_k_end: &Tensor,
) -> Result<()> {
    for (name, tensor) in [
        ("kv", kv),
        ("kv_scale", kv_scale),
        ("weights", weights),
        ("cu_seq_len_k_start", cu_seq_len_k_start),
        ("cu_seq_len_k_end", cu_seq_len_k_end),
    ] {
        ensure_same_device(q, tensor, name)?;
    }
    if let Some(q_scale) = q_scale {
        ensure_same_device(q, q_scale, "q_scale")?;
    }
    Ok(())
}

fn validate_paged_devices(
    q: &Tensor,
    q_scale: Option<&Tensor>,
    fused_kv_cache: &Tensor,
    weights: &Tensor,
    context_lens: &Tensor,
    block_table: &Tensor,
    schedule_meta: &Tensor,
    indices: Option<&Tensor>,
) -> Result<()> {
    for (name, tensor) in [
        ("fused_kv_cache", fused_kv_cache),
        ("weights", weights),
        ("context_lens", context_lens),
        ("block_table", block_table),
        ("schedule_meta", schedule_meta),
    ] {
        ensure_same_device(q, tensor, name)?;
    }
    if let Some(q_scale) = q_scale {
        ensure_same_device(q, q_scale, "q_scale")?;
    }
    if let Some(indices) = indices {
        ensure_same_device(q, indices, "indices")?;
    }
    Ok(())
}

fn mqa_logits_spec(
    q: &Tensor,
    q_scale: Option<&Tensor>,
    kv: &Tensor,
    kv_scale: &Tensor,
    weights: &Tensor,
    cu_seq_len_k_start: &Tensor,
    cu_seq_len_k_end: &Tensor,
    config: MqaLogitsConfig,
) -> Result<MqaLogitsSpec> {
    let is_fp4 = q_scale.is_some();
    ensure_rank(q, 3, "q")?;
    ensure_rank(kv, 2, "kv")?;
    ensure_rank(kv_scale, 1, "kv_scale")?;
    ensure_rank(weights, 2, "weights")?;
    ensure_rank(cu_seq_len_k_start, 1, "cu_seq_len_k_start")?;
    ensure_rank(cu_seq_len_k_end, 1, "cu_seq_len_k_end")?;

    if is_fp4 {
        ensure_dtype(q, CandleDType::U8, "q")?;
        ensure_dtype(kv, CandleDType::U8, "kv")?;
        ensure_dtype(kv_scale, CandleDType::I32, "kv_scale")?;
        if let Some(q_scale) = q_scale {
            ensure_rank(q_scale, 2, "q_scale")?;
            ensure_dtype(q_scale, CandleDType::I32, "q_scale")?;
        }
    } else {
        ensure_dtype(q, CandleDType::F8E4M3, "q")?;
        ensure_dtype(kv, CandleDType::F8E4M3, "kv")?;
        ensure_dtype(kv_scale, CandleDType::F32, "kv_scale")?;
    }
    ensure_dtype(cu_seq_len_k_start, CandleDType::I32, "cu_seq_len_k_start")?;
    ensure_dtype(cu_seq_len_k_end, CandleDType::I32, "cu_seq_len_k_end")?;

    Ok(MqaLogitsSpec {
        q: tensor_spec(
            q,
            if is_fp4 {
                DeepGemmDType::PackedFp4E2M1
            } else {
                DeepGemmDType::Fp8E4M3
            },
            "q",
        )?,
        q_scale: q_scale
            .map(|q_scale| tensor_spec(q_scale, DeepGemmDType::PackedUe8M0, "q_scale"))
            .transpose()?,
        kv: tensor_spec(
            kv,
            if is_fp4 {
                DeepGemmDType::PackedFp4E2M1
            } else {
                DeepGemmDType::Fp8E4M3
            },
            "kv",
        )?,
        kv_scale: tensor_spec(
            kv_scale,
            if is_fp4 {
                DeepGemmDType::PackedUe8M0
            } else {
                DeepGemmDType::F32
            },
            "kv_scale",
        )?,
        weights: tensor_spec(
            weights,
            candle_to_deepgemm_dtype(weights.dtype(), "weights")?,
            "weights",
        )?,
        cu_seq_len_k_start: tensor_spec(
            cu_seq_len_k_start,
            DeepGemmDType::I32,
            "cu_seq_len_k_start",
        )?,
        cu_seq_len_k_end: tensor_spec(cu_seq_len_k_end, DeepGemmDType::I32, "cu_seq_len_k_end")?,
        clean_logits: config.clean_logits,
        max_seqlen_k: config.max_seqlen_k,
        logits_dtype: candle_to_logits_dtype(config.logits_dtype)?,
    })
}

fn paged_metadata_spec(
    context_lens: &Tensor,
    indices: Option<&Tensor>,
    block_kv: usize,
) -> Result<PagedMqaLogitsMetadataSpec> {
    ensure_rank(context_lens, 2, "context_lens")?;
    ensure_dtype(context_lens, CandleDType::I32, "context_lens")?;
    if let Some(indices) = indices {
        ensure_rank(indices, 1, "indices")?;
        ensure_dtype(indices, CandleDType::I32, "indices")?;
    }
    let num_sms = usize::try_from(deepgemm::num_sms()?)
        .map_err(|_| crate::Error::Tensor("num_sms overflow".to_string()))?;
    Ok(PagedMqaLogitsMetadataSpec {
        context_lens: tensor_spec(context_lens, DeepGemmDType::I32, "context_lens")?,
        indices: indices
            .map(|indices| tensor_spec(indices, DeepGemmDType::I32, "indices"))
            .transpose()?,
        block_kv,
        num_sms,
    })
}

#[allow(clippy::too_many_arguments)]
fn paged_mqa_logits_spec(
    q: &Tensor,
    q_scale: Option<&Tensor>,
    fused_kv_cache: &Tensor,
    weights: &Tensor,
    context_lens: &Tensor,
    block_table: &Tensor,
    schedule_meta: &Tensor,
    indices: Option<&Tensor>,
    config: PagedMqaLogitsConfig,
    num_sms: usize,
) -> Result<PagedMqaLogitsSpec> {
    let is_fp4 = q_scale.is_some();
    ensure_rank(q, 4, "q")?;
    ensure_rank(fused_kv_cache, 4, "fused_kv_cache")?;
    ensure_rank(weights, 2, "weights")?;
    ensure_rank(context_lens, 2, "context_lens")?;
    ensure_rank(block_table, 2, "block_table")?;
    ensure_rank(schedule_meta, 2, "schedule_meta")?;

    if is_fp4 {
        ensure_dtype(q, CandleDType::U8, "q")?;
        if let Some(q_scale) = q_scale {
            ensure_rank(q_scale, 3, "q_scale")?;
            ensure_dtype(q_scale, CandleDType::I32, "q_scale")?;
        }
    } else {
        ensure_dtype(q, CandleDType::F8E4M3, "q")?;
    }
    ensure_dtype(fused_kv_cache, CandleDType::U8, "fused_kv_cache")?;
    ensure_dtype(context_lens, CandleDType::I32, "context_lens")?;
    ensure_dtype(block_table, CandleDType::I32, "block_table")?;
    ensure_dtype(schedule_meta, CandleDType::I32, "schedule_meta")?;
    if let Some(indices) = indices {
        ensure_rank(indices, 1, "indices")?;
        ensure_dtype(indices, CandleDType::I32, "indices")?;
    }

    Ok(PagedMqaLogitsSpec {
        q: tensor_spec(
            q,
            if is_fp4 {
                DeepGemmDType::PackedFp4E2M1
            } else {
                DeepGemmDType::Fp8E4M3
            },
            "q",
        )?,
        q_scale: q_scale
            .map(|q_scale| tensor_spec(q_scale, DeepGemmDType::PackedUe8M0, "q_scale"))
            .transpose()?,
        fused_kv_cache: tensor_spec(fused_kv_cache, DeepGemmDType::U8, "fused_kv_cache")?,
        weights: tensor_spec(
            weights,
            candle_to_deepgemm_dtype(weights.dtype(), "weights")?,
            "weights",
        )?,
        context_lens: tensor_spec(context_lens, DeepGemmDType::I32, "context_lens")?,
        block_table: tensor_spec(block_table, DeepGemmDType::I32, "block_table")?,
        schedule_meta: tensor_spec(schedule_meta, DeepGemmDType::I32, "schedule_meta")?,
        indices: indices
            .map(|indices| tensor_spec(indices, DeepGemmDType::I32, "indices"))
            .transpose()?,
        max_context_len: config.max_context_len,
        clean_logits: config.clean_logits,
        logits_dtype: candle_to_logits_dtype(config.logits_dtype)?,
        num_sms,
    })
}

fn tensor_spec<const RANK: usize>(
    tensor: &Tensor,
    dtype: DeepGemmDType,
    name: &str,
) -> Result<TensorSpec<RANK>> {
    ensure_rank(tensor, RANK, name)?;
    let mut shape = [0usize; RANK];
    let mut strides = [0isize; RANK];
    for (index, dim) in tensor.dims().iter().copied().enumerate() {
        shape[index] = dim;
    }
    for (index, stride) in tensor.stride().iter().copied().enumerate() {
        strides[index] = isize::try_from(stride)
            .map_err(|_| crate::Error::Tensor(format!("{name} stride overflow")))?;
    }
    Ok(TensorSpec {
        dtype,
        shape,
        strides,
    })
}

fn tensor_arg<const RANK: usize>(
    data: *const std::ffi::c_void,
    spec: TensorSpec<RANK>,
) -> TensorArg<RANK> {
    TensorArg { data, spec }
}

fn candle_to_logits_dtype(dtype: CandleDType) -> Result<DeepGemmDType> {
    match dtype {
        CandleDType::F32 => Ok(DeepGemmDType::F32),
        CandleDType::BF16 => Ok(DeepGemmDType::BF16),
        dtype => invalid_arg(format!("logits_dtype must be F32 or BF16, got {dtype:?}")),
    }
}

fn candle_to_deepgemm_dtype(dtype: CandleDType, name: &str) -> Result<DeepGemmDType> {
    match dtype {
        CandleDType::F32 => Ok(DeepGemmDType::F32),
        CandleDType::BF16 => Ok(DeepGemmDType::BF16),
        CandleDType::I32 => Ok(DeepGemmDType::I32),
        CandleDType::U8 => Ok(DeepGemmDType::U8),
        CandleDType::F8E4M3 => Ok(DeepGemmDType::Fp8E4M3),
        dtype => invalid_arg(format!("{name} has unsupported dtype {dtype:?}")),
    }
}
