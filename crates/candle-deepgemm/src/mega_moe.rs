//! Candle CUDA integration for DeepGEMM BF16 Mega MoE.

use candle::{DType as CandleDType, Tensor};
use deepgemm::{
    Bf16MegaMoeLaunch, Bf16MegaMoeSpec, DType as DeepGemmDType, Sm90Fp8MegaMoeLaunch,
    Sm90Fp8MegaMoeSpec, TensorArg, TensorOut, TensorSpec,
};

use crate::{
    Result,
    error::invalid_arg,
    tensor::cuda::{
        ensure_dtype, ensure_rank, ensure_same_device, stream_and_device_id, tensor_ptr_by_dtype,
    },
};

/// A reusable single-rank symmetric allocation for BF16 Mega MoE.
///
/// The U8 CUDA tensor has exactly `spec.buffer_layout.total_bytes` bytes and
/// persists across layer calls so the kernel's dispatch and combine regions do
/// not need to be allocated in the token loop.
#[derive(Debug)]
pub struct Bf16MegaMoeWorkspace {
    spec: Bf16MegaMoeSpec,
    buffer: Tensor,
}

/// A reusable single-rank symmetric allocation for Hopper FP8 Mega MoE.
#[derive(Debug)]
pub struct Sm90Fp8MegaMoeWorkspace {
    spec: Sm90Fp8MegaMoeSpec,
    buffer: Tensor,
}

impl Sm90Fp8MegaMoeWorkspace {
    /// Allocates a zero-initialized single-rank SM90 FP8 symmetric buffer.
    pub fn new_single_rank(spec: Sm90Fp8MegaMoeSpec, device: &candle::Device) -> Result<Self> {
        if spec.num_experts == 0
            || spec.num_topk == 0
            || spec.num_max_tokens_per_rank == 0
            || spec.buffer_layout.total_bytes == 0
        {
            return invalid_arg("SM90 FP8 Mega MoE workspace dimensions must be positive");
        }
        let buffer = Tensor::zeros((spec.buffer_layout.total_bytes,), CandleDType::U8, device)?;
        Ok(Self { spec, buffer })
    }

    /// Returns the immutable launch specification associated with this allocation.
    pub fn spec(&self) -> &Sm90Fp8MegaMoeSpec {
        &self.spec
    }

    /// Returns the backing symmetric allocation size in bytes.
    pub fn allocation_bytes(&self) -> usize {
        self.spec.buffer_layout.total_bytes
    }
}

impl Bf16MegaMoeWorkspace {
    /// Allocates a zero-initialized single-rank symmetric buffer on `device`.
    ///
    /// The current constructor intentionally supports one rank only. Multi-rank
    /// use requires CUDA-IPC or NVSHMEM addresses that are valid from every
    /// process; accepting ordinary per-process tensors would violate the kernel
    /// contract.
    pub fn new_single_rank(spec: Bf16MegaMoeSpec, device: &candle::Device) -> Result<Self> {
        if spec.num_experts == 0
            || spec.num_topk == 0
            || spec.num_max_tokens_per_rank == 0
            || spec.num_ring_tokens == 0
            || spec.buffer_layout.total_bytes == 0
        {
            return invalid_arg("BF16 Mega MoE workspace dimensions must be positive");
        }
        let buffer = Tensor::zeros((spec.buffer_layout.total_bytes,), CandleDType::U8, device)?;
        Ok(Self { spec, buffer })
    }

    /// Returns the immutable launch specification associated with this allocation.
    pub fn spec(&self) -> &Bf16MegaMoeSpec {
        &self.spec
    }

    /// Returns the backing symmetric allocation size in bytes.
    pub fn allocation_bytes(&self) -> usize {
        self.spec.buffer_layout.total_bytes
    }
}

/// Interleaves BF16 L1 expert gate/up weights in DeepGEMM's eight-row layout.
///
/// `weights` must be contiguous BF16
/// `[experts, 2 * intermediate, hidden]`, with the complete gate matrix first
/// and the complete up matrix second. The returned contiguous tensor has the
/// same shape and orders rows as gate[0..8], up[0..8], gate[8..16], ... .
pub fn interleave_bf16_mega_moe_l1_weights(weights: &Tensor) -> Result<Tensor> {
    if weights.dtype() != CandleDType::BF16 {
        return invalid_arg(format!(
            "l1_weights must use BF16, got {:?}",
            weights.dtype()
        ));
    }
    interleave_mega_moe_l1_weights(weights)
}

/// Interleaves BF16 or FP8 L1 gate/up weights in DeepGEMM's eight-row layout.
///
/// `weights` must be contiguous `[experts, 2 * intermediate, hidden]`, with
/// gate rows followed by up rows. The returned tensor has the same dtype,
/// shape, and device and orders rows in alternating eight-row groups.
pub fn interleave_mega_moe_l1_weights(weights: &Tensor) -> Result<Tensor> {
    interleave_mega_moe_l1_weights_impl(weights, false)
}

/// Converts modeld-style FP8 `[up, gate]` L1 weights to Mega MoE's interleave.
///
/// `weights` must be contiguous CUDA FP8 E4M3
/// `[experts, 2 * intermediate, hidden]`. The returned tensor orders rows as
/// gate[0..8], up[0..8], gate[8..16], ... .
pub fn interleave_up_gate_fp8_mega_moe_l1_weights(weights: &Tensor) -> Result<Tensor> {
    if weights.dtype() != CandleDType::F8E4M3 || !weights.device().is_cuda() {
        return invalid_arg("up/gate FP8 Mega MoE interleave requires a CUDA F8E4M3 tensor");
    }
    interleave_mega_moe_l1_weights_impl(weights, true)
}

fn interleave_mega_moe_l1_weights_impl(weights: &Tensor, input_up_gate: bool) -> Result<Tensor> {
    ensure_rank(weights, 3, "l1_weights")?;
    if !matches!(weights.dtype(), CandleDType::BF16 | CandleDType::F8E4M3) {
        return invalid_arg(format!(
            "l1_weights must use BF16 or F8E4M3, got {:?}",
            weights.dtype()
        ));
    }
    let (experts, twice_intermediate, hidden) = weights.dims3()?;
    if twice_intermediate == 0 || twice_intermediate % 16 != 0 {
        return invalid_arg(format!(
            "l1_weights second dimension must be positive and divisible by 16, got {twice_intermediate}"
        ));
    }
    let intermediate = twice_intermediate / 2;
    if weights.dtype() == CandleDType::F8E4M3 && weights.device().is_cuda() {
        let weights = weights.contiguous()?;
        // SAFETY: the interleave kernel below overwrites every output byte.
        let output = unsafe { Tensor::empty_like(&weights)? };
        let (stream, device_id) = stream_and_device_id(&weights)?;
        let info = deepgemm::device_info()?;
        if info.device != device_id {
            return invalid_arg(format!(
                "weights are on CUDA device {device_id}, but DeepGEMM current device is {}",
                info.device
            ));
        }
        {
            let (input_storage, input_layout) = weights.storage_and_layout();
            let input_ptr = tensor_ptr_by_dtype(
                &input_storage,
                CandleDType::F8E4M3,
                input_layout.start_offset(),
                &stream,
                "weights",
            )?;
            let (output_storage, output_layout) = output.storage_and_layout();
            let output_ptr = tensor_ptr_by_dtype(
                &output_storage,
                CandleDType::F8E4M3,
                output_layout.start_offset(),
                &stream,
                "output",
            )?;
            deepgemm::fp8_mega_moe_interleave_l1_weights(
                TensorArg {
                    data: input_ptr.as_const_void(),
                    spec: tensor_spec(&weights, DeepGemmDType::Fp8E4M3, "weights")?,
                },
                TensorOut {
                    data: output_ptr.as_mut_void(),
                    spec: tensor_spec(&output, DeepGemmDType::Fp8E4M3, "output")?,
                },
                input_up_gate,
                stream.cu_stream() as *mut std::ffi::c_void,
            )?;
        }
        return Ok(output);
    }
    let weights = if input_up_gate {
        let up = weights.narrow(1, 0, intermediate)?;
        let gate = weights.narrow(1, intermediate, intermediate)?;
        Tensor::cat(&[&gate, &up], 1)?
    } else {
        weights.clone()
    };
    weights
        .contiguous()?
        .reshape((experts, 2, intermediate / 8, 8, hidden))?
        .permute((0, 2, 1, 3, 4))?
        .contiguous()?
        .reshape((experts, twice_intermediate, hidden))
        .map_err(Into::into)
}

/// Runs fused dispatch, FP8 expert SwiGLU, and combine on an SM90 CUDA device.
///
/// Tensor contract:
/// - `x`: contiguous FP8 E4M3 `[tokens, hidden]`.
/// - `x_scale`: contiguous F32 `[tokens, hidden / 128]`.
/// - `topk_indices`: contiguous I64 `[tokens, topk]` global expert IDs.
/// - `topk_weights`: contiguous F32 `[tokens, topk]` routing weights.
/// - `l1_weights`: interleaved FP8 `[experts, 2 * intermediate, hidden]`.
/// - `l1_weights_scale`: F32 `[experts, 2 * intermediate / 128, hidden / 128]`.
/// - `l2_weights`: FP8 `[experts, hidden, intermediate]`.
/// - `l2_weights_scale`: F32 `[experts, hidden / 128, intermediate / 128]`.
/// - returns contiguous BF16 `[tokens, hidden]`.
pub fn sm90_fp8_mega_moe(
    workspace: &Sm90Fp8MegaMoeWorkspace,
    x: &Tensor,
    x_scale: &Tensor,
    topk_indices: &Tensor,
    topk_weights: &Tensor,
    l1_weights: &Tensor,
    l1_weights_scale: &Tensor,
    l2_weights: &Tensor,
    l2_weights_scale: &Tensor,
) -> Result<Tensor> {
    for (name, tensor) in [
        ("x_scale", x_scale),
        ("topk_indices", topk_indices),
        ("topk_weights", topk_weights),
        ("l1_weights", l1_weights),
        ("l1_weights_scale", l1_weights_scale),
        ("l2_weights", l2_weights),
        ("l2_weights_scale", l2_weights_scale),
        ("workspace", &workspace.buffer),
    ] {
        ensure_same_device(x, tensor, name)?;
    }
    for (tensor, rank, dtype, name) in [
        (x, 2, CandleDType::F8E4M3, "x"),
        (x_scale, 2, CandleDType::F32, "x_scale"),
        (topk_indices, 2, CandleDType::I64, "topk_indices"),
        (topk_weights, 2, CandleDType::F32, "topk_weights"),
        (l1_weights, 3, CandleDType::F8E4M3, "l1_weights"),
        (l1_weights_scale, 3, CandleDType::F32, "l1_weights_scale"),
        (l2_weights, 3, CandleDType::F8E4M3, "l2_weights"),
        (l2_weights_scale, 3, CandleDType::F32, "l2_weights_scale"),
    ] {
        ensure_rank(tensor, rank, name)?;
        ensure_dtype(tensor, dtype, name)?;
    }

    let x = x.contiguous()?;
    let x_scale = x_scale.contiguous()?;
    let topk_indices = topk_indices.contiguous()?;
    let topk_weights = topk_weights.contiguous()?;
    let l1_weights = l1_weights.contiguous()?;
    let l1_weights_scale = l1_weights_scale.contiguous()?;
    let l2_weights = l2_weights.contiguous()?;
    let l2_weights_scale = l2_weights_scale.contiguous()?;
    let [tokens, hidden] = <[usize; 2]>::try_from(x.dims())
        .map_err(|_| crate::Error::Tensor("x dimensions changed during validation".into()))?;
    let output = Tensor::zeros((tokens, hidden), CandleDType::BF16, x.device())?;
    let (stream, device_id) = stream_and_device_id(&x)?;
    let info = deepgemm::device_info()?;
    if info.device != device_id {
        return invalid_arg(format!(
            "x is on CUDA device {device_id}, but DeepGEMM current device is {}",
            info.device
        ));
    }

    {
        macro_rules! tensor_ptr {
            ($pointer:ident, $storage:ident, $layout:ident, $tensor:expr, $dtype:expr, $name:literal) => {
                let ($storage, $layout) = $tensor.storage_and_layout();
                let $pointer =
                    tensor_ptr_by_dtype(&$storage, $dtype, $layout.start_offset(), &stream, $name)?;
            };
        }
        tensor_ptr!(x_ptr, x_storage, x_layout, x, CandleDType::F8E4M3, "x");
        tensor_ptr!(
            x_scale_ptr,
            x_scale_storage,
            x_scale_layout,
            x_scale,
            CandleDType::F32,
            "x_scale"
        );
        tensor_ptr!(
            indices_ptr,
            indices_storage,
            indices_layout,
            topk_indices,
            CandleDType::I64,
            "topk_indices"
        );
        tensor_ptr!(
            routing_ptr,
            routing_storage,
            routing_layout,
            topk_weights,
            CandleDType::F32,
            "topk_weights"
        );
        tensor_ptr!(
            l1_ptr,
            l1_storage,
            l1_layout,
            l1_weights,
            CandleDType::F8E4M3,
            "l1_weights"
        );
        tensor_ptr!(
            l1_scale_ptr,
            l1_scale_storage,
            l1_scale_layout,
            l1_weights_scale,
            CandleDType::F32,
            "l1_weights_scale"
        );
        tensor_ptr!(
            l2_ptr,
            l2_storage,
            l2_layout,
            l2_weights,
            CandleDType::F8E4M3,
            "l2_weights"
        );
        tensor_ptr!(
            l2_scale_ptr,
            l2_scale_storage,
            l2_scale_layout,
            l2_weights_scale,
            CandleDType::F32,
            "l2_weights_scale"
        );
        tensor_ptr!(
            output_ptr,
            output_storage,
            output_layout,
            output,
            CandleDType::BF16,
            "output"
        );
        tensor_ptr!(
            buffer_ptr,
            buffer_storage,
            buffer_layout,
            workspace.buffer,
            CandleDType::U8,
            "workspace"
        );
        let sym_buffer_ptrs = [buffer_ptr.as_mut_void() as usize as u64];

        let launch = Sm90Fp8MegaMoeLaunch {
            x: TensorArg {
                data: x_ptr.as_const_void(),
                spec: tensor_spec(&x, DeepGemmDType::Fp8E4M3, "x")?,
            },
            x_scale: TensorArg {
                data: x_scale_ptr.as_const_void(),
                spec: tensor_spec(&x_scale, DeepGemmDType::F32, "x_scale")?,
            },
            topk_indices: TensorArg {
                data: indices_ptr.as_const_void(),
                spec: tensor_spec(&topk_indices, DeepGemmDType::I64, "topk_indices")?,
            },
            topk_weights: TensorArg {
                data: routing_ptr.as_const_void(),
                spec: tensor_spec(&topk_weights, DeepGemmDType::F32, "topk_weights")?,
            },
            l1_weights: TensorArg {
                data: l1_ptr.as_const_void(),
                spec: tensor_spec(&l1_weights, DeepGemmDType::Fp8E4M3, "l1_weights")?,
            },
            l1_weights_scale: TensorArg {
                data: l1_scale_ptr.as_const_void(),
                spec: tensor_spec(&l1_weights_scale, DeepGemmDType::F32, "l1_weights_scale")?,
            },
            l2_weights: TensorArg {
                data: l2_ptr.as_const_void(),
                spec: tensor_spec(&l2_weights, DeepGemmDType::Fp8E4M3, "l2_weights")?,
            },
            l2_weights_scale: TensorArg {
                data: l2_scale_ptr.as_const_void(),
                spec: tensor_spec(&l2_weights_scale, DeepGemmDType::F32, "l2_weights_scale")?,
            },
            y: TensorOut {
                data: output_ptr.as_mut_void(),
                spec: tensor_spec(&output, DeepGemmDType::BF16, "output")?,
            },
            sym_buffer: TensorOut {
                data: buffer_ptr.as_mut_void(),
                spec: TensorSpec::contiguous(DeepGemmDType::U8, [workspace.allocation_bytes()]),
            },
            sym_buffer_ptrs: &sym_buffer_ptrs,
            rank_idx: 0,
            activation_clamp: f32::INFINITY,
            fast_math: true,
            stream: stream.cu_stream() as *mut std::ffi::c_void,
        };
        deepgemm::sm90_fp8_mega_moe(workspace.spec(), launch)?;
    }
    Ok(output)
}

/// Runs fused dispatch, BF16 expert SwiGLU, and combine on CUDA.
///
/// Tensor contract:
/// - `x`: contiguous BF16 `[tokens, hidden]`.
/// - `topk_indices`: contiguous I64 `[tokens, topk]` global expert IDs.
/// - `topk_weights`: contiguous F32 `[tokens, topk]` routing weights.
/// - `l1_weights`: contiguous, pre-interleaved BF16
///   `[experts, 2 * intermediate, hidden]`.
/// - `l2_weights`: contiguous BF16 `[experts, hidden, intermediate]`.
/// - returns contiguous BF16 `[tokens, hidden]` on the same device.
///
/// The upstream implementation is available only on SM100. Callers must gate
/// dispatch by compute capability and retain a fallback on other devices.
pub fn bf16_mega_moe(
    workspace: &Bf16MegaMoeWorkspace,
    x: &Tensor,
    topk_indices: &Tensor,
    topk_weights: &Tensor,
    l1_weights: &Tensor,
    l2_weights: &Tensor,
) -> Result<Tensor> {
    for (name, tensor) in [
        ("topk_indices", topk_indices),
        ("topk_weights", topk_weights),
        ("l1_weights", l1_weights),
        ("l2_weights", l2_weights),
        ("workspace", &workspace.buffer),
    ] {
        ensure_same_device(x, tensor, name)?;
    }
    ensure_rank(x, 2, "x")?;
    ensure_rank(topk_indices, 2, "topk_indices")?;
    ensure_rank(topk_weights, 2, "topk_weights")?;
    ensure_rank(l1_weights, 3, "l1_weights")?;
    ensure_rank(l2_weights, 3, "l2_weights")?;
    ensure_dtype(x, CandleDType::BF16, "x")?;
    ensure_dtype(topk_indices, CandleDType::I64, "topk_indices")?;
    ensure_dtype(topk_weights, CandleDType::F32, "topk_weights")?;
    ensure_dtype(l1_weights, CandleDType::BF16, "l1_weights")?;
    ensure_dtype(l2_weights, CandleDType::BF16, "l2_weights")?;

    let x = x.contiguous()?;
    let topk_indices = topk_indices.contiguous()?;
    let topk_weights = topk_weights.contiguous()?;
    let l1_weights = l1_weights.contiguous()?;
    let l2_weights = l2_weights.contiguous()?;
    let [tokens, hidden] = <[usize; 2]>::try_from(x.dims())
        .map_err(|_| crate::Error::Tensor("x dimensions changed during validation".into()))?;
    let output = Tensor::zeros((tokens, hidden), CandleDType::BF16, x.device())?;
    let (stream, device_id) = stream_and_device_id(&x)?;
    let info = deepgemm::device_info()?;
    if info.device != device_id {
        return invalid_arg(format!(
            "x is on CUDA device {device_id}, but DeepGEMM current device is {}",
            info.device
        ));
    }

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let x_ptr = tensor_ptr_by_dtype(
            &x_storage,
            CandleDType::BF16,
            x_layout.start_offset(),
            &stream,
            "x",
        )?;
        let (indices_storage, indices_layout) = topk_indices.storage_and_layout();
        let indices_ptr = tensor_ptr_by_dtype(
            &indices_storage,
            CandleDType::I64,
            indices_layout.start_offset(),
            &stream,
            "topk_indices",
        )?;
        let (weights_storage, weights_layout) = topk_weights.storage_and_layout();
        let weights_ptr = tensor_ptr_by_dtype(
            &weights_storage,
            CandleDType::F32,
            weights_layout.start_offset(),
            &stream,
            "topk_weights",
        )?;
        let (l1_storage, l1_layout) = l1_weights.storage_and_layout();
        let l1_ptr = tensor_ptr_by_dtype(
            &l1_storage,
            CandleDType::BF16,
            l1_layout.start_offset(),
            &stream,
            "l1_weights",
        )?;
        let (l2_storage, l2_layout) = l2_weights.storage_and_layout();
        let l2_ptr = tensor_ptr_by_dtype(
            &l2_storage,
            CandleDType::BF16,
            l2_layout.start_offset(),
            &stream,
            "l2_weights",
        )?;
        let (output_storage, output_layout) = output.storage_and_layout();
        let output_ptr = tensor_ptr_by_dtype(
            &output_storage,
            CandleDType::BF16,
            output_layout.start_offset(),
            &stream,
            "output",
        )?;
        let (buffer_storage, buffer_layout) = workspace.buffer.storage_and_layout();
        let buffer_ptr = tensor_ptr_by_dtype(
            &buffer_storage,
            CandleDType::U8,
            buffer_layout.start_offset(),
            &stream,
            "workspace",
        )?;
        let sym_buffer_ptrs = [buffer_ptr.as_mut_void() as usize as u64];

        let launch = Bf16MegaMoeLaunch {
            x: TensorArg {
                data: x_ptr.as_const_void(),
                spec: tensor_spec(&x, DeepGemmDType::BF16, "x")?,
            },
            topk_indices: TensorArg {
                data: indices_ptr.as_const_void(),
                spec: tensor_spec(&topk_indices, DeepGemmDType::I64, "topk_indices")?,
            },
            topk_weights: TensorArg {
                data: weights_ptr.as_const_void(),
                spec: tensor_spec(&topk_weights, DeepGemmDType::F32, "topk_weights")?,
            },
            l1_weights: TensorArg {
                data: l1_ptr.as_const_void(),
                spec: tensor_spec(&l1_weights, DeepGemmDType::BF16, "l1_weights")?,
            },
            l2_weights: TensorArg {
                data: l2_ptr.as_const_void(),
                spec: tensor_spec(&l2_weights, DeepGemmDType::BF16, "l2_weights")?,
            },
            y: TensorOut {
                data: output_ptr.as_mut_void(),
                spec: tensor_spec(&output, DeepGemmDType::BF16, "output")?,
            },
            sym_buffer: TensorOut {
                data: buffer_ptr.as_mut_void(),
                spec: TensorSpec::contiguous(DeepGemmDType::U8, [workspace.allocation_bytes()]),
            },
            sym_buffer_ptrs: &sym_buffer_ptrs,
            rank_idx: 0,
            activation_clamp: f32::INFINITY,
            fast_math: true,
            stream: stream.cu_stream() as *mut std::ffi::c_void,
        };
        deepgemm::bf16_mega_moe(workspace.spec(), launch)?;
    }
    Ok(output)
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

#[cfg(test)]
mod tests {
    use candle::Device;
    use deepgemm::{
        MegaMoeBufferConfig, MegaMoeMmaKind, MegaMoeRingConfig, mega_moe_buffer_layout,
        mega_moe_ring_limits, sm90_fp8_mega_moe_buffer_layout,
    };

    use super::*;

    #[test]
    fn interleaves_gate_and_up_in_eight_row_groups() -> Result<()> {
        let values = (0..32)
            .map(|value| half::bf16::from_f32(value as f32))
            .collect::<Vec<_>>();
        let weights = Tensor::from_vec(values, (1, 32, 1), &Device::Cpu)?;
        let transformed = interleave_bf16_mega_moe_l1_weights(&weights)?;
        let values = transformed.flatten_all()?.to_vec1::<half::bf16>()?;
        let values = values.into_iter().map(f32::from).collect::<Vec<_>>();
        assert_eq!(
            values,
            vec![
                0., 1., 2., 3., 4., 5., 6., 7., 16., 17., 18., 19., 20., 21., 22., 23., 8., 9.,
                10., 11., 12., 13., 14., 15., 24., 25., 26., 27., 28., 29., 30., 31.,
            ]
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires an SM100 CUDA device and the DeepGEMM JIT toolchain"]
    fn launches_single_rank_bf16_mega_moe() -> Result<()> {
        let root = deepgemm::source_root()
            .to_str()
            .ok_or_else(|| crate::Error::Tensor("DeepGEMM root is not UTF-8".into()))?;
        let cuda_home = std::env::var("CUDA_HOME")
            .or_else(|_| std::env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        deepgemm::init(root, &cuda_home)?;
        let info = deepgemm::device_info()?;
        if info.compute_capability_major != 10 {
            println!(
                "Skipping test: BF16 Mega MoE requires SM10x, got SM{}{}",
                info.compute_capability_major, info.compute_capability_minor
            );
            return Ok(());
        }
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(_) => {
                println!("Skipping test: no CUDA device available");
                return Ok(());
            }
        };
        let num_experts = 256;
        let num_topk = 8;
        let hidden = 2048;
        let intermediate = 512;
        let max_tokens = 384;
        let ring = mega_moe_ring_limits(MegaMoeRingConfig {
            num_ranks: 1,
            num_experts,
            num_max_tokens_per_rank: max_tokens,
            num_topk,
        })?;
        let buffer_layout = mega_moe_buffer_layout(MegaMoeBufferConfig {
            num_ranks: 1,
            num_experts,
            num_max_tokens_per_rank: max_tokens,
            num_topk,
            hidden,
            intermediate_hidden: intermediate,
            num_ring_tokens: ring.max_tokens,
            mma_kind: MegaMoeMmaKind::Bf16,
        })?;
        let workspace = Bf16MegaMoeWorkspace::new_single_rank(
            Bf16MegaMoeSpec {
                num_experts,
                num_topk,
                num_max_tokens_per_rank: max_tokens,
                num_ring_tokens: ring.max_tokens,
                buffer_layout,
            },
            &device,
        )?;
        let x = Tensor::ones((1, hidden), CandleDType::BF16, &device)?;
        let topk_indices =
            Tensor::from_vec((0..num_topk as i64).collect(), (1, num_topk), &device)?;
        let topk_weights = Tensor::ones((1, num_topk), CandleDType::F32, &device)?;
        let l1_weights = Tensor::zeros(
            (num_experts, 2 * intermediate, hidden),
            CandleDType::BF16,
            &device,
        )?;
        let l1_weights = interleave_bf16_mega_moe_l1_weights(&l1_weights)?;
        let l2_weights = Tensor::zeros(
            (num_experts, hidden, intermediate),
            CandleDType::BF16,
            &device,
        )?;

        let output = bf16_mega_moe(
            &workspace,
            &x,
            &topk_indices,
            &topk_weights,
            &l1_weights,
            &l2_weights,
        )?;
        let output = output.to_dtype(CandleDType::F32)?.to_vec2::<f32>()?;
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), hidden);
        assert!(output[0].iter().all(|value| *value == 0.0));
        Ok(())
    }

    #[test]
    #[ignore = "requires an SM90 CUDA device and the DeepGEMM JIT toolchain"]
    fn launches_single_rank_sm90_fp8_mega_moe() -> Result<()> {
        let root = deepgemm::source_root()
            .to_str()
            .ok_or_else(|| crate::Error::Tensor("DeepGEMM root is not UTF-8".into()))?;
        let cuda_home = std::env::var("CUDA_HOME")
            .or_else(|_| std::env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        deepgemm::init(root, &cuda_home)?;
        let info = deepgemm::device_info()?;
        if info.compute_capability_major != 9 {
            println!(
                "Skipping test: FP8 Mega MoE requires SM9x, got SM{}{}",
                info.compute_capability_major, info.compute_capability_minor
            );
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let num_experts = 8;
        let num_topk = 2;
        let hidden = 256;
        let intermediate = 128;
        let max_tokens = 128;
        let buffer_layout = sm90_fp8_mega_moe_buffer_layout(MegaMoeBufferConfig {
            num_ranks: 1,
            num_experts,
            num_max_tokens_per_rank: max_tokens,
            num_topk,
            hidden,
            intermediate_hidden: intermediate,
            num_ring_tokens: 0,
            mma_kind: MegaMoeMmaKind::Fp8Fp8Sm90,
        })?;
        let workspace = Sm90Fp8MegaMoeWorkspace::new_single_rank(
            Sm90Fp8MegaMoeSpec {
                num_experts,
                num_topk,
                num_max_tokens_per_rank: max_tokens,
                buffer_layout,
            },
            &device,
        )?;
        let x =
            Tensor::zeros((1, hidden), CandleDType::F8E4M3, &Device::Cpu)?.to_device(&device)?;
        let x_scale = Tensor::ones((1, hidden / 128), CandleDType::F32, &device)?;
        let topk_indices = Tensor::from_vec(vec![0i64, 1], (1, num_topk), &device)?;
        let topk_weights = Tensor::ones((1, num_topk), CandleDType::F32, &device)?;
        let l1_weights = Tensor::zeros(
            (num_experts, 2 * intermediate, hidden),
            CandleDType::F8E4M3,
            &Device::Cpu,
        )?
        .to_device(&device)?;
        let l1_weights = interleave_mega_moe_l1_weights(&l1_weights)?;
        let l1_scale = Tensor::ones(
            (num_experts, 2 * intermediate / 128, hidden / 128),
            CandleDType::F32,
            &device,
        )?;
        let l2_weights = Tensor::zeros(
            (num_experts, hidden, intermediate),
            CandleDType::F8E4M3,
            &Device::Cpu,
        )?
        .to_device(&device)?;
        let l2_scale = Tensor::ones(
            (num_experts, hidden / 128, intermediate / 128),
            CandleDType::F32,
            &device,
        )?;
        let output = sm90_fp8_mega_moe(
            &workspace,
            &x,
            &x_scale,
            &topk_indices,
            &topk_weights,
            &l1_weights,
            &l1_scale,
            &l2_weights,
            &l2_scale,
        )?;
        assert_eq!(output.dims(), &[1, hidden]);
        assert_eq!(output.dtype(), CandleDType::BF16);
        let values = output.to_dtype(CandleDType::F32)?.to_vec2::<f32>()?;
        assert!(values[0].iter().all(|value| *value == 0.0));
        Ok(())
    }
}
