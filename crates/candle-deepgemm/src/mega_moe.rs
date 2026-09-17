//! Candle CUDA integration for DeepGEMM BF16 Mega MoE.

use candle::{DType as CandleDType, Tensor};
use deepgemm::{
    Bf16MegaMoeLaunch, Bf16MegaMoeSpec, DType as DeepGemmDType, TensorArg, TensorOut, TensorSpec,
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
    ensure_rank(weights, 3, "l1_weights")?;
    ensure_dtype(weights, CandleDType::BF16, "l1_weights")?;
    let (experts, twice_intermediate, hidden) = weights.dims3()?;
    if twice_intermediate == 0 || twice_intermediate % 16 != 0 {
        return invalid_arg(format!(
            "l1_weights second dimension must be positive and divisible by 16, got {twice_intermediate}"
        ));
    }
    let intermediate = twice_intermediate / 2;
    weights
        .contiguous()?
        .reshape((experts, 2, intermediate / 8, 8, hidden))?
        .permute((0, 2, 1, 3, 4))?
        .contiguous()?
        .reshape((experts, twice_intermediate, hidden))
        .map_err(Into::into)
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
}
