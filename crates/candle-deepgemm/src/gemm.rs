//! Candle CUDA bindings for dense FP8 GEMM.

use std::sync::Arc;

use candle::{DType as CandleDType, Tensor, cuda::cudarc::driver::CudaStream};
use deepgemm::{
    Arch, DType as DeepGemmDType, Fp8GemmNtLaunch, Fp8GemmNtSpec, Fp8GemmScaleTransformLaunch,
    TensorArg, TensorOut, TensorSpec,
};

use crate::{
    Result,
    error::invalid_arg,
    tensor::cuda::{
        ensure_dtype, ensure_rank, ensure_same_device, stream_and_device_id, tensor_ptr_by_dtype,
    },
};

/// Launches dense DeepGEMM FP8 `nt` GEMM on Candle CUDA tensors.
///
/// Computes `d = a @ b.T`.
///
/// Tensor contract:
/// - `a`: `F8E4M3 [m, k]`, row-major contiguous.
/// - `a_scale`: `F32 [m, ceil(k / 128)]`, row-major contiguous raw scales.
/// - `b`: `F8E4M3 [n, k]`, row-major contiguous. This is the logical
///   transposed RHS.
/// - `b_scale`:
///   - SM90: `F32 [ceil(n / 128), ceil(k / 128)]`, row-major contiguous
///     raw block scales.
///   - SM100: `F32 [n, ceil(k / 128)]`, row-major contiguous raw per-row
///     scales.
///
/// Returns `BF16 [m, n]`, row-major contiguous.
pub fn fp8_gemm_nt(a: &Tensor, a_scale: &Tensor, b: &Tensor, b_scale: &Tensor) -> Result<Tensor> {
    validate_devices(a, a_scale, b, b_scale)?;
    ensure_rank(a, 2, "a")?;
    ensure_rank(a_scale, 2, "a_scale")?;
    ensure_rank(b, 2, "b")?;
    ensure_rank(b_scale, 2, "b_scale")?;
    ensure_dtype(a, CandleDType::F8E4M3, "a")?;
    ensure_dtype(a_scale, CandleDType::F32, "a_scale")?;
    ensure_dtype(b, CandleDType::F8E4M3, "b")?;
    ensure_dtype(b_scale, CandleDType::F32, "b_scale")?;

    let m = a.dims()[0];
    let k = a.dims()[1];
    let n = b.dims()[0];
    if b.dims()[1] != k {
        return invalid_arg("b shape must be [n, k] with the same k as a");
    }

    let (stream, device_id) = stream_and_device_id(a)?;
    let device_info = deepgemm::device_info()?;
    if device_info.device != device_id {
        return invalid_arg(format!(
            "a is on CUDA device {device_id}, but DeepGEMM current device is {}",
            device_info.device
        ));
    }
    let arch = device_info.arch()?;

    let a_spec = tensor_spec(a, DeepGemmDType::Fp8E4M3, "a")?;
    let b_spec = tensor_spec(b, DeepGemmDType::Fp8E4M3, "b")?;
    let a_scale_raw_spec = tensor_spec(a_scale, DeepGemmDType::F32, "a_scale")?;
    require_raw_scale(a_scale_raw_spec, [m, ceil_div(k, 128)?], "a_scale")?;

    let a_scale_layout = match arch {
        Arch::Sm90 => deepgemm::fp8_gemm_scale_layout(m, k, 128, DeepGemmDType::F32)?,
        Arch::Sm100 => deepgemm::fp8_gemm_scale_layout(m, k, 128, DeepGemmDType::PackedUe8M0)?,
    };
    let a_scale_transformed =
        transform_scale(a_scale, a_scale_raw_spec, a_scale_layout, m, k, &stream)?;

    let b_scale_tensor;
    let b_scale_launch_spec;
    match arch {
        Arch::Sm90 => {
            let expected = [ceil_div(n, 128)?, ceil_div(k, 128)?];
            b_scale_launch_spec = tensor_spec(b_scale, DeepGemmDType::F32, "b_scale")?;
            require_raw_scale(b_scale_launch_spec, expected, "b_scale")?;
            b_scale_tensor = b_scale.clone();
        }
        Arch::Sm100 => {
            let b_scale_raw_spec = tensor_spec(b_scale, DeepGemmDType::F32, "b_scale")?;
            require_raw_scale(b_scale_raw_spec, [n, ceil_div(k, 128)?], "b_scale")?;
            let b_scale_layout =
                deepgemm::fp8_gemm_scale_layout(n, k, 128, DeepGemmDType::PackedUe8M0)?;
            b_scale_tensor =
                transform_scale(b_scale, b_scale_raw_spec, b_scale_layout, n, k, &stream)?;
            b_scale_launch_spec = b_scale_layout.logical_spec();
        }
    }

    let d_spec = TensorSpec {
        dtype: DeepGemmDType::BF16,
        shape: [m, n],
        strides: [
            isize::try_from(n).map_err(|_| crate::Error::Tensor("d stride overflow".into()))?,
            1,
        ],
    };
    let launch_spec = Fp8GemmNtSpec {
        a: a_spec,
        a_scale: a_scale_layout.logical_spec(),
        b: b_spec,
        b_scale: b_scale_launch_spec,
        d: d_spec,
    };
    let d_layout = deepgemm::fp8_gemm_nt_output_layout(&launch_spec, arch)?;
    let d = Tensor::zeros(
        (d_layout.allocation_shape[0], d_layout.allocation_shape[1]),
        CandleDType::BF16,
        a.device(),
    )?;

    {
        let (a_storage, a_layout) = a.storage_and_layout();
        let a_ptr = tensor_ptr_by_dtype(
            &a_storage,
            CandleDType::F8E4M3,
            a_layout.start_offset(),
            &stream,
            "a",
        )?;
        let (a_scale_storage, a_scale_layout_storage) = a_scale_transformed.storage_and_layout();
        let a_scale_dtype = candle_dtype_for_deepgemm(a_scale_layout.dtype);
        let a_scale_ptr = tensor_ptr_by_dtype(
            &a_scale_storage,
            a_scale_dtype,
            a_scale_layout_storage.start_offset(),
            &stream,
            "a_scale",
        )?;
        let (b_storage, b_layout) = b.storage_and_layout();
        let b_ptr = tensor_ptr_by_dtype(
            &b_storage,
            CandleDType::F8E4M3,
            b_layout.start_offset(),
            &stream,
            "b",
        )?;
        let (b_scale_storage, b_scale_layout_storage) = b_scale_tensor.storage_and_layout();
        let b_scale_dtype = match arch {
            Arch::Sm90 => CandleDType::F32,
            Arch::Sm100 => CandleDType::I32,
        };
        let b_scale_ptr = tensor_ptr_by_dtype(
            &b_scale_storage,
            b_scale_dtype,
            b_scale_layout_storage.start_offset(),
            &stream,
            "b_scale",
        )?;
        let (d_storage, d_storage_layout) = d.storage_and_layout();
        let d_ptr = tensor_ptr_by_dtype(
            &d_storage,
            CandleDType::BF16,
            d_storage_layout.start_offset(),
            &stream,
            "d",
        )?;

        let launch = Fp8GemmNtLaunch {
            a: tensor_arg(a_ptr.as_const_void(), a_spec),
            a_scale: tensor_arg(a_scale_ptr.as_const_void(), a_scale_layout.logical_spec()),
            b: tensor_arg(b_ptr.as_const_void(), b_spec),
            b_scale: tensor_arg(b_scale_ptr.as_const_void(), b_scale_launch_spec),
            d: TensorOut {
                data: d_ptr.as_mut_void(),
                spec: d_layout.logical_spec(),
            },
            stream: stream.cu_stream() as *mut std::ffi::c_void,
        };

        // SAFETY: all pointers come from live Candle CUDA tensors on the launch stream,
        // and specs were validated by the DeepGEMM layout path before launch.
        unsafe { deepgemm::fp8_gemm_nt(&launch)? };
    }

    Ok(d)
}

fn transform_scale(
    scale: &Tensor,
    scale_spec: TensorSpec<2>,
    transformed_layout: deepgemm::TensorLayout2D,
    mn: usize,
    k: usize,
    stream: &Arc<CudaStream>,
) -> Result<Tensor> {
    let transformed = allocate_transformed_scale(transformed_layout, scale.device())?;
    {
        let (scale_storage, scale_layout) = scale.storage_and_layout();
        let scale_ptr = tensor_ptr_by_dtype(
            &scale_storage,
            CandleDType::F32,
            scale_layout.start_offset(),
            stream,
            "scale",
        )?;
        let (transformed_storage, transformed_storage_layout) = transformed.storage_and_layout();
        let transformed_dtype = candle_dtype_for_deepgemm(transformed_layout.dtype);
        let transformed_ptr = tensor_ptr_by_dtype(
            &transformed_storage,
            transformed_dtype,
            transformed_storage_layout.start_offset(),
            stream,
            "transformed",
        )?;

        let launch = Fp8GemmScaleTransformLaunch {
            scale: tensor_arg(scale_ptr.as_const_void(), scale_spec),
            transformed: TensorOut {
                data: transformed_ptr.as_mut_void(),
                spec: transformed_layout.logical_spec(),
            },
            mn,
            k,
            gran_k: 128,
            stream: stream.cu_stream() as *mut std::ffi::c_void,
        };

        // SAFETY: pointers come from live Candle CUDA tensors on the launch stream,
        // and specs were checked by the safe DeepGEMM wrapper.
        unsafe { deepgemm::fp8_gemm_transform_scale(&launch)? };
    }
    Ok(transformed)
}

fn allocate_transformed_scale(
    layout: deepgemm::TensorLayout2D,
    device: &candle::Device,
) -> Result<Tensor> {
    let allocation = Tensor::zeros(
        (layout.allocation_shape[0], layout.allocation_shape[1]),
        candle_dtype_for_deepgemm(layout.dtype),
        device,
    )?;
    Ok(allocation
        .transpose(0, 1)?
        .narrow(0, 0, layout.logical_shape[0])?)
}

fn validate_devices(a: &Tensor, a_scale: &Tensor, b: &Tensor, b_scale: &Tensor) -> Result<()> {
    for (name, tensor) in [("a_scale", a_scale), ("b", b), ("b_scale", b_scale)] {
        ensure_same_device(a, tensor, name)?;
    }
    Ok(())
}

fn require_raw_scale(spec: TensorSpec<2>, expected_shape: [usize; 2], name: &str) -> Result<()> {
    if !spec.is_contiguous() {
        return invalid_arg(format!("{name} must be row-major contiguous"));
    }
    if spec.dtype != DeepGemmDType::F32 {
        return invalid_arg(format!("{name} must have dtype F32"));
    }
    if spec.shape != expected_shape {
        return invalid_arg(format!(
            "{name} shape must be {expected_shape:?}, got {:?}",
            spec.shape
        ));
    }
    Ok(())
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

fn candle_dtype_for_deepgemm(dtype: DeepGemmDType) -> CandleDType {
    match dtype {
        DeepGemmDType::Fp8E4M3 => CandleDType::F8E4M3,
        DeepGemmDType::PackedUe8M0 | DeepGemmDType::I32 => CandleDType::I32,
        DeepGemmDType::F32 => CandleDType::F32,
        DeepGemmDType::BF16 => CandleDType::BF16,
        DeepGemmDType::PackedFp4E2M1 | DeepGemmDType::U8 => CandleDType::U8,
    }
}

fn ceil_div(value: usize, divisor: usize) -> Result<usize> {
    if divisor == 0 {
        return invalid_arg("divisor must be positive");
    }
    value
        .checked_add(divisor - 1)
        .map(|value| value / divisor)
        .ok_or_else(|| crate::Error::Tensor("ceil_div overflowed".into()))
}
