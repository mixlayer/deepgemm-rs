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
///   - SM100: UE8M0-compatible row-major F32 scales shaped either
///     `[n, ceil(k / 128)]` (per row) or
///     `[ceil(n / 128), ceil(k / 128)]` (128-row blocks). This function
///     prepares both operands on every call; retain prepared weight scales and
///     use [`fp8_gemm_nt_prepared_scales`] when weights are reused.
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

    let (_, device_id) = stream_and_device_id(a)?;
    let device_info = deepgemm::device_info()?;
    if device_info.device != device_id {
        return invalid_arg(format!(
            "a is on CUDA device {device_id}, but DeepGEMM current device is {}",
            device_info.device
        ));
    }
    let arch = device_info.arch()?;

    let a_scale_raw_spec = tensor_spec(a_scale, DeepGemmDType::F32, "a_scale")?;
    require_raw_scale(a_scale_raw_spec, [m, ceil_div(k, 128)?], "a_scale")?;

    let scale_dtype = match arch {
        Arch::Sm90 => DeepGemmDType::F32,
        Arch::Sm100 => DeepGemmDType::PackedUe8M0,
    };
    let a_scale_transformed = prepare_fp8_gemm_scale(a_scale, m, k, 1, 128, scale_dtype)?;

    let b_scale_tensor;
    match arch {
        Arch::Sm90 => {
            let expected = [ceil_div(n, 128)?, ceil_div(k, 128)?];
            let b_scale_spec = tensor_spec(b_scale, DeepGemmDType::F32, "b_scale")?;
            require_raw_scale(b_scale_spec, expected, "b_scale")?;
            b_scale_tensor = b_scale.clone();
        }
        Arch::Sm100 => {
            let b_scale_raw_spec = tensor_spec(b_scale, DeepGemmDType::F32, "b_scale")?;
            let scale_cols = ceil_div(k, 128)?;
            let gran_mn = if b_scale_raw_spec.shape == [n, scale_cols] {
                1
            } else if b_scale_raw_spec.shape == [ceil_div(n, 128)?, scale_cols] {
                128
            } else {
                return invalid_arg(format!(
                    "SM100 b_scale shape must be [{n}, {scale_cols}] or [{}, {scale_cols}], got {:?}",
                    ceil_div(n, 128)?,
                    b_scale_raw_spec.shape
                ));
            };
            b_scale_tensor =
                prepare_fp8_gemm_scale(b_scale, n, k, gran_mn, 128, DeepGemmDType::PackedUe8M0)?;
        }
    }

    fp8_gemm_nt_prepared_scales(a, &a_scale_transformed, b, &b_scale_tensor)
}

/// Transforms raw F32 FP8 scales into a DeepGEMM architecture-native layout.
///
/// Tensor contract:
/// - `scale`: contiguous CUDA
///   `F32 [ceil(mn / gran_mn), ceil(k / gran_k)]`.
/// - returns an F32 or packed-UE8M0 (`I32` in Candle) logical tensor with
///   shape and strides from `deepgemm::fp8_gemm_scale_layout`.
///
/// For packed UE8M0 output every input F32 value must already be a finite,
/// positive, exponent-only power of two. This operation changes layout and
/// broadcasts MN blocks; it does not numerically requantize weights.
pub fn prepare_fp8_gemm_scale(
    scale: &Tensor,
    mn: usize,
    k: usize,
    gran_mn: usize,
    gran_k: usize,
    transformed_dtype: DeepGemmDType,
) -> Result<Tensor> {
    ensure_rank(scale, 2, "scale")?;
    ensure_dtype(scale, CandleDType::F32, "scale")?;
    let scale_spec = tensor_spec(scale, DeepGemmDType::F32, "scale")?;
    require_raw_scale(
        scale_spec,
        [ceil_div(mn, gran_mn)?, ceil_div(k, gran_k)?],
        "scale",
    )?;

    let (stream, device_id) = stream_and_device_id(scale)?;
    let device_info = deepgemm::device_info()?;
    if device_info.device != device_id {
        return invalid_arg(format!(
            "scale is on CUDA device {device_id}, but DeepGEMM current device is {}",
            device_info.device
        ));
    }
    let transformed_layout = deepgemm::fp8_gemm_scale_layout(mn, k, gran_k, transformed_dtype)?;
    transform_scale(
        scale,
        scale_spec,
        transformed_layout,
        mn,
        k,
        gran_mn,
        gran_k,
        &stream,
    )
}

/// Launches dense FP8 `nt` GEMM with architecture-native scale tensors.
///
/// Tensor contract:
/// - `a`: contiguous CUDA `F8E4M3 [m, k]`.
/// - `b`: contiguous CUDA `F8E4M3 [n, k]`.
/// - SM90 `a_scale`: transformed F32 `[m, ceil(k / 128)]`; `b_scale`:
///   contiguous checkpoint F32 `[ceil(n / 128), ceil(k / 128)]`.
/// - SM100 `a_scale` and `b_scale`: packed UE8M0 I32 tensors returned by
///   [`prepare_fp8_gemm_scale`] for logical MN sizes `m` and `n`.
/// - returns contiguous CUDA `BF16 [m, n]`.
pub fn fp8_gemm_nt_prepared_scales(
    a: &Tensor,
    a_scale: &Tensor,
    b: &Tensor,
    b_scale: &Tensor,
) -> Result<Tensor> {
    validate_devices(a, a_scale, b, b_scale)?;
    ensure_rank(a, 2, "a")?;
    ensure_rank(a_scale, 2, "a_scale")?;
    ensure_rank(b, 2, "b")?;
    ensure_rank(b_scale, 2, "b_scale")?;
    ensure_dtype(a, CandleDType::F8E4M3, "a")?;
    ensure_dtype(b, CandleDType::F8E4M3, "b")?;

    let (stream, device_id) = stream_and_device_id(a)?;
    let device_info = deepgemm::device_info()?;
    if device_info.device != device_id {
        return invalid_arg(format!(
            "a is on CUDA device {device_id}, but DeepGEMM current device is {}",
            device_info.device
        ));
    }
    let arch = device_info.arch()?;
    let scale_dtype = match arch {
        Arch::Sm90 => {
            ensure_dtype(a_scale, CandleDType::F32, "a_scale")?;
            ensure_dtype(b_scale, CandleDType::F32, "b_scale")?;
            DeepGemmDType::F32
        }
        Arch::Sm100 => {
            ensure_dtype(a_scale, CandleDType::I32, "a_scale")?;
            ensure_dtype(b_scale, CandleDType::I32, "b_scale")?;
            DeepGemmDType::PackedUe8M0
        }
    };

    let a_spec = tensor_spec(a, DeepGemmDType::Fp8E4M3, "a")?;
    let a_scale_spec = tensor_spec(a_scale, scale_dtype, "a_scale")?;
    let b_spec = tensor_spec(b, DeepGemmDType::Fp8E4M3, "b")?;
    let b_scale_spec = tensor_spec(b_scale, scale_dtype, "b_scale")?;
    let m = a_spec.shape[0];
    let k = a_spec.shape[1];
    let n = b_spec.shape[0];
    if b_spec.shape[1] != k {
        return invalid_arg("b shape must be [n, k] with the same k as a");
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
        a_scale: a_scale_spec,
        b: b_spec,
        b_scale: b_scale_spec,
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
        let (a_scale_storage, a_scale_layout_storage) = a_scale.storage_and_layout();
        let a_scale_dtype = candle_dtype_for_deepgemm(scale_dtype);
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
        let (b_scale_storage, b_scale_layout_storage) = b_scale.storage_and_layout();
        let b_scale_dtype = candle_dtype_for_deepgemm(scale_dtype);
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
            a_scale: tensor_arg(a_scale_ptr.as_const_void(), a_scale_spec),
            b: tensor_arg(b_ptr.as_const_void(), b_spec),
            b_scale: tensor_arg(b_scale_ptr.as_const_void(), b_scale_spec),
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
    gran_mn: usize,
    gran_k: usize,
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
            gran_mn,
            gran_k,
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

#[cfg(test)]
mod tests {
    use candle::Device;

    use super::*;

    fn cuda_device_or_skip() -> Option<Device> {
        match Device::new_cuda(0) {
            Ok(device) => Some(device),
            Err(_) => {
                println!("Skipping test: no CUDA device available");
                None
            }
        }
    }

    fn init_deepgemm() -> Result<()> {
        let root = deepgemm::source_root()
            .to_str()
            .ok_or_else(|| crate::Error::Tensor("DeepGEMM root is not UTF-8".into()))?;
        let cuda_home = std::env::var("CUDA_HOME")
            .or_else(|_| std::env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        deepgemm::init(root, &cuda_home)?;
        Ok(())
    }

    fn packed_ue8m0(values: [f32; 4]) -> i32 {
        values
            .into_iter()
            .enumerate()
            .fold(0u32, |packed, (lane, value)| {
                packed | ((value.to_bits() >> 23) << (lane * 8))
            }) as i32
    }

    #[test]
    #[ignore = "requires CUDA and the DeepGEMM JIT toolchain"]
    fn prepares_compact_block_scales_for_sm100_layout() -> Result<()> {
        let Some(device) = cuda_device_or_skip() else {
            return Ok(());
        };
        init_deepgemm()?;

        let first = [0.5f32, 1.0, 2.0, 4.0];
        let second = [8.0f32, 16.0, 32.0, 64.0];
        let scales = Tensor::from_slice(
            &[first.as_slice(), second.as_slice()].concat(),
            (2, 4),
            &device,
        )?;
        let prepared =
            prepare_fp8_gemm_scale(&scales, 136, 512, 128, 128, DeepGemmDType::PackedUe8M0)?;

        assert_eq!(prepared.dtype(), CandleDType::I32);
        assert_eq!(prepared.dims(), &[136, 1]);
        assert_eq!(prepared.stride(), &[1, 136]);
        let values = prepared.contiguous()?.to_vec2::<i32>()?;
        let first_packed = packed_ue8m0(first);
        let second_packed = packed_ue8m0(second);
        assert!(values[..128].iter().all(|row| row[0] == first_packed));
        assert!(values[128..].iter().all(|row| row[0] == second_packed));
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA, SM100, and the DeepGEMM JIT toolchain"]
    fn sm100_prepared_block_scale_gemm_smoke() -> Result<()> {
        let Some(device) = cuda_device_or_skip() else {
            return Ok(());
        };
        init_deepgemm()?;
        if deepgemm::device_info()?.arch()? != Arch::Sm100 {
            println!("Skipping test: prepared block-scale smoke requires SM100");
            return Ok(());
        }

        let m = 64usize;
        let n = 128usize;
        let k = 128usize;
        let one = float8::F8E4M3::from_f32(1.0);
        let a = Tensor::from_vec(vec![one; m * k], (m, k), &device)?;
        let b = Tensor::from_vec(vec![one; n * k], (n, k), &device)?;
        let a_scale = Tensor::ones((m, 1), CandleDType::F32, &device)?;
        let b_scale = Tensor::ones((1, 1), CandleDType::F32, &device)?;
        let prepared_a =
            prepare_fp8_gemm_scale(&a_scale, m, k, 1, 128, DeepGemmDType::PackedUe8M0)?;
        let prepared_b =
            prepare_fp8_gemm_scale(&b_scale, n, k, 128, 128, DeepGemmDType::PackedUe8M0)?;

        let output = fp8_gemm_nt_prepared_scales(&a, &prepared_a, &b, &prepared_b)?;
        let output = output.to_dtype(CandleDType::F32)?.to_vec2::<f32>()?;
        assert!(
            output
                .iter()
                .flatten()
                .all(|value| (*value - k as f32).abs() < 1.0)
        );
        Ok(())
    }
}
