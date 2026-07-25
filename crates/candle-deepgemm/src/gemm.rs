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

/// Reusable caller-owned buffers for dense FP8 `nt` GEMM.
///
/// The workspace keeps architecture-specific transformed scales and the GEMM
/// output alive at stable addresses. This makes repeated launches safe to
/// capture in a CUDA graph. A workspace is tied to one `(m, n, k)` shape and
/// must not be used concurrently from multiple streams.
#[derive(Debug, Clone)]
pub struct Fp8GemmNtWorkspace {
    arch: Arch,
    m: usize,
    n: usize,
    k: usize,
    a_scale_layout: deepgemm::TensorLayout2D,
    a_scale_transformed: Tensor,
    b_scale_launch_spec: TensorSpec<2>,
    b_scale_transformed: Tensor,
    d_layout: deepgemm::TensorLayout2D,
    output: Tensor,
}

impl Fp8GemmNtWorkspace {
    /// Allocates reusable buffers and transforms the static RHS scales.
    ///
    /// `m` is the number of input rows that each launch will process. `b` and
    /// `b_scale` follow the same tensor contract as [`fp8_gemm_nt`].
    pub fn new(m: usize, b: &Tensor, b_scale: &Tensor) -> Result<Self> {
        ensure_rank(b, 2, "b")?;
        ensure_rank(b_scale, 2, "b_scale")?;
        ensure_dtype(b, CandleDType::F8E4M3, "b")?;
        ensure_dtype(b_scale, CandleDType::F32, "b_scale")?;
        ensure_same_device(b, b_scale, "b_scale")?;
        if m == 0 {
            return invalid_arg("m must be positive");
        }

        let n = b.dims()[0];
        let k = b.dims()[1];
        let (stream, device_id) = stream_and_device_id(b)?;
        let device_info = deepgemm::device_info()?;
        if device_info.device != device_id {
            return invalid_arg(format!(
                "b is on CUDA device {device_id}, but DeepGEMM current device is {}",
                device_info.device
            ));
        }
        let arch = device_info.arch()?;
        let b_spec = tensor_spec(b, DeepGemmDType::Fp8E4M3, "b")?;
        let a_spec = TensorSpec::contiguous(DeepGemmDType::Fp8E4M3, [m, k]);
        let a_scale_layout = match arch {
            Arch::Sm90 => deepgemm::fp8_gemm_scale_layout(m, k, 128, DeepGemmDType::F32)?,
            Arch::Sm100 => deepgemm::fp8_gemm_scale_layout(m, k, 128, DeepGemmDType::PackedUe8M0)?,
        };
        let a_scale_transformed = allocate_transformed_scale(a_scale_layout, b.device())?;

        let (b_scale_launch_spec, b_scale_transformed) = match arch {
            Arch::Sm90 => {
                let expected = [ceil_div(n, 128)?, ceil_div(k, 128)?];
                let spec = tensor_spec(b_scale, DeepGemmDType::F32, "b_scale")?;
                require_raw_scale(spec, expected, "b_scale")?;
                (spec, b_scale.clone())
            }
            Arch::Sm100 => {
                let raw_spec = tensor_spec(b_scale, DeepGemmDType::F32, "b_scale")?;
                require_raw_scale(raw_spec, [n, ceil_div(k, 128)?], "b_scale")?;
                let layout =
                    deepgemm::fp8_gemm_scale_layout(n, k, 128, DeepGemmDType::PackedUe8M0)?;
                let transformed = allocate_transformed_scale(layout, b.device())?;
                transform_scale_into(b_scale, raw_spec, &transformed, layout, n, k, &stream)?;
                (layout.logical_spec(), transformed)
            }
        };

        let d_spec = TensorSpec::contiguous(DeepGemmDType::BF16, [m, n]);
        let launch_spec = Fp8GemmNtSpec {
            a: a_spec,
            a_scale: a_scale_layout.logical_spec(),
            b: b_spec,
            b_scale: b_scale_launch_spec,
            d: d_spec,
        };
        let d_layout = deepgemm::fp8_gemm_nt_output_layout(&launch_spec, arch)?;
        let output = Tensor::zeros(
            (d_layout.allocation_shape[0], d_layout.allocation_shape[1]),
            CandleDType::BF16,
            b.device(),
        )?;

        Ok(Self {
            arch,
            m,
            n,
            k,
            a_scale_layout,
            a_scale_transformed,
            b_scale_launch_spec,
            b_scale_transformed,
            d_layout,
            output,
        })
    }

    /// Launches into this workspace's persistent output tensor.
    ///
    /// The returned tensor aliases the workspace output and is overwritten by
    /// the next launch on this workspace.
    pub fn forward(&self, a: &Tensor, a_scale: &Tensor, b: &Tensor) -> Result<Tensor> {
        validate_workspace_inputs(self, a, a_scale, b)?;
        let (stream, _) = stream_and_device_id(a)?;
        let a_scale_raw_spec = tensor_spec(a_scale, DeepGemmDType::F32, "a_scale")?;
        transform_scale_into(
            a_scale,
            a_scale_raw_spec,
            &self.a_scale_transformed,
            self.a_scale_layout,
            self.m,
            self.k,
            &stream,
        )?;

        let a_spec = tensor_spec(a, DeepGemmDType::Fp8E4M3, "a")?;
        let b_spec = tensor_spec(b, DeepGemmDType::Fp8E4M3, "b")?;
        let (a_storage, a_layout) = a.storage_and_layout();
        let a_ptr = tensor_ptr_by_dtype(
            &a_storage,
            CandleDType::F8E4M3,
            a_layout.start_offset(),
            &stream,
            "a",
        )?;
        let (a_scale_storage, a_scale_storage_layout) =
            self.a_scale_transformed.storage_and_layout();
        let a_scale_ptr = tensor_ptr_by_dtype(
            &a_scale_storage,
            candle_dtype_for_deepgemm(self.a_scale_layout.dtype),
            a_scale_storage_layout.start_offset(),
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
        let (b_scale_storage, b_scale_layout) = self.b_scale_transformed.storage_and_layout();
        let b_scale_ptr = tensor_ptr_by_dtype(
            &b_scale_storage,
            match self.arch {
                Arch::Sm90 => CandleDType::F32,
                Arch::Sm100 => CandleDType::I32,
            },
            b_scale_layout.start_offset(),
            &stream,
            "b_scale",
        )?;
        let (d_storage, d_layout) = self.output.storage_and_layout();
        let d_ptr = tensor_ptr_by_dtype(
            &d_storage,
            CandleDType::BF16,
            d_layout.start_offset(),
            &stream,
            "d",
        )?;
        let launch = Fp8GemmNtLaunch {
            a: tensor_arg(a_ptr.as_const_void(), a_spec),
            a_scale: tensor_arg(
                a_scale_ptr.as_const_void(),
                self.a_scale_layout.logical_spec(),
            ),
            b: tensor_arg(b_ptr.as_const_void(), b_spec),
            b_scale: tensor_arg(b_scale_ptr.as_const_void(), self.b_scale_launch_spec),
            d: TensorOut {
                data: d_ptr.as_mut_void(),
                spec: self.d_layout.logical_spec(),
            },
            stream: stream.cu_stream() as *mut std::ffi::c_void,
        };

        // SAFETY: the workspace owns live buffers matching the validated
        // launch specs, all on the launch stream's device.
        unsafe { deepgemm::fp8_gemm_nt(&launch)? };
        Ok(self.output.clone())
    }

    /// Returns the persistent BF16 output tensor.
    pub fn output(&self) -> &Tensor {
        &self.output
    }
}

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
    if b.dims()[1] != k {
        return invalid_arg("b shape must be [n, k] with the same k as a");
    }

    let a_scale_raw_spec = tensor_spec(a_scale, DeepGemmDType::F32, "a_scale")?;
    require_raw_scale(a_scale_raw_spec, [m, ceil_div(k, 128)?], "a_scale")?;
    Fp8GemmNtWorkspace::new(m, b, b_scale)?.forward(a, a_scale, b)
}

fn transform_scale_into(
    scale: &Tensor,
    scale_spec: TensorSpec<2>,
    transformed: &Tensor,
    transformed_layout: deepgemm::TensorLayout2D,
    mn: usize,
    k: usize,
    stream: &Arc<CudaStream>,
) -> Result<()> {
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
    Ok(())
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

fn validate_workspace_inputs(
    workspace: &Fp8GemmNtWorkspace,
    a: &Tensor,
    a_scale: &Tensor,
    b: &Tensor,
) -> Result<()> {
    ensure_same_device(&workspace.output, b, "b")?;
    ensure_same_device(b, a, "a")?;
    ensure_same_device(b, a_scale, "a_scale")?;
    ensure_rank(a, 2, "a")?;
    ensure_rank(a_scale, 2, "a_scale")?;
    ensure_rank(b, 2, "b")?;
    ensure_dtype(a, CandleDType::F8E4M3, "a")?;
    ensure_dtype(a_scale, CandleDType::F32, "a_scale")?;
    ensure_dtype(b, CandleDType::F8E4M3, "b")?;
    if a.dims() != [workspace.m, workspace.k] {
        return invalid_arg(format!(
            "a shape must be [{}, {}], got {:?}",
            workspace.m,
            workspace.k,
            a.dims()
        ));
    }
    if a_scale.dims() != [workspace.m, ceil_div(workspace.k, 128)?] {
        return invalid_arg(format!(
            "a_scale shape must be [{}, {}], got {:?}",
            workspace.m,
            ceil_div(workspace.k, 128)?,
            a_scale.dims()
        ));
    }
    if b.dims() != [workspace.n, workspace.k] {
        return invalid_arg(format!(
            "b shape must be [{}, {}], got {:?}",
            workspace.n,
            workspace.k,
            b.dims()
        ));
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
