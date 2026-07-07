use crate::{
    Arch, DType, Error, Result, TensorArg, TensorLayout2D, TensorOut, TensorSpec,
    tensor::{i64_to_isize, i64_to_usize, require_contiguous, require_dtype, usize_to_i64},
};

/// Shape and dtype contract for transforming FP8 GEMM scales.
///
/// Tensor contract:
/// - `scale`: `F32 [mn, ceil(k / gran_k)]`, row-major contiguous.
/// - `transformed`: either `F32 [mn, ceil(k / gran_k)]` or packed UE8M0
///   `[mn, ceil(k / (gran_k * 4))]`, with strides `[1, aligned_mn]`.
///   `aligned_mn` is the TMA alignment of `mn` for the transformed dtype.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Fp8GemmScaleTransformSpec {
    /// Raw FP32 scale tensor, `F32 [mn, ceil(k / gran_k)]`.
    pub scale: TensorSpec<2>,
    /// Transformed scale tensor consumed by GEMM kernels.
    pub transformed: TensorSpec<2>,
    /// Number of logical rows in the scale tensor.
    pub mn: usize,
    /// GEMM K dimension before scale blocking.
    pub k: usize,
    /// K-block granularity represented by one raw FP32 scale.
    pub gran_k: usize,
}

/// Shape and dtype contract for dense FP8 `nt` GEMM.
///
/// Tensor contract:
/// - `a`: `FP8_E4M3 [m, k]`, row-major contiguous.
/// - `b`: `FP8_E4M3 [n, k]`, row-major contiguous. This is the logical
///   transposed RHS; the operation computes `a @ b.T`.
/// - `d`: `BF16 [m, n]`, row-major contiguous.
///
/// Scale contract:
/// - SM90:
///   - `a_scale`: transformed `F32 [m, ceil(k / 128)]`, strides
///     `[1, aligned_m]`.
///   - `b_scale`: raw block scale `F32 [ceil(n / 128), ceil(k / 128)]`,
///     row-major contiguous.
/// - SM100:
///   - `a_scale`: transformed packed UE8M0 `[m, ceil(k / 512)]`, strides
///     `[1, aligned_m]`.
///   - `b_scale`: transformed packed UE8M0 `[n, ceil(k / 512)]`, strides
///     `[1, aligned_n]`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Fp8GemmNtSpec {
    /// LHS tensor: `FP8_E4M3 [m, k]`.
    pub a: TensorSpec<2>,
    /// LHS scale tensor, architecture-specific as described in the type docs.
    pub a_scale: TensorSpec<2>,
    /// RHS tensor: `FP8_E4M3 [n, k]`, interpreted as `b.T`.
    pub b: TensorSpec<2>,
    /// RHS scale tensor, architecture-specific as described in the type docs.
    pub b_scale: TensorSpec<2>,
    /// Output tensor: `BF16 [m, n]`.
    pub d: TensorSpec<2>,
}

/// Raw launch arguments for transforming FP8 GEMM scales.
#[derive(Debug, Copy, Clone)]
pub struct Fp8GemmScaleTransformLaunch {
    /// Raw FP32 scale tensor: `F32 [mn, ceil(k / gran_k)]`.
    pub scale: TensorArg<2>,
    /// Transformed scale tensor. Dtype and dims are defined by
    /// `fp8_gemm_scale_layout(mn, k, gran_k, transformed_dtype)`.
    pub transformed: TensorOut<2>,
    /// Number of logical rows in `scale`.
    pub mn: usize,
    /// GEMM K dimension before scale blocking.
    pub k: usize,
    /// K-block granularity represented by one raw FP32 scale.
    pub gran_k: usize,
    /// CUDA stream for the launch.
    pub stream: deepgemm_sys::deepgemm_cuda_stream_t,
}

impl Fp8GemmScaleTransformLaunch {
    /// Returns the shape and dtype contract for this launch.
    pub fn spec(&self) -> Fp8GemmScaleTransformSpec {
        Fp8GemmScaleTransformSpec {
            scale: self.scale.spec,
            transformed: self.transformed.spec,
            mn: self.mn,
            k: self.k,
            gran_k: self.gran_k,
        }
    }
}

/// Raw launch arguments for dense FP8 `nt` GEMM.
#[derive(Debug, Copy, Clone)]
pub struct Fp8GemmNtLaunch {
    /// LHS tensor: `FP8_E4M3 [m, k]`.
    pub a: TensorArg<2>,
    /// LHS scale tensor, architecture-specific as described in `Fp8GemmNtSpec`.
    pub a_scale: TensorArg<2>,
    /// RHS tensor: `FP8_E4M3 [n, k]`, interpreted as `b.T`.
    pub b: TensorArg<2>,
    /// RHS scale tensor, architecture-specific as described in `Fp8GemmNtSpec`.
    pub b_scale: TensorArg<2>,
    /// Output tensor: `BF16 [m, n]`.
    pub d: TensorOut<2>,
    /// CUDA stream for the launch.
    pub stream: deepgemm_sys::deepgemm_cuda_stream_t,
}

impl Fp8GemmNtLaunch {
    /// Returns the shape and dtype contract for this launch.
    pub fn spec(&self) -> Fp8GemmNtSpec {
        Fp8GemmNtSpec {
            a: self.a.spec,
            a_scale: self.a_scale.spec,
            b: self.b.spec,
            b_scale: self.b_scale.spec,
            d: self.d.spec,
        }
    }
}

/// Computes the required dense FP8 `nt` GEMM output layout.
///
/// The returned logical tensor is `BF16 [m, n]` with row-major strides `[n, 1]`.
pub fn fp8_gemm_nt_output_layout(spec: &Fp8GemmNtSpec, arch: Arch) -> Result<TensorLayout2D> {
    let dims = validate_fp8_gemm_nt_spec(spec, arch)?;
    let params = deepgemm_sys::deepgemm_fp8_gemm_nt_output_layout_params_t {
        m: usize_to_i64(dims.m, "m")?,
        n: usize_to_i64(dims.n, "n")?,
        output_dtype: spec.d.dtype.to_sys(),
    };
    call_layout(|out| {
        // SAFETY: `params` and `out` are valid for this call.
        unsafe { deepgemm_sys::deepgemm_fp8_gemm_nt_output_layout(&params, out) }
    })
}

/// Computes the transformed scale layout used by FP8 GEMM kernels.
///
/// Returned logical dims and dtype:
/// - `F32`: `[mn, ceil(k / gran_k)]`.
/// - `PackedUe8M0`: `[mn, ceil(k / (gran_k * 4))]`.
///
/// Returned strides are `[1, aligned_mn]`; allocate the returned
/// `allocation_shape` and view it with the returned logical shape/strides.
pub fn fp8_gemm_scale_layout(
    mn: usize,
    k: usize,
    gran_k: usize,
    scale_dtype: DType,
) -> Result<TensorLayout2D> {
    match scale_dtype {
        DType::F32 | DType::PackedUe8M0 => {}
        _ => {
            return Err(Error::InvalidArgument(
                "scale dtype must be f32 or packed UE8M0".into(),
            ));
        }
    }
    let params = deepgemm_sys::deepgemm_fp8_gemm_scale_layout_params_t {
        mn: usize_to_i64(mn, "mn")?,
        k: usize_to_i64(k, "k")?,
        gran_k: usize_to_i64(gran_k, "gran_k")?,
        scale_dtype: scale_dtype.to_sys(),
    };
    call_layout(|out| {
        // SAFETY: `params` and `out` are valid for this call.
        unsafe { deepgemm_sys::deepgemm_fp8_gemm_scale_layout(&params, out) }
    })
}

/// Launches the FP8 GEMM scale transform through the raw DeepGEMM C ABI.
///
/// # Safety
///
/// All pointers must refer to valid CUDA device buffers matching the attached
/// shape and dtype metadata. Buffers must remain live until work enqueued on
/// `stream` has completed.
pub unsafe fn fp8_gemm_transform_scale(params: &Fp8GemmScaleTransformLaunch) -> Result<()> {
    validate_fp8_gemm_scale_transform_spec(&params.spec())?;
    let raw = deepgemm_sys::deepgemm_fp8_gemm_scale_transform_params_t {
        scale: params.scale.to_raw()?,
        transformed: params.transformed.to_raw()?,
        mn: usize_to_i64(params.mn, "mn")?,
        k: usize_to_i64(params.k, "k")?,
        gran_k: usize_to_i64(params.gran_k, "gran_k")?,
        stream: params.stream,
    };
    // SAFETY: the caller upholds pointer and stream validity; `raw` is valid for this call.
    let status = unsafe { deepgemm_sys::deepgemm_fp8_gemm_transform_scale(&raw) };
    Error::check_raw_status(status)
}

/// Launches dense FP8 `nt` GEMM through the raw DeepGEMM C ABI.
///
/// Computes `d = a @ b.T` with `a: FP8_E4M3 [m, k]`,
/// `b: FP8_E4M3 [n, k]`, and `d: BF16 [m, n]`.
///
/// # Safety
///
/// All pointers must refer to valid CUDA device buffers matching the attached
/// shape and dtype metadata. Buffers must remain live until work enqueued on
/// `stream` has completed.
pub unsafe fn fp8_gemm_nt(params: &Fp8GemmNtLaunch) -> Result<()> {
    let arch = crate::runtime::device_info()?.arch()?;
    let expected_d = fp8_gemm_nt_output_layout(&params.spec(), arch)?.logical_spec();
    require_spec(params.d.spec, expected_d, "d")?;

    let raw = deepgemm_sys::deepgemm_fp8_gemm_nt_params_t {
        a: params.a.to_raw()?,
        a_scale: params.a_scale.to_raw()?,
        b: params.b.to_raw()?,
        b_scale: params.b_scale.to_raw()?,
        d: params.d.to_raw()?,
        stream: params.stream,
    };
    // SAFETY: the caller upholds pointer and stream validity; `raw` is valid for this call.
    let status = unsafe { deepgemm_sys::deepgemm_fp8_gemm_nt(&raw) };
    Error::check_raw_status(status)
}

#[derive(Debug, Copy, Clone)]
struct GemmDims {
    m: usize,
    n: usize,
}

fn validate_fp8_gemm_scale_transform_spec(spec: &Fp8GemmScaleTransformSpec) -> Result<()> {
    require_positive(spec.mn, "mn")?;
    require_positive(spec.k, "k")?;
    require_positive(spec.gran_k, "gran_k")?;
    require_contiguous(&spec.scale, "scale")?;
    require_dtype(&spec.scale, DType::F32, "scale")?;
    require_shape(
        spec.scale.shape,
        [spec.mn, ceil_div(spec.k, spec.gran_k)?],
        "scale",
    )?;
    let expected =
        fp8_gemm_scale_layout(spec.mn, spec.k, spec.gran_k, spec.transformed.dtype)?.logical_spec();
    require_spec(spec.transformed, expected, "transformed")
}

fn validate_fp8_gemm_nt_spec(spec: &Fp8GemmNtSpec, arch: Arch) -> Result<GemmDims> {
    require_contiguous(&spec.a, "a")?;
    require_contiguous(&spec.b, "b")?;
    require_contiguous(&spec.d, "d")?;
    require_dtype(&spec.a, DType::Fp8E4M3, "a")?;
    require_dtype(&spec.b, DType::Fp8E4M3, "b")?;
    require_dtype(&spec.d, DType::BF16, "d")?;

    let m = spec.a.shape[0];
    let k = spec.a.shape[1];
    let n = spec.b.shape[0];
    require_positive(m, "m")?;
    require_positive(n, "n")?;
    require_positive(k, "k")?;
    if spec.b.shape[1] != k {
        return Err(Error::InvalidArgument(
            "b shape must be [n, k] with the same k as a".into(),
        ));
    }
    require_shape(spec.d.shape, [m, n], "d")?;
    if n % 8 != 0 {
        return Err(Error::InvalidArgument("n must be a multiple of 8".into()));
    }

    match arch {
        Arch::Sm90 => {
            let expected_a_scale = fp8_gemm_scale_layout(m, k, 128, DType::F32)?.logical_spec();
            require_spec(spec.a_scale, expected_a_scale, "a_scale")?;
            require_contiguous(&spec.b_scale, "b_scale")?;
            require_dtype(&spec.b_scale, DType::F32, "b_scale")?;
            require_shape(
                spec.b_scale.shape,
                [ceil_div(n, 128)?, ceil_div(k, 128)?],
                "b_scale",
            )?;
        }
        Arch::Sm100 => {
            let expected_a_scale =
                fp8_gemm_scale_layout(m, k, 128, DType::PackedUe8M0)?.logical_spec();
            let expected_b_scale =
                fp8_gemm_scale_layout(n, k, 128, DType::PackedUe8M0)?.logical_spec();
            require_spec(spec.a_scale, expected_a_scale, "a_scale")?;
            require_spec(spec.b_scale, expected_b_scale, "b_scale")?;
        }
    }

    Ok(GemmDims { m, n })
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
    layout_from_sys(raw)
}

fn layout_from_sys(raw: deepgemm_sys::deepgemm_tensor_layout_2d_t) -> Result<TensorLayout2D> {
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

fn ceil_div(value: usize, divisor: usize) -> Result<usize> {
    if divisor == 0 {
        return Err(Error::InvalidArgument("divisor must be positive".into()));
    }
    value
        .checked_add(divisor - 1)
        .map(|value| value / divisor)
        .ok_or_else(|| Error::InvalidArgument("ceil_div overflowed".into()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_layout_f32_is_mn_major_and_tma_aligned() {
        let layout = fp8_gemm_scale_layout(4097, 7168, 128, DType::F32).unwrap();
        assert_eq!(layout.dtype, DType::F32);
        assert_eq!(layout.logical_shape, [4097, 56]);
        assert_eq!(layout.allocation_shape, [56, 4100]);
        assert_eq!(layout.strides, [1, 4100]);
        assert_eq!(layout.element_count, 56 * 4100);
    }

    #[test]
    fn scale_layout_packed_ue8m0_packs_four_k_scales_per_i32() {
        let layout = fp8_gemm_scale_layout(4097, 7168, 128, DType::PackedUe8M0).unwrap();
        assert_eq!(layout.dtype, DType::PackedUe8M0);
        assert_eq!(layout.logical_shape, [4097, 14]);
        assert_eq!(layout.allocation_shape, [14, 4100]);
        assert_eq!(layout.strides, [1, 4100]);
        assert_eq!(layout.element_count, 14 * 4100);
    }

    #[test]
    fn sm90_spec_accepts_transformed_a_scale_and_block_b_scale() {
        let spec = Fp8GemmNtSpec {
            a: TensorSpec::contiguous(DType::Fp8E4M3, [128, 7168]),
            a_scale: fp8_gemm_scale_layout(128, 7168, 128, DType::F32)
                .unwrap()
                .logical_spec(),
            b: TensorSpec::contiguous(DType::Fp8E4M3, [2112, 7168]),
            b_scale: TensorSpec::contiguous(DType::F32, [17, 56]),
            d: TensorSpec::contiguous(DType::BF16, [128, 2112]),
        };
        assert!(validate_fp8_gemm_nt_spec(&spec, Arch::Sm90).is_ok());
    }

    #[test]
    fn sm90_spec_rejects_per_row_b_scale() {
        let spec = Fp8GemmNtSpec {
            a: TensorSpec::contiguous(DType::Fp8E4M3, [128, 7168]),
            a_scale: fp8_gemm_scale_layout(128, 7168, 128, DType::F32)
                .unwrap()
                .logical_spec(),
            b: TensorSpec::contiguous(DType::Fp8E4M3, [2112, 7168]),
            b_scale: TensorSpec::contiguous(DType::F32, [2112, 56]),
            d: TensorSpec::contiguous(DType::BF16, [128, 2112]),
        };
        assert!(matches!(
            validate_fp8_gemm_nt_spec(&spec, Arch::Sm90),
            Err(Error::InvalidArgument(_))
        ));
    }

    #[test]
    fn sm100_spec_accepts_packed_transformed_scales() {
        let spec = Fp8GemmNtSpec {
            a: TensorSpec::contiguous(DType::Fp8E4M3, [128, 7168]),
            a_scale: fp8_gemm_scale_layout(128, 7168, 128, DType::PackedUe8M0)
                .unwrap()
                .logical_spec(),
            b: TensorSpec::contiguous(DType::Fp8E4M3, [2112, 7168]),
            b_scale: fp8_gemm_scale_layout(2112, 7168, 128, DType::PackedUe8M0)
                .unwrap()
                .logical_spec(),
            d: TensorSpec::contiguous(DType::BF16, [128, 2112]),
        };
        assert!(validate_fp8_gemm_nt_spec(&spec, Arch::Sm100).is_ok());
    }
}
