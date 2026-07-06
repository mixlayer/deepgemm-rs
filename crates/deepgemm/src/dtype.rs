/// Tensor element dtype used by DeepGEMM bindings.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DType {
    /// NVIDIA FP8 E4M3 values.
    Fp8E4M3,
    /// Packed FP4 E2M1 values, two logical values per byte.
    PackedFp4E2M1,
    /// Packed UE8M0 scale values stored in `i32` lanes.
    PackedUe8M0,
    /// 32-bit float.
    F32,
    /// bfloat16.
    BF16,
    /// 32-bit signed integer.
    I32,
    /// Raw byte.
    U8,
}

impl DType {
    /// Returns the storage size in bytes for one ABI element.
    pub const fn element_size(self) -> usize {
        match self {
            Self::Fp8E4M3 | Self::PackedFp4E2M1 | Self::U8 => 1,
            Self::BF16 => 2,
            Self::PackedUe8M0 | Self::F32 | Self::I32 => 4,
        }
    }

    pub(crate) const fn to_sys(self) -> deepgemm_sys::deepgemm_dtype_t {
        match self {
            Self::Fp8E4M3 => deepgemm_sys::DEEPGEMM_DTYPE_FP8_E4M3,
            Self::PackedFp4E2M1 => deepgemm_sys::DEEPGEMM_DTYPE_PACKED_FP4_E2M1,
            Self::PackedUe8M0 => deepgemm_sys::DEEPGEMM_DTYPE_PACKED_UE8M0,
            Self::F32 => deepgemm_sys::DEEPGEMM_DTYPE_F32,
            Self::BF16 => deepgemm_sys::DEEPGEMM_DTYPE_BF16,
            Self::I32 => deepgemm_sys::DEEPGEMM_DTYPE_I32,
            Self::U8 => deepgemm_sys::DEEPGEMM_DTYPE_U8,
        }
    }

    pub(crate) fn from_sys(dtype: deepgemm_sys::deepgemm_dtype_t) -> Option<Self> {
        match dtype {
            deepgemm_sys::DEEPGEMM_DTYPE_FP8_E4M3 => Some(Self::Fp8E4M3),
            deepgemm_sys::DEEPGEMM_DTYPE_PACKED_FP4_E2M1 => Some(Self::PackedFp4E2M1),
            deepgemm_sys::DEEPGEMM_DTYPE_PACKED_UE8M0 => Some(Self::PackedUe8M0),
            deepgemm_sys::DEEPGEMM_DTYPE_F32 => Some(Self::F32),
            deepgemm_sys::DEEPGEMM_DTYPE_BF16 => Some(Self::BF16),
            deepgemm_sys::DEEPGEMM_DTYPE_I32 => Some(Self::I32),
            deepgemm_sys::DEEPGEMM_DTYPE_U8 => Some(Self::U8),
            _ => None,
        }
    }
}
