use core::ffi::c_void;

use crate::{DType, Error, Result};

/// Shape, stride, and dtype for a tensor argument.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TensorSpec<const RANK: usize> {
    /// Tensor dtype.
    pub dtype: DType,
    /// Tensor shape in elements.
    pub shape: [usize; RANK],
    /// Tensor strides in elements.
    pub strides: [isize; RANK],
}

impl<const RANK: usize> TensorSpec<RANK> {
    /// Creates a row-major contiguous tensor spec.
    pub fn contiguous(dtype: DType, shape: [usize; RANK]) -> Self {
        let mut strides = [0; RANK];
        let mut stride = 1isize;
        let mut index = RANK;
        while index > 0 {
            index -= 1;
            strides[index] = stride;
            stride = stride
                .checked_mul(shape[index] as isize)
                .expect("contiguous stride overflowed isize");
        }
        Self {
            dtype,
            shape,
            strides,
        }
    }

    /// Returns true when the tensor is row-major contiguous.
    pub fn is_contiguous(&self) -> bool {
        *self == Self::contiguous(self.dtype, self.shape)
    }

    pub(crate) fn to_raw_tensor(
        self,
        data: *const c_void,
    ) -> Result<deepgemm_sys::deepgemm_tensor_t> {
        let mut shape = [0i64; 4];
        let mut stride = [0i64; 4];
        for index in 0..RANK {
            shape[index] = usize_to_i64(self.shape[index], "shape")?;
            stride[index] = isize_to_i64(self.strides[index], "stride")?;
        }

        Ok(deepgemm_sys::deepgemm_tensor_t {
            data,
            dtype: self.dtype.to_sys(),
            rank: RANK as u32,
            shape,
            stride,
        })
    }

    pub(crate) fn to_raw_tensor_mut(
        self,
        data: *mut c_void,
    ) -> Result<deepgemm_sys::deepgemm_tensor_mut_t> {
        let mut shape = [0i64; 4];
        let mut stride = [0i64; 4];
        for index in 0..RANK {
            shape[index] = usize_to_i64(self.shape[index], "shape")?;
            stride[index] = isize_to_i64(self.strides[index], "stride")?;
        }

        Ok(deepgemm_sys::deepgemm_tensor_mut_t {
            data,
            dtype: self.dtype.to_sys(),
            rank: RANK as u32,
            shape,
            stride,
        })
    }
}

/// Immutable tensor pointer plus explicit shape, stride, and dtype metadata.
#[derive(Debug, Copy, Clone)]
pub struct TensorArg<const RANK: usize> {
    /// Device pointer.
    pub data: *const c_void,
    /// Shape, stride, and dtype metadata.
    pub spec: TensorSpec<RANK>,
}

impl<const RANK: usize> TensorArg<RANK> {
    pub(crate) fn to_raw(self) -> Result<deepgemm_sys::deepgemm_tensor_t> {
        self.spec.to_raw_tensor(self.data)
    }
}

/// Mutable tensor pointer plus explicit shape, stride, and dtype metadata.
#[derive(Debug, Copy, Clone)]
pub struct TensorOut<const RANK: usize> {
    /// Device pointer.
    pub data: *mut c_void,
    /// Shape, stride, and dtype metadata.
    pub spec: TensorSpec<RANK>,
}

impl<const RANK: usize> TensorOut<RANK> {
    pub(crate) fn to_raw(self) -> Result<deepgemm_sys::deepgemm_tensor_mut_t> {
        self.spec.to_raw_tensor_mut(self.data)
    }
}

/// Required layout for a 2D output tensor.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TensorLayout2D {
    /// Tensor dtype.
    pub dtype: DType,
    /// Logical shape returned to users.
    pub logical_shape: [usize; 2],
    /// Physical allocation shape required by the kernel.
    pub allocation_shape: [usize; 2],
    /// Tensor strides in elements.
    pub strides: [isize; 2],
    /// Number of elements to allocate.
    pub element_count: usize,
}

impl TensorLayout2D {
    /// Returns the tensor spec for the logical output view.
    pub fn logical_spec(self) -> TensorSpec<2> {
        TensorSpec {
            dtype: self.dtype,
            shape: self.logical_shape,
            strides: self.strides,
        }
    }

    /// Returns the tensor spec for the physical allocation.
    pub fn allocation_spec(self) -> TensorSpec<2> {
        TensorSpec {
            dtype: self.dtype,
            shape: self.allocation_shape,
            strides: self.strides,
        }
    }
}

pub(crate) fn require_dtype<const RANK: usize>(
    tensor: &TensorSpec<RANK>,
    expected: DType,
    name: &str,
) -> Result<()> {
    if tensor.dtype != expected {
        return Err(Error::InvalidArgument(format!(
            "{name} dtype must be {expected:?}, got {:?}",
            tensor.dtype
        )));
    }
    Ok(())
}

pub(crate) fn require_contiguous<const RANK: usize>(
    tensor: &TensorSpec<RANK>,
    name: &str,
) -> Result<()> {
    if !tensor.is_contiguous() {
        return Err(Error::InvalidArgument(format!("{name} must be contiguous")));
    }
    Ok(())
}

pub(crate) fn usize_to_i64(value: usize, name: &str) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| Error::InvalidArgument(format!("{name} value {value} does not fit in i64")))
}

fn isize_to_i64(value: isize, name: &str) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| Error::InvalidArgument(format!("{name} value {value} does not fit in i64")))
}

pub(crate) fn i64_to_usize(value: i64, name: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidArgument(format!("{name} value {value} does not fit in usize")))
}

pub(crate) fn i64_to_isize(value: i64, name: &str) -> Result<isize> {
    isize::try_from(value)
        .map_err(|_| Error::InvalidArgument(format!("{name} value {value} does not fit in isize")))
}
