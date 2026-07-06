pub(crate) mod cuda {
    use std::{ffi::c_void, sync::Arc};

    use candle::{
        DType, Storage, Tensor,
        cuda::{
            CudaDType,
            cudarc::driver::{CudaStream, DevicePtr, DeviceRepr, SyncOnDrop},
        },
    };

    use crate::{
        Result,
        error::{Error, invalid_arg},
    };

    pub(crate) fn ensure_rank(t: &Tensor, rank: usize, name: &str) -> Result<()> {
        if t.rank() == rank {
            Ok(())
        } else {
            invalid_arg(format!("{name} must have rank {rank}, got {}", t.rank()))
        }
    }

    pub(crate) fn ensure_same_device(reference: &Tensor, t: &Tensor, name: &str) -> Result<()> {
        if reference.device().same_device(t.device()) {
            Ok(())
        } else {
            invalid_arg(format!("{name} must be on the same device as q"))
        }
    }

    pub(crate) fn ensure_dtype(t: &Tensor, dtype: DType, name: &str) -> Result<()> {
        if t.dtype() == dtype {
            Ok(())
        } else {
            invalid_arg(format!(
                "{name} must have dtype {dtype:?}, got {:?}",
                t.dtype()
            ))
        }
    }

    pub(crate) fn stream_and_device_id(t: &Tensor) -> Result<(Arc<CudaStream>, i32)> {
        let cuda_dev = t.device().as_cuda_device()?;
        let stream = cuda_dev.cuda_stream();
        let device_id = i32::try_from(stream.context().ordinal())
            .map_err(|_| Error::Tensor("device id overflow".to_string()))?;
        Ok((stream, device_id))
    }

    pub(crate) struct TensorPtr<'a> {
        ptr: *const c_void,
        _guard: SyncOnDrop<'a>,
    }

    impl TensorPtr<'_> {
        pub(crate) fn as_const_void(&self) -> *const c_void {
            self.ptr
        }

        pub(crate) fn as_mut_void(&self) -> *mut c_void {
            self.ptr as *mut c_void
        }
    }

    pub(crate) fn tensor_ptr_by_dtype<'a>(
        storage: &'a Storage,
        dtype: DType,
        start_offset: usize,
        stream: &'a Arc<CudaStream>,
        name: &str,
    ) -> Result<TensorPtr<'a>> {
        match dtype {
            DType::BF16 => tensor_ptr_typed::<half::bf16>(storage, start_offset, stream, name),
            DType::F32 => tensor_ptr_typed::<f32>(storage, start_offset, stream, name),
            DType::I32 => tensor_ptr_typed::<i32>(storage, start_offset, stream, name),
            DType::U8 => tensor_ptr_typed::<u8>(storage, start_offset, stream, name),
            DType::F8E4M3 => {
                tensor_ptr_typed::<float8::F8E4M3>(storage, start_offset, stream, name)
            }
            dtype => invalid_arg(format!("{name} has unsupported dtype {dtype:?}")),
        }
    }

    fn tensor_ptr_typed<'a, T: CudaDType + DeviceRepr + 'a>(
        storage: &'a Storage,
        start_offset: usize,
        stream: &'a Arc<CudaStream>,
        name: &str,
    ) -> Result<TensorPtr<'a>> {
        let (ptr, guard) = match storage {
            Storage::Cuda(storage) => storage.as_cuda_slice::<T>()?.device_ptr(stream),
            _ => return invalid_arg(format!("{name} must be a CUDA tensor")),
        };
        let offset_bytes = start_offset
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::Tensor(format!("{name} start_offset overflow")))?;
        let ptr = ptr
            .checked_add(offset_bytes as u64)
            .ok_or_else(|| Error::Tensor(format!("{name} pointer overflow")))?;
        Ok(TensorPtr {
            ptr: ptr as usize as *const c_void,
            _guard: guard,
        })
    }
}
