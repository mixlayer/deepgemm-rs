//! Candle CUDA integration for DeepGEMM BF16 and SM90 FP8 Mega MoE.

use candle::{
    DType as CandleDType, Tensor,
    cuda::cudarc::driver::{
        CudaStream,
        sys::{self, CUdeviceptr, CUipcMem_flags_enum, CUipcMemHandle_st},
    },
};
use deepgemm::{
    Bf16MegaMoeLaunch, Bf16MegaMoeSpec, DType as DeepGemmDType, Sm90Fp8MegaMoeLaunch,
    Sm90Fp8MegaMoeSpec, TensorArg, TensorOut, TensorSpec,
};
use std::{ffi::c_char, sync::Arc};

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

/// Number of opaque bytes in a CUDA IPC memory handle.
pub const SM90_FP8_MEGA_MOE_IPC_HANDLE_BYTES: usize = 64;

/// Serializable CUDA IPC handle for one rank's SM90 FP8 Mega MoE allocation.
pub type Sm90Fp8MegaMoeIpcHandle = [u8; SM90_FP8_MEGA_MOE_IPC_HANDLE_BYTES];

#[derive(Debug)]
enum Sm90Fp8MegaMoeBuffer {
    SingleRank(Tensor),
    Distributed(Sm90Fp8MegaMoeIpcBuffer),
}

/// A reusable local or CUDA-IPC symmetric allocation for Hopper FP8 Mega MoE.
#[derive(Debug)]
pub struct Sm90Fp8MegaMoeWorkspace {
    spec: Sm90Fp8MegaMoeSpec,
    buffer: Sm90Fp8MegaMoeBuffer,
}

/// An allocated SM90 FP8 buffer waiting for peer CUDA IPC handles.
///
/// Every rank must allocate one pending workspace, exchange [`Self::ipc_handle`]
/// in rank order, and call [`Self::connect`] with the complete handle table.
#[derive(Debug)]
pub struct PendingSm90Fp8MegaMoeWorkspace {
    spec: Sm90Fp8MegaMoeSpec,
    rank_idx: usize,
    world_size: usize,
    local_ptr: CUdeviceptr,
    ipc_handle: Sm90Fp8MegaMoeIpcHandle,
    stream: Arc<CudaStream>,
}

#[derive(Debug)]
struct Sm90Fp8MegaMoeIpcBuffer {
    rank_idx: usize,
    local_ptr: CUdeviceptr,
    imported_peer_ptrs: Vec<CUdeviceptr>,
    peer_ptrs: Vec<u64>,
    stream: Arc<CudaStream>,
}

impl Sm90Fp8MegaMoeWorkspace {
    /// Allocates a zero-initialized single-rank SM90 FP8 symmetric buffer.
    pub fn new_single_rank(spec: Sm90Fp8MegaMoeSpec, device: &candle::Device) -> Result<Self> {
        validate_sm90_fp8_workspace_spec(&spec)?;
        let buffer = Tensor::zeros((spec.buffer_layout.total_bytes,), CandleDType::U8, device)?;
        Ok(Self {
            spec,
            buffer: Sm90Fp8MegaMoeBuffer::SingleRank(buffer),
        })
    }

    /// Allocates this rank's CUDA-IPC-exportable symmetric buffer.
    ///
    /// `rank_idx` is this process's rank in a `world_size` expert-parallel
    /// group. The allocation uses `cuMemAlloc` because CUDA IPC handles cannot
    /// be exported from cudarc's stream-ordered memory pool.
    pub fn begin_distributed(
        spec: Sm90Fp8MegaMoeSpec,
        rank_idx: usize,
        world_size: usize,
        device: &candle::Device,
    ) -> Result<PendingSm90Fp8MegaMoeWorkspace> {
        validate_sm90_fp8_workspace_spec(&spec)?;
        if world_size <= 1 || rank_idx >= world_size || spec.num_experts % world_size != 0 {
            return invalid_arg(format!(
                "distributed SM90 FP8 Mega MoE requires world_size > 1, rank_idx < world_size, and experts divisible by world_size; got experts={}, rank_idx={rank_idx}, world_size={world_size}",
                spec.num_experts
            ));
        }

        let stream = device.as_cuda_device()?.cuda_stream();
        stream
            .context()
            .bind_to_thread()
            .map_err(cuda_driver_error("bind CUDA context"))?;

        let mut local_ptr: CUdeviceptr = 0;
        check_cuda(
            // SAFETY: `local_ptr` is writable and the requested allocation size was validated.
            unsafe { sys::cuMemAlloc_v2(&mut local_ptr, spec.buffer_layout.total_bytes) },
            "allocate CUDA-IPC SM90 FP8 Mega MoE buffer",
        )?;
        let mut pending = PendingSm90Fp8MegaMoeWorkspace {
            spec,
            rank_idx,
            world_size,
            local_ptr,
            ipc_handle: [0; SM90_FP8_MEGA_MOE_IPC_HANDLE_BYTES],
            stream,
        };
        check_cuda(
            // SAFETY: `local_ptr` owns at least `total_bytes` writable device bytes.
            unsafe {
                sys::cuMemsetD8Async(
                    pending.local_ptr,
                    0,
                    pending.spec.buffer_layout.total_bytes,
                    pending.stream.cu_stream(),
                )
            },
            "zero CUDA-IPC SM90 FP8 Mega MoE buffer",
        )?;
        pending
            .stream
            .synchronize()
            .map_err(cuda_driver_error("synchronize initialized CUDA-IPC buffer"))?;

        let mut handle = CUipcMemHandle_st {
            reserved: [0 as c_char; SM90_FP8_MEGA_MOE_IPC_HANDLE_BYTES],
        };
        check_cuda(
            // SAFETY: `handle` is writable and `local_ptr` is a live `cuMemAlloc` allocation.
            unsafe { sys::cuIpcGetMemHandle(&mut handle, pending.local_ptr) },
            "export SM90 FP8 Mega MoE CUDA IPC handle",
        )?;
        pending.ipc_handle = ipc_handle_to_bytes(handle);
        Ok(pending)
    }

    /// Returns the immutable launch specification associated with this allocation.
    pub fn spec(&self) -> &Sm90Fp8MegaMoeSpec {
        &self.spec
    }

    /// Returns the backing symmetric allocation size in bytes.
    pub fn allocation_bytes(&self) -> usize {
        self.spec.buffer_layout.total_bytes
    }

    /// Returns the expert-parallel rank represented by this workspace.
    pub fn rank_idx(&self) -> usize {
        match &self.buffer {
            Sm90Fp8MegaMoeBuffer::SingleRank(_) => 0,
            Sm90Fp8MegaMoeBuffer::Distributed(buffer) => buffer.rank_idx,
        }
    }

    /// Returns the number of peer-visible symmetric allocations.
    pub fn world_size(&self) -> usize {
        match &self.buffer {
            Sm90Fp8MegaMoeBuffer::SingleRank(_) => 1,
            Sm90Fp8MegaMoeBuffer::Distributed(buffer) => buffer.peer_ptrs.len(),
        }
    }
}

impl PendingSm90Fp8MegaMoeWorkspace {
    /// Returns this rank's opaque CUDA IPC handle for all-gather.
    pub fn ipc_handle(&self) -> Sm90Fp8MegaMoeIpcHandle {
        self.ipc_handle
    }

    /// Opens every remote rank's allocation and completes the workspace.
    ///
    /// `peer_handles` must contain exactly one handle per rank in communicator
    /// rank order, including this rank's own handle at `rank_idx`.
    pub fn connect(
        mut self,
        peer_handles: &[Sm90Fp8MegaMoeIpcHandle],
    ) -> Result<Sm90Fp8MegaMoeWorkspace> {
        if peer_handles.len() != self.world_size {
            return invalid_arg(format!(
                "SM90 FP8 Mega MoE received {} CUDA IPC handles for world_size {}",
                peer_handles.len(),
                self.world_size
            ));
        }
        if peer_handles[self.rank_idx] != self.ipc_handle {
            return invalid_arg("SM90 FP8 Mega MoE CUDA IPC handles are not in rank order");
        }
        self.stream
            .context()
            .bind_to_thread()
            .map_err(cuda_driver_error("bind CUDA context for IPC peer import"))?;

        let mut imported_peer_ptrs = Vec::with_capacity(self.world_size - 1);
        let mut peer_ptrs = Vec::with_capacity(self.world_size);
        for (peer_rank, peer_handle) in peer_handles.iter().enumerate() {
            if peer_rank == self.rank_idx {
                peer_ptrs.push(self.local_ptr as u64);
                continue;
            }
            let mut peer_ptr: CUdeviceptr = 0;
            let handle = ipc_handle_from_bytes(*peer_handle);
            let status = unsafe {
                // SAFETY: `peer_ptr` is writable and `handle` came from a live peer allocation.
                sys::cuIpcOpenMemHandle_v2(
                    &mut peer_ptr,
                    handle,
                    CUipcMem_flags_enum::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS as u32,
                )
            };
            if let Err(error) = check_cuda(status, "open SM90 FP8 Mega MoE CUDA IPC peer") {
                close_imported_peers(&imported_peer_ptrs);
                return Err(error);
            }
            imported_peer_ptrs.push(peer_ptr);
            peer_ptrs.push(peer_ptr as u64);
        }

        let local_ptr = self.local_ptr;
        self.local_ptr = 0;
        Ok(Sm90Fp8MegaMoeWorkspace {
            spec: self.spec,
            buffer: Sm90Fp8MegaMoeBuffer::Distributed(Sm90Fp8MegaMoeIpcBuffer {
                rank_idx: self.rank_idx,
                local_ptr,
                imported_peer_ptrs,
                peer_ptrs,
                stream: self.stream.clone(),
            }),
        })
    }
}

impl Drop for PendingSm90Fp8MegaMoeWorkspace {
    fn drop(&mut self) {
        if self.local_ptr == 0 {
            return;
        }
        let _ = self.stream.context().bind_to_thread();
        // SAFETY: this pending allocation was created by `cuMemAlloc_v2` and was not transferred.
        unsafe {
            let _ = sys::cuMemFree_v2(self.local_ptr);
        }
    }
}

impl Drop for Sm90Fp8MegaMoeIpcBuffer {
    fn drop(&mut self) {
        let _ = self.stream.context().bind_to_thread();
        let _ = self.stream.context().synchronize();
        close_imported_peers(&self.imported_peer_ptrs);
        if self.local_ptr != 0 {
            // SAFETY: the local allocation was created by `cuMemAlloc_v2` and is owned here.
            unsafe {
                let _ = sys::cuMemFree_v2(self.local_ptr);
            }
        }
    }
}

fn validate_sm90_fp8_workspace_spec(spec: &Sm90Fp8MegaMoeSpec) -> Result<()> {
    if spec.num_experts == 0
        || spec.num_topk == 0
        || spec.num_max_tokens_per_rank == 0
        || spec.buffer_layout.total_bytes == 0
    {
        return invalid_arg("SM90 FP8 Mega MoE workspace dimensions must be positive");
    }
    Ok(())
}

fn check_cuda(status: sys::CUresult, operation: &'static str) -> Result<()> {
    if status == sys::cudaError_enum::CUDA_SUCCESS {
        Ok(())
    } else {
        invalid_arg(format!("{operation} failed: {status:?}"))
    }
}

fn cuda_driver_error<E: std::fmt::Debug>(
    operation: &'static str,
) -> impl FnOnce(E) -> crate::Error {
    move |error| crate::Error::Tensor(format!("{operation} failed: {error:?}"))
}

fn ipc_handle_to_bytes(handle: CUipcMemHandle_st) -> Sm90Fp8MegaMoeIpcHandle {
    handle.reserved.map(|byte| byte as u8)
}

fn ipc_handle_from_bytes(bytes: Sm90Fp8MegaMoeIpcHandle) -> CUipcMemHandle_st {
    CUipcMemHandle_st {
        reserved: bytes.map(|byte| byte as c_char),
    }
}

fn close_imported_peers(peer_ptrs: &[CUdeviceptr]) {
    for peer_ptr in peer_ptrs.iter().copied() {
        // SAFETY: every pointer in this slice was returned by `cuIpcOpenMemHandle_v2`.
        unsafe {
            let _ = sys::cuIpcCloseMemHandle(peer_ptr);
        }
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
/// - `l1_weights`: interleaved FP8 `[local_experts, 2 * intermediate, hidden]`.
/// - `l1_weights_scale`: F32 `[local_experts, 2 * intermediate / 128, hidden / 128]`.
/// - `l2_weights`: FP8 `[local_experts, hidden, intermediate]`.
/// - `l2_weights_scale`: F32 `[local_experts, hidden / 128, intermediate / 128]`.
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
    ] {
        ensure_same_device(x, tensor, name)?;
    }
    if let Sm90Fp8MegaMoeBuffer::SingleRank(buffer) = &workspace.buffer {
        ensure_same_device(x, buffer, "workspace")?;
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
        let single_rank_storage = match &workspace.buffer {
            Sm90Fp8MegaMoeBuffer::SingleRank(buffer) => Some(buffer.storage_and_layout()),
            Sm90Fp8MegaMoeBuffer::Distributed(buffer) => {
                if buffer.stream.context().ordinal() != device_id as usize {
                    return invalid_arg(format!(
                        "workspace is on CUDA device {}, but input is on CUDA device {device_id}",
                        buffer.stream.context().ordinal()
                    ));
                }
                None
            }
        };
        let single_rank_buffer_ptr = single_rank_storage
            .as_ref()
            .map(|(storage, layout)| {
                tensor_ptr_by_dtype(
                    storage,
                    CandleDType::U8,
                    layout.start_offset(),
                    &stream,
                    "workspace",
                )
            })
            .transpose()?;
        let local_buffer_ptr = match (&single_rank_buffer_ptr, &workspace.buffer) {
            (Some(pointer), Sm90Fp8MegaMoeBuffer::SingleRank(_)) => {
                pointer.as_mut_void() as usize as u64
            }
            (None, Sm90Fp8MegaMoeBuffer::Distributed(buffer)) => buffer.local_ptr as u64,
            _ => unreachable!("workspace pointer variant mismatch"),
        };
        let single_rank_peer_ptrs = [local_buffer_ptr];
        let peer_ptrs = match &workspace.buffer {
            Sm90Fp8MegaMoeBuffer::SingleRank(_) => single_rank_peer_ptrs.as_slice(),
            Sm90Fp8MegaMoeBuffer::Distributed(buffer) => buffer.peer_ptrs.as_slice(),
        };

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
                data: local_buffer_ptr as usize as *mut std::ffi::c_void,
                spec: TensorSpec::contiguous(DeepGemmDType::U8, [workspace.allocation_bytes()]),
            },
            sym_buffer_ptrs: peer_ptrs,
            rank_idx: workspace.rank_idx(),
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
    fn cuda_ipc_handle_bytes_round_trip_without_signedness_loss() {
        let bytes = std::array::from_fn(|index| (index as u8).wrapping_mul(131));
        assert_eq!(ipc_handle_to_bytes(ipc_handle_from_bytes(bytes)), bytes);
    }

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
