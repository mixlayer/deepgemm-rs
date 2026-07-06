# DeepGEMM Binding Plan

## Initial Scope

The first binding surface is the MQA logits family used by DeepGEMM's attention APIs:

- `fp8_fp4_mqa_logits`
- `get_paged_mqa_logits_metadata`
- `fp8_fp4_paged_mqa_logits`

The paged logits API requires schedule metadata shaped `[num_sms + 1, 2]`, so metadata generation is part of the initial surface.

## Architecture Dispatch

Architecture dispatch belongs in the `deepgemm-sys` C++ shim, at runtime, per launch.

The shim should query the current CUDA device and dispatch as follows:

- SM90:
  - `sm90_fp8_mqa_logits`
  - `sm90_paged_mqa_logits_metadata`
  - `sm90_fp8_paged_mqa_logits`
  - FP4 is rejected.
- SM100:
  - `sm100_mqa_logits`
  - `sm100_paged_mqa_logits_metadata`
  - `sm100_paged_mqa_logits`
  - FP8 and FP4 are both supported.

Rust wrappers may perform early validation and return nicer errors, but they should not own kernel selection. Kernel selection depends on TMA descriptors, JIT template generation, shared-memory sizing, supported dtypes, and CUDA compiler flags, which are C++/CUDA concerns. Dispatch must be runtime-based because the same Rust binary can run on different GPU architectures.

## Binding Strategy

Avoid binding DeepGEMM's PyTorch-facing functions directly. They validate `torch::Tensor`s, allocate output tensors, and select the current PyTorch CUDA stream. Instead, build a raw C ABI around the DeepGEMM JIT/runtime path and use caller-owned CUDA buffers.

The C++ shim should:

- Initialize DeepGEMM's JIT runtime with a DeepGEMM source root and CUDA home.
- Validate raw pointer, shape, stride, dtype, and architecture constraints.
- Construct the required `CUtensorMap` descriptors.
- Build or retrieve JIT kernels through DeepGEMM's existing compiler/cache code.
- Launch on a caller-provided `cudaStream_t`.
- Catch DeepGEMM/CUDA exceptions and expose status codes plus `deepgemm_last_error()`.

The Rust layers should be split as follows:

- `deepgemm-sys`: `#[repr(C)]` parameter structs, raw externs, status codes, and build/link setup.
- `deepgemm`: safe wrappers, shape validation, output layout calculators, metadata sizing helpers, and error conversion.
- `candle-deepgemm`: Candle tensor/storage integration and ergonomic tensor APIs.

## Ownership Model

All launch APIs should write into caller-provided buffers.

Reasons:

- Rust/Candle should own allocation and tensor lifetimes.
- The FFI boundary stays independent of PyTorch allocation semantics.
- Output layout can be explicit and testable.

The safe Rust layer should expose helper functions for:

- Required non-paged logits allocation shape and aligned row stride.
- Required paged logits allocation shape and aligned row stride.
- Required paged metadata length: `(num_sms + 1) * 2` `i32` elements.

## C ABI Surface

Add runtime/setup functions:

- `deepgemm_init(const char* deepgemm_root, const char* cuda_home)`
- `deepgemm_get_device_info(...)`
- `deepgemm_get_num_sms(...)`
- `deepgemm_set_num_sms(int num_sms)`
- `deepgemm_set_pdl(bool enabled)`
- `deepgemm_last_error()`

Add launch functions:

- `deepgemm_fp8_fp4_mqa_logits(const deepgemm_mqa_logits_params_t*)`
- `deepgemm_paged_mqa_logits_metadata(const deepgemm_paged_mqa_logits_metadata_params_t*)`
- `deepgemm_fp8_fp4_paged_mqa_logits(const deepgemm_paged_mqa_logits_params_t*)`

Parameter structs should include:

- Raw device pointers.
- Shape fields.
- Element strides where non-contiguous layouts are allowed.
- Dtype enums for Q/KV format, weights, logits, and scale tensors.
- Optional pointer fields for FP4 Q scales and varlen indices.
- `cudaStream_t`.
- Output pointer and output row stride.

## Shape And Dtype Constraints

### `fp8_fp4_mqa_logits`

Common:

- `q`: `[seq_len, num_heads, head_dim]` for FP8, or packed FP4 with last logical dim halved.
- `kv`: `[seq_len_kv, head_dim]` for FP8, or packed FP4 with last logical dim halved.
- `weights`: `[seq_len, num_heads]`, stride on head dimension must be 1.
- `cu_seq_len_k_start`: `[seq_len]`, `i32`, contiguous.
- `cu_seq_len_k_end`: `[seq_len]`, `i32`, contiguous.
- Logits dtype: `f32` or `bf16`.
- Logits row stride must be 1024-byte aligned.
- `max_seqlen_k == 0` means full logits width `seq_len_kv`; otherwise compressed logits width is `max_seqlen_k`.
- If compressed logits are used, cleaning unfilled logits is not supported.

SM90:

- FP8 only.
- `num_heads`: `32` or `64`.
- `head_dim`: `32`, `64`, or `128`.
- KV scale dtype: `f32`.
- weights dtype: `f32`.

SM100:

- FP8 or FP4.
- `num_heads`: `8`, `16`, `32`, or `64`.
- FP8 `head_dim`: `32`, `64`, or `128`.
- FP4 `head_dim`: `64` or `128`.
- FP4 scale dtype: packed UE8M0 in `i32`.
- weights dtype: `f32` or `bf16`; `bf16` weights require `bf16` logits.

### `get_paged_mqa_logits_metadata`

Common:

- `context_lens`: `[batch_size, next_n]`, `i32`, contiguous.
- Only 2D context lengths are in scope initially.
- Output metadata: `[num_sms + 1, 2]`, `i32`, contiguous.

SM90:

- `block_kv == 64`.
- No varlen `indices`.

SM100:

- `block_kv == 32` or `64`.
- Varlen `indices` are supported only with `next_n == 1`.

### `fp8_fp4_paged_mqa_logits`

Common:

- `q`: `[batch_size, next_n, num_heads, head_dim]` for FP8, or packed FP4 with last logical dim halved.
- `fused_kv_cache`: byte tensor with upstream DeepGEMM fused data-plus-scale layout.
- `weights`: `[batch_size * next_n, num_heads]`, stride on head dimension must be 1.
- `context_lens`: `[batch_size, next_n]`, `i32`, contiguous.
- `block_table`: `[batch_size, max_block_len]`, `i32`, stride on block dimension must be 1.
- `schedule_meta`: `[num_sms + 1, 2]`, `i32`, contiguous.
- Logits dtype: `f32` or `bf16`.
- Logits row stride must be 1024-byte aligned and a multiple of `split_kv = 256`.
- `clean_logits` should remain unsupported initially because upstream currently asserts against 2D context lengths when cleaning.

SM90:

- FP8 only.
- `num_heads`: `32` or `64`.
- `head_dim`: `32`, `64`, or `128`.
- `block_kv == 64`.
- `next_n == 1` or `2`.
- No varlen `indices`.
- weights dtype: `f32`.

SM100:

- FP8 or FP4.
- `num_heads`: `8`, `16`, `32`, or `64`.
- FP8 `head_dim`: `32`, `64`, or `128`.
- FP4 `head_dim`: `64` or `128`.
- `block_kv == 32` or `64`.
- Varlen `indices` are supported only with `next_n == 1`.
- weights dtype: `f32` or `bf16`; `bf16` weights require `bf16` logits.

## Implementation Phases

1. Add C ABI types and runtime initialization.
2. Add raw metadata generation for paged MQA logits.
3. Add raw SM90 FP8 non-paged launch.
4. Add raw SM90 FP8 paged launch.
5. Add raw SM100 FP8 non-paged and paged launch.
6. Add raw SM100 FP4 non-paged and paged launch.
7. Add safe Rust layout calculators and validation wrappers.
8. Add Candle tensor integration.
9. Port a focused subset of DeepGEMM attention tests.

## Test Strategy

Start with small Rust tests for pure layout math and validation that do not require CUDA.

CUDA tests should run in this order:

1. Metadata generation only.
2. SM90 FP8 non-paged logits.
3. SM90 FP8 paged logits.
4. SM100 FP8 non-paged and paged logits.
5. SM100 FP4 non-paged and paged logits.

Reference checks can initially use fixtures generated from DeepGEMM's Python tests or a small Python reference. Once Candle integration exists, compare Candle outputs against the Python reference for the same randomized inputs.

## Open Questions

- Should the first safe Rust API expose only contiguous tensors, or should it accept explicit strides from the start?
- Where should FP8/FP4 quantization helpers live: `deepgemm`, `candle-deepgemm`, or a separate test-only module?
- Should the C ABI expose precomputed output layout helpers, or should layout calculation remain Rust-only?
- Do we need a public API for JIT cache directory and compiler selection, or is environment-variable configuration sufficient for the first version?
