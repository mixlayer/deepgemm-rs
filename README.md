# deepgemm-rs

Work-in-progress Rust and Candle bindings for [DeepSeek DeepGEMM](https://github.com/deepseek-ai/DeepGEMM).

### Crates

- `crates/deepgemm-sys`: C ABI, C++ runtime shim, DeepGEMM source discovery, CUDA driver/JIT integration.
- `crates/deepgemm`: Candle-independent Rust validation, layout calculators, launch descriptors, and error handling.
- `crates/candle-deepgemm`: Candle CUDA tensor integration and ergonomic tensor-returning APIs.

## DeepGEMM

DeepGEMM is DeepSeek's CUDA kernel library for modern LLM primitives. Upstream covers FP8/FP4/BF16 GEMMs, grouped GEMMs, Mega MoE, HyperConnection, and attention/indexer kernels. Unlike a conventional Rust crate that links precompiled CUDA objects, DeepGEMM is built around runtime code generation: it emits specialized CUDA/C++ source for the concrete operation, shape, dtype, and GPU architecture, compiles it with the CUDA toolchain, loads it through the CUDA driver API, and caches the result.

The upstream design matters for bindings because kernel selection is not just a Rust enum. It depends on the active CUDA device, SM architecture, tensor-map descriptors, shared-memory sizing, dtype combinations, and template parameters that are materialized into generated CUDA source.

### JIT Compilation Model

DeepGEMM uses JIT compilation so it can specialize kernels for the exact runtime case instead of shipping a large matrix of precompiled kernels.

This has a few practical consequences:

- First launch for a new kernel configuration may compile CUDA code.
- Later launches reuse the compiled kernel from the DeepGEMM cache.
- The runtime needs a readable DeepGEMM source checkout because generated kernels include upstream headers.
- The runtime needs a CUDA toolkit compiler, not only the CUDA driver.
- Kernel dispatch must happen close to the C++/CUDA layer, where generated source, TMA descriptors, and launch attributes are assembled.

In this wrapper, Rust does not reimplement DeepGEMM's JIT. The C++ shim in `deepgemm-sys` generates the same style of specialized wrapper source, invokes the CUDA compiler, loads the resulting CUDA library with the driver API, and launches the selected kernel on a caller-provided stream.

By default the JIT cache is:

```text
$HOME/.deepgemm_rs
```

Set `DG_JIT_CACHE_DIR` to override it.

## CUDA Toolchain Integration

There are two separate CUDA requirements:

- Build time: Cargo compiles a small C++ shim and needs CUDA headers such as `cuda.h`.
- Runtime: DeepGEMM JIT compilation needs a CUDA toolkit with `nvcc`.

`deepgemm-sys` discovers CUDA headers from:

1. `CUDA_HOME`
2. `CUDA_PATH`
3. `/usr/local/cuda`

It checks both `include/` and `targets/*/include/`.

The runtime compiler path is discovered from the configured CUDA home. Use `CUDA_HOME` or `CUDA_PATH` if the toolkit is not installed at `/usr/local/cuda`.

Recommended CUDA versions follow upstream DeepGEMM:

- SM90: CUDA 12.3 or newer; CUDA 12.9 or newer is recommended for best performance.
- SM100: CUDA 12.9 or newer.

## DeepGEMM Source Checkout

The default source root is the vendored submodule:

```text
crates/deepgemm-sys/vendor/DeepGEMM
```

The submodule is pinned to DeepGEMM `nv_dev` revision
`8b1392b978f5a03c828dd1711090d7fb50958b8a`; it does not follow the moving
branch head.

Initialize it with nested submodules:

```bash
git submodule update --init --recursive
```

To develop against another DeepGEMM checkout:

```bash
DEEPGEMM_ROOT=/path/to/DeepGEMM cargo check --workspace
```

The build script validates that the checkout contains DeepGEMM headers plus nested CUTLASS and `{fmt}` submodules.

## Runtime Controls

The `deepgemm` crate exposes:

- `deepgemm::device_info()`
- `deepgemm::num_sms()`
- `deepgemm::set_num_sms(...)`
- `deepgemm::set_pdl(...)`
- `deepgemm::init(...)`

The wrapper also honors `DG_JIT_CACHE_DIR` for compiled-kernel caching.

## API Layers

### `deepgemm-sys`

This is the raw C ABI layer. It exposes status codes, FFI structs, and native functions. It is intentionally pointer-based and does not know about Candle or PyTorch.

### `deepgemm`

This layer validates explicit tensor specs:

- dtype
- rank
- shape
- element strides
- output layout and padded row stride
- architecture constraints

It takes raw CUDA pointers in `TensorArg` / `TensorOut`, computes required layouts, and forwards launches to the C ABI.

### `candle-deepgemm`

This layer accepts Candle CUDA tensors, extracts stream-safe CUDA pointers, maps Candle dtypes into DeepGEMM dtypes, allocates padded output tensors, and returns logical Candle views.

## Kernel Status

| Kernel surface | SM90 | SM100 | Rust safe API | Candle API | Notes |
| --- | --- | --- | --- | --- | --- |
| `fp8_fp4_mqa_logits` FP8 | Implemented, smoke-tested on GH200 | Implemented, needs SM100 hardware validation | Yes | Yes | Non-paged MQA logits |
| `fp8_fp4_mqa_logits` FP4 | Rejected | Implemented, needs SM100 hardware validation | Yes | Yes | FP4 requires `q_scale` and packed UE8M0 scales |
| `get_paged_mqa_logits_metadata` | Implemented, smoke-tested on GH200 | Implemented, needs SM100 hardware validation | Yes | Yes | Candle API exposes this as `paged_mqa_logits_plan` |
| `fp8_fp4_paged_mqa_logits` FP8 | Implemented, smoke-tested on GH200 | Implemented, needs SM100 hardware validation | Yes | Yes | Paged decode/indexer logits |
| `fp8_fp4_paged_mqa_logits` FP4 | Rejected | Implemented, needs SM100 hardware validation | Yes | Yes | Requires SM100 |
| Other DeepGEMM GEMMs | Planned | Planned | No | No | Not in current binding scope |
| Mega MoE | Planned | Planned | No | No | Requires additional symmetric-memory and multi-rank work |
| HyperConnection | Planned | Planned | No | No | Not in current binding scope |

Smoke-tested means a native CUDA launch completed on the available SM90 GH200 development machine. The ignored-by-default SM90 `next_n=4` regression compares native output with four flattened `next_n=1` launches for batch sizes `1`, `4`, and `16`, head counts `32` and `64`, both supported page sizes, and varying per-position context lengths; broader numerical coverage is still planned.

SM90 paged-MQA/indexer logits support `next_n` exactly in `{1, 2, 4}` and
`block_kv` in `{32, 64}`. `next_n=4` uses the native two-CTA multicast path:
the physical grid still contains `num_sms` CTAs, while scheduler metadata has
`num_sms / 2` slots. An odd SM-count override is rejected. SM90 `next_n=3` and
values above four remain unsupported; higher-level callers may partition wider
verification blocks (for example, `6` as `4 + 2`).

The raw metadata-layout ABI now takes the architecture major, batch size,
`next_n`, `block_kv`, and the physical SM count so it can derive cluster-sized
metadata. The safe Rust and Candle entry points retain their existing spec/plan
shape and derive this internally.

## Development

Common checks:

```bash
cargo fmt --all --check
cargo check --workspace
cargo test --workspace
```

On an SM90 GPU, run the clustered correctness regression with:

```bash
cargo test -p candle-deepgemm \
  mqa::tests::sm90_next_n_four_matches_flattened_launches \
  -- --ignored --nocapture
```

Useful environment variables:

```bash
CUDA_HOME=/usr/local/cuda
DEEPGEMM_ROOT=/path/to/DeepGEMM
DG_JIT_CACHE_DIR=/tmp/deepgemm-rs-cache
```
