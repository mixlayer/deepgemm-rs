# deepgemm-rs

Rust bindings scaffold for DeepSeek's DeepGEMM.

## Layout

- `crates/deepgemm-sys`: raw C ABI surface and build-time discovery of DeepGEMM.
- `crates/deepgemm`: Candle-independent safe wrapper layer.
- `crates/candle-deepgemm`: placeholder for Candle tensor integration.
- `crates/deepgemm-sys/vendor/DeepGEMM`: upstream DeepGEMM submodule.

## Submodules

DeepGEMM depends on nested CUTLASS and fmt submodules. Initialize everything with:

```bash
git submodule update --init --recursive
```

The `deepgemm-sys` build script uses the vendored checkout by default. Set
`DEEPGEMM_ROOT=/path/to/DeepGEMM` to build against another checkout while
developing the bindings.
