use candle::{DType, Device, Tensor};
use candle_deepgemm::{bf16_m_grouped_gemm_nt_contiguous_into, deepgemm};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args()
        .skip(1)
        .map(|value| value.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    let [m, n, k, groups, active_rows, iterations] = args.as_slice() else {
        return Err("usage: bench_sm12x_grouped M N K GROUPS ACTIVE_ROWS ITERATIONS".into());
    };
    if m % 64 != 0 || active_rows % groups != 0 || active_rows > m {
        return Err("M must be 64-aligned and ACTIVE_ROWS must divide evenly across GROUPS".into());
    }

    let device = Device::new_cuda(0)?;
    let source_root = deepgemm::source_root()
        .to_str()
        .ok_or("DeepGEMM source root is not UTF-8")?;
    let cuda_home = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_owned());
    deepgemm::init(source_root, &cuda_home)?;
    if let Ok(num_sms) = std::env::var("DG_TEST_NUM_SMS") {
        deepgemm::set_num_sms(num_sms.parse()?)?;
    }
    let rows_per_group = active_rows / groups;
    if rows_per_group % 64 != 0 {
        return Err("rows per group must be 64-aligned".into());
    }
    let mut layout = vec![-1i32; *m];
    for group in 0..*groups {
        layout[group * rows_per_group..(group + 1) * rows_per_group].fill(i32::try_from(group)?);
    }

    let a = Tensor::zeros((*m, *k), DType::BF16, &device)?;
    let b = Tensor::zeros((*groups, *n, *k), DType::BF16, &device)?;
    let d = Tensor::zeros((*m, *n), DType::BF16, &device)?;
    let grouped_layout = Tensor::from_vec(layout, *m, &device)?;
    for _ in 0..10 {
        bf16_m_grouped_gemm_nt_contiguous_into(&a, &b, &grouped_layout, &d, 64)?;
    }
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..*iterations {
        bf16_m_grouped_gemm_nt_contiguous_into(&a, &b, &grouped_layout, &d, 64)?;
    }
    device.synchronize()?;
    println!(
        "m={m} n={n} k={k} groups={groups} active_rows={active_rows}: {:.3} us",
        start.elapsed().as_secs_f64() * 1e6 / *iterations as f64
    );
    Ok(())
}
