#pragma once

#include "deepgemm_c_api.h"

#include <cuda.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace deepgemm_rs {

class StatusError final : public std::runtime_error {
 public:
  StatusError(deepgemm_status_t status, const std::string& message);

  deepgemm_status_t status() const;

 private:
  deepgemm_status_t status_;
};

struct DeviceInfo {
  int device = 0;
  int major = 0;
  int minor = 0;
  int num_sms = 0;
};

struct LaunchArgs {
  int grid_x = 1;
  int grid_y = 1;
  int num_threads = 1;
  int smem_size = 0;
  int cluster_dim = 1;
  bool enable_pdl = false;
};

class KernelRuntime final {
 public:
  explicit KernelRuntime(std::string cubin_path);
  ~KernelRuntime();

  KernelRuntime(const KernelRuntime&) = delete;
  KernelRuntime& operator=(const KernelRuntime&) = delete;

  CUfunction kernel() const;

 private:
  CUlibrary library_{};
  CUfunction kernel_{};
};

void throw_status(deepgemm_status_t status, const std::string& message);
void check_cuda(CUresult result, const char* expression);

void runtime_init(const std::string& deepgemm_root, const std::string& cuda_home);
bool runtime_is_initialized();
DeviceInfo current_device_info();
int effective_num_sms();
void set_num_sms_override(int num_sms);
void set_pdl(bool enabled);
bool pdl_enabled();

std::shared_ptr<KernelRuntime> build_kernel(
    const std::string& name,
    const std::string& code);

CUlaunchConfig make_launch_config(
    CUfunction kernel,
    CUstream stream,
    const LaunchArgs& launch_args);
void launch_kernel_ex(
    const CUlaunchConfig& config,
    CUfunction kernel,
    void** kernel_args);

template <typename... ArgTypes>
void launch_kernel(
    const std::shared_ptr<KernelRuntime>& runtime,
    CUstream stream,
    const LaunchArgs& launch_args,
    ArgTypes&&... args) {
  auto config = make_launch_config(runtime->kernel(), stream, launch_args);
  void* kernel_args[] = {const_cast<void*>(static_cast<const void*>(&args))...};
  launch_kernel_ex(config, runtime->kernel(), kernel_args);
}

int64_t dtype_element_size(deepgemm_dtype_t dtype);
std::string kernel_dtype_name(deepgemm_dtype_t dtype);
int64_t get_tma_aligned_size(int64_t value, int64_t element_size);

CUtensorMap make_tma_2d_desc(
    const void* data,
    deepgemm_dtype_t dtype,
    int gmem_inner_dim,
    int gmem_outer_dim,
    int smem_inner_dim,
    int smem_outer_dim,
    int gmem_outer_stride,
    int swizzle_mode,
    int swizzle_base = 0,
    bool allow_tf32 = false,
    bool fp4_unpacked_smem = true);

}  // namespace deepgemm_rs
