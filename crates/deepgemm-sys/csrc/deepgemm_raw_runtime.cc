#include "deepgemm_raw_runtime.h"

#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <regex>
#include <sstream>
#include <unordered_map>

namespace deepgemm_rs {
namespace {

std::mutex g_runtime_mutex;
std::string g_deepgemm_root;
std::string g_cuda_home;
std::string g_include_path;
std::string g_cutlass_include_path;
std::string g_cutlass_util_include_path;
int g_num_sms_override = 0;
bool g_pdl_enabled = false;
std::unordered_map<std::string, std::shared_ptr<KernelRuntime>> g_kernel_cache;

struct CompilerConfig {
  std::filesystem::path nvcc;
  std::string flags;
};

std::optional<CompilerConfig> g_compiler_config_cache;

void* cuda_driver_handle() {
  static void* handle = nullptr;
  if (handle == nullptr) {
    handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
      throw_status(
          DEEPGEMM_STATUS_CUDA_ERROR,
          std::string("failed to load CUDA driver libcuda.so.1: ") + dlerror());
    }
  }
  return handle;
}

#define DECL_LAZY_CUDA_DRIVER_FUNCTION(name)                                    \
  template <typename... Args>                                                   \
  static auto lazy_##name(Args&&... args) -> decltype(name(args...)) {          \
    using FuncType = decltype(&(name));                                         \
    static FuncType func = nullptr;                                             \
    if (func == nullptr) {                                                      \
      func = reinterpret_cast<FuncType>(dlsym(cuda_driver_handle(), #name));    \
      if (func == nullptr) {                                                    \
        throw_status(                                                           \
            DEEPGEMM_STATUS_CUDA_ERROR,                                         \
            std::string("failed to load CUDA driver symbol ") + #name);         \
      }                                                                         \
    }                                                                           \
    return func(std::forward<Args>(args)...);                                   \
  }

DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorName);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorString);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuInit);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuCtxGetCurrent);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuCtxGetDevice);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuDeviceGet);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuDeviceGetAttribute);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuFuncSetAttribute);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryLoadFromFile);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryUnload);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryGetKernelCount);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryEnumerateKernels);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuKernelGetFunction);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLaunchKernelEx);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuTensorMapEncodeTiled);

#undef DECL_LAZY_CUDA_DRIVER_FUNCTION

int env_int(const char* name, int default_value = 0) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  return std::atoi(value);
}

std::string env_string(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

std::string shell_quote(const std::string& value) {
  std::string out = "'";
  for (const char c : value) {
    if (c == '\'') {
      out += "'\"'\"'";
    } else {
      out += c;
    }
  }
  out += "'";
  return out;
}

std::pair<int, std::string> run_command(const std::string& command) {
  FILE* pipe = popen((command + " 2>&1").c_str(), "r");
  if (pipe == nullptr) {
    throw_status(DEEPGEMM_STATUS_INTERNAL_ERROR, "failed to run external command");
  }

  std::string output;
  char buffer[4096];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
  }
  const int raw_status = pclose(pipe);
  if (raw_status == -1) {
    return {1, output};
  }
  if (WIFEXITED(raw_status)) {
    return {WEXITSTATUS(raw_status), output};
  }
  return {1, output};
}

void require_file(const std::filesystem::path& path, const char* label) {
  if (!std::filesystem::exists(path)) {
    throw_status(
        DEEPGEMM_STATUS_INVALID_ARGUMENT,
        std::string(label) + " does not exist: " + path.string());
  }
}

std::filesystem::path cache_root() {
  const auto env_cache = env_string("DG_JIT_CACHE_DIR");
  if (!env_cache.empty()) {
    return env_cache;
  }
  const auto home = env_string("HOME");
  if (home.empty()) {
    throw_status(DEEPGEMM_STATUS_INTERNAL_ERROR, "HOME is not set for JIT cache");
  }
  return std::filesystem::path(home) / ".deepgemm_rs";
}

uint64_t fnv1a64(const std::string& input) {
  uint64_t hash = 1469598103934665603ull;
  for (const unsigned char byte : input) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string hex_u64(uint64_t value) {
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << value;
  return out.str();
}

std::filesystem::path nvcc_path() {
  const auto env_nvcc = env_string("DG_JIT_NVCC_COMPILER");
  if (!env_nvcc.empty()) {
    return env_nvcc;
  }

  std::lock_guard<std::mutex> lock(g_runtime_mutex);
  return std::filesystem::path(g_cuda_home) / "bin" / "nvcc";
}

std::pair<int, int> nvcc_version(const std::filesystem::path& nvcc) {
  require_file(nvcc, "nvcc");
  const auto [status, output] = run_command(shell_quote(nvcc.string()) + " --version");
  if (status != 0) {
    throw_status(
        DEEPGEMM_STATUS_INTERNAL_ERROR,
        "nvcc --version failed: " + output);
  }

  std::smatch match;
  if (!std::regex_search(output, match, std::regex(R"(release ([0-9]+)\.([0-9]+))"))) {
    throw_status(
        DEEPGEMM_STATUS_INTERNAL_ERROR,
        "failed to parse nvcc version: " + output);
  }

  const int major = std::stoi(match[1].str());
  const int minor = std::stoi(match[2].str());
  if (major < 12 || (major == 12 && minor < 3)) {
    throw_status(DEEPGEMM_STATUS_UNSUPPORTED_ARCH, "DeepGEMM requires nvcc >= 12.3");
  }
  return {major, minor};
}

std::string nvcc_arch(const DeviceInfo& info, bool supports_arch_family) {
  if (info.major == 10 && info.minor != 1) {
    return supports_arch_family ? "100f" : "100a";
  }
  return std::to_string(info.major * 10 + info.minor) + "a";
}

std::string compiler_flags(const std::filesystem::path& nvcc) {
  const auto [nvcc_major, nvcc_minor] = nvcc_version(nvcc);
  const auto info = current_device_info();
  const bool supports_arch_family = nvcc_major > 12 || (nvcc_major == 12 && nvcc_minor >= 9);
  const auto cpp_standard = env_int("DG_JIT_CPP_STANDARD", 20);

  std::string flags =
      "-std=c++" + std::to_string(cpp_standard) +
      " --diag-suppress=39,161,174,177,186,940"
      " --ptxas-options=--register-usage-level=10";
  if (env_int("DG_JIT_DEBUG") || env_int("DG_JIT_PTXAS_VERBOSE") ||
      env_int("DG_JIT_PTXAS_CHECK")) {
    flags += " --ptxas-options=--verbose,--warn-on-local-memory-usage";
  }
  if (env_int("DG_JIT_WITH_LINEINFO")) {
    flags += " -Xcompiler -rdynamic -lineinfo";
  }

  std::lock_guard<std::mutex> lock(g_runtime_mutex);
  flags += " -I" + shell_quote(g_include_path);
  flags += " -I" + shell_quote(g_cutlass_include_path);
  flags += " -I" + shell_quote(g_cutlass_util_include_path);
  flags += " --gpu-architecture=sm_" + nvcc_arch(info, supports_arch_family);
  flags += " --compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi";
  flags += " -O3 --expt-relaxed-constexpr --expt-extended-lambda";
  return flags;
}

CompilerConfig compiler_config() {
  {
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    if (g_compiler_config_cache.has_value()) {
      return *g_compiler_config_cache;
    }
  }

  CompilerConfig config{nvcc_path(), ""};
  config.flags = compiler_flags(config.nvcc);

  std::lock_guard<std::mutex> lock(g_runtime_mutex);
  if (!g_compiler_config_cache.has_value()) {
    g_compiler_config_cache = config;
  }
  return *g_compiler_config_cache;
}

void write_file(const std::filesystem::path& path, const std::string& contents) {
  std::ofstream out(path, std::ios::binary);
  if (!out.write(contents.data(), static_cast<std::streamsize>(contents.size()))) {
    throw_status(DEEPGEMM_STATUS_INTERNAL_ERROR, "failed to write " + path.string());
  }
}

std::shared_ptr<KernelRuntime> load_cached_kernel(
    const std::string& cache_key,
    const std::filesystem::path& dir_path) {
  std::lock_guard<std::mutex> lock(g_runtime_mutex);
  auto entry = g_kernel_cache.find(cache_key);
  if (entry != g_kernel_cache.end()) {
    return entry->second;
  }
  if (!std::filesystem::exists(dir_path / "kernel.cubin")) {
    return nullptr;
  }

  auto runtime = std::make_shared<KernelRuntime>((dir_path / "kernel.cubin").string());
  g_kernel_cache[cache_key] = runtime;
  return runtime;
}

CUtensorMapDataType tensor_map_dtype(
    deepgemm_dtype_t dtype,
    bool allow_tf32,
    bool fp4_unpacked_smem) {
  if (allow_tf32 && dtype == DEEPGEMM_DTYPE_F32) {
    return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32;
  }

  switch (dtype) {
    case DEEPGEMM_DTYPE_I32:
    case DEEPGEMM_DTYPE_PACKED_UE8M0:
      return CU_TENSOR_MAP_DATA_TYPE_INT32;
    case DEEPGEMM_DTYPE_F32:
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    case DEEPGEMM_DTYPE_BF16:
      return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    case DEEPGEMM_DTYPE_FP8_E4M3:
    case DEEPGEMM_DTYPE_U8:
      return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    case DEEPGEMM_DTYPE_PACKED_FP4_E2M1:
      return fp4_unpacked_smem ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
                               : CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    default:
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "unsupported tensor map dtype");
      return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  }
}

CUtensorMapSwizzle tensor_map_swizzle(int mode, int base) {
  if (base != 0) {
    if (base == 32 && mode == 128) {
      return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
    }
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "unsupported tensor map swizzle base");
  }

  switch (mode) {
    case 0:
    case 16:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
    default:
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "unsupported tensor map swizzle mode");
      return CU_TENSOR_MAP_SWIZZLE_NONE;
  }
}

}  // namespace

StatusError::StatusError(deepgemm_status_t status, const std::string& message)
    : std::runtime_error(message), status_(status) {}

deepgemm_status_t StatusError::status() const {
  return status_;
}

void throw_status(deepgemm_status_t status, const std::string& message) {
  throw StatusError(status, message);
}

void check_cuda(CUresult result, const char* expression) {
  if (result == CUDA_SUCCESS) {
    return;
  }

  const char* name = nullptr;
  const char* message = nullptr;
  lazy_cuGetErrorName(result, &name);
  lazy_cuGetErrorString(result, &message);

  std::string error = expression;
  error += " failed";
  if (name != nullptr) {
    error += " with ";
    error += name;
  }
  if (message != nullptr) {
    error += ": ";
    error += message;
  }
  throw_status(DEEPGEMM_STATUS_CUDA_ERROR, error);
}

void runtime_init(const std::string& deepgemm_root, const std::string& cuda_home) {
  if (deepgemm_root.empty()) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "deepgemm_root must not be empty");
  }
  if (cuda_home.empty()) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "cuda_home must not be empty");
  }

  const auto root = std::filesystem::weakly_canonical(deepgemm_root);
  const auto cuda = std::filesystem::weakly_canonical(cuda_home);
  require_file(root / "deep_gemm/include/deep_gemm/scheduler/sm90_paged_mqa_logits.cuh", "DeepGEMM SM90 scheduler header");
  require_file(root / "deep_gemm/include/deep_gemm/scheduler/sm100_paged_mqa_logits.cuh", "DeepGEMM SM100 scheduler header");
  require_file(root / "third-party/cutlass/include/cutlass/cutlass.h", "CUTLASS header");
  require_file(cuda / "bin/nvcc", "CUDA nvcc");

  std::lock_guard<std::mutex> lock(g_runtime_mutex);
  g_deepgemm_root = root.string();
  g_cuda_home = cuda.string();
  g_include_path = (root / "deep_gemm/include").string();
  g_cutlass_include_path = (root / "third-party/cutlass/include").string();
  g_cutlass_util_include_path = (root / "third-party/cutlass/tools/util/include").string();
  g_compiler_config_cache.reset();
}

bool runtime_is_initialized() {
  std::lock_guard<std::mutex> lock(g_runtime_mutex);
  return !g_deepgemm_root.empty() && !g_cuda_home.empty();
}

DeviceInfo current_device_info() {
  check_cuda(lazy_cuInit(0), "cuInit");

  CUdevice device = 0;
  CUcontext context = nullptr;
  check_cuda(lazy_cuCtxGetCurrent(&context), "cuCtxGetCurrent");
  if (context != nullptr) {
    check_cuda(lazy_cuCtxGetDevice(&device), "cuCtxGetDevice");
  } else {
    check_cuda(lazy_cuDeviceGet(&device, 0), "cuDeviceGet");
  }

  DeviceInfo info;
  info.device = static_cast<int>(device);
  check_cuda(lazy_cuDeviceGetAttribute(
                 &info.major,
                 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                 device),
             "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR)");
  check_cuda(lazy_cuDeviceGetAttribute(
                 &info.minor,
                 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                 device),
             "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR)");
  check_cuda(lazy_cuDeviceGetAttribute(
                 &info.num_sms,
                 CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                 device),
             "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)");
  return info;
}

int effective_num_sms() {
  std::lock_guard<std::mutex> lock(g_runtime_mutex);
  if (g_num_sms_override > 0) {
    return g_num_sms_override;
  }
  return current_device_info().num_sms;
}

void set_num_sms_override(int num_sms) {
  if (num_sms < 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "num_sms must be non-negative");
  }
  if (num_sms > 0 && num_sms > current_device_info().num_sms) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "num_sms exceeds current device SM count");
  }

  std::lock_guard<std::mutex> lock(g_runtime_mutex);
  g_num_sms_override = num_sms;
}

void set_pdl(bool enabled) {
  std::lock_guard<std::mutex> lock(g_runtime_mutex);
  g_pdl_enabled = enabled;
}

bool pdl_enabled() {
  std::lock_guard<std::mutex> lock(g_runtime_mutex);
  return g_pdl_enabled;
}

std::shared_ptr<KernelRuntime> build_kernel(
    const std::string& name,
    const std::string& code) {
  if (!runtime_is_initialized()) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "deepgemm_init must be called before launching kernels");
  }

  const auto config = compiler_config();
  const auto cache_key = name + "$$" + config.nvcc.string() + "$$" + config.flags + "$$" + code;
  const auto hash = hex_u64(fnv1a64(cache_key));
  const auto dir_path = cache_root() / "cache" / ("kernel." + name + "." + hash);
  if (auto runtime = load_cached_kernel(cache_key, dir_path)) {
    return runtime;
  }

  const auto tmp_root = cache_root() / "tmp";
  std::filesystem::create_directories(tmp_root);
  const auto tmp_dir = tmp_root / (name + "." + std::to_string(getpid()) + "." + hash);
  std::filesystem::remove_all(tmp_dir);
  std::filesystem::create_directories(tmp_dir);

  const auto code_path = tmp_dir / "kernel.cu";
  const auto cubin_path = tmp_dir / "kernel.cubin";
  write_file(code_path, code);

  const auto command =
      "cd " + shell_quote(tmp_root.string()) + " && " +
      shell_quote(config.nvcc.string()) + " " +
      shell_quote(code_path.string()) + " -cubin -o " +
      shell_quote(cubin_path.string()) + " " + config.flags;
  if (env_int("DG_JIT_DEBUG") || env_int("DG_JIT_PRINT_COMPILER_COMMAND")) {
    std::fprintf(stderr, "Running NVCC command: %s\n", command.c_str());
  }

  const auto [status, output] = run_command(command);
  if (status != 0) {
    std::filesystem::remove_all(tmp_dir);
    throw_status(DEEPGEMM_STATUS_INTERNAL_ERROR, "NVCC compilation failed: " + output);
  }

  std::filesystem::create_directories(dir_path.parent_path());
  std::error_code rename_error;
  std::filesystem::rename(tmp_dir, dir_path, rename_error);
  if (rename_error) {
    std::filesystem::remove_all(tmp_dir);
    if (!std::filesystem::exists(dir_path / "kernel.cubin")) {
      throw_status(
          DEEPGEMM_STATUS_INTERNAL_ERROR,
          "failed to move compiled kernel into JIT cache: " + rename_error.message());
    }
  }

  auto runtime = std::make_shared<KernelRuntime>((dir_path / "kernel.cubin").string());
  {
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    g_kernel_cache[cache_key] = runtime;
  }
  return runtime;
}

CUlaunchConfig make_launch_config(
    CUfunction kernel,
    CUstream stream,
    const LaunchArgs& launch_args) {
  if (launch_args.smem_size > 0) {
    check_cuda(
        lazy_cuFuncSetAttribute(
            kernel,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            launch_args.smem_size),
        "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)");
  }

  CUlaunchConfig config{};
  config.gridDimX = static_cast<unsigned int>(launch_args.grid_x);
  config.gridDimY = static_cast<unsigned int>(launch_args.grid_y);
  config.gridDimZ = 1;
  config.blockDimX = static_cast<unsigned int>(launch_args.num_threads);
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = static_cast<unsigned int>(launch_args.smem_size);
  config.hStream = stream;

  thread_local CUlaunchAttribute attrs[2];
  config.numAttrs = 0;
  config.attrs = attrs;

  if (launch_args.cluster_dim > 1) {
    auto& attr = attrs[config.numAttrs++];
    attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attr.value.clusterDim.x = static_cast<unsigned int>(launch_args.cluster_dim);
    attr.value.clusterDim.y = 1;
    attr.value.clusterDim.z = 1;
  }
  if (launch_args.enable_pdl) {
    auto& attr = attrs[config.numAttrs++];
    attr.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attr.value.programmaticStreamSerializationAllowed = 1;
  }
  return config;
}

void launch_kernel_ex(
    const CUlaunchConfig& config,
    CUfunction kernel,
    void** kernel_args) {
  check_cuda(lazy_cuLaunchKernelEx(&config, kernel, kernel_args, nullptr), "cuLaunchKernelEx");
}

int64_t dtype_element_size(deepgemm_dtype_t dtype) {
  switch (dtype) {
    case DEEPGEMM_DTYPE_FP8_E4M3:
    case DEEPGEMM_DTYPE_PACKED_FP4_E2M1:
    case DEEPGEMM_DTYPE_U8:
      return 1;
    case DEEPGEMM_DTYPE_BF16:
      return 2;
    case DEEPGEMM_DTYPE_PACKED_UE8M0:
    case DEEPGEMM_DTYPE_F32:
    case DEEPGEMM_DTYPE_I32:
      return 4;
    default:
      return 0;
  }
}

std::string kernel_dtype_name(deepgemm_dtype_t dtype) {
  switch (dtype) {
    case DEEPGEMM_DTYPE_I32:
      return "int";
    case DEEPGEMM_DTYPE_F32:
      return "float";
    case DEEPGEMM_DTYPE_BF16:
      return "cutlass::bfloat16_t";
    case DEEPGEMM_DTYPE_FP8_E4M3:
      return "cutlass::float_e4m3_t";
    case DEEPGEMM_DTYPE_PACKED_FP4_E2M1:
      return "cutlass::detail::float_e2m1_unpacksmem_t";
    default:
      throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "unsupported kernel dtype");
      return {};
  }
}

int64_t get_tma_aligned_size(int64_t value, int64_t element_size) {
  constexpr int64_t kNumTmaAlignmentBytes = 16;
  if (value < 0 || element_size <= 0 || kNumTmaAlignmentBytes % element_size != 0) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "invalid TMA alignment input");
  }
  const int64_t alignment = kNumTmaAlignmentBytes / element_size;
  return ((value + alignment - 1) / alignment) * alignment;
}

CUtensorMap make_tma_2d_desc(
    const void* data,
    deepgemm_dtype_t dtype,
    int gmem_inner_dim,
    int gmem_outer_dim,
    int smem_inner_dim,
    int smem_outer_dim,
    int gmem_outer_stride,
    int swizzle_mode,
    int swizzle_base,
    bool allow_tf32,
    bool fp4_unpacked_smem) {
  if (data == nullptr) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "tensor map data must not be null");
  }

  const int64_t elem_size_i64 = dtype_element_size(dtype);
  if (elem_size_i64 <= 0 || elem_size_i64 > INT32_MAX) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "invalid tensor map element size");
  }
  const int elem_size = static_cast<int>(elem_size_i64);

  if (swizzle_mode != 0) {
    smem_inner_dim = swizzle_mode / elem_size;
  }

  if (dtype == DEEPGEMM_DTYPE_PACKED_FP4_E2M1) {
    if (fp4_unpacked_smem && gmem_inner_dim % 128 != 0) {
      throw_status(
          DEEPGEMM_STATUS_INVALID_ARGUMENT,
          "FP4 unpacked TMA inner dimension must be a multiple of 128");
    }
    if (!fp4_unpacked_smem && swizzle_mode != 0) {
      smem_inner_dim = swizzle_mode * 2;
    }
  }

  CUtensorMap tensor_map{};
  const cuuint64_t gmem_dims[2] = {
      static_cast<cuuint64_t>(gmem_inner_dim),
      static_cast<cuuint64_t>(gmem_outer_dim)};
  const cuuint32_t smem_dims[2] = {
      static_cast<cuuint32_t>(smem_inner_dim),
      static_cast<cuuint32_t>(smem_outer_dim)};
  const cuuint64_t gmem_strides[1] = {
      static_cast<cuuint64_t>(gmem_outer_stride) * static_cast<cuuint64_t>(elem_size)};
  const cuuint32_t elem_strides[2] = {1, 1};

  check_cuda(
      lazy_cuTensorMapEncodeTiled(
          &tensor_map,
          tensor_map_dtype(dtype, allow_tf32, fp4_unpacked_smem),
          2,
          const_cast<void*>(data),
          gmem_dims,
          gmem_strides,
          smem_dims,
          elem_strides,
          CU_TENSOR_MAP_INTERLEAVE_NONE,
          tensor_map_swizzle(swizzle_mode, swizzle_base),
          CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");
  return tensor_map;
}

CUtensorMap make_tma_3d_desc(
    const void* data,
    deepgemm_dtype_t dtype,
    int gmem_inner_dim,
    int gmem_outer_dim,
    int gmem_batch_dim,
    int smem_inner_dim,
    int smem_outer_dim,
    int smem_batch_dim,
    int gmem_outer_stride,
    int gmem_batch_stride,
    int swizzle_mode,
    int swizzle_base,
    bool allow_tf32,
    bool fp4_unpacked_smem) {
  if (data == nullptr) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "tensor map data must not be null");
  }

  const int64_t elem_size_i64 = dtype_element_size(dtype);
  if (elem_size_i64 <= 0 || elem_size_i64 > INT32_MAX) {
    throw_status(DEEPGEMM_STATUS_INVALID_ARGUMENT, "invalid tensor map element size");
  }
  const int elem_size = static_cast<int>(elem_size_i64);

  if (swizzle_mode != 0) {
    smem_inner_dim = swizzle_mode / elem_size;
  }

  if (dtype == DEEPGEMM_DTYPE_PACKED_FP4_E2M1) {
    if (fp4_unpacked_smem && gmem_inner_dim % 128 != 0) {
      throw_status(
          DEEPGEMM_STATUS_INVALID_ARGUMENT,
          "FP4 unpacked TMA inner dimension must be a multiple of 128");
    }
    if (!fp4_unpacked_smem && swizzle_mode != 0) {
      smem_inner_dim = swizzle_mode * 2;
    }
  }

  CUtensorMap tensor_map{};
  const cuuint64_t gmem_dims[3] = {
      static_cast<cuuint64_t>(gmem_inner_dim),
      static_cast<cuuint64_t>(gmem_outer_dim),
      static_cast<cuuint64_t>(gmem_batch_dim)};
  const cuuint32_t smem_dims[3] = {
      static_cast<cuuint32_t>(smem_inner_dim),
      static_cast<cuuint32_t>(smem_outer_dim),
      static_cast<cuuint32_t>(smem_batch_dim)};
  const cuuint64_t gmem_strides[2] = {
      static_cast<cuuint64_t>(gmem_outer_stride) * static_cast<cuuint64_t>(elem_size),
      static_cast<cuuint64_t>(gmem_batch_stride) * static_cast<cuuint64_t>(elem_size)};
  const cuuint32_t elem_strides[3] = {1, 1, 1};

  check_cuda(
      lazy_cuTensorMapEncodeTiled(
          &tensor_map,
          tensor_map_dtype(dtype, allow_tf32, fp4_unpacked_smem),
          3,
          const_cast<void*>(data),
          gmem_dims,
          gmem_strides,
          smem_dims,
          elem_strides,
          CU_TENSOR_MAP_INTERLEAVE_NONE,
          tensor_map_swizzle(swizzle_mode, swizzle_base),
          CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");
  return tensor_map;
}

KernelRuntime::KernelRuntime(std::string cubin_path) {
  check_cuda(
      lazy_cuLibraryLoadFromFile(
          &library_,
          cubin_path.c_str(),
          nullptr,
          nullptr,
          0,
          nullptr,
          nullptr,
          0),
      "cuLibraryLoadFromFile");

  unsigned int kernel_count = 0;
  check_cuda(lazy_cuLibraryGetKernelCount(&kernel_count, library_), "cuLibraryGetKernelCount");
  if (kernel_count != 1) {
    throw_status(DEEPGEMM_STATUS_INTERNAL_ERROR, "JIT cubin does not contain exactly one kernel");
  }

  CUkernel cu_kernel{};
  check_cuda(lazy_cuLibraryEnumerateKernels(&cu_kernel, 1, library_), "cuLibraryEnumerateKernels");
  check_cuda(lazy_cuKernelGetFunction(&kernel_, cu_kernel), "cuKernelGetFunction");
}

KernelRuntime::~KernelRuntime() {
  if (library_ != nullptr) {
    try {
      const auto result = lazy_cuLibraryUnload(library_);
      if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
        std::fprintf(stderr, "warning: cuLibraryUnload failed during DeepGEMM shutdown\n");
      }
    } catch (...) {
      std::fprintf(stderr, "warning: cuLibraryUnload failed during DeepGEMM shutdown\n");
    }
  }
}

CUfunction KernelRuntime::kernel() const {
  return kernel_;
}

}  // namespace deepgemm_rs
