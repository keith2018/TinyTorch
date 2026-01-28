/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "CUDAUtils.h"

#include <vector>

#include "Logger.h"
#include "Tensor.h"

namespace tinytorch::cuda {

#ifdef USE_CUDA

constexpr int defaultKernelBlockSize = 512;

int getDeviceCount() {
  static int cnt = -1;
  if (cnt == -1) {
    CUDA_CHECK(cudaGetDeviceCount(&cnt));
  }
  return cnt;
}

static int getMaxDevices() {
  static int maxDevices = getDeviceCount();
  return maxDevices;
}

CudaDeviceGuard::CudaDeviceGuard(int newIndex) {
  ASSERT(newIndex >= 0 && newIndex < getMaxDevices());
  CUDA_CHECK(cudaGetDevice(&oldIndex_));
  if (newIndex != oldIndex_) {
    CUDA_CHECK(cudaSetDevice(newIndex));
    switched_ = true;
  } else {
    switched_ = false;
  }
}

CudaDeviceGuard::~CudaDeviceGuard() {
  if (switched_) {
    CUDA_CHECK(cudaSetDevice(oldIndex_));
  }
}

thread_local std::vector<CUDAStream>* currentStreams = nullptr;
thread_local std::vector<cublasHandle_t>* cublasHandles = nullptr;
static std::vector<int>* gpuComputeCapabilities = nullptr;

static std::vector<CUDAStream>& getCurrentStreamsVec() {
  if (!currentStreams) {
    currentStreams = new std::vector<CUDAStream>(getMaxDevices());
  }
  return *currentStreams;
}

static std::vector<cublasHandle_t>& getCublasHandlesVec() {
  if (!cublasHandles) {
    cublasHandles = new std::vector<cublasHandle_t>(getMaxDevices(), nullptr);
  }
  return *cublasHandles;
}

static std::vector<int>& getGpuComputeCapabilitiesVec() {
  if (!gpuComputeCapabilities) {
    gpuComputeCapabilities = new std::vector<int>(getMaxDevices(), -1);
  }
  return *gpuComputeCapabilities;
}

struct CublasHandleInitializer {
  CublasHandleInitializer() = default;
  ~CublasHandleInitializer() {
    auto& handles = getCublasHandlesVec();
    for (auto i = 0; i < getMaxDevices(); i++) {
      if (handles[i]) {
        CUBLAS_CHECK(cublasDestroy(handles[i]));
        handles[i] = nullptr;
      }
    }
  }
};
thread_local CublasHandleInitializer _cublasHandleInit;

void setDevice(int device) {
  if (device < 0 || device >= getMaxDevices()) {
    LOGE("setCurrentDevice: invalid device: %d", device);
    return;
  }
  CUDA_CHECK(cudaSetDevice(device));
}

Device getCurrentDevice() {
  int currentDevice = -1;
  CUDA_CHECK(cudaGetDevice(&currentDevice));
  return {DeviceType::CUDA, static_cast<DeviceIndex>(currentDevice)};
}

void CUDAStream::waitStream(const CUDAStream& other) const {
  if (!valid() || !other.valid()) {
    LOGE("Invalid CUDAStream in waitStream");
    return;
  }
  if (deviceIdx_ != other.deviceIdx_) {
    LOGE("Cannot wait on a stream from a different device");
    return;
  }

  CudaDeviceGuard guard(deviceIdx_);
  CUDAEvent event(deviceIdx_, cudaEventDisableTiming);
  event.record(other);
  event.block(*this);
}

CUDAStream createCUDAStream(int device) { return CUDAStream(device); }

CUDAEvent createCUDAEvent(int device, unsigned int flags) { return CUDAEvent(device, flags); }

CUDAStream& getCurrentCUDAStream(int device) {
  if (device < 0 || device >= getMaxDevices()) {
    LOGE("getCurrentCUDAStream: invalid device: %d", device);
    static CUDAStream empty{};
    return empty;
  }
  auto& streams = getCurrentStreamsVec();
  auto& stream = streams[device];
  if (!stream.valid()) {
    stream = createCUDAStream(device);
  }
  return stream;
}

cublasHandle_t& getCublasHandle(int device) {
  if (device < 0 || device >= getMaxDevices()) {
    LOGE("getCublasHandle: invalid device: %d", device);
    static cublasHandle_t empty = nullptr;
    return empty;
  }
  auto& handles = getCublasHandlesVec();
  cublasHandle_t& handle = handles[device];
  if (!handle) {
    CudaDeviceGuard guard(device);
    CUBLAS_CHECK(cublasCreate(&handle));
    int cc = getGpuComputeCapability(device);
    if (cc >= 80) {
      // TF32 Tensor Core
      CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    } else if (cc >= 70) {
      // Tensor Core
      CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    }
  }
  // bind to stream
  CUDAStream& s = getCurrentCUDAStream(device);
  CUBLAS_CHECK(cublasSetStream(handle, s.stream()));
  return handle;
}

const cudaDeviceProp& getDeviceProperties(int device) {
  static std::vector<cudaDeviceProp> cache(getMaxDevices());
  static std::vector<bool> isCached(getMaxDevices(), false);

  if (device < 0 || device >= getMaxDevices()) {
    LOGE("getDeviceProperties: invalid device: %d", device);
    static cudaDeviceProp empty;
    return empty;
  }

  if (!isCached[device]) {
    CUDA_CHECK(cudaGetDeviceProperties(&cache[device], device));
    isCached[device] = true;
  }
  return cache[device];
}

int getGpuComputeCapability(int device) {
  if (device < 0 || device >= getMaxDevices()) {
    return 0;
  }
  auto& capabilities = getGpuComputeCapabilitiesVec();
  if (capabilities[device] < 0) {
    const auto& props = getDeviceProperties(device);
    capabilities[device] = props.major * 10 + props.minor;
  }
  return capabilities[device];
}

int getMaxThreadsPerBlock(int device) { return getDeviceProperties(device).maxThreadsPerBlock; }

size_t getMaxSharedMemoryPerBlock(int device) { return getDeviceProperties(device).sharedMemPerBlock; }

unsigned int getKernelBlockSize(int device) {
  auto maxSize = getMaxThreadsPerBlock(device);
  return std::min(defaultKernelBlockSize, maxSize);
}

unsigned int getKernelGridSize(unsigned int blockSize, int64_t n, size_t batch) {
  return (n + (blockSize * batch) - 1) / (blockSize * batch);
}

KernelLaunchParams getKernelLaunchParams(int device, int64_t n, size_t batch, size_t sharedMemBytes) {
  auto blockSize = cuda::getKernelBlockSize(device);
  auto gridSize = cuda::getKernelGridSize(blockSize, n, batch);
  auto stream = cuda::getCurrentCUDAStream(device).stream();
  return {gridSize, blockSize, sharedMemBytes, stream};
}

TensorCudaCtx getTensorCudaCtx(const Tensor& t) {
  TensorCudaCtx ret{};
  ret.ndim = t.dim();
  ret.numel = t.numel();
  std::memcpy(ret.shape, t.shape().data(), t.dim() * sizeof(int64_t));
  std::memcpy(ret.strides, t.strides().data(), t.dim() * sizeof(int64_t));
  ret.data = const_cast<Tensor&>(t).dataPtr<>();
  return ret;
}

const char* curandGetErrorString(curandStatus_t status) {
  switch (status) {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown cuRAND error";
}

const char* cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "Unknown cuBLAS error";
}
#endif

bool deviceAvailable() {
#ifdef USE_CUDA
  return getDeviceCount() > 0;
#else
  return false;
#endif
}

}  // namespace tinytorch::cuda