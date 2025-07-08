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

constexpr int defaultKernelBlockSize = 512;

size_t getDeviceCount() {
  static int cnt = -1;
  if (cnt == -1) {
    CUDA_CHECK(cudaGetDeviceCount(&cnt));
  }
  return cnt;
}

bool deviceAvailable() { return getDeviceCount() > 0; }

static auto kMaxDevices = getDeviceCount();

CudaDeviceGuard::CudaDeviceGuard(int newIndex) {
  if (newIndex < 0 || newIndex >= kMaxDevices) {
    return;
  }
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

thread_local int currentDevice = 0;
thread_local std::vector<CUDAStream> currentStreams(kMaxDevices);
thread_local std::vector<cublasHandle_t> cublasHandles(kMaxDevices);

struct DefaultStreamInitializer {
  DefaultStreamInitializer() {
    for (auto i = 0; i < kMaxDevices; i++) {
      currentStreams[i] = CUDAStream(nullptr, i);
    }
  }
};
thread_local DefaultStreamInitializer _streamInit;

struct CublasHandleInitializer {
  CublasHandleInitializer() {
    for (auto i = 0; i < kMaxDevices; i++) {
      cublasHandles[i] = nullptr;
    }
  }
  ~CublasHandleInitializer() {
    for (auto i = 0; i < kMaxDevices; i++) {
      if (cublasHandles[i]) {
        CUBLAS_CHECK(cublasDestroy(cublasHandles[i]));
        cublasHandles[i] = nullptr;
      }
    }
  }
};
thread_local CublasHandleInitializer _cublasHandleInit;

void setCurrentDevice(int device) {
  if (device < 0 || device >= kMaxDevices) {
    LOGE("setCurrentDevice: invalid device: %d", device);
    return;
  }
  currentDevice = device;
}

int getCurrentDevice() { return currentDevice; }

void setCurrentCUDAStream(CUDAStream stream, int device) {
  if (device < 0 || device >= kMaxDevices) {
    LOGE("setCurrentCUDAStream: invalid device: %d", device);
    return;
  }
  currentStreams[device] = stream;
}

CUDAStream getCurrentCUDAStream(int device) {
  if (device < 0 || device >= kMaxDevices) {
    LOGE("getCurrentCUDAStream: invalid device: %d", device);
    ASSERT(false);
    return {};
  }
  return currentStreams[device];
}

cublasHandle_t getCublasHandle(int device) {
  if (device < 0 || device >= kMaxDevices) {
    LOGE("getCublasHandle: invalid device: %d", device);
    return nullptr;
  }
  cublasHandle_t& handle = cublasHandles[device];
  if (!handle) {
    CUBLAS_CHECK(cublasCreate(&handle));
  }
  // bind to stream
  CUDAStream s = getCurrentCUDAStream(device);
  CUBLAS_CHECK(cublasSetStream(handle, s.stream));
  return handle;
}

int getMaxThreadsPerBlock(int device) {
  static std::vector<int> cache(kMaxDevices, -1);

  if (device < 0 || device >= kMaxDevices) {
    LOGE("getMaxThreadsPerBlock: invalid device: %d", device);
    return 0;
  }

  if (cache[device] == -1) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    cache[device] = prop.maxThreadsPerBlock;
  }
  return cache[device];
}

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
  auto stream = cuda::getCurrentCUDAStream(device).stream;
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

}  // namespace tinytorch::cuda