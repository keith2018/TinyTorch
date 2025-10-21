/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#endif

#include <cfloat>
#include <iostream>

#include "CUDAMath.h"
#include "CUDATypes.h"
#include "Macros.h"
#include "Tensor/DType.h"
#include "Tensor/Device.h"

namespace tinytorch {
class Tensor;
}  // namespace tinytorch

namespace tinytorch::cuda {

#ifdef USE_CUDA

const char* curandGetErrorString(curandStatus_t status);
const char* cublasGetErrorString(cublasStatus_t status);

// clang-format off
#define CUDA_CHECK(call)                                                                                              \
  do {                                                                                                                \
    cudaError_t err = call;                                                                                           \
    if (err != cudaSuccess) {                                                                                         \
      std::cerr << "CUDA error in file '" << __FILE__ << "' in line " << __LINE__ << ": " << cudaGetErrorString(err)  \
                << " (" << err << ")" << std::endl;                                                                   \
      std::abort();                                                                                                   \
    }                                                                                                                 \
  } while (0)

#define CUDA_ERROR(err)                                                                                               \
  do {                                                                                                                \
    if ((err) != cudaSuccess) {                                                                                       \
      std::cerr << "CUDA error in file '" << __FILE__ << "' in line " << __LINE__ << ": " << cudaGetErrorString(err)  \
                << " (" << (err) << ")" << std::endl;                                                                 \
    }                                                                                                                 \
  } while (0)

#define CURAND_CHECK(call)                                                                                            \
  do {                                                                                                                \
    curandStatus_t err = call;                                                                                        \
    if (err != CURAND_STATUS_SUCCESS) {                                                                               \
      std::cerr << "CURAND error in file '" << __FILE__ << "' in line " << __LINE__ << ": "                           \
                << tinytorch::cuda::curandGetErrorString(err) << " (" << err << ")" << std::endl;                     \
      std::abort();                                                                                                   \
    }                                                                                                                 \
  } while (0)

#define CUBLAS_CHECK(call)                                                                                            \
  do {                                                                                                                \
    cublasStatus_t err = call;                                                                                        \
    if (err != CUBLAS_STATUS_SUCCESS) {                                                                               \
      std::cerr << "CUBLAS error in file '" << __FILE__ << "' in line " << __LINE__ << ": "                           \
                << tinytorch::cuda::cublasGetErrorString(err) << " (" << err << ")" << std::endl;                     \
      std::abort();                                                                                                   \
    }                                                                                                                 \
  } while (0)

#define CUDA_KERNEL_CHECK()                                                                                           \
  do {                                                                                                                \
    cudaError_t err = cudaGetLastError();                                                                             \
    if (err != cudaSuccess) {                                                                                         \
      std::cerr << "CUDA kernel error in file '" << __FILE__ << "' in line " << __LINE__ << ": "                      \
                << cudaGetErrorString(err) << " (" << err << ")" << std::endl;                                        \
      std::abort();                                                                                                   \
    }                                                                                                                 \
  } while (0)
// clang-format on

#define CUDA_WARP_SIZE 32

#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FETCH_FLOAT3(pointer) (reinterpret_cast<float3*>(&(pointer))[0])
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define FETCH_CONST_FLOAT2(pointer) (reinterpret_cast<const float2*>(&(pointer))[0])
#define FETCH_CONST_FLOAT3(pointer) (reinterpret_cast<const float3*>(&(pointer))[0])
#define FETCH_CONST_FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

// device utils
int getDeviceCount();
void setDevice(int device);
Device getCurrentDevice();

class CudaDeviceGuard {
 public:
  explicit CudaDeviceGuard(int newIndex);
  ~CudaDeviceGuard();

  CudaDeviceGuard(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard(CudaDeviceGuard&&) = delete;
  CudaDeviceGuard& operator=(CudaDeviceGuard&&) = delete;

 private:
  int oldIndex_ = -1;
  bool switched_ = false;
};

// stream utils
struct CUDAStream {
  CUDAStream() : stream_(nullptr), deviceIdx_(-1) {}

  explicit CUDAStream(int device) : stream_(nullptr), deviceIdx_(device) {
    CudaDeviceGuard guard(deviceIdx_);
    CUDA_CHECK(cudaStreamCreate(&stream_));
  }

  CUDAStream(const CUDAStream&) = delete;
  CUDAStream& operator=(const CUDAStream&) = delete;

  CUDAStream(CUDAStream&& other) noexcept : stream_(other.stream_), deviceIdx_(other.deviceIdx_) {
    other.stream_ = nullptr;
    other.deviceIdx_ = -1;
  }

  CUDAStream& operator=(CUDAStream&& other) noexcept {
    if (this != &other) {
      destroy();
      stream_ = other.stream_;
      deviceIdx_ = other.deviceIdx_;
      other.stream_ = nullptr;
      other.deviceIdx_ = -1;
    }
    return *this;
  }

  ~CUDAStream() { destroy(); }

  void synchronize() const {
    if (stream_) {
      CudaDeviceGuard guard(deviceIdx_);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
  }

  void waitStream(const CUDAStream& other) const;

  cudaStream_t stream() const { return stream_; }
  int deviceIdx() const { return deviceIdx_; }
  bool valid() const { return stream_ != nullptr; }

 private:
  void destroy() {
    if (stream_) {
      CudaDeviceGuard guard(deviceIdx_);
      CUDA_CHECK(cudaStreamDestroy(stream_));
      stream_ = nullptr;
      deviceIdx_ = -1;
    }
  }

  cudaStream_t stream_;
  int deviceIdx_;
};

// event utils
struct CUDAEvent {
  CUDAEvent() : event_(nullptr), deviceIdx_(-1) {}

  explicit CUDAEvent(int device, unsigned int flags = cudaEventDisableTiming) : event_(nullptr), deviceIdx_(device) {
    CudaDeviceGuard guard(deviceIdx_);
    CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
  }

  CUDAEvent(const CUDAEvent&) = delete;
  CUDAEvent& operator=(const CUDAEvent&) = delete;

  CUDAEvent(CUDAEvent&& other) noexcept : event_(other.event_), deviceIdx_(other.deviceIdx_) {
    other.event_ = nullptr;
    other.deviceIdx_ = -1;
  }

  CUDAEvent& operator=(CUDAEvent&& other) noexcept {
    if (this != &other) {
      destroy();
      event_ = other.event_;
      deviceIdx_ = other.deviceIdx_;
      other.event_ = nullptr;
      other.deviceIdx_ = -1;
    }
    return *this;
  }

  ~CUDAEvent() { destroy(); }

  void record(const CUDAStream& stream) const {
    if (event_ && stream.stream()) {
      CUDA_CHECK(cudaEventRecord(event_, stream.stream()));
    }
  }

  void block(const CUDAStream& stream) const {
    if (event_ && stream.stream()) {
      CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), event_));
    }
  }

  bool query() const {
    if (!event_) {
      return false;
    }

    auto err = cudaEventQuery(event_);
    if (err == cudaSuccess) {
      return true;
    }
    if (err == cudaErrorNotReady) {
      return false;
    }
    CUDA_ERROR(err);
    return false;
  }

  cudaEvent_t event() const { return event_; }
  int deviceIdx() const { return deviceIdx_; }
  bool valid() const { return event_ != nullptr; }

 private:
  void destroy() {
    if (event_) {
      CudaDeviceGuard guard(deviceIdx_);
      CUDA_CHECK(cudaEventDestroy(event_));
      event_ = nullptr;
      deviceIdx_ = -1;
    }
  }

  cudaEvent_t event_;
  int deviceIdx_;
};

CUDAStream createCUDAStream(int device);
CUDAEvent createCUDAEvent(int device, unsigned int flags = cudaEventDisableTiming);

CUDAStream& getCurrentCUDAStream(int device);
cublasHandle_t& getCublasHandle(int device);

// kernel launch utils
struct KernelLaunchParams {
  dim3 grid;
  dim3 block;
  size_t sharedMemBytes = 0;
  cudaStream_t stream = nullptr;
};

int getMaxThreadsPerBlock(int device);
size_t getMaxSharedMemoryPerBlock(int device);
unsigned int getKernelBlockSize(int device);
unsigned int getKernelGridSize(unsigned int blockSize, int64_t n, size_t batch = 1);
KernelLaunchParams getKernelLaunchParams(int device, int64_t n, size_t batch = 1, size_t sharedMemBytes = 0);

#define CUDA_LAUNCH_KERNEL(KERNEL, PARAMS, ...)                                                     \
  KERNEL<<<(PARAMS).grid, (PARAMS).block, (PARAMS).sharedMemBytes, (PARAMS).stream>>>(__VA_ARGS__); \
  CUDA_KERNEL_CHECK();

struct ALIGN(16) TensorCudaCtx {
  int64_t ndim;
  int64_t numel;
  int64_t shape[MAX_TENSOR_DIM];
  int64_t strides[MAX_TENSOR_DIM];
  void* data;
};

TensorCudaCtx getTensorCudaCtx(const Tensor& t);

#endif

bool deviceAvailable();

}  // namespace tinytorch::cuda
