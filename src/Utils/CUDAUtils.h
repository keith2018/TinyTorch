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
      abort();                                                                                                        \
    }                                                                                                                 \
  } while (0)

#define CURAND_CHECK(call)                                                                                            \
  do {                                                                                                                \
    curandStatus_t err = call;                                                                                        \
    if (err != CURAND_STATUS_SUCCESS) {                                                                               \
      std::cerr << "CURAND error in file '" << __FILE__ << "' in line " << __LINE__ << ": "                           \
                << tinytorch::cuda::curandGetErrorString(err) << " (" << err << ")" << std::endl;                     \
      abort();                                                                                                        \
    }                                                                                                                 \
  } while (0)

#define CUBLAS_CHECK(call)                                                                                            \
  do {                                                                                                                \
    cublasStatus_t err = call;                                                                                        \
    if (err != CUBLAS_STATUS_SUCCESS) {                                                                               \
      std::cerr << "CUBLAS error in file '" << __FILE__ << "' in line " << __LINE__ << ": "                           \
                << tinytorch::cuda::cublasGetErrorString(err) << " (" << err << ")" << std::endl;                     \
      abort();                                                                                                        \
    }                                                                                                                 \
  } while (0)

#define CUDA_KERNEL_CHECK()                                                                                           \
  do {                                                                                                                \
    cudaError_t err = cudaGetLastError();                                                                             \
    if (err != cudaSuccess) {                                                                                         \
      std::cerr << "CUDA kernel error in file '" << __FILE__ << "' in line " << __LINE__ << ": "                      \
                << cudaGetErrorString(err) << " (" << err << ")" << std::endl;                                        \
      abort();                                                                                                        \
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
void setCurrentDevice(int device);
int getCurrentDevice();

class CudaDeviceGuard {
 public:
  explicit CudaDeviceGuard(int newIndex);
  ~CudaDeviceGuard();

  CudaDeviceGuard(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

 private:
  int oldIndex_ = -1;
  bool switched_ = false;
};

// stream utils
struct CUDAStream {
  cudaStream_t stream;
  int deviceIdx;

  CUDAStream() : stream(nullptr), deviceIdx(-1) {}
  CUDAStream(cudaStream_t s, int d) : stream(s), deviceIdx(d) {}

  bool operator==(const CUDAStream& other) const { return stream == other.stream && deviceIdx == other.deviceIdx; }
};

void setCurrentCUDAStream(CUDAStream stream, int device);
CUDAStream getCurrentCUDAStream(int device);

cublasHandle_t getCublasHandle(int device);

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

template <typename T>
__host__ __device__ static T Inf();

template <>
inline __host__ __device__ uint8_t Inf<uint8_t>() {
  return UINT8_MAX;
}

template <>
inline __host__ __device__ uint16_t Inf<uint16_t>() {
  return UINT16_MAX;
}

template <>
inline __host__ __device__ int32_t Inf<int32_t>() {
  return INT32_MAX;
}

template <>
inline __host__ __device__ int64_t Inf<int64_t>() {
  return INT64_MAX;
}

template <>
inline __host__ __device__ float Inf<float>() {
  return FLT_MAX;
}
#endif

bool deviceAvailable();

}  // namespace tinytorch::cuda
