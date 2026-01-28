/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdio>

namespace tfa {

#define TFA_CUDA_CHECK(call)                                                                  \
  do {                                                                                        \
    cudaError_t err = call;                                                                   \
    if (err != cudaSuccess) {                                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE);                                                                     \
    }                                                                                         \
  } while (0)

template <typename T>
__host__ __device__ __forceinline__ constexpr T ceilDiv(T m, T n) {
  return (m + n - 1) / n;
}

template <typename T>
__host__ __device__ __forceinline__ float toFloat(T val) {
  return static_cast<float>(val);
}

template <>
__host__ __device__ __forceinline__ float toFloat(__half val) {
  return __half2float(val);
}

template <>
__host__ __device__ __forceinline__ float toFloat(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <typename T>
__host__ __device__ __forceinline__ T fromFloat(float val) {
  return static_cast<T>(val);
}

template <>
__host__ __device__ __forceinline__ __half fromFloat(float val) {
  return __float2half(val);
}

template <>
__host__ __device__ __forceinline__ __nv_bfloat16 fromFloat(float val) {
  return __float2bfloat16(val);
}

__device__ __forceinline__ float fastExp(float x) { return __expf(x); }

template <int kHeadDim>
struct AttentionScale {
  // 1/sqrt(kHeadDim)
  static constexpr float value = (kHeadDim == 32)    ? 0.17677669529f
                                 : (kHeadDim == 64)  ? 0.125f
                                 : (kHeadDim == 128) ? 0.08838834764f
                                 : (kHeadDim == 256) ? 0.0625f
                                                     : 0.f;
  static_assert(value != 0.f, "Unsupported kHeadDim for AttentionScale");
};

template <typename KernelFunc>
bool setDynamicSharedMemory(KernelFunc kernel, size_t requestedSize) {
  int device;
  TFA_CUDA_CHECK(cudaGetDevice(&device));

  int maxOptin = 0;
  TFA_CUDA_CHECK(cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  if (requestedSize > static_cast<size_t>(maxOptin)) {
    fprintf(stderr,
            "Error: requested shared memory %zu exceeds "
            "cudaDevAttrMaxSharedMemoryPerBlockOptin (%d bytes)\n",
            requestedSize, maxOptin);
    return false;
  }

  int maxPerBlock = 0;
  TFA_CUDA_CHECK(cudaDeviceGetAttribute(&maxPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device));
  if (requestedSize > static_cast<size_t>(maxPerBlock)) {
    TFA_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, requestedSize));
  }
  return true;
}

}  // namespace tfa
