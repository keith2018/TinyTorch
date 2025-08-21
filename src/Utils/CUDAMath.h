/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#ifdef USE_CUDA

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "Macros.h"

namespace tinytorch::cuda {

// ======================= limits ====================
template <typename T>
HOST_DEVICE static T max() {
  return std::numeric_limits<T>::max();
}

// ======================= min =======================
template <typename T>
HOST_DEVICE T min(T a, T b) {
  return a < b ? a : b;
}

// ======================= max =======================
template <typename T>
HOST_DEVICE T max(T a, T b) {
  return a > b ? a : b;
}

// ======================= exp =======================
template <typename T>
HOST_DEVICE T exp(T x);

// float
template <>
HOST_DEVICE inline float exp<float>(float x) {
#ifdef __CUDA_ARCH__
  return expf(x);
#else
  return ::expf(x);
#endif
}

// __half
template <>
HOST_DEVICE inline __half exp<__half>(__half x) {
#ifdef __CUDA_ARCH__
  return hexp(x);
#else
  return __float2half(::expf(__half2float(x)));
#endif
}

// __nv_bfloat16
template <>
HOST_DEVICE inline __nv_bfloat16 exp<__nv_bfloat16>(__nv_bfloat16 x) {
#ifdef __CUDA_ARCH__
  return hexp(x);
#else
  return __float2bfloat16(::expf(__bfloat162float(x)));
#endif
}

// ======================= log =======================
template <typename T>
HOST_DEVICE T log(T x);

template <>
HOST_DEVICE inline float log<float>(float x) {
#ifdef __CUDA_ARCH__
  return logf(x);
#else
  return ::logf(x);
#endif
}

template <>
HOST_DEVICE inline __half log<__half>(__half x) {
#ifdef __CUDA_ARCH__
  return hlog(x);
#else
  return __float2half(::logf(__half2float(x)));
#endif
}

template <>
HOST_DEVICE inline __nv_bfloat16 log<__nv_bfloat16>(__nv_bfloat16 x) {
#ifdef __CUDA_ARCH__
  return hlog(x);
#else
  return __float2bfloat16(::logf(__bfloat162float(x)));
#endif
}

// ======================= rsqrt =======================
template <typename T>
HOST_DEVICE T rsqrt(T x);

template <>
HOST_DEVICE inline float rsqrt<float>(float x) {
#ifdef __CUDA_ARCH__
  return rsqrtf(x);
#else
  return 1.0f / ::sqrtf(x);
#endif
}

template <>
HOST_DEVICE inline __half rsqrt<__half>(__half x) {
#ifdef __CUDA_ARCH__
#if CUDART_VERSION >= 11020
  return hrsqrt(x);
#else
  return __float2half(rsqrtf(__half2float(x)));
#endif
#else
  return __float2half(1.0f / ::sqrtf(__half2float(x)));
#endif
}

template <>
HOST_DEVICE inline __nv_bfloat16 rsqrt<__nv_bfloat16>(__nv_bfloat16 x) {
#ifdef __CUDA_ARCH__
  return __float2bfloat16(rsqrtf(__bfloat162float(x)));
#else
  return __float2bfloat16(1.0f / ::sqrtf(__bfloat162float(x)));
#endif
}

}  // namespace tinytorch::cuda

#endif
