/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#ifdef USE_CUDA

#include "Macros.h"

namespace tinytorch::cuda {

template <typename T>
HOST_DEVICE T abs(T x) {
  return x >= T(0) ? x : -x;
}

template <typename T>
HOST_DEVICE T sign(T x) {
  return (x > T(0)) ? T(1) : ((x < T(0)) ? T(-1) : T(0));
}

template <typename T>
HOST_DEVICE T min(T a, T b) {
  return a < b ? a : b;
}

template <typename T>
HOST_DEVICE T max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
HOST_DEVICE T sin(T x) {
  return static_cast<T>(sinf(static_cast<float>(x)));
}

template <typename T>
HOST_DEVICE T cos(T x) {
  return static_cast<T>(cosf(static_cast<float>(x)));
}

template <typename T>
HOST_DEVICE T tanh(T x) {
  return static_cast<T>(tanhf(static_cast<float>(x)));
}

template <typename T>
HOST_DEVICE T sqrt(T x) {
  return static_cast<T>(sqrtf(static_cast<float>(x)));
}

template <typename T>
HOST_DEVICE T rsqrt(T x) {
#ifdef __CUDA_ARCH__
  return static_cast<T>(rsqrtf(static_cast<float>(x)));
#else
  return static_cast<T>(1.f / sqrtf(static_cast<float>(x)));
#endif
}

template <typename T>
HOST_DEVICE T pow(T a, T b) {
  return static_cast<T>(powf(static_cast<float>(a), static_cast<float>(b)));
}

template <typename T>
HOST_DEVICE T exp(T x) {
  return static_cast<T>(expf(static_cast<float>(x)));
}

template <typename T>
HOST_DEVICE T log(T x) {
  return static_cast<T>(logf(static_cast<float>(x)));
}

}  // namespace tinytorch::cuda

#endif
