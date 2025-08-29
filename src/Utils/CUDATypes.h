/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#ifdef USE_CUDA

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "Utils/BFloat16.h"
#include "Utils/Half.h"

namespace tinytorch::cuda {

// type cast
template <typename T>
struct CudaTypeCast {
  static_assert(sizeof(T) == 0, "Unsupported type for CUDA");
};
template <>
struct CudaTypeCast<float> {
  using type = float;
};
template <>
struct CudaTypeCast<Half> {
  using type = __half;
};
template <>
struct CudaTypeCast<BFloat16> {
  using type = __nv_bfloat16;
};
template <>
struct CudaTypeCast<__half> {
  using type = __half;
};
template <>
struct CudaTypeCast<__nv_bfloat16> {
  using type = __nv_bfloat16;
};
template <>
struct CudaTypeCast<int32_t> {
  using type = int32_t;
};
template <>
struct CudaTypeCast<int64_t> {
  using type = int64_t;
};
template <>
struct CudaTypeCast<uint8_t> {
  using type = uint8_t;
};

// compute type
template <typename T>
struct CudaComputeType {
  static_assert(sizeof(T) == 0, "Unsupported type for CUDA");
};
template <>
struct CudaComputeType<float> {
  using type = float;
};
template <>
struct CudaComputeType<Half> {
  using type = float;
};
template <>
struct CudaComputeType<BFloat16> {
  using type = float;
};
template <>
struct CudaComputeType<__half> {
  using type = float;
};
template <>
struct CudaComputeType<__nv_bfloat16> {
  using type = float;
};
template <>
struct CudaComputeType<int32_t> {
  using type = int32_t;
};
template <>
struct CudaComputeType<int64_t> {
  using type = int64_t;
};
template <>
struct CudaComputeType<uint8_t> {
  using type = uint8_t;
};

}  // namespace tinytorch::cuda

#endif
