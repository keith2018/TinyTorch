/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "OpFilling.h"
#include "Utils/CUDAUtils.h"
#include "Utils/RandomGenerator.h"

namespace tinytorch::op {

template <typename T>
__global__ void kFill(T* t, const T val, const int64_t n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index + 3 < n) {
    t[index] = val;
    t[index + 1] = val;
    t[index + 2] = val;
    t[index + 3] = val;
  } else {
    if (index < n) t[index] = val;
    if (index + 1 < n) t[index + 1] = val;
    if (index + 2 < n) t[index + 2] = val;
  }
}

template <typename T>
__global__ void kFill(T* t, const T* valPtr, const int64_t n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const T val = *valPtr;
  if (index + 3 < n) {
    t[index] = val;
    t[index + 1] = val;
    t[index + 2] = val;
    t[index + 3] = val;
  } else {
    if (index < n) t[index] = val;
    if (index + 1 < n) t[index + 1] = val;
    if (index + 2 < n) t[index + 2] = val;
  }
}

template <typename T>
__global__ void kFillMasked(T* outPtr, const T* selfPtr, const DTypeToType_t<DType::Bool>* maskPtr, const T val,
                            const int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    outPtr[index] = maskPtr[index] ? val : selfPtr[index];
  }
}

template <typename T>
__global__ void kFillLinSpace(T* dst, const T start, const T step, const int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    dst[index] = start + static_cast<T>(index) * step;
  }
}

template <typename T>
__global__ void kFillRandUniform(T* t, const float minVal, const float maxVal, const unsigned long seed,
                                 const unsigned long seq, const int64_t n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index < n) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, seq, index, &state);
    const auto rand = curand_uniform4(&state);
    const auto range = maxVal - minVal;

    if (index + 3 < n) {
      t[index] = static_cast<T>(rand.x * range + minVal);
      t[index + 1] = static_cast<T>(rand.y * range + minVal);
      t[index + 2] = static_cast<T>(rand.z * range + minVal);
      t[index + 3] = static_cast<T>(rand.w * range + minVal);
    } else {
      if (index < n) t[index] = static_cast<T>(rand.x * range + minVal);
      if (index + 1 < n) t[index + 1] = static_cast<T>(rand.y * range + minVal);
      if (index + 2 < n) t[index + 2] = static_cast<T>(rand.z * range + minVal);
    }
  }
}

template <typename T>
__global__ void kFillRandNormal(T* t, const float mean, const float stddev, const unsigned long seed,
                                const unsigned long seq, const int64_t n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index < n) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, seq, index, &state);
    const auto rand = curand_normal4(&state);

    if (index + 3 < n) {
      t[index] = static_cast<T>(rand.x * stddev + mean);
      t[index + 1] = static_cast<T>(rand.y * stddev + mean);
      t[index + 2] = static_cast<T>(rand.z * stddev + mean);
      t[index + 3] = static_cast<T>(rand.w * stddev + mean);
    } else {
      if (index < n) t[index] = static_cast<T>(rand.x * stddev + mean);
      if (index + 1 < n) t[index + 1] = static_cast<T>(rand.y * stddev + mean);
      if (index + 2 < n) t[index + 2] = static_cast<T>(rand.z * stddev + mean);
    }
  }
}

template <typename T>
__global__ void kFillRandBernoulli(T* t, const float p, const unsigned long seed, const unsigned long seq,
                                   const int64_t n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index < n) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, seq, index, &state);
    const auto rand = curand_uniform4(&state);

    if (index + 3 < n) {
      t[index] = rand.x < p ? T(1.f) : T(0.f);
      t[index + 1] = rand.y < p ? T(1.f) : T(0.f);
      t[index + 2] = rand.z < p ? T(1.f) : T(0.f);
      t[index + 3] = rand.w < p ? T(1.f) : T(0.f);
    } else {
      if (index < n) t[index] = rand.x < p ? T(1.f) : T(0.f);
      if (index + 1 < n) t[index + 1] = rand.y < p ? T(1.f) : T(0.f);
      if (index + 2 < n) t[index + 2] = rand.z < p ? T(1.f) : T(0.f);
    }
  }
}

template <typename T>
void fillOpOffsetCudaImpl(Tensor& self, const Tensor& val, int64_t offset, int64_t count) {
  ASSERT(val.isScalar());
  self.copyOnWrite();
  auto* selfPtr = self.dataPtr<T>();
  const auto* valPtr = val.dataPtr<T>();

  auto params = cuda::getKernelLaunchParams(self.device().index, count, 4);
  if (val.device().isCpu()) {
    CUDA_LAUNCH_KERNEL(kFill<T>, params, selfPtr + offset, *valPtr, count);
  } else {
    CUDA_LAUNCH_KERNEL(kFill<T>, params, selfPtr + offset, valPtr, count);
  }
}

template <typename T>
void fillOpMaskedCudaImplDetail(Tensor& out, const Tensor& self, const Tensor& mask, const Scalar& val) {
  T* outPtr = out.dataPtr<T>();
  const T* selfPtr = self.dataPtr<T>();
  const auto* maskPtr = mask.dataPtr<DTypeToType_t<DType::Bool>>();
  T valToFill = val.to<T>();

  int64_t n = self.numel();
  auto params = cuda::getKernelLaunchParams(self.device().index, n);
  CUDA_LAUNCH_KERNEL(kFillMasked<T>, params, outPtr, selfPtr, maskPtr, valToFill, n);
}

template <typename T>
void fillOpCudaImpl(Tensor& self, const Scalar& val) {
  self.copyOnWrite();
  auto* selfPtr = self.dataPtr<T>();
  auto params = cuda::getKernelLaunchParams(self.device().index, self.numel(), 4);
  CUDA_LAUNCH_KERNEL(kFill<T>, params, selfPtr, val.to<T>(), self.numel());
}

template <typename T>
Tensor fillOpMaskedCudaImpl(const Tensor& self, const Tensor& mask, const Scalar& val) {
  ASSERT(mask.dtype() == DType::Bool);
  ASSERT(mask.shape() == self.shape());
  Tensor out(self.shape(), self.options().noGrad());
  fillOpMaskedCudaImplDetail<T>(out, self, mask, val);
  return out;
}

template <typename T>
void fillOpMaskedOutCudaImpl(Tensor& out, const Tensor& self, const Tensor& mask, const Scalar& val) {
  ASSERT(mask.dtype() == DType::Bool);
  ASSERT(self.shape() == mask.shape());
  out.copyOnWrite();
  fillOpMaskedCudaImplDetail<T>(out, self, mask, val);
}

template <typename T>
void fillOpMaskedInplaceCudaImpl(Tensor& self, const Tensor& mask, const Scalar& val) {
  ASSERT(mask.dtype() == DType::Bool);
  ASSERT(mask.shape() == self.shape());
  self.copyOnWrite();
  fillOpMaskedCudaImplDetail<T>(self, self, mask, val);
}

template <typename T>
void fillOpLinSpaceCudaImpl(Tensor& self, const Scalar& start, const Scalar& step, int64_t steps) {
  ASSERT(self.numel() == steps);
  self.copyOnWrite();
  auto* selfPtr = self.dataPtr<T>();
  T startVal = start.to<T>();
  T stepVal = step.to<T>();
  int64_t n = self.numel();

  auto params = cuda::getKernelLaunchParams(self.device().index, n);
  CUDA_LAUNCH_KERNEL(kFillLinSpace, params, selfPtr, startVal, stepVal, n);
}

template <typename T>
void fillOpRandUniformCudaImpl(Tensor& self, float min, float max) {
  self.copyOnWrite();
  auto* selfPtr = self.dataPtr<T>();
  auto seed = RandomGeneratorCUDA::getSeed();
  auto seq = RandomGeneratorCUDA::nextSequence();
  int64_t n = self.numel();

  auto params = cuda::getKernelLaunchParams(self.device().index, n, 4);
  CUDA_LAUNCH_KERNEL(kFillRandUniform<T>, params, selfPtr, min, max, seed, seq, n);
}

template <typename T>
void fillOpRandNormalCudaImpl(Tensor& self, float mean, float stddev) {
  self.copyOnWrite();
  auto* selfPtr = self.dataPtr<T>();
  auto seed = RandomGeneratorCUDA::getSeed();
  auto seq = RandomGeneratorCUDA::nextSequence();
  int64_t n = self.numel();

  auto params = cuda::getKernelLaunchParams(self.device().index, n, 4);
  CUDA_LAUNCH_KERNEL(kFillRandNormal<T>, params, selfPtr, mean, stddev, seed, seq, n);
}

template <typename T>
void fillOpRandBernoulliCudaImpl(Tensor& self, float p) {
  self.copyOnWrite();
  auto* selfPtr = self.dataPtr<T>();
  auto seed = RandomGeneratorCUDA::getSeed();
  auto seq = RandomGeneratorCUDA::nextSequence();
  int64_t n = self.numel();

  auto params = cuda::getKernelLaunchParams(self.device().index, n, 4);
  CUDA_LAUNCH_KERNEL(kFillRandBernoulli<T>, params, selfPtr, p, seed, seq, n);
}

}  // namespace tinytorch::op
