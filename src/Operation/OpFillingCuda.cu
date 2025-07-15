/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpFillingCuda.cuh"
#include "Utils/Macros.h"
#include "Utils/RandomGenerator.h"

namespace tinytorch::op {

__global__ void kFillRandUniform(float* t, const float minVal, const float maxVal, const unsigned long seed,
                                 const unsigned long seq, const int64_t n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index < n) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, seq, index, &state);
    const auto rand = curand_uniform4(&state);
    const auto range = maxVal - minVal;

    if (index + 3 < n) {
      FETCH_FLOAT4(t[index]) = make_float4(rand.x * range + minVal, rand.y * range + minVal, rand.z * range + minVal,
                                           rand.w * range + minVal);
    } else {
      if (index < n) t[index] = rand.x * range + minVal;
      if (index + 1 < n) t[index + 1] = rand.y * range + minVal;
      if (index + 2 < n) t[index + 2] = rand.z * range + minVal;
    }
  }
}

__global__ void kFillRandNormal(float* t, const float mean, const float stddev, const unsigned long seed,
                                const unsigned long seq, const int64_t n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index < n) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, seq, index, &state);
    const auto rand = curand_normal4(&state);

    if (index + 3 < n) {
      FETCH_FLOAT4(t[index]) =
          make_float4(rand.x * stddev + mean, rand.y * stddev + mean, rand.z * stddev + mean, rand.w * stddev + mean);
    } else {
      if (index < n) t[index] = rand.x * stddev + mean;
      if (index + 1 < n) t[index + 1] = rand.y * stddev + mean;
      if (index + 2 < n) t[index + 2] = rand.z * stddev + mean;
    }
  }
}

__global__ void kFillRandBernoulli(float* t, const float p, const unsigned long seed, const unsigned long seq,
                                   const int64_t n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index < n) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, seq, index, &state);
    const auto rand = curand_uniform4(&state);

    if (index + 3 < n) {
      FETCH_FLOAT4(t[index]) =
          make_float4(rand.x < p ? 1.f : 0.f, rand.y < p ? 1.f : 0.f, rand.z < p ? 1.f : 0.f, rand.w < p ? 1.f : 0.f);
    } else {
      if (index < n) t[index] = rand.x < p ? 1.f : 0.f;
      if (index + 1 < n) t[index + 1] = rand.y < p ? 1.f : 0.f;
      if (index + 2 < n) t[index + 2] = rand.z < p ? 1.f : 0.f;
    }
  }
}

void fillOpRandUniformCudaImpl(Tensor& self, float min, float max) {
  ASSERT(self.dtype() == DType::Float32);
  self.copyOnWrite();
  auto* selfPtr = self.dataPtr<float>();
  auto seed = RandomGeneratorCUDA::getSeed();
  auto seq = RandomGeneratorCUDA::nextSequence();
  int64_t n = self.numel();

  auto params = cuda::getKernelLaunchParams(self.device().index, n, 4);
  CUDA_LAUNCH_KERNEL(kFillRandUniform, params, selfPtr, min, max, seed, seq, n);
}

void fillOpRandNormalCudaImpl(Tensor& self) {
  ASSERT(self.dtype() == DType::Float32);
  self.copyOnWrite();
  auto* selfPtr = self.dataPtr<float>();
  auto seed = RandomGeneratorCUDA::getSeed();
  auto seq = RandomGeneratorCUDA::nextSequence();
  int64_t n = self.numel();

  auto params = cuda::getKernelLaunchParams(self.device().index, n, 4);
  CUDA_LAUNCH_KERNEL(kFillRandNormal, params, selfPtr, 0.f, 1.f, seed, seq, n);
}

void fillOpRandBernoulliCudaImpl(Tensor& self, float p) {
  ASSERT(self.dtype() == DType::Float32);
  self.copyOnWrite();
  auto* selfPtr = self.dataPtr<float>();
  auto seed = RandomGeneratorCUDA::getSeed();
  auto seq = RandomGeneratorCUDA::nextSequence();
  int64_t n = self.numel();

  auto params = cuda::getKernelLaunchParams(self.device().index, n, 4);
  CUDA_LAUNCH_KERNEL(kFillRandBernoulli, params, selfPtr, p, seed, seq, n);
}

void registerFillCudaFloat32() {
  REGISTER_OP_IMPL(fillRandUniform, CUDA, Float32, &fillOpRandUniformCudaImpl);
  REGISTER_OP_IMPL(fillRandNormal, CUDA, Float32, &fillOpRandNormalCudaImpl);
  REGISTER_OP_IMPL(fillRandBernoulli, CUDA, Float32, &fillOpRandBernoulliCudaImpl);
}

void registerFillCuda() {
  REGISTER_OP_IMPL_DTYPE_TPL(fill, CUDA, fillOpCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(fillOffset, CUDA, fillOpOffsetCudaImpl);

  REGISTER_OP_IMPL_DTYPE_TPL(fillMasked, CUDA, fillOpMaskedCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(fillMaskedOut, CUDA, fillOpMaskedOutCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(fillMaskedInplace, CUDA, fillOpMaskedInplaceCudaImpl);

  REGISTER_OP_IMPL_DTYPE_TPL(fillLinSpace, CUDA, fillOpLinSpaceCudaImpl);

  registerFillCudaFloat32();
}

}  // namespace tinytorch::op