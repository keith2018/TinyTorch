/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "OpElemWiseCuda.cuh"
#include "OpFused.h"
#include "Utils/CUDAUtils.h"

namespace tinytorch::op {

template <typename T>
__global__ void kSiluMul(T* retPtr, const T* selfPtr, const int64_t halfLastDim, const int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const int64_t sliceIdx = index / halfLastDim;
    const int64_t innerIdx = index % halfLastDim;

    const int64_t gateIdx = sliceIdx * halfLastDim * 2 + innerIdx;
    const int64_t upIdx = gateIdx + halfLastDim;

    const T gateVal = selfPtr[gateIdx];
    const T upVal = selfPtr[upIdx];
    retPtr[index] = OpCudaSilu::apply(gateVal) * upVal;
  }
}

template <typename T>
Tensor siluMulOpCudaImpl(const Tensor& self) {
  ASSERT(self.size(-1) % 2 == 0);
  SizeVector retShape = self.shape();
  retShape.back() /= 2;
  auto ret = Tensor::empty(retShape, self.options().noGrad());

  using CudaT = typename cuda::CudaTypeCast<T>::type;
  const CudaT* selfPtr = self.dataPtr<CudaT>();
  CudaT* retPtr = ret.dataPtr<CudaT>();

  const int64_t lastDim = self.size(-1);
  const int64_t halfLastDim = lastDim / 2;
  const int64_t n = ret.numel();
  auto params = cuda::getKernelLaunchParams(self.device().index, n);
  CUDA_LAUNCH_KERNEL(kSiluMul<CudaT>, params, retPtr, selfPtr, halfLastDim, n);
  return ret;
}

}  // namespace tinytorch::op
