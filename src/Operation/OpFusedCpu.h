/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "OpElemWiseCpu.h"
#include "OpFused.h"

namespace tinytorch::op {

template <typename T>
Tensor siluMulOpCpuImpl(const Tensor& self) {
  ASSERT(self.size(-1) % 2 == 0);
  SizeVector retShape = self.shape();
  retShape.back() /= 2;
  auto ret = Tensor::empty(retShape, self.options().noGrad());

  const T* selfPtr = self.dataPtr<T>();
  T* retPtr = ret.dataPtr<T>();

  const int64_t lastDim = self.size(-1);
  const int64_t halfLastDim = lastDim / 2;
  const int64_t numSlices = self.numel() / lastDim;

  for (int64_t i = 0; i < numSlices; i++) {
    const int64_t offsetIn = i * lastDim;
    const int64_t offsetOut = i * halfLastDim;

    for (int64_t j = 0; j < halfLastDim; j++) {
      const T gateVal = selfPtr[offsetIn + j];
      const T upVal = selfPtr[offsetIn + j + halfLastDim];
      retPtr[offsetOut + j] = OpCpuSilu::apply(gateVal) * upVal;
    }
  }
  return ret;
}

template <typename T>
void fusedAddRmsNormOpCpuImpl(Tensor& input, Tensor& residual, const Tensor& weight, float eps) {
  ASSERT(input.shape() == residual.shape());
  int64_t dim = input.size(-1);
  int64_t numRows = input.numel() / dim;

  T* inputPtr = input.dataPtr<T>();
  T* residualPtr = residual.dataPtr<T>();
  const T* weightPtr = weight.dataPtr<T>();

  for (int64_t row = 0; row < numRows; row++) {
    int64_t base = row * dim;

    // add residual + accumulate sum‑of‑squares
    float sumSq = 0.f;
    for (int64_t i = 0; i < dim; i++) {
      float r = static_cast<float>(inputPtr[base + i]) + static_cast<float>(residualPtr[base + i]);
      residualPtr[base + i] = static_cast<T>(r);
      sumSq += r * r;
    }

    float invRms = 1.f / std::sqrt(sumSq / static_cast<float>(dim) + eps);

    // normalize + affine
    for (int64_t i = 0; i < dim; i++) {
      auto r = static_cast<float>(residualPtr[base + i]);
      inputPtr[base + i] = static_cast<T>(r * invRms * static_cast<float>(weightPtr[i]));
    }
  }
}

}  // namespace tinytorch::op
