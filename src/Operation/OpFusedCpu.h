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

}  // namespace tinytorch::op
