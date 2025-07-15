/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "OpFilling.h"

namespace tinytorch::op {

template <typename T>
void fillOpOffsetCpuImpl(Tensor& self, const Tensor& val, int64_t offset, int64_t count) {
  ASSERT(val.isScalar());
  self.copyOnWrite();
  T* selfPtr = self.dataPtr<T>();
  T valToFill = val.item<T>();
  std::fill_n(selfPtr + offset, count, valToFill);
}

template <typename T>
void fillOpCpuImpl(Tensor& self, const Scalar& val) {
  self.copyOnWrite();
  T* selfPtr = self.dataPtr<T>();
  std::fill_n(selfPtr, self.numel(), val.to<T>());
}

template <typename T>
void fillOpMaskedCpuImplDetail(Tensor& out, const Tensor& self, const Tensor& mask, const Scalar& val) {
  T* outPtr = out.dataPtr<T>();
  const T* selfPtr = self.dataPtr<T>();
  const auto* maskPtr = mask.dataPtr<DTypeToType_t<DType::Bool>>();
  T valToFill = val.to<T>();
  for (int64_t i = 0; i < self.numel(); i++) {
    outPtr[i] = maskPtr[i] ? valToFill : selfPtr[i];
  }
}

template <typename T>
Tensor fillOpMaskedCpuImpl(const Tensor& self, const Tensor& mask, const Scalar& val) {
  ASSERT(mask.dtype() == DType::Bool);
  ASSERT(mask.shape() == self.shape());
  Tensor out(self.shape(), self.options().noGrad());
  fillOpMaskedCpuImplDetail<T>(out, self, mask, val);
  return out;
}

template <typename T>
void fillOpMaskedOutCpuImpl(Tensor& out, const Tensor& self, const Tensor& mask, const Scalar& val) {
  ASSERT(mask.dtype() == DType::Bool);
  ASSERT(self.shape() == mask.shape());
  out.copyOnWrite();
  fillOpMaskedCpuImplDetail<T>(out, self, mask, val);
}

template <typename T>
void fillOpMaskedInplaceCpuImpl(Tensor& self, const Tensor& mask, const Scalar& val) {
  ASSERT(mask.dtype() == DType::Bool);
  ASSERT(mask.shape() == self.shape());
  self.copyOnWrite();
  fillOpMaskedCpuImplDetail<T>(self, self, mask, val);
}

template <typename T>
void fillOpLinSpaceCpuImpl(Tensor& self, const Scalar& start, const Scalar& step, int64_t steps) {
  ASSERT(self.numel() == steps);
  self.copyOnWrite();
  auto* selfPtr = self.dataPtr<T>();
  T startVal = start.to<T>();
  T stepVal = step.to<T>();
  for (int64_t i = 0; i < steps; i++) {
    selfPtr[i] = startVal + static_cast<T>(i) * stepVal;
  }
}

}  // namespace tinytorch::op
