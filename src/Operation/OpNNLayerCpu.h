/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cmath>

#include "OpNNLayer.h"
#include "Tensor/TensorIterator.h"

namespace tinytorch::op {

template <typename T, SoftmaxType type>
void softmaxForwardCpuImpl(Tensor& out, const Tensor& self, int64_t dim) {
  ASSERT(out.shape() == self.shape());
  auto info = getSoftmaxDimInfo(self, dim);

  const T* selfPtr = self.dataPtr<T>();
  T* outPtr = out.dataPtr<T>();

  for (int64_t outer = 0; outer < info.outerSize; outer++) {
    int64_t outerBase = outer * info.dimSize * info.innerSize;
    for (int64_t i = 0; i < info.innerSize; i++) {
      int64_t base = outerBase + i;
      // max
      T maxVal = -std::numeric_limits<T>::max();
      int64_t offset = base;
      for (int64_t d = 0; d < info.dimSize; d++) {
        T v = selfPtr[offset];
        if (v > maxVal) {
          maxVal = v;
        }
        offset += info.innerSize;
      }
      // exp & sum
      T sum = 0;
      offset = base;
      for (int64_t d = 0; d < info.dimSize; d++) {
        T e = std::exp(selfPtr[offset] - maxVal);
        outPtr[offset] = e;
        sum += e;
        offset += info.innerSize;
      }
      // output
      offset = base;
      if constexpr (type == SoftmaxType::Softmax) {
        T invSum = 1.0 / sum;
        for (int64_t d = 0; d < info.dimSize; d++) {
          outPtr[offset] *= invSum;
          offset += info.innerSize;
        }
      } else {  // LogSoftmax
        T logSum = std::log(sum);
        for (int64_t d = 0; d < info.dimSize; d++) {
          outPtr[offset] = selfPtr[offset] - maxVal - logSum;
          offset += info.innerSize;
        }
      }
    }
  }
}

template <typename T>
void softmaxOpOutCpuImpl(Tensor& out, const Tensor& self, int64_t dim) {
  softmaxForwardCpuImpl<T, SoftmaxType::Softmax>(out, self, dim);
}

template <typename T>
Tensor softmaxOpCpuImpl(const Tensor& self, int64_t dim) {
  Tensor out(self.shape(), self.options().noGrad());
  softmaxOpOutCpuImpl<T>(out, self, dim);
  return out;
}

template <typename T>
Tensor softmaxOpBackwardCpuImpl(const Tensor& grad, const Tensor& output, int64_t dim) {
  ASSERT(output.shape() == grad.shape());
  auto info = getSoftmaxDimInfo(output, dim);
  Tensor out(output.shape(), output.options().noGrad());

  const T* outputPtr = output.dataPtr<T>();
  const T* gradPtr = grad.dataPtr<T>();
  T* outPtr = out.dataPtr<T>();

  for (int64_t outer = 0; outer < info.outerSize; outer++) {
    int64_t outerBase = outer * info.dimSize * info.innerSize;
    for (int64_t i = 0; i < info.innerSize; i++) {
      int64_t base = outerBase + i;
      // sum_j(y_j * dL/dy_j)
      T sum = 0;
      int64_t offset = base;
      for (int64_t d = 0; d < info.dimSize; d++) {
        sum += outputPtr[offset] * gradPtr[offset];
        offset += info.innerSize;
      }
      // dL/dx_i = y_i * (dL/dy_i - sum)
      offset = base;
      for (int64_t d = 0; d < info.dimSize; d++) {
        outPtr[offset] = outputPtr[offset] * (gradPtr[offset] - sum);
        offset += info.innerSize;
      }
    }
  }

  return out;
}

template <typename T>
void logSoftmaxOpOutCpuImpl(Tensor& out, const Tensor& self, int64_t dim) {
  softmaxForwardCpuImpl<T, SoftmaxType::LogSoftmax>(out, self, dim);
}

template <typename T>
Tensor logSoftmaxOpCpuImpl(const Tensor& self, int64_t dim) {
  Tensor out(self.shape(), self.options().noGrad());
  logSoftmaxOpOutCpuImpl<T>(out, self, dim);
  return out;
}

template <typename T>
Tensor logSoftmaxOpBackwardCpuImpl(const Tensor& grad, const Tensor& output, int64_t dim) {
  ASSERT(output.shape() == grad.shape());
  auto info = getSoftmaxDimInfo(output, dim);
  Tensor out(output.shape(), output.options().noGrad());

  const T* outputPtr = output.dataPtr<T>();
  const T* gradPtr = grad.dataPtr<T>();
  T* outPtr = out.dataPtr<T>();

  for (int64_t outer = 0; outer < info.outerSize; outer++) {
    int64_t outerBase = outer * info.dimSize * info.innerSize;
    for (int64_t i = 0; i < info.innerSize; i++) {
      int64_t base = outerBase + i;
      // sum_j(dL/dy_j)
      T sum = 0;
      int64_t offset = base;
      for (int64_t d = 0; d < info.dimSize; d++) {
        sum += gradPtr[offset];
        offset += info.innerSize;
      }
      // dL/dx_i = dL/dy_i - exp(y_i) * sum
      offset = base;
      for (int64_t d = 0; d < info.dimSize; d++) {
        outPtr[offset] = gradPtr[offset] - std::exp(outputPtr[offset]) * sum;
        offset += info.innerSize;
      }
    }
  }

  return out;
}

template <typename T>
Tensor dropoutOpCpuImpl(const Tensor& grad, const Tensor& mask, float p) {
  TensorIteratorCpu iterator(grad, mask);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Tensor out(outShape, grad.options().noGrad());
  iterator.template forEach<T>(out, [p](const T& a, const T& b) -> T { return a * b / p; });
  return out;
}

template <typename T>
Tensor layerNormOpCpuImpl(const Tensor& self, const Tensor& weight, const Tensor& bias, float eps) {
  int64_t d = self.shape().back();
  int64_t n = self.numel() / d;

  Tensor out(self.shape(), self.options().noGrad());

  const auto* selfPtr = self.dataPtr<T>();
  const auto* weightPtr = weight.defined() ? weight.dataPtr<T>() : nullptr;
  const auto* biasPtr = bias.defined() ? bias.dataPtr<T>() : nullptr;

  auto* outPtr = out.dataPtr<T>();

  for (auto i = 0; i < n; i++) {
    const T* src = selfPtr + i * d;
    T* dst = outPtr + i * d;
    // mean
    T mean = 0.f;
    for (auto j = 0; j < d; j++) {
      mean += src[j];
    }
    mean /= static_cast<T>(d);
    // var
    T var = 0.f;
    for (auto j = 0; j < d; j++) {
      T diff = src[j] - mean;
      var += diff * diff;
    }
    var /= static_cast<T>(d);
    T invStd = 1.f / std::sqrt(var + eps);
    // norm + affine
    for (auto j = 0; j < d; j++) {
      T normed = (src[j] - mean) * invStd;
      if (weightPtr) normed *= weightPtr[j];
      if (biasPtr) normed += biasPtr[j];
      dst[j] = normed;
    }
  }

  return out;
}

}  // namespace tinytorch::op
