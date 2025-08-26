/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "OpNNLayer.h"
#include "Tensor/TensorIterator.h"
#include "Utils/MathUtils.h"
#include "Utils/RandomGenerator.h"

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
Tensor dropoutOpCpuImpl(const Tensor& self, float p) {
  Tensor out(self.shape(), self.options().noGrad());
  const auto* selfPtr = self.dataPtr<T>();
  auto* outPtr = out.dataPtr<T>();
  auto generator = RandomGeneratorCPU::getGenerator();
  std::bernoulli_distribution distribution(p);
  for (int64_t i = 0; i < self.numel(); i++) {
    outPtr[i] = static_cast<T>(static_cast<float>(selfPtr[i]) * static_cast<float>(distribution(generator)) / p);
  }
  return out;
}

template <typename T>
Tensor dropoutMaskedOpCpuImpl(const Tensor& self, const Tensor& mask, float p) {
  TensorIteratorCpu iterator(self, mask);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Tensor out(outShape, self.options().noGrad());
  iterator.template forEach<T>(out, [p](const T& a, const T& b) -> T { return a * b / p; });
  return out;
}

template <typename T, NormType normType>
Tensor normOpCpuImplDetail(const Tensor& self, IntArrayView normalizedShape, const Tensor& weight, const Tensor& bias,
                           float eps) {
  int64_t dim = self.shape().back();
  int64_t numRows = self.numel() / dim;
  ASSERT(normalizedShape.size() == 1);
  ASSERT(normalizedShape.front() == dim);

  Tensor out(self.shape(), self.options().noGrad());

  const auto* inputPtr = self.dataPtr<T>();
  const auto* weightPtr = weight.defined() ? weight.dataPtr<T>() : nullptr;
  const auto* biasPtr = bias.defined() ? bias.dataPtr<T>() : nullptr;
  auto* outPtr = out.dataPtr<T>();

  for (auto i = 0; i < numRows; i++) {
    const T* src = inputPtr + i * dim;
    T* dst = outPtr + i * dim;

    if constexpr (normType == NormType::LayerNorm) {
      T mean = 0;
      for (auto j = 0; j < dim; j++) {
        mean += src[j];
      }
      mean /= static_cast<T>(dim);

      T var = 0;
      for (auto j = 0; j < dim; j++) {
        T diff = src[j] - mean;
        var += diff * diff;
      }
      var /= static_cast<T>(dim);
      T invStd = T(1) / std::sqrt(var + eps);

      for (auto j = 0; j < dim; j++) {
        T normed = (src[j] - mean) * invStd;
        if (weightPtr) normed *= weightPtr[j];
        if (biasPtr) normed += biasPtr[j];
        dst[j] = normed;
      }
    } else {
      UNUSED(biasPtr);
      T meanSquare = 0;
      for (auto j = 0; j < dim; j++) {
        meanSquare += src[j] * src[j];
      }
      meanSquare /= static_cast<T>(dim);
      T invRms = T(1) / std::sqrt(meanSquare + eps);

      for (auto j = 0; j < dim; j++) {
        T normed = src[j] * invRms;
        if (weightPtr) normed *= weightPtr[j];
        dst[j] = normed;
      }
    }
  }

  return out;
}

template <typename T>
Tensor layerNormOpCpuImpl(const Tensor& self, IntArrayView normalizedShape, const Tensor& weight, const Tensor& bias,
                          float eps) {
  return normOpCpuImplDetail<T, NormType::LayerNorm>(self, normalizedShape, weight, bias, eps);
}

template <typename T>
Tensor rmsNormOpCpuImpl(const Tensor& self, IntArrayView normalizedShape, const Tensor& weight, float eps) {
  return normOpCpuImplDetail<T, NormType::RMSNorm>(self, normalizedShape, weight, {}, eps);
}

template <typename T>
TensorPair ropeInitOpCpuImpl(int64_t headDim, int64_t contextLength, float thetaBase,
                             std::optional<RopeScalingConfig> scaling, Options options) {
  ASSERT(!options.requiresGrad_);
  ASSERT(options.device_.type == DeviceType::CPU);

  ASSERT(headDim % 2 == 0);
  int64_t halfDim = headDim >> 1;

  // inverse frequency
  Tensor invFreq({halfDim}, options);
  auto* invFreqPtr = invFreq.dataPtr<T>();
  for (int64_t i = 0; i < halfDim; i++) {
    invFreqPtr[i] = 1.f / std::pow(thetaBase, static_cast<float>(i << 1) / static_cast<float>(headDim));
  }

  // apply scaling if needed
  if (scaling.has_value()) {
    auto originCtxLen = static_cast<float>(scaling->originalContextLength);
    auto lowWaveLen = originCtxLen / scaling->lowFreqFactor;
    auto highWaveLen = originCtxLen / scaling->highFreqFactor;
    for (int64_t i = 0; i < halfDim; i++) {
      auto waveLen = 2.f * static_cast<float>(M_PI) / invFreqPtr[i];
      if (waveLen > lowWaveLen) {
        invFreqPtr[i] /= scaling->factor;
      } else if (waveLen < highWaveLen) {
        // do nothing
      } else {
        auto smoothFactor =
            (originCtxLen / waveLen - scaling->lowFreqFactor) / (scaling->highFreqFactor - scaling->lowFreqFactor);
        auto scaled = invFreqPtr[i] / scaling->factor;
        invFreqPtr[i] = (1.f - smoothFactor) * scaled + smoothFactor * invFreqPtr[i];
      }
    }
  }

  // precompute cos/sin
  Tensor cos({contextLength, headDim}, options);
  Tensor sin({contextLength, headDim}, options);
  auto* cosPtr = cos.dataPtr<T>();
  auto* sinPtr = sin.dataPtr<T>();

  for (int64_t pos = 0; pos < contextLength; pos++) {
    for (int64_t i = 0; i < halfDim; i++) {
      float angle = static_cast<T>(pos) * invFreqPtr[i];
      int64_t offset1 = pos * headDim + i;
      int64_t offset2 = pos * headDim + halfDim + i;
      cosPtr[offset1] = std::cos(angle);
      sinPtr[offset1] = std::sin(angle);
      cosPtr[offset2] = cosPtr[offset1];
      sinPtr[offset2] = sinPtr[offset1];
    }
  }

  return {cos, sin};
}

template <typename T>
Tensor ropeApplyOpCpuImpl(const Tensor& input, const TensorPair& rope, int64_t offset) {
  const auto& shape = input.shape();  // [batch, numHead, seqLen, headDim]
  ASSERT(shape.size() == 4);

  int64_t batch = shape[0];
  int64_t numHead = shape[1];
  int64_t seqLen = shape[2];
  int64_t headDim = shape[3];

  ASSERT(headDim % 2 == 0);
  int64_t halfDim = headDim >> 1;

  const auto* inputPtr = input.dataPtr<T>();
  const auto* cosPtr = rope.first.dataPtr<T>();
  const auto* sinPtr = rope.second.dataPtr<T>();

  Tensor out(shape, input.options().noGrad());
  auto* outPtr = out.dataPtr<T>();

  for (int64_t b = 0; b < batch; b++) {
    for (int64_t h = 0; h < numHead; h++) {
      for (int64_t t = 0; t < seqLen; t++) {
        int64_t base = ((b * numHead + h) * seqLen + t) * headDim;
        const T* xPtr = inputPtr + base;
        T* yPtr = outPtr + base;

        int64_t posIndex = offset + t;
        const T* cosRow = cosPtr + posIndex * headDim;
        const T* sinRow = sinPtr + posIndex * headDim;

        for (int64_t i = 0; i < halfDim; i++) {
          float x1 = xPtr[i];
          float x2 = xPtr[halfDim + i];
          float c = cosRow[i];
          float s = sinRow[i];
          yPtr[i] = x1 * c - x2 * s;
          yPtr[halfDim + i] = x2 * c + x1 * s;
        }
      }
    }
  }
  return out;
}

}  // namespace tinytorch::op
