/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor/Dispatch.h"

namespace tinytorch::op {

enum class SoftmaxType : int8_t {
  Softmax,
  LogSoftmax
};

enum class NormType : int8_t {
  LayerNorm,
  RMSNorm
};

struct SoftmaxDimInfo {
  int64_t outerSize;
  int64_t dimSize;
  int64_t innerSize;
};

SoftmaxDimInfo getSoftmaxDimInfo(const Tensor& self, int64_t dim);

using SoftmaxOpFn = Tensor (*)(const Tensor& self, int64_t dim);
using SoftmaxOpOutFn = void (*)(Tensor& out, const Tensor& self, int64_t dim);
using SoftmaxOpBackwardFn = Tensor (*)(const Tensor& grad, const Tensor& output, int64_t dim);

using DropoutOpFn = Tensor (*)(const Tensor& self, float p);
using DropoutMaskedOpFn = Tensor (*)(const Tensor& self, const Tensor& mask, float p);

using LayerNormOpFn = Tensor (*)(const Tensor& self, IntArrayView normalizedShape, const Tensor& weight,
                                 const Tensor& bias, float eps);

using RMSNormOpFn = Tensor (*)(const Tensor& self, IntArrayView normalizedShape, const Tensor& weight, float eps);

// softmax
DEFINE_OP(softmax, SoftmaxOpFn);
DEFINE_OP(softmaxOut, SoftmaxOpOutFn);
DEFINE_OP(softmaxBackward, SoftmaxOpBackwardFn);

// logSoftmax
DEFINE_OP(logSoftmax, SoftmaxOpFn);
DEFINE_OP(logSoftmaxOut, SoftmaxOpOutFn);
DEFINE_OP(logSoftmaxBackward, SoftmaxOpBackwardFn);

// dropout
DEFINE_OP(dropout, DropoutOpFn);
DEFINE_OP(dropoutMasked, DropoutMaskedOpFn);

// layerNorm
DEFINE_OP(layerNorm, LayerNormOpFn);

// rmsNorm
DEFINE_OP(rmsNorm, RMSNormOpFn);

void registerNNLayerCpu();
STATIC_CALL(registerNNLayerCpu);

#ifdef USE_CUDA
void registerNNLayerCuda();
STATIC_CALL(registerNNLayerCuda);
#endif

}  // namespace tinytorch::op
