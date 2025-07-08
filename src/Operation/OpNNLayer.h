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

struct SoftmaxDimInfo {
  int64_t outerSize;
  int64_t dimSize;
  int64_t innerSize;
};

SoftmaxDimInfo getSoftmaxDimInfo(const Tensor& self, int64_t dim);

using SoftmaxOpFn = Tensor (*)(const Tensor& self, int64_t dim);
using SoftmaxOpOutFn = void (*)(Tensor& out, const Tensor& self, int64_t dim);
using SoftmaxOpBackwardFn = Tensor (*)(const Tensor& grad, const Tensor& output, int64_t dim);

using DropoutOpFn = Tensor (*)(const Tensor& grad, const Tensor& mask, float p);

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

void registerNNLayerCpu();
void registerNNLayerCuda();

STATIC_CALL(registerNNLayerCpu);
STATIC_CALL(registerNNLayerCuda);

}  // namespace tinytorch::op
