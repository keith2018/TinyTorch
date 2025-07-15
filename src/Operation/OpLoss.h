/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Loss.h"
#include "Tensor/Dispatch.h"

namespace tinytorch::op {

using LossOpFn = Tensor (*)(const Tensor& input, const Tensor& target, LossReduction reduction);
using LossOpBackwardFn = Tensor (*)(const Tensor& grad, const Tensor& input, const Tensor& target,
                                    LossReduction reduction);

// mseLoss
DEFINE_OP(mseLoss, LossOpFn);
DEFINE_OP(mseLossBackward, LossOpBackwardFn);

// nllLoss
DEFINE_OP(nllLoss, LossOpFn);
DEFINE_OP(nllLossBackward, LossOpBackwardFn);

void registerLossCpu();
STATIC_CALL(registerLossCpu);

#ifdef USE_CUDA
void registerLossCuda();
STATIC_CALL(registerLossCuda);
#endif

}  // namespace tinytorch::op
