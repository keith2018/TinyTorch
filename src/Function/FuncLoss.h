/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Operations.h"
#include "Tensor/Function.h"

namespace tinytorch::function {

#define DEFINE_LOSS_FUNCTION(CLASS_NAME, FORWARD_OP, BACKWARD_OP)                                                     \
  class CLASS_NAME : public Function<CLASS_NAME> {                                                                    \
   public:                                                                                                            \
    static Tensor forward(AutogradContext* ctx, const Tensor& input, const Tensor& target, LossReduction reduction) { \
      auto output = FORWARD_OP(input, target, reduction);                                                             \
      if (ctx) {                                                                                                      \
        ctx->pushData(reduction);                                                                                     \
      }                                                                                                               \
      return output;                                                                                                  \
    }                                                                                                                 \
                                                                                                                      \
    static void backward(AutogradContext* ctx, const Tensor& grad) {                                                  \
      auto& input = ctx->savedInputs[0];                                                                              \
      auto& target = ctx->savedInputs[1];                                                                             \
      auto reduction = ctx->popData().toEnum<LossReduction>();                                                        \
                                                                                                                      \
      if (input.requiresGrad()) {                                                                                     \
        input.addGrad(BACKWARD_OP(grad, input, target, reduction));                                                   \
      }                                                                                                               \
      ASSERT(!target.requiresGrad());                                                                                 \
    }                                                                                                                 \
  };

DEFINE_LOSS_FUNCTION(FuncMSELoss, op::mseLoss, op::mseLossBackward);

DEFINE_LOSS_FUNCTION(FuncNLLLoss, op::nllLoss, op::nllLossBackward);

inline Tensor mseLoss(const Tensor& input, const Tensor& target, LossReduction reduction = LossReduction::MEAN) {
  return FuncMSELoss::apply(input, target, reduction);
}

inline Tensor nllLoss(const Tensor& input, const Tensor& target, LossReduction reduction = LossReduction::MEAN) {
  return FuncNLLLoss::apply(input, target, reduction);
}

}  // namespace tinytorch::function
