/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Loss.h"

#include "Function.h"

namespace tinytorch::nn {

Tensor MSELoss::forward(Tensor& input, Tensor& target) {
  return Function::mseLoss(input, target, reduction_);
}

Tensor NLLLoss::forward(Tensor& input, Tensor& target) {
  return Function::nllloss(input, target, reduction_);
}

}  // namespace tinytorch::nn
