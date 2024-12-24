/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Loss.h"

#include "Function.h"

namespace TinyTorch::nn {

Tensor MSELoss::forward(const Tensor& input, const Tensor& target) const {
  return Function::mseLoss(input, target, reduction_);
}

Tensor NLLLoss::forward(const Tensor& input, const Tensor& target) const {
  return Function::nllloss(input, target, reduction_);
}

}  // namespace TinyTorch::nn
