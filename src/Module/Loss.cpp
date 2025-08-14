/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Loss.h"

#include "Functions.h"

namespace tinytorch::nn {

Tensor MSELoss::forward(Tensor& input, Tensor& target) { return function::mseLoss(input, target, reduction_); }

Tensor NLLLoss::forward(Tensor& input, Tensor& target) { return function::nllLoss(input, target, reduction_); }

}  // namespace tinytorch::nn
