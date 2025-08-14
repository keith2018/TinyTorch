/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Norm.h"

#include "Function/FuncNNLayer.h"
#include "Initializer.h"

namespace tinytorch::nn {

LayerNorm::LayerNorm(IntArrayView normalizedShape, float eps, bool bias, Options options)
    : normalizedShape_(normalizedShape), eps_(eps), useBias_(bias) {
  options.requiresGrad(true);
  weight_ = Tensor::empty(normalizedShape, options);
  if (bias) {
    bias_ = Tensor::empty(normalizedShape, options);
  }
}

Tensor LayerNorm::forward(const Tensor &input) {
  return function::layerNorm(input, normalizedShape_, weight_, bias_, eps_);
}

std::vector<std::pair<std::string, TensorPtr>> LayerNorm::namedParameters_() {
  if (useBias_) {
    return {{"weight", &weight_}, {"bias", &bias_}};
  }
  return {{"weight", &weight_}};
}

void LayerNorm::resetParameters() {
  Initializer::ones(weight_);
  if (useBias_) {
    Initializer::zeros(bias_);
  }
}

RMSNorm::RMSNorm(IntArrayView normalizedShape, float eps, Options options)
    : normalizedShape_(normalizedShape), eps_(eps) {
  options.requiresGrad(true);
  weight_ = Tensor::empty(normalizedShape, options);
}

Tensor RMSNorm::forward(const Tensor &input) { return function::rmsNorm(input, normalizedShape_, weight_, eps_); }

std::vector<std::pair<std::string, TensorPtr>> RMSNorm::namedParameters_() { return {{"weight", &weight_}}; }

void RMSNorm::resetParameters() { Initializer::ones(weight_); }

}  // namespace tinytorch::nn