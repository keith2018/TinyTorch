/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Module.h"

#include "Functions.h"
#include "Initializer.h"

namespace tinytorch::nn {

Tensor Sequential::forward(Tensor &input) {
  Tensor ret = {input};
  for (auto &module : modules_) {
    ret = (*module)(ret);
  }

  return ret;
}

Linear::Linear(int64_t inFeatures, int64_t outFeatures, bool bias, Options options) : useBias_(bias) {
  options.requiresGrad(true);
  weight_ = Tensor::empty({outFeatures, inFeatures}, options);
  if (bias) {
    bias_ = Tensor::empty({outFeatures}, options);
  }
  Linear::resetParameters();
}

Tensor Linear::forward(Tensor &input) { return function::linear(input, weight_, bias_); }

std::vector<std::pair<std::string, TensorPtr>> Linear::namedParameters_() {
  if (useBias_) {
    return {{"weight", &weight_}, {"bias", &bias_}};
  }
  return {{"weight", &weight_}};
}

std::vector<std::pair<std::string, TensorPtr>> Linear::namedStates_() { return namedParameters_(); }

void Linear::resetParameters() {
  Initializer::kaimingUniform(weight_, std::sqrt(5.f));
  if (useBias_) {
    auto fanIn = Initializer::calculateFan(weight_).first;
    const auto bound = fanIn > 0 ? 1.f / std::sqrt(static_cast<float>(fanIn)) : 0;
    Initializer::uniform(bias_, -bound, bound);
  }
}

Tensor Flatten::forward(Tensor &input) { return function::flatten(input, startDim_, endDim_); }

Tensor Relu::forward(Tensor &input) { return function::relu(input); }

Tensor Gelu::forward(Tensor &input) { return function::gelu(input); }

Tensor Silu::forward(Tensor &input) { return function::silu(input); }

Tensor Dropout::forward(Tensor &input) { return function::dropout(input, p_, training_); }

Tensor Softmax::forward(Tensor &input) { return function::softmax(input, dim_); }

Tensor LogSoftmax::forward(Tensor &input) { return function::logSoftmax(input, dim_); }

Tensor MaxPool2D::forward(Tensor &input) { return function::maxPool2d(input, kernel_, stride_, padding_); }

Conv2D::Conv2D(int64_t inFeatures, int64_t outFeatures, Dim2D kernel, Dim2D stride, Dim2D padding, bool bias,
               Options options)
    : kernel_(kernel), stride_(stride), padding_(padding), useBias_(bias) {
  options.requiresGrad(true);
  weight_ = Tensor::empty({outFeatures, inFeatures, kernel_.h, kernel_.w}, options);
  if (bias) {
    bias_ = Tensor::empty({outFeatures}, options);
  }
  Conv2D::resetParameters();
}

Tensor Conv2D::forward(Tensor &input) { return function::conv2d(input, weight_, bias_, stride_, padding_); }

std::vector<std::pair<std::string, TensorPtr>> Conv2D::namedParameters_() {
  if (useBias_) {
    return {{"weight", &weight_}, {"bias", &bias_}};
  }
  return {{"weight", &weight_}};
}
std::vector<std::pair<std::string, TensorPtr>> Conv2D::namedStates_() { return namedParameters_(); }

void Conv2D::resetParameters() {
  Initializer::kaimingUniform(weight_, std::sqrt(5.f));
  if (useBias_) {
    auto fanIn = Initializer::calculateFan(weight_).first;
    if (fanIn != 0) {
      const auto bound = 1.f / std::sqrt(static_cast<float>(fanIn));
      Initializer::uniform(bias_, -bound, bound);
    }
  }
}

Embedding::Embedding(int64_t numEmbeddings, int64_t embeddingDim, Options options) {
  options.requiresGrad(true);
  weight_ = Tensor::empty({numEmbeddings, embeddingDim}, options);
  Embedding::resetParameters();
}

Tensor Embedding::forward(Tensor &input) { return function::embedding(input, weight_); }

std::vector<std::pair<std::string, TensorPtr>> Embedding::namedParameters_() { return {{"weight", &weight_}}; }

std::vector<std::pair<std::string, TensorPtr>> Embedding::namedStates_() { return namedParameters_(); }

void Embedding::resetParameters() { Initializer::normal(weight_); }

LayerNorm::LayerNorm(IntArrayView normalizedShape, float eps, bool bias, Options options)
    : normalizedShape_(normalizedShape), eps_(eps), useBias_(bias) {
  options.requiresGrad(true);
  weight_ = Tensor::empty(normalizedShape, options);
  if (bias) {
    bias_ = Tensor::empty(normalizedShape, options);
  }
  LayerNorm::resetParameters();
}

Tensor LayerNorm::forward(Tensor &input) {
  return function::layerNorm(input, normalizedShape_, weight_, bias_, eps_);
}

std::vector<std::pair<std::string, TensorPtr>> LayerNorm::namedParameters_() {
  if (useBias_) {
    return {{"weight", &weight_}, {"bias", &bias_}};
  }
  return {{"weight", &weight_}};
}

std::vector<std::pair<std::string, TensorPtr>> LayerNorm::namedStates_() { return namedParameters_(); }

void LayerNorm::resetParameters() {
  Initializer::ones(weight_);
  if (useBias_) {
    Initializer::zeros(bias_);
  }
}

}  // namespace tinytorch::nn
