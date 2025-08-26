/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Basic.h"

#include "Functions.h"
#include "Initializer.h"

namespace tinytorch::nn {

Tensor Flatten::forward(const Tensor &input) { return function::flatten(input, startDim_, endDim_); }

Tensor Relu::forward(const Tensor &input) { return function::relu(input); }

Tensor Gelu::forward(const Tensor &input) { return function::gelu(input); }

Tensor Silu::forward(const Tensor &input) { return function::silu(input); }

Tensor Dropout::forward(const Tensor &input) { return function::dropout(input, p_, training_); }

Tensor Softmax::forward(const Tensor &input) { return function::softmax(input, dim_); }

Tensor LogSoftmax::forward(const Tensor &input) { return function::logSoftmax(input, dim_); }

Tensor MaxPool2D::forward(const Tensor &input) { return function::maxPool2d(input, kernel_, stride_, padding_); }

Conv2D::Conv2D(int64_t inFeatures, int64_t outFeatures, Dim2D kernel, Dim2D stride, Dim2D padding, bool bias,
               Options options)
    : kernel_(kernel), stride_(stride), padding_(padding), useBias_(bias) {
  options.requiresGrad(true);
  weight_ = Tensor::empty({outFeatures, inFeatures, kernel_.h, kernel_.w}, options);
  if (bias) {
    bias_ = Tensor::empty({outFeatures}, options);
  }
}

Tensor Conv2D::forward(const Tensor &input) { return function::conv2d(input, weight_, bias_, stride_, padding_); }

std::vector<std::pair<std::string, TensorPtr>> Conv2D::namedParameters_() {
  if (useBias_) {
    return {{"weight", &weight_}, {"bias", &bias_}};
  }
  return {{"weight", &weight_}};
}

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

Linear::Linear(int64_t inFeatures, int64_t outFeatures, bool bias, Options options) : useBias_(bias) {
  options.requiresGrad(true);
  weight_ = Tensor::empty({outFeatures, inFeatures}, options);
  if (bias) {
    bias_ = Tensor::empty({outFeatures}, options);
  }
}

Tensor Linear::forward(const Tensor &input) { return function::linear(input, weight_, bias_); }

std::vector<std::pair<std::string, TensorPtr>> Linear::namedParameters_() {
  if (useBias_) {
    return {{"weight", &weight_}, {"bias", &bias_}};
  }
  return {{"weight", &weight_}};
}

void Linear::resetParameters() {
  Initializer::kaimingUniform(weight_, std::sqrt(5.f));
  if (useBias_) {
    auto fanIn = Initializer::calculateFan(weight_).first;
    const auto bound = fanIn > 0 ? 1.f / std::sqrt(static_cast<float>(fanIn)) : 0;
    Initializer::uniform(bias_, -bound, bound);
  }
}

Embedding::Embedding(int64_t numEmbeddings, int64_t embeddingDim, Options options) {
  options.requiresGrad(true);
  weight_ = Tensor::empty({numEmbeddings, embeddingDim}, options);
}

Tensor Embedding::forward(const Tensor &input) { return function::embedding(input, weight_); }

std::vector<std::pair<std::string, TensorPtr>> Embedding::namedParameters_() { return {{"weight", &weight_}}; }

void Embedding::resetParameters() { Initializer::normal(weight_); }

RoPE::RoPE(int64_t headDim, int64_t contextLength, float thetaBase, std::optional<RopeScalingConfig> scaling,
           Options options)
    : headDim_(headDim), contextLength_(contextLength), thetaBase_(thetaBase), scaling_(scaling), options_(options) {
  RoPE::resetParameters();
}

Tensor RoPE::forward(const Tensor &input) { return function::ropeApply(input, rope_); }

Tensor RoPE::forward(const Tensor &input, int64_t offset) { return function::ropeApply(input, rope_, offset); }

void RoPE::resetParameters() { rope_ = op::ropeInit(headDim_, contextLength_, thetaBase_, scaling_, options_); }

std::vector<std::pair<std::string, TensorPtr>> RoPE::namedStates_() {
  return {{"cos", &rope_.first}, {"sin", &rope_.second}};
}

}  // namespace tinytorch::nn