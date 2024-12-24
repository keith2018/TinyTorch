/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Module.h"

#include "Function.h"
#include "Init.h"

namespace TinyTorch::nn {

std::vector<Tensor *> Module::parameters() {
  std::vector<Tensor *> ret;
  for (auto &module : subModules_) {
    for (auto p : module->parameters()) {
      ret.push_back(p);
    }
  }
  return ret;
}

void Module::resetParameters() {
  for (auto &module : subModules_) {
    module->resetParameters();
  }
}

void Module::zeroGrad() {
  for (auto &module : subModules_) {
    module->zeroGrad();
  }
}

Tensor Sequential::forward(const Tensor &input) const {
  Tensor ret = {input};
  for (auto &module : modules_) {
    ret = (*module)(ret);
  }

  return ret;
}

std::vector<Tensor *> Sequential::parameters() {
  std::vector<Tensor *> ret;
  for (auto &module : modules_) {
    for (auto p : module->parameters()) {
      ret.push_back(p);
    }
  }
  return ret;
}

void Sequential::resetParameters() {
  for (auto &module : modules_) {
    module->resetParameters();
  }
}

void Sequential::zeroGrad() {
  for (auto &module : modules_) {
    module->zeroGrad();
  }
}

void Sequential::setTraining(bool mode) {
  Module::setTraining(mode);
  for (auto &module : modules_) {
    module->train(mode);
  }
}

Linear::Linear(int32_t inFeatures, int32_t outFeatures, bool bias)
    : inFeatures_(inFeatures), outFeatures_(outFeatures), useBias_(bias) {
  weights_ = Tensor::shape({outFeatures, inFeatures}, true);
  if (bias) {
    bias_ = Tensor::shape({outFeatures}, true);
  }
  Linear::resetParameters();
}

Tensor Linear::forward(const Tensor &input) const {
  return Function::linear(input, weights_, bias_);
}

std::vector<Tensor *> Linear::parameters() {
  if (useBias_) {
    return {&weights_, &bias_};
  }
  return {&weights_};
}

void Linear::resetParameters() {
  Init::kaimingUniform(weights_, std::sqrtf(5.f));
  if (useBias_) {
    auto fanIn = Init::calculateFan(weights_).first;
    const auto bound = fanIn > 0 ? 1.f / std::sqrtf((float)fanIn) : 0;
    Init::uniform(bias_, -bound, bound);
  }
}

void Linear::zeroGrad() {
  weights_.zeroGrad();
  if (useBias_) {
    bias_.zeroGrad();
  }
}

Tensor Flatten::forward(const Tensor &input) const {
  return Function::flatten(input, startDim_, endDim_);
}

Tensor Relu::forward(const Tensor &input) const {
  return Function::relu(input);
}

Tensor Dropout::forward(const Tensor &input) const {
  return Function::dropout(input, p_, training_);
}

Tensor Softmax::forward(const Tensor &input) const {
  return Function::softmax(input, dim_);
}

Tensor LogSoftmax::forward(const Tensor &input) const {
  return Function::logSoftmax(input, dim_);
}

Tensor MaxPool2D::forward(const Tensor &input) const {
  return Function::maxPool2d(input, kernelSize_, stride_, padding_);
}

Conv2D::Conv2D(int32_t inFeatures, int32_t outFeatures, Size2D kernelSize,
               Size2D stride, Size2D padding, bool bias)
    : inFeatures_(inFeatures),
      outFeatures_(outFeatures),
      kernelSize_(kernelSize),
      stride_(stride),
      padding_(padding),
      useBias_(bias) {
  weights_ = Tensor::shape(
      {outFeatures, inFeatures, kernelSize_.h, kernelSize_.w}, true);
  if (bias) {
    bias_ = Tensor::shape({outFeatures}, true);
  }
  Conv2D::resetParameters();
}

Tensor Conv2D::forward(const Tensor &input) const {
  return Function::conv2d(input, weights_, bias_, stride_, padding_);
}

std::vector<Tensor *> Conv2D::parameters() {
  if (useBias_) {
    return {&weights_, &bias_};
  }
  return {&weights_};
}

void Conv2D::resetParameters() {
  Init::kaimingUniform(weights_, std::sqrtf(5.f));
  if (useBias_) {
    auto fanIn = Init::calculateFan(weights_).first;
    if (fanIn != 0) {
      const auto bound = 1.f / std::sqrtf((float)fanIn);
      Init::uniform(bias_, -bound, bound);
    }
  }
}

void Conv2D::zeroGrad() {
  weights_.zeroGrad();
  if (useBias_) {
    bias_.zeroGrad();
  }
}

}  // namespace TinyTorch::nn
